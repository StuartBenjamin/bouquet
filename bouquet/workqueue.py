"""Shared-filesystem work queue: many workers on many nodes claim units by ``mkdir``.

Stdlib only. Correct on NFSv4 and Lustre: ``mkdir``'s EEXIST is the claim, ``rename`` is the
single point that serialises stealing an expired lease. Vendored byte-identically as
``bouquet/workqueue.py`` and ``tearing_physics_suite/drivers/fsqueue.py``; edit the eq_stab copy
(``D3D_V1/FUSE_IDAlite_BOUQUET/common/fsqueue.py``) and copy it over.

Layout under ``root`` (one queue per tool)::

    config.json                 policy overrides (STALE_S, MAX_ATTEMPTS, MAX_FAILURES, ...)
    manifest.d/<batch>.json     {"units": [{"id": ..., "est": ..., ...}]}; append-only
    claims/<id>/                mkdir == the claim; owner.json, heartbeat (epoch seconds)
    claims/.dead/               stolen claims
    leases/<name>/              named leases (e.g. a feeder), same protocol as claims
    state/<id>/attempt_*.json   one per started attempt
    state/<id>/fail_*.json      one per recorded failure
    done/<id>                   terminal: {"status": "done" | "failed", ...}
    workers/<worker>.json       worker status; <job>.broken markers
    STOP, DRAIN                 STOP: workers exit after their unit. DRAIN: no new claims
"""
import json
import os
import random
import socket
import sys
import threading
import time
import traceback
import uuid

DEFAULTS = dict(STALE_S=300.0, HEARTBEAT_S=30.0, MAX_ATTEMPTS=4, MAX_FAILURES=2,
                MAX_CONSECUTIVE_FAILURES=3, BROKEN_JOBS_STOP=2, BROKEN_WINDOW_S=3600.0,
                RESCAN_S=600.0)
HEARTBEAT_EXPIRED = 1   # epoch written on the way out: the lease is stealable at once


def _now():
    return time.time()


def _tag():
    return f'{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}'


def _ls(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def write_json(path, obj):
    """Atomic JSON write (tmp + rename)."""
    tmp = f'{path}.tmp.{_tag()}'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def error_signature(exc):
    """Exception type + first line of its message: groups failures in ``status``."""
    msg = str(exc).strip().splitlines()
    return f'{type(exc).__name__}: {msg[0][:200] if msg else ""}'


def slurm_info():
    return {k: os.environ.get(f'SLURM_{k.upper()}', '') for k in ('job_id', 'array_task_id', 'procid')}


class Queue:
    """One tool's queue at ``root``. Policy values: DEFAULTS, then ``config.json``, then kwargs."""

    def __init__(self, root, **policy):
        self.root = os.path.abspath(root)
        for d in ('manifest.d', 'claims', 'leases', 'state', 'done', 'workers'):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.policy = dict(DEFAULTS)
        self.policy.update({k: v for k, v in (read_json(self.path('config.json')) or {}).items() if k in DEFAULTS})
        self.policy.update({k: v for k, v in policy.items() if v is not None})
        self._units, self._seen, self._beats = {}, set(), {}

    def __getattr__(self, name):
        if name in DEFAULTS:
            return self.policy[name]
        raise AttributeError(name)

    def path(self, *p):
        return os.path.join(self.root, *p)

    # --- manifest -------------------------------------------------------------------------

    def units(self, refresh=False):
        """{id: unit dict}, in manifest order (first definition of an id wins)."""
        if refresh or not self._seen:
            for name in sorted(_ls(self.path('manifest.d'))):
                if not name.endswith('.json') or name in self._seen:
                    continue
                for u in (read_json(self.path('manifest.d', name)) or {}).get('units', []):
                    self._units.setdefault(u['id'], u)
                self._seen.add(name)
        return self._units

    def add_units(self, units, batch=None):
        """Append the units whose ids are not in the manifest yet; returns how many were added."""
        known = self.units(refresh=True)
        new = [u for u in units if u['id'] not in known]
        for u in new:
            if '/' in u['id'] or u['id'].startswith('.'):
                raise ValueError(f'bad unit id {u["id"]!r}')
        if new:
            name = f'{batch or time.strftime("%Y%m%dT%H%M%S")}_{_tag()}.json'
            write_json(self.path('manifest.d', name), {'units': new, 'added': _now()})
            self.units(refresh=True)
        return len(new)

    # --- leases (claims and named leases share one protocol) -----------------------------

    def _lease_dir(self, kind, name):
        return self.path(kind, name)

    def beat(self, kind, name, t=None, check=False):
        """Write the epoch into the lease's heartbeat file in place, and set the lease dir's mtime to it.

        In place: a renamed file leaves NFS readers on the old inode until their directory cache expires.
        The dir mtime is the fallback when a reader cannot read the file."""
        d = self._lease_dir(kind, name)
        t = int(_now() if t is None else t)
        try:
            with open(os.path.join(d, 'heartbeat'), 'w') as fh:
                fh.write(str(t))
            os.utime(d, (t, t))
        except OSError:
            if check:
                raise

    def _hb(self, kind, name):
        """(heartbeat epoch or None, lease dir mtime or None when the lease is gone)."""
        d = self._lease_dir(kind, name)
        try:
            names = os.listdir(d)                      # also revalidates the dir on NFS
            mtime = os.stat(d).st_mtime
        except OSError:
            return None, None
        for _ in range(2 if 'heartbeat' in names else 0):
            try:
                with open(os.path.join(d, 'heartbeat')) as fh:
                    return float(fh.read().strip()), mtime
            except (OSError, ValueError):              # caught mid-rewrite
                time.sleep(0.5)
        return None, mtime

    def age(self, kind, name):
        """Seconds since the last heartbeat (file, else the lease dir's mtime); inf if the lease is gone."""
        t, mtime = self._hb(kind, name)
        return float('inf') if mtime is None else _now() - (t if t is not None else mtime)

    def stale(self, kind, name):
        """A lease may be stolen: expired on purpose, its heartbeat seen unchanged here for STALE_S (immune
        to clock skew and NFS caching), or older than 3 x STALE_S by the clocks (a lease long dead)."""
        t, mtime = self._hb(kind, name)
        if mtime is None or (t is not None and t <= HEARTBEAT_EXPIRED):
            return True
        v, now = (t if t is not None else mtime), _now()
        first = self._beats.get((kind, name))
        if first is None or first[0] != v:
            self._beats[(kind, name)] = first = (v, now)
        return now - first[1] >= self.STALE_S or now - v >= 3 * self.STALE_S

    def owns(self, kind, name, owner):
        return (read_json(os.path.join(self._lease_dir(kind, name), 'owner.json')) or {}).get('owner') == owner

    def _steal(self, kind, name):
        if not self.stale(kind, name):
            return False
        dead = self.path(kind, '.dead')
        os.makedirs(dead, exist_ok=True)
        try:
            os.rename(self._lease_dir(kind, name), os.path.join(dead, f'{name}.{_tag()}'))
        except OSError:
            return False                    # another worker stole it first
        try:
            os.mkdir(self._lease_dir(kind, name))
        except FileExistsError:
            return False                    # re-claimed in the gap
        return True

    def acquire(self, kind, name, owner):
        """mkdir the lease (or steal a stale one); True if we hold it."""
        try:
            os.mkdir(self._lease_dir(kind, name))
        except FileExistsError:
            if not self._steal(kind, name):
                return False
        try:
            self.beat(kind, name, check=True)
            write_json(os.path.join(self._lease_dir(kind, name), 'owner.json'),
                       dict(owner=owner, host=socket.gethostname(), pid=os.getpid(), t=_now(), **slurm_info()))
        except OSError:
            return False                    # stolen between mkdir and owner.json
        return True

    def periodic(self, name, every, fn):
        """Run ``fn()`` under lease ``name`` unless some worker ran it in the last ``every`` s (e.g. a feeder).

        Returns fn's result, or None when skipped."""
        last = self.path('leases', f'{name}.last')
        try:
            with open(last) as fh:
                if _now() - float(fh.read().strip()) < every:
                    return None
        except (OSError, ValueError):
            pass
        if not self.acquire('leases', name, socket.gethostname()):
            return None
        try:
            with open(last, 'w') as fh:
                fh.write(str(int(_now())))
            return fn()
        finally:
            self.drop('leases', name)

    def drop(self, kind, name):
        d = self._lease_dir(kind, name)
        for f in _ls(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass

    def expire(self, kind, name):
        self.beat(kind, name, t=HEARTBEAT_EXPIRED)

    # --- unit state ----------------------------------------------------------------------

    def done_ids(self):
        return set(_ls(self.path('done')))

    def outcome(self, uid):
        return read_json(self.path('done', uid))

    def n_attempts(self, uid):
        return sum(f.startswith('attempt_') for f in _ls(self.path('state', uid)))

    def n_failures(self, uid):
        return sum(f.startswith('fail_') for f in _ls(self.path('state', uid)))

    def _record(self, uid, prefix, obj):
        d = self.path('state', uid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'{prefix}_{uuid.uuid4().hex[:12]}.json')
        write_json(path, dict(obj, host=socket.gethostname(), t=_now(), **slurm_info()))
        return path

    def record_attempt(self, uid, worker):
        return self._record(uid, 'attempt', dict(worker=worker))

    def record_failure(self, uid, worker, exc=None, signature=None, detail=''):
        sig = signature or (error_signature(exc) if exc is not None else 'unknown')
        tb = ''.join(traceback.format_exception(exc))[-8000:] if isinstance(exc, BaseException) else detail
        self._record(uid, 'fail', dict(worker=worker, signature=sig, traceback=tb))
        return sig

    def finish(self, uid, status='done', owner=None, **info):
        """Mark ``uid`` terminal and drop its claim (unless ``owner`` is given and no longer holds it).

        A 'done' unit is never marked 'failed' (a late duplicate run failing after another finished)."""
        if not (status == 'failed' and (self.outcome(uid) or {}).get('status') == 'done'):
            write_json(self.path('done', uid), dict(info, status=status, t=_now(), host=socket.gethostname()))
        if owner is None or self.owns('claims', uid, owner):
            self.drop('claims', uid)

    def reopen(self, uid):
        """Make a terminal unit claimable again (clears its done marker and failure records)."""
        for f in _ls(self.path('state', uid)):
            if f.startswith('fail_'):
                os.remove(self.path('state', uid, f))
        try:
            os.remove(self.path('done', uid))
        except FileNotFoundError:
            pass

    # --- control -------------------------------------------------------------------------

    def stopped(self):
        return os.path.exists(self.path('STOP'))

    def draining(self):
        return os.path.exists(self.path('DRAIN'))

    def set_flag(self, name, reason=''):
        write_json(self.path(name), dict(reason=reason, t=_now(), host=socket.gethostname()))

    def write_worker(self, worker, **status):
        write_json(self.path('workers', f'{worker}.json'), dict(status, worker=worker, t=_now(),
                                                                host=socket.gethostname(), pid=os.getpid()))

    def mark_broken(self, job, reason=''):
        """Record a broken job; write STOP once BROKEN_JOBS_STOP jobs broke within BROKEN_WINDOW_S."""
        write_json(self.path('workers', f'{job}.broken'), dict(reason=reason, t=_now()))
        recent = [f for f in _ls(self.path('workers')) if f.endswith('.broken')
                  and _now() - (read_json(self.path('workers', f)) or {}).get('t', 0) < self.BROKEN_WINDOW_S]
        if len(recent) >= self.BROKEN_JOBS_STOP and not self.stopped():
            self.set_flag('STOP', f'{len(recent)} broken jobs within {self.BROKEN_WINDOW_S:.0f} s: {reason}')

    # --- selection -----------------------------------------------------------------------

    def order(self):
        """Unit ids, longest first by 'est' (stable on manifest order)."""
        u = self.units(refresh=True)
        return sorted(u, key=lambda k: -float(u[k].get('est', 0) or 0))

    def next_unit(self, cursor, worker):
        """Claim the next open unit and record the attempt; None if nothing is claimable now."""
        if self.stopped() or self.draining():
            return None
        ids = cursor.ids(self)
        done = self.done_ids()
        now = _now()
        for _ in range(len(ids)):
            uid = ids[cursor.i % len(ids)]
            cursor.i += 1
            if uid in done or uid in cursor.skip or cursor.defer.get(uid, 0) > now:
                continue
            if self.n_failures(uid) >= self.MAX_FAILURES or self.n_attempts(uid) >= self.MAX_ATTEMPTS:
                why = 'max_failures' if self.n_failures(uid) >= self.MAX_FAILURES else 'max_attempts'
                if self.acquire('claims', uid, worker):
                    self.finish(uid, 'failed', reason=why)
                cursor.skip.add(uid)
                continue
            if not self.acquire('claims', uid, worker):
                cursor.defer[uid] = now + self.STALE_S
                continue
            if os.path.exists(self.path('done', uid)):     # finished in the gap
                self.drop('claims', uid)
                cursor.skip.add(uid)
                continue
            cursor.attempt = self.record_attempt(uid, worker)
            cursor.defer.pop(uid, None)
            return uid
        return None

    def pending_elsewhere(self, cursor):
        """Units deferred because another live worker holds them."""
        now = _now()
        for k in [k for k, t in cursor.defer.items() if t <= now]:
            del cursor.defer[k]
        return len(cursor.defer)

    # --- status --------------------------------------------------------------------------

    def status(self):
        units = self.units(refresh=True)
        done = {u: self.outcome(u) or {} for u in self.done_ids()}
        claims = [c for c in _ls(self.path('claims')) if not c.startswith('.')]
        live = [c for c in claims if self.age('claims', c) < self.STALE_S]
        sigs = {}
        for uid in _ls(self.path('state')):
            for f in _ls(self.path('state', uid)):
                if f.startswith('fail_'):
                    r = read_json(self.path('state', uid, f)) or {}
                    s = sigs.setdefault(r.get('signature', 'unknown'), dict(n=0, units=[]))
                    s['n'] += 1
                    if uid not in s['units']:
                        s['units'].append(uid)
        workers = [read_json(self.path('workers', f)) for f in _ls(self.path('workers')) if f.endswith('.json')]
        return dict(total=len(units), done=sum(v.get('status') == 'done' for v in done.values()),
                    failed=sum(v.get('status') == 'failed' for v in done.values()),
                    running=len(live), stale=len(claims) - len(live),
                    open=len(set(units) - set(done) - set(claims)), failures=sigs,
                    workers=[w for w in workers if w], stopped=self.stopped(), draining=self.draining())


class Cursor:
    """A worker's walk over the manifest; starts at a random offset to spread contention."""

    def __init__(self, seed=None):
        self.i = random.Random(seed).randrange(1 << 30)
        self.skip, self.defer, self.attempt = set(), {}, None
        self._ids, self._t = [], 0.0

    def ids(self, q, force=False):
        if force or not self._ids or _now() - self._t > q.RESCAN_S:
            self._ids, self._t = q.order(), _now()
        return self._ids


class Heartbeat:
    """Context manager: beats ``claims/<uid>`` every HEARTBEAT_S in a thread; failed or slow beats are logged
    (a lease whose beats stall for STALE_S can be taken over while its owner still runs)."""

    def __init__(self, q, uid, kind='claims'):
        self.q, self.uid, self.kind = q, uid, kind
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.q.HEARTBEAT_S):
            t0 = _now()
            try:
                self.q.beat(self.kind, self.uid, check=True)
            except OSError as e:
                print(f'[heartbeat] {self.uid}: {e}', file=sys.stderr, flush=True)
            if _now() - t0 > self.q.HEARTBEAT_S:
                print(f'[heartbeat] {self.uid}: a beat took {_now() - t0:.0f} s (slow filesystem)',
                      file=sys.stderr, flush=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        return False


def work(q, worker, run, deadline=None, idle_exit_s=600.0, poll_s=30.0, on_idle=None, log=print):
    """Worker loop: claim -> ``run(uid, unit)`` -> finish, until no work, STOP, the deadline or
    MAX_CONSECUTIVE_FAILURES failures in a row. ``run`` returning means done; raising means a failure.
    ``Requeue`` gives the unit back uncounted and ends the loop; ``Broken`` records one failure and ends it.
    ``on_idle()`` (e.g. a feeder) runs when a pass finds nothing; True means it added work.
    Returns 'drained' | 'stopped' | 'deadline' | 'broken' | 'idle'."""
    cur = Cursor()
    n_bad, n_ok, idle_since = 0, 0, None
    last = None

    def done(why):
        q.write_worker(worker, state=f'exited: {why}', n_ok=n_ok, n_failed_in_row=n_bad, last_error=last)
        return why
    while True:
        if q.stopped():
            return done('stopped')
        if deadline is not None and _now() >= deadline:
            return done('deadline')
        uid = q.next_unit(cur, worker)
        if uid is None:
            if q.draining():
                return done('drained')
            if on_idle is not None and on_idle():
                cur.ids(q, force=True)
                continue
            if (idle_exit_s <= 0 and not q.pending_elsewhere(cur)
                    and set(q.units(refresh=True)) <= q.done_ids() | cur.skip):
                return done('drained')
            idle_since = idle_since or _now()
            if _now() - idle_since > idle_exit_s:
                return done('idle')
            q.write_worker(worker, state='idle', n_ok=n_ok, n_failed_in_row=n_bad, last_error=last)
            time.sleep(poll_s)
            cur.ids(q, force=True)
            continue
        idle_since = None
        q.write_worker(worker, state='running', unit=uid, n_ok=n_ok, n_failed_in_row=n_bad, last_error=last)
        try:
            with Heartbeat(q, uid):
                info = run(uid, q.units()[uid])
        except Requeue as e:                        # not an attempt: hand it back and stop
            log(f'[{worker}] {uid}: requeued ({e})')
            try:
                os.remove(cur.attempt)
            except OSError:
                pass
            q.drop('claims', uid)
            return done('deadline')
        except Broken as e:                         # environment broken: give the unit back and stop
            last = q.record_failure(uid, worker, e)
            log(f'[{worker}] {uid}: BROKEN ({last})')
            q.drop('claims', uid)
            return done('broken')
        except Exception as e:
            last = q.record_failure(uid, worker, e)
            n_bad += 1
            log(f'[{worker}] {uid}: FAILED ({last}); {n_bad} in a row')
            if q.n_failures(uid) >= q.MAX_FAILURES:
                q.finish(uid, 'failed', owner=worker, reason='max_failures', signature=last)
            elif q.owns('claims', uid, worker):
                q.drop('claims', uid)
            if n_bad >= q.MAX_CONSECUTIVE_FAILURES:
                return done('broken')
            continue
        n_bad, n_ok = 0, n_ok + 1
        if not q.owns('claims', uid, worker):
            log(f'[{worker}] {uid}: lease was taken over while running (a duplicate run)')
        q.finish(uid, 'done', owner=worker, worker=worker, **(info if isinstance(info, dict) else {}))
        log(f'[{worker}] {uid}: done')


class Requeue(Exception):
    """Raised by a runner to hand its unit back without counting a failure (e.g. a deadline)."""


class Broken(Exception):
    """Raised by a runner when this worker's environment is broken: one failure is recorded, the worker exits."""


# --- running units in child processes, and the per-node launcher ---------------------------

_CHILD = {}


def run_child(argv, log_path, timeout=None, result_path=None, env=None, cwd=None):
    """Run one unit as ``argv`` in its own process group, output to ``log_path``.

    Raises TimeoutError (group killed), or RuntimeError whose first line is the child's error signature
    (``result_path`` JSON ``{"error": {"signature": ...}}`` if written, else the return code), then the log
    path and tail."""
    import subprocess
    t0 = _now()
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'a') as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=cwd,
                             preexec_fn=die_with_parent)
        _CHILD['p'] = p
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(p)
            raise TimeoutError(f'unit exceeded {timeout:.0f} s\nlog {log_path}') from None
        finally:
            _CHILD.pop('p', None)
    res = read_json(result_path) if result_path else None
    if res is not None and os.path.getmtime(result_path) < t0:
        res = None                                  # stale result from an earlier attempt
    err = (res or {}).get('error')
    if rc != 0 or err:
        sig = (err or {}).get('signature') or (f'killed by signal {-rc}' if rc < 0 else f'exit code {rc}')
        raise RuntimeError(f'{sig}\nlog {log_path}\n' + '\n'.join(_tail(log_path)))
    return res


def _tail(path, n=8):
    try:
        with open(path, errors='replace') as fh:
            return [ln.rstrip() for ln in fh.readlines()[-200:] if ln.strip()][-n:]
    except OSError:
        return []


def die_with_parent():
    """preexec_fn: SIGKILL this child when the process that started it dies (Linux), so a killed worker
    leaves no orphan running a unit whose lease will be taken over."""
    import ctypes
    import signal
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def kill_group(p, grace=10.0):
    import signal
    import subprocess
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except OSError:
            return
        try:
            p.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass


def job_deadline(margin_s=600.0):
    """Epoch at which to stop claiming: SLURM_JOB_END_TIME (or squeue's end time) minus margin."""
    t = os.environ.get('SLURM_JOB_END_TIME')
    if not t and os.environ.get('SLURM_JOB_ID'):
        import subprocess
        try:
            out = subprocess.run(['squeue', '-h', '-j', os.environ['SLURM_JOB_ID'], '-o', '%e'],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            t = time.mktime(time.strptime(out, '%Y-%m-%dT%H:%M:%S')) if out[:2].isdigit() else None
        except (OSError, ValueError, subprocess.SubprocessError):
            t = None
    return float(t) - margin_s if t else None


def _worker_main(root, policy, worker, make_run, deadline, work_kw):
    import signal
    q = Queue(root, **policy)
    run = make_run(q, worker)

    def on_term(signum, frame):                     # preemption / walltime: give the unit back now
        p = _CHILD.get('p')
        if p is not None:
            kill_group(p, grace=5.0)
        cur = (read_json(q.path('workers', f'{worker}.json')) or {}).get('unit')
        if cur:
            q.expire('claims', cur)
        q.write_worker(worker, state='terminated', signal=signum)
        os._exit(128 + signum)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR1, on_term)
    return work(q, worker, run, deadline=deadline, **work_kw)


def launch(root, n_workers, make_run, tag=None, policy=None, deadline=None, **work_kw):
    """Start ``n_workers`` worker processes on this node; returns {worker: exit reason}.

    ``make_run(q, worker)`` (picklable, module level) returns the worker's ``run(uid, unit)``.
    Workers are '<tag>.w<i>'; tag defaults to '<host>.<job>.<procid>'. If every worker exits
    'broken', the job is marked broken (see Queue.mark_broken)."""
    import multiprocessing
    import signal
    s = slurm_info()
    tag = tag or f'{socket.gethostname()}.{s["job_id"] or os.getpid()}.{s["procid"] or 0}'
    ctx = multiprocessing.get_context('spawn')
    names = [f'{tag}.w{i}' for i in range(n_workers)]
    procs = [ctx.Process(target=_worker_main, args=(root, policy or {}, w, make_run, deadline, work_kw), name=w)
             for w in names]
    for p in procs:
        p.start()

    def forward(signum, frame):
        for p in procs:
            if p.is_alive():
                os.kill(p.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGUSR1, forward)
    for p in procs:
        p.join()
    q = Queue(root, **(policy or {}))
    out = {}
    for w, p in zip(names, procs):
        st = (read_json(q.path('workers', f'{w}.json')) or {}).get('state', '')
        out[w] = st[len('exited: '):] if p.exitcode == 0 and st.startswith('exited: ') else f'died (exit {p.exitcode})'
    if out and all(v == 'broken' for v in out.values()):
        q.mark_broken(tag, f'all {n_workers} workers broken')
    return out
