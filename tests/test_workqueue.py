"""bouquet.workqueue (solver-free): claims, leases, failure policy, the feeder hook, the launcher."""
import json
import multiprocessing
import os
import subprocess
import sys
import time

import pytest

from bouquet import workqueue as fq

FAST = dict(STALE_S=2.0, HEARTBEAT_S=0.3, RESCAN_S=0.5)


def _units(n, prefix='u'):
    return [{'id': f'{prefix}{i:03d}', 'est': i % 7} for i in range(n)]


def _drill_worker(root, worker, out):
    q = fq.Queue(root, **FAST)

    def run(uid, unit):
        with open(os.path.join(out, uid), 'a') as fh:
            fh.write(worker + '\n')
        time.sleep(0.01)
    return fq.work(q, worker, run, idle_exit_s=0, poll_s=0.05, log=lambda *a: None)


def test_race_drill(tmp_path):
    q = fq.Queue(tmp_path / 'q', **FAST)
    assert q.add_units(_units(120)) == 120
    assert q.add_units(_units(120)) == 0
    out = tmp_path / 'out'
    out.mkdir()
    ctx = multiprocessing.get_context('spawn')
    ps = [ctx.Process(target=_drill_worker, args=(str(tmp_path / 'q'), f'w{i}', str(out))) for i in range(8)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(600)
        assert p.exitcode == 0
    runs = {f: open(out / f).read().split() for f in os.listdir(out)}
    assert len(runs) == 120 and all(len(v) == 1 for v in runs.values())
    s = q.status()
    assert (s['done'], s['failed'], s['running'], s['open']) == (120, 0, 0, 0)
    assert all(q.n_attempts(u['id']) == 1 for u in _units(120))


def test_steal_stale_claim(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))
    assert q.next_unit(fq.Cursor(), 'a') == 'u000'
    assert q.next_unit(fq.Cursor(), 'b') is None          # live claim
    q.beat('claims', 'u000', t=time.time() - 10)          # owner died 10 s ago
    assert q.next_unit(fq.Cursor(), 'b') == 'u000'
    assert json.load(open(q.path('claims', 'u000', 'owner.json')))['owner'] == 'b'
    assert len(os.listdir(q.path('claims', '.dead'))) == 1


def test_duplicate_run_keeps_the_new_owners_claim(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))
    assert q.next_unit(fq.Cursor(), 'a') == 'u000'
    q.expire('claims', 'u000')
    assert q.next_unit(fq.Cursor(), 'b') == 'u000'      # a's lease taken over while a still runs
    q.finish('u000', 'done', owner='a')
    assert q.owns('claims', 'u000', 'b') and 'u000' in q.done_ids()
    q.finish('u000', 'failed', owner='b')                # b's duplicate fails later: still done
    assert q.outcome('u000')['status'] == 'done'


def test_child_dies_with_parent(tmp_path):
    pidfile = tmp_path / 'pid'
    code = (f'import subprocess, sys, time; sys.path.insert(0, {os.path.dirname(fq.__file__)!r}); '
            f'from {fq.__name__.split(".")[-1]} import die_with_parent; '
            f'p = subprocess.Popen(["sleep", "60"], start_new_session=True, preexec_fn=die_with_parent); '
            f'open({str(pidfile)!r}, "w").write(str(p.pid)); time.sleep(60)')
    parent = subprocess.Popen([sys.executable, '-c', code])
    for _ in range(300):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    parent.kill()
    parent.wait()
    time.sleep(1)
    status = f'/proc/{pidfile.read_text()}/status'
    assert not os.path.exists(status) or 'State:\tZ' in open(status).read()


def test_stale_by_observation_despite_clock_skew(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))
    q.next_unit(fq.Cursor(), 'a')
    q.beat('claims', 'u000', t=time.time() + 1000)      # the owner's clock runs 1000 s ahead, then it dies
    thief = fq.Queue(tmp_path, **FAST)
    assert not thief.stale('claims', 'u000')
    time.sleep(FAST['STALE_S'] + 0.2)
    assert thief.stale('claims', 'u000')
    q.beat('claims', 'u000')                            # a live owner keeps changing it
    assert not thief.stale('claims', 'u000')


def test_periodic(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    calls = []
    assert q.periodic('feeder', 60, lambda: calls.append(1) or 'fed') == 'fed'
    assert q.periodic('feeder', 60, lambda: calls.append(1)) is None and calls == [1]
    assert q.periodic('feeder', 0, lambda: 'again') == 'again'


def test_expired_heartbeat_is_stolen_at_once(tmp_path):
    q = fq.Queue(tmp_path, STALE_S=300)
    q.add_units(_units(1))
    q.next_unit(fq.Cursor(), 'a')
    q.expire('claims', 'u000')
    assert q.next_unit(fq.Cursor(), 'b') == 'u000'


def test_failures_and_broken_worker(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(5))

    def run(uid, unit):
        raise ValueError(f'bad input\nmore {uid}')
    assert fq.work(q, 'w', run, idle_exit_s=0, poll_s=0.01, log=lambda *a: None) == 'broken'
    assert q.MAX_CONSECUTIVE_FAILURES == 3
    sig = q.status()['failures']
    assert list(sig) == ['ValueError: bad input'] and sig['ValueError: bad input']['n'] == 3
    # a second worker takes every unit to MAX_FAILURES (2): all end 'failed', nothing left open
    for i in range(5):
        fq.work(q, f'w{i + 2}', run, idle_exit_s=0, poll_s=0.01, log=lambda *a: None)
    s = q.status()
    assert (s['failed'], s['open']) == (5, 0)
    q.reopen('u000')
    assert q.next_unit(fq.Cursor(), 'w4') == 'u000'


def test_max_attempts(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))
    for i in range(q.MAX_ATTEMPTS):
        assert q.next_unit(fq.Cursor(), f'w{i}') == 'u000'
        q.expire('claims', 'u000')                         # preempted
    assert q.next_unit(fq.Cursor(), 'x') is None
    assert q.outcome('u000')['reason'] == 'max_attempts'


def test_feeder_adds_units_mid_run(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(2))
    seen, fed = [], []

    def feed():
        if not fed:
            fed.append(1)
            return q.add_units(_units(3, prefix='new')) > 0
        return False
    fq.work(q, 'w', lambda uid, u: seen.append(uid), idle_exit_s=0, poll_s=0.01, on_idle=feed,
            log=lambda *a: None)
    assert sorted(seen) == ['new000', 'new001', 'new002', 'u000', 'u001']


def test_stop_and_drain(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(3))
    q.set_flag('DRAIN')
    assert q.next_unit(fq.Cursor(), 'w') is None
    assert fq.work(q, 'w', lambda *a: None, idle_exit_s=0, poll_s=0.01, log=lambda *a: None) == 'drained'
    os.remove(q.path('DRAIN'))
    q.set_flag('STOP')
    assert fq.work(q, 'w', lambda *a: None, poll_s=0.01, log=lambda *a: None) == 'stopped'


def test_requeue_and_deadline(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))

    def run(uid, unit):
        raise fq.Requeue('no time')
    assert fq.work(q, 'w', run, poll_s=0.01, log=lambda *a: None) == 'deadline'
    assert (q.n_failures('u000'), q.n_attempts('u000')) == (0, 0) and 'u000' not in q.done_ids()
    assert q.next_unit(fq.Cursor(), 'w2') == 'u000'


def test_broken_runner(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(2))

    def run(uid, unit):
        raise fq.Broken('executable missing')
    assert fq.work(q, 'w', run, idle_exit_s=0, poll_s=0.01, log=lambda *a: None) == 'broken'
    s = q.status()
    assert (s['failed'], s['open'], s['running']) == (0, 2, 0)


def test_mark_broken_writes_stop(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.mark_broken('job1', 'x')
    assert not q.stopped()
    q.mark_broken('job2', 'x')
    assert q.stopped()


def test_run_child(tmp_path):
    res = tmp_path / 'r.json'
    ok = [sys.executable, '-c', f'import json; json.dump({{"n": 3}}, open({str(res)!r}, "w"))']
    assert fq.run_child(ok, str(tmp_path / 'log'), result_path=str(res)) == {'n': 3}
    bad = [sys.executable, '-c', f'import json; json.dump({{"error": {{"signature": "E: x"}}}}, '
                                 f'open({str(res)!r}, "w")); raise SystemExit(1)']
    with pytest.raises(RuntimeError, match='E: x'):
        fq.run_child(bad, str(tmp_path / 'log'), result_path=str(res))
    t0 = time.time()
    with pytest.raises(TimeoutError):
        fq.run_child(['bash', '-c', 'sleep 60 & sleep 60'], str(tmp_path / 'log'), timeout=1)
    assert time.time() - t0 < 30


def _make_run(q, worker):
    return _ok_run


def _ok_run(uid, unit):
    time.sleep(0.05)


def test_launch(tmp_path):
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(10))
    out = fq.launch(str(tmp_path), 3, _make_run, tag='t', policy=FAST, idle_exit_s=0, poll_s=0.01)
    assert sorted(out) == ['t.w0', 't.w1', 't.w2'] and set(out.values()) <= {'drained', 'idle'}
    assert q.status()['done'] == 10


@pytest.mark.skipif(not os.environ.get("FSQUEUE_SRC"), reason="FSQUEUE_SRC (the shared fsqueue.py) not set")
def test_vendored_copy_is_identical():
    assert open(fq.__file__, "rb").read() == open(os.environ["FSQUEUE_SRC"], "rb").read()
