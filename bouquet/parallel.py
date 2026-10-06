"""Process-parallel bouquet generation, on two independent axes.

**Tier 1 -- draws** (:func:`parallel_generate`): one bouquet, its ``n_equils``
draws sharded across workers and merged back into one archive. Use it for a
SINGLE case you want finished sooner.

**Tier 2 -- cases** (:func:`parallel_cases`): a queue of whole, independent
bouquets -- one per g-file/p-file pair, per IDA time slice, per shot -- run on a
persistent pool where each worker stands up its solver once and then swaps case
after case onto it. Use it whenever there are at least as many cases as cores;
it needs no baseline agreement between workers, and one failed case cannot take
down the sweep. Inputs are expanded into cases by a
:class:`~bouquet.config.ParallelSource`, and
:func:`merge_cases` reassembles the results along the input structure (one
archive per IDA file, its slices as ``scan/<key>/`` groups).

The tiers are alternatives, not layers: never call :func:`parallel_generate`
inside a :func:`parallel_cases` worker -- each would claim the whole machine.

Tier 1 in detail:

``OFT_env`` is a per-process singleton, so a second TokaMaker cannot live in the
same interpreter -- parallelism is across **processes**, each with its own
solver. Draws are embarrassingly parallel, and the baseline forward-solve is
**bit-identical across processes at ``nthreads=1``** (verified: a fresh process
reproduces li_1/li_3/Ip/psi to 0), so every worker simply runs the ordinary
serial path (``setup_solver -> prepare_baseline -> generate``) on its shard of
``n_equils`` and the per-worker archives are concatenated.

Two launchers, one ``run_shard`` entry point:

* **laptop / single node** -- :func:`parallel_generate` drives a
  ``ProcessPoolExecutor`` (spawn). Budget ``n_workers x threads_per_worker`` to
  the physical cores; the ``nthreads=1`` regime (``threads_per_worker=1``,
  ``n_workers = cores``) is both the most reproducible and the most parallel.
* **cluster** -- :func:`emit_slurm_script` writes a SLURM job-array + dependent
  merge job that call ``python -m bouquet.parallel`` on the same ``run_shard``.

Note: parallel draws are NOT bit-identical to a serial run of the same seed --
each worker's ``GenerationConfig.seed`` (and hence the single
``numpy.random.Generator`` its ``generate_bouquet`` builds via
``sampling.make_rng``) is derived from ``(seed, worker_id, scan_key)`` via
``np.random.SeedSequence`` (see :func:`_derive_seed`), so the union is a
statistically-equivalent *different* draw set. The baseline is identical.
Folding the ``scan_key`` into the derivation decorrelates a multi-slice sweep
run with one ``seed``: draw *i* of slice A and draw *i* of slice B no longer
share a perturbation stream.

The parallel run IS reproducible as a whole: the derivation is a pure function
of ``(seed, worker_id, scan_key)``, so re-running the same ``seed`` over the
same ``n_workers`` regenerates every shard's draws bitwise. Changing
``n_workers`` re-partitions the draws and therefore changes the ensemble --
record it alongside the seed.
"""
from __future__ import annotations

import copy
import hashlib
import os

from threadpoolctl import threadpool_limits

__all__ = [
    # tier 1: split ONE bouquet's draws across workers
    "run_shard",
    "merge_archives",
    "parallel_generate",
    "emit_slurm_script",
    # tier 2: split a queue of WHOLE bouquets across workers
    "run_case",
    "merge_cases",
    "parallel_cases",
]


def _shard_size(total, n_workers, worker_id):
    """Draws assigned to *worker_id* when *total* is split over *n_workers*."""
    base, rem = divmod(int(total), int(n_workers))
    return base + (1 if worker_id < rem else 0)


def _derive_seed(seed_base, worker_id, scan_key):
    """Per-shard seed from ``(seed_base, worker_id, scan_key)``.

    The returned int becomes the shard's ``GenerationConfig.seed``, which
    ``generate_bouquet`` turns into that shard's one
    ``numpy.random.Generator`` (``sampling.make_rng``). Shards are therefore
    *independently* seeded but *deterministically derived* from the master
    seed: same triple -> same shard, always.

    ``SeedSequence`` scrambles the entropy tuple into a well-separated 32-bit
    seed, replacing the old ``seed_base + worker_id`` scheme, which (a) gave
    adjacent-integer seeds across workers and (b) reused the *same* streams
    for every slice of a timeseries swept with one ``seed_base`` --
    correlating draw *i* across time slices. The ``scan_key`` enters through
    its canonical string form (``utils._scan_key``), so ``2000`` and
    ``"2000"`` derive the same stream (mirroring the archive layout).
    """
    import numpy as np
    from .utils import _scan_key

    entropy = [int(seed_base), int(worker_id)]
    bkey = _scan_key(scan_key)
    if bkey is not None:
        entropy.append(int.from_bytes(
            hashlib.sha256(bkey.encode()).digest()[:8], "little"))
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _warn_multithreaded(threads_per_worker):
    """Warn when a worker's TokaMaker would run nthreads>1.

    One solver per core is the validated regime: nthreads>1 makes the OpenMP
    reduction order non-deterministic (~±1% li_1 jitter -- so workers converge
    to DIFFERENT l_i targets and the merged draw set mixes acceptance bands)
    and can tip the GS DLSODE solve into a non-convergence hang on stiff
    slices (on SLURM that burns the task's whole time limit and loses the
    shard). For throughput, raise n_workers instead.
    """
    if int(threads_per_worker) > 1:
        import warnings
        warnings.warn(
            f"threads_per_worker={threads_per_worker} sets nthreads="
            f"{threads_per_worker} on every worker's TokaMaker. This breaks "
            "run-to-run determinism (~±1% li_1 jitter -> workers accept draws "
            "against different l_i targets) and risks DLSODE hangs on stiff "
            "slices. Use threads_per_worker=1 and more workers instead.",
            stacklevel=3,
        )


# --------------------------------------------------------------------------
#  worker: generate one shard
# --------------------------------------------------------------------------
def run_shard(config, worker_id, n_workers, *, n_equils_total, seed_base,
              out_header, scan_key, threads_per_worker, verbose=False,
              progress_q=None):
    """Generate worker *worker_id*'s shard of draws in THIS process.

    Builds its own TokaMaker (own ``OFT_env``), forward-solves the baseline, and
    writes its draws to ``{out_header}_w{worker_id}.h5``. Returns a metadata dict
    (shard path, count, baseline ``li``/``Ip``) consumed by the merge and the
    cross-worker baseline check. A worker assigned zero draws is a no-op.

    ``verbose=False`` (default) captures the worker's native solver/mesh chatter
    (otherwise N workers x every slice floods the parent's stdout); set True to
    stream it for debugging.
    """
    n = _shard_size(n_equils_total, n_workers, worker_id)
    if n == 0:
        return dict(worker_id=worker_id, path=None, n=0,
                    li_target=None, Ip_target=None)

    # Silence the worker's entire stdout/stderr at the fd level BEFORE importing
    # OFT, which caches fd 1 at init -- a later redirect leaks the banner + N x
    # the mesh/solver chatter into the parent (notebook cell). The shard result
    # returns through the pool and exceptions propagate as pickled objects, so
    # nothing needed for results or debugging is lost. verbose=True streams it.
    _saved = None
    if not verbose:
        _dn = os.open(os.devnull, os.O_WRONLY)
        _saved = (os.dup(1), os.dup(2))
        os.dup2(_dn, 1)
        os.dup2(_dn, 2)
        os.close(_dn)
    try:
        import bouquet as bq
        cfg = copy.deepcopy(config)
        cfg.solver.nthreads = int(threads_per_worker)
        cfg.generation.n_equils = int(n)
        # Independent, slice-decorrelated, deterministically-derived stream:
        # generate_bouquet consumes this into the shard's single Generator
        # (see _derive_seed and sampling.make_rng), so re-running the same
        # (seed_base, n_workers, scan_key) regenerates the shard bitwise.
        cfg.generation.seed = _derive_seed(seed_base, worker_id, scan_key)
        cfg.generation.scan_key = scan_key
        cfg.output_header = f"{out_header}_w{worker_id}"

        # Fresh shard: the archive writer opens the h5 in append mode and only
        # deletes the draw groups it rewrites, so a leftover shard from a
        # previous run (crash before cleanup, cancelled SLURM array, or a
        # re-run with fewer draws per worker) would keep its stale
        # higher-index draws -- and merge_archives copies every draw group it
        # finds. Remove any pre-existing file so the shard holds ONLY this
        # run's draws.
        _shard_h5 = os.path.abspath(f"{cfg.output_header}.h5")
        if os.path.exists(_shard_h5):
            os.remove(_shard_h5)

        # per-draw progress -> parent aggregate bar (one tick per draw attempt)
        cb = None
        if progress_q is not None:
            def cb(_count, _q=progress_q, _w=worker_id):
                try:
                    _q.put(_w)
                except Exception:
                    pass

        b = bq.Bouquet(cfg)
        b.setup_solver()
        b.prepare_baseline()
        b.generate(progress_callback=cb)
        return dict(worker_id=worker_id, path=f"{cfg.output_header}.h5", n=int(n),
                    li_target=float(b.baseline.l_i_target),
                    Ip_target=float(b.baseline.Ip_target))
    finally:
        if _saved is not None:
            os.dup2(_saved[0], 1)
            os.dup2(_saved[1], 2)
            os.close(_saved[0])
            os.close(_saved[1])


# --------------------------------------------------------------------------
#  merge per-worker shards into one archive
# --------------------------------------------------------------------------
def merge_archives(shard_paths, out_header, scan_key=None, *, cleanup=False,
                   baseline_match_rtol=1e-6, config=None):
    """Concatenate per-worker shard archives into ``{out_header}.h5``.

    Draw groups are renumbered to a contiguous running index; under schema v2
    the raw bytes keep their fixed ``eqdsk`` / ``pfile`` dataset names, so no
    rename is needed (the group path carries the coordinates). The
    ``_baseline`` group is copied once (the baseline is identical across
    workers). Returns ``(out_path, n_draws)``.

    Pass the RUN-level ``config`` (the original :class:`~bouquet.BouquetConfig`,
    not a per-worker mutation) to stamp ``config_json`` provenance onto the
    merged archive -- the shards' own provenance records per-worker configs
    (derived seed, shard n_equils, ``_w{i}`` header) and is not copied, and
    the shards themselves are deleted under ``cleanup=True``, so without this
    the deliverable file carries no config record.

    Every shard's stored baseline (``_baseline`` attrs ``l_i_target`` /
    ``Ip_target``) is verified against the first shard's before anything is
    copied; a drift beyond ``baseline_match_rtol`` raises. This is the merge's
    own guard -- unlike the pre-merge check in :func:`parallel_generate` it
    also covers the SLURM CLI path, where drifted baselines (heterogeneous
    nodes, a stray ``nthreads>1``) would otherwise merge silently, mixing
    draws accepted against different l_i targets. A listed shard that does not
    exist on disk raises (missing workers must be handled by the caller, not
    dropped silently).
    """
    import warnings
    import h5py
    from .utils import (initialize_equilibrium_database, _scan_key,
                        _group_path)

    bkey = _scan_key(scan_key)
    base_path = f"scan/{bkey}" if bkey is not None else None
    bl_dst = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"

    # ---- pre-pass: shard existence + cross-shard baseline consistency ----
    def _baseline_targets(src):
        parent = src[base_path] if base_path else src
        if "_baseline" not in parent:
            return None
        a = parent["_baseline"].attrs
        if "l_i_target" not in a or "Ip_target" not in a:
            return None
        return float(a["l_i_target"]), float(a["Ip_target"])

    targets = []
    for sp in shard_paths:
        if sp is None:
            continue
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"shard archive not found: {sp}. Merging a partial set "
                "silently shrinks the bouquet -- re-run the missing worker, "
                "or drop the path explicitly from shard_paths.")
        with h5py.File(sp, "r") as src:
            targets.append((sp, _baseline_targets(src)))
    present = [(sp, t) for sp, t in targets if t is not None]
    unchecked = [sp for sp, t in targets if t is None]
    if unchecked and present:
        warnings.warn(
            "shard(s) without stored baseline targets (l_i_target/Ip_target "
            f"attrs) skipped the baseline consistency check: {unchecked}")
    if len(present) >= 2:
        sp0, (li0, ip0) = present[0]
        for sp, (li, ip) in present[1:]:
            if (abs(li - li0) > baseline_match_rtol * abs(li0) or
                    abs(ip - ip0) > baseline_match_rtol * abs(ip0)):
                raise RuntimeError(
                    f"shard baseline drifted: {sp} has l_i={li:.8f}, "
                    f"Ip={ip:.6e} vs {sp0} l_i={li0:.8f}, Ip={ip0:.6e} "
                    f"(rtol={baseline_match_rtol:g}). Workers did not "
                    "converge to the same baseline -- check nthreads=1 "
                    "(threads_per_worker=1), identical source/config, and "
                    "on SLURM that all array tasks ran on the same node "
                    "architecture. Nothing was merged.")

    out_path = os.path.abspath(f"{out_header}.h5")
    if os.path.exists(out_path):
        os.remove(out_path)                  # fresh archive (init opens append)
    initialize_equilibrium_database(out_header)

    offset = 0
    with h5py.File(out_path, "a") as out:
        if bkey is not None and base_path not in out:
            out.create_group(base_path)
        for sp in shard_paths:
            if sp is None:
                continue
            with h5py.File(sp, "r") as src:
                parent = src[base_path] if base_path else src
                if "_baseline" in parent and bl_dst not in out:
                    out.copy(parent["_baseline"], bl_dst)
                idxs = sorted(
                    int(k) for k in parent.keys()
                    if k not in ("_baseline", "scan")
                    and str(k).lstrip("-").isdigit()
                )
                for i in idxs:
                    dst = _group_path(scan_key, offset)
                    out.copy(parent[str(i)], dst)
                    g = out[dst]
                    g.attrs["count"] = offset
                    # Schema v2: draws store fixed `eqdsk`/`pfile` names, so the
                    # copy needs no rename (F11) -- the group path carries the
                    # coordinate. load_equilibrium resolves by the fixed name.
                    offset += 1

    # Run-level provenance on the merged archive (schema/version/timestamp
    # always; config_json when the caller supplied the run config).
    from .utils import write_provenance
    write_provenance(out_header, config=config, scan_key=scan_key)

    if cleanup:
        for sp in shard_paths:
            if sp and os.path.exists(sp):
                os.remove(sp)
    return out_path, offset


# --------------------------------------------------------------------------
#  orchestration
# --------------------------------------------------------------------------
def parallel_generate(config, *, n_workers=None, threads_per_worker=1, seed=0,
                      backend="laptop", baseline_match_rtol=1e-6, cleanup=True,
                      verbose=False, progress=True, slurm=None):
    """Fan ``config.generation.n_equils`` draws across worker processes, merge.

    ``backend="laptop"`` runs a ``ProcessPoolExecutor`` (spawn) now;
    ``backend="slurm"`` writes a job-array + merge script via
    :func:`emit_slurm_script` (pass options as the ``slurm`` dict) and returns
    their paths without running.

    ``n_workers=None`` defaults to the machine's **physical** core count
    (one single-threaded TokaMaker per physical core; logical/SMT cores
    oversubscribe the solver). Pass it explicitly on shared machines.

    The cross-worker **baseline check**: every worker reports its forward-solved
    baseline ``l_i``/``Ip``; if any drifts beyond ``baseline_match_rtol`` the run
    raises (a worker did not converge to the shared baseline -- e.g. a stray
    ``nthreads>1`` or a mismatched source). Returns a summary dict.
    """
    n_total = int(config.generation.n_equils)
    scan_key = config.generation.scan_key
    out_header = config.output_header
    if n_workers is None:
        # physical-core budget, but honouring the job's affinity mask --
        # see _get_num_cpus (which replaced the psutil/sysctl _physical_cores).
        n_workers = _get_num_cpus(use_logical=False)[0]
    nw = max(1, min(int(n_workers), n_total))

    if backend == "slurm":
        return emit_slurm_script(
            config, n_workers=nw, seed=seed,
            threads_per_worker=threads_per_worker, **(slurm or {}))

    if backend != "laptop":
        raise ValueError(f"backend must be 'laptop' or 'slurm', got {backend!r}")

    _warn_multithreaded(threads_per_worker)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # Pin each worker's BLAS/LAPACK + OpenMP to threads_per_worker BEFORE the
    # workers spawn (they inherit this env). Without it, OFT runs nthreads=1 but
    # the underlying BLAS grabs ~all cores per worker, so N workers oversubscribe
    # the machine (~Nx cores) and thrash -- the dominant cause of slow runs. The
    # spawned workers read these at their fresh numpy/OFT import.
    _thr = str(max(1, int(threads_per_worker)))
    _tvars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    _saved_env = {k: os.environ.get(k) for k in _tvars}
    for k in _tvars:
        os.environ[k] = _thr

    ctx = mp.get_context("spawn")
    results = [None] * nw

    # Optional live progress: workers post one item per draw attempt to a shared
    # queue; a daemon thread drains it into a single aggregate bar (per-worker
    # tqdm can't surface across processes, and worker stderr is suppressed).
    import threading
    import queue as _queue
    mgr = ctx.Manager() if progress else None
    pq = mgr.Queue() if progress else None
    bar = None
    drain_stop = threading.Event()
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=n_total, desc=f"draws ({nw}w x{threads_per_worker}t)",
                       unit="draw")
        except ImportError:
            bar = None

    def _drain():
        per_worker = {}
        while not drain_stop.is_set() or not pq.empty():
            try:
                w = pq.get(timeout=0.2)
            except (_queue.Empty, OSError):
                continue
            per_worker[w] = per_worker.get(w, 0) + 1
            if bar is not None:
                bar.update(1)
                bar.set_postfix_str(
                    " ".join(f"w{k}:{v}" for k, v in sorted(per_worker.items())))

    drain_thread = None
    if progress:
        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            futs = {
                ex.submit(run_shard, config, i, nw,
                          n_equils_total=n_total, seed_base=seed,
                          out_header=out_header, scan_key=scan_key,
                          threads_per_worker=threads_per_worker, verbose=verbose,
                          progress_q=pq): i
                for i in range(nw)
            }
            for fut in as_completed(futs):
                r = fut.result()
                results[r["worker_id"]] = r
    finally:
        drain_stop.set()
        if drain_thread is not None:
            drain_thread.join(timeout=2.0)
        if bar is not None:
            bar.close()
        if mgr is not None:
            mgr.shutdown()
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # cross-worker baseline check (the one-line guard)
    solved = [r for r in results if r and r["li_target"] is not None]
    if not solved:
        raise RuntimeError("no worker produced a baseline")
    li0, ip0 = solved[0]["li_target"], solved[0]["Ip_target"]
    for r in solved:
        if (abs(r["li_target"] - li0) > baseline_match_rtol * abs(li0) or
                abs(r["Ip_target"] - ip0) > baseline_match_rtol * abs(ip0)):
            raise RuntimeError(
                f"worker {r['worker_id']} baseline drifted from worker "
                f"{solved[0]['worker_id']}: li {r['li_target']:.8f} vs "
                f"{li0:.8f}, Ip {r['Ip_target']:.6e} vs {ip0:.6e}. Workers did "
                "not converge to the same baseline -- check nthreads=1 and that "
                "every worker used the identical source/config.")

    paths = [r["path"] for r in results if r and r["path"]]
    out_path, n_merged = merge_archives(paths, out_header, scan_key=scan_key,
                                        cleanup=cleanup, config=config)
    return dict(out_path=out_path, n_draws=n_merged, n_workers=nw,
                threads_per_worker=threads_per_worker,
                li_target=li0, Ip_target=ip0)


# --------------------------------------------------------------------------
#  cluster: emit a SLURM job-array + dependent merge
# --------------------------------------------------------------------------
def emit_slurm_script(config, *, n_workers, seed, threads_per_worker,
                      out_dir=".", job_name="bouquet", partition=None,
                      time_limit="02:00:00", mem_per_task="16G",
                      python="python", setup=None):
    """Write a SLURM job-array (one shard per task) + a dependent merge job.

    Serialises the run into ``{job_name}_bundle.json`` (config via
    ``BouquetConfig.to_dict``) and writes two sbatch
    scripts plus a ``{job_name}_submit.sh`` that chains them with
    ``--dependency=afterany`` (the merge validates shards itself: it aborts
    loudly on missing workers or a drifted baseline, so it is safe -- and
    more informative -- to let it run after a partial array instead of
    leaving it pending forever behind ``afterok``). Nothing is launched.
    Returns the written paths.

    ``setup`` is an optional list of shell lines inserted before the payload
    in BOTH sbatch scripts -- environment activation the compute node needs,
    e.g. ``["module load conda", "conda activate bouquet",
    "export OFT_PYTHONPATH=/path/to/OFT/python"]``. Without OFT importable,
    every shard dies at ``import OpenFUSIONToolkit``.

    Launch with ``{job_name}_submit.sh`` -- it ``cd``'s to its own directory
    first, so it works from any CWD. (The sbatch scripts reference the bundle
    by basename and therefore assume the submission CWD is ``out_dir``;
    shard/merged ``.h5`` outputs land there too unless
    ``config.output_header`` is an absolute path.)
    """
    _warn_multithreaded(threads_per_worker)
    os.makedirs(out_dir, exist_ok=True)
    # Config JSON bundle (not pickle): portable across package/Python versions,
    # human-inspectable, and the same serialization the h5 provenance uses (F25).
    import json
    bundle = dict(
        config=config.to_dict(), n_workers=int(n_workers), seed=int(seed),
        threads_per_worker=int(threads_per_worker),
        n_equils_total=int(config.generation.n_equils),
        scan_key=config.generation.scan_key,
        out_header=config.output_header,
    )
    bname = f"{job_name}_bundle.json"
    bpath = os.path.join(out_dir, bname)
    with open(bpath, "w") as fh:
        json.dump(bundle, fh, indent=2)

    part = f"#SBATCH --partition={partition}\n" if partition else ""
    extra = "".join(f"{line}\n" for line in (setup or []))
    # threads pinned to the task's cores; BLAS held to the same to avoid nesting.
    env = (f"export OMP_NUM_THREADS={threads_per_worker}\n"
           "export OMP_PROC_BIND=close OMP_PLACES=cores\n"
           f"export OPENBLAS_NUM_THREADS={threads_per_worker} "
           f"MKL_NUM_THREADS={threads_per_worker}\n")

    # The bundle is referenced by BASENAME: sbatch tasks run in the submission
    # CWD, and submit.sh cd's to this directory first -- so the pair works
    # from any CWD without baking a machine-specific absolute path into the
    # scripts.
    array = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}\n"
        f"#SBATCH --array=0-{n_workers - 1}\n"
        f"#SBATCH --cpus-per-task={threads_per_worker}\n"
        f"#SBATCH --time={time_limit}\n#SBATCH --mem={mem_per_task}\n{part}"
        f"{extra}{env}"
        f"{python} -m bouquet.parallel shard {bname} $SLURM_ARRAY_TASK_ID\n"
    )
    merge = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}_merge\n"
        f"#SBATCH --cpus-per-task=1\n#SBATCH --time=00:20:00\n"
        f"#SBATCH --mem={mem_per_task}\n{part}{extra}"
        f"{python} -m bouquet.parallel merge {bname}\n"
    )
    submit = (
        "#!/bin/bash\n"
        "# run from anywhere: everything below is relative to this script\n"
        'cd "$(dirname "$0")"\n'
        "# chain array + merge; afterany (not afterok) so the merge still\n"
        "# runs -- and reports exactly which shards are missing -- after a\n"
        "# partial array, instead of pending forever.\n"
        f"aid=$(sbatch --parsable {job_name}_array.sbatch)\n"
        f"sbatch --dependency=afterany:$aid {job_name}_merge.sbatch\n"
    )

    apath = os.path.join(out_dir, f"{job_name}_array.sbatch")
    mpath = os.path.join(out_dir, f"{job_name}_merge.sbatch")
    spath = os.path.join(out_dir, f"{job_name}_submit.sh")
    for p, txt in ((apath, array), (mpath, merge), (spath, submit)):
        with open(p, "w") as fh:
            fh.write(txt)
    os.chmod(spath, 0o755)
    return dict(bundle=bpath, array=apath, merge=mpath, submit=spath)


# --------------------------------------------------------------------------
#  CLI used by the SLURM scripts:  python -m bouquet.parallel {shard|merge} ...
# --------------------------------------------------------------------------
def _cli(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit("usage: python -m bouquet.parallel {shard <bundle> "
                         "<worker_id> | merge <bundle> [--allow-missing]}")
    cmd, bpath = argv[0], argv[1]
    import json
    from .config import BouquetConfig
    with open(bpath) as fh:
        b = json.load(fh)
    b["config"] = BouquetConfig.from_dict(b["config"])
    if cmd == "shard":
        # verbose=True: SLURM already isolates each task's output in its own
        # slurm-*.out, so the notebook flood rationale for fd-suppression does
        # not apply -- and an empty log is useless when a shard dies.
        run_shard(b["config"], int(argv[2]), b["n_workers"],
                  n_equils_total=b["n_equils_total"], seed_base=b["seed"],
                  out_header=b["out_header"], scan_key=b["scan_key"],
                  threads_per_worker=b["threads_per_worker"], verbose=True)
    elif cmd == "merge":
        allow_missing = "--allow-missing" in argv[2:]
        # workers assigned zero draws legitimately produce no shard file;
        # only count the ones that were expected to write one.
        expected = [i for i in range(b["n_workers"])
                    if _shard_size(b["n_equils_total"], b["n_workers"], i) > 0]
        paths = {i: f"{b['out_header']}_w{i}.h5" for i in expected}
        missing = sorted(i for i, p in paths.items() if not os.path.exists(p))
        if missing:
            msg = (f"missing shard archive(s) for worker(s) {missing}: "
                   f"expected {len(expected)}, found "
                   f"{len(expected) - len(missing)}")
            if not allow_missing:
                raise SystemExit(
                    "merge aborted: " + msg + ". Re-run those array indices "
                    "(sbatch --array=" + ",".join(map(str, missing)) +
                    " <array.sbatch>), or pass --allow-missing to merge the "
                    "partial set.")
            print(f"WARN: {msg} -- merging the partial set")
        shard_list = [p for i, p in sorted(paths.items()) if i not in missing]
        out_path, n = merge_archives(shard_list, b["out_header"],
                                     scan_key=b["scan_key"], cleanup=True,
                                     config=b["config"])
        print(f"merged {n} draws -> {out_path}")
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    _cli()


# ==========================================================================
#  TIER 2: case-level parallelism -- many independent bouquets at once
# ==========================================================================
#  Everything above splits ONE bouquet's draws across workers.  This tier
#  splits a QUEUE OF WHOLE BOUQUETS -- one per g-file/p-file pair, per IDA time
#  slice, per shot -- across a persistent pool, each worker standing up its own
#  solver once and then running case after case on it (`Bouquet.set_case`).
#
#  Ported from the standalone `parallel_ext.py`, which remains in the tree as
#  the reference implementation.  The per-case body there (`re_generate_bouquet`)
#  and its config dict are gone: a case is now just a `CaseSpec` swapped onto a
#  standing `Bouquet`, so the sweep inherits the whole validated pipeline
#  (reconstruction, workflow guard, filtering, provenance) instead of
#  reimplementing it.
#
#  Which tier: case-parallel whenever there are at least as many cases as cores
#  -- it needs no baseline agreement between workers and merges nothing per
#  case.  Draw-parallel (`parallel_generate`) is for a SINGLE case.  Do not nest
#  them: `parallel_generate` must never be called inside a case worker.

# Module-level state populated by _init_case_worker in each spawned worker.
_worker_state: dict = {}


class _IndexMap:
    """Picklable ``map_object``: ``map_object(idx)`` returns ``flat_list[idx]``.

    Saved to ``map_object.pkl`` by :func:`parallel_cases` so a finished sweep
    can be traced back from a run index to the case that produced it (the run
    order is not the completion order).
    """

    def __init__(self, flat_list):
        self.flat_list = list(flat_list)

    def __call__(self, idx):
        return self.flat_list[idx]

    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)


def _get_num_cpus(use_logical=True):
    """Return ``(n_workers, nthreads_per_worker)`` for spawning OFT workers.

    Works on Linux HPC clusters (SLURM, PBS, LSF, SGE) and degrades gracefully
    on non-Linux systems (macOS, Windows).  Preferred over a bare
    ``os.cpu_count()``: it respects the cgroup/taskset affinity mask a batch
    scheduler hands the job, so a 4-CPU allocation on a 128-core node reports 4.

    Parameters
    ----------
    use_logical : bool
        ``True`` (default): one worker per logical CPU (hyperthread),
        ``nthreads=1`` -- the reproducible regime (see
        :func:`_warn_multithreaded`).

        ``False``: one worker per physical core, ``nthreads = logical/physical``,
        using OFT's OpenMP intra-core parallelism.

    Returns
    -------
    n_workers : int
    nthreads_per_worker : int
    """
    # --- Logical CPU count from OS affinity (Linux) or cpu_count (other) ---
    try:
        affinity = os.sched_getaffinity(0)          # Linux: respects cgroup/taskset
        n_logical = len(affinity)
    except AttributeError:
        affinity = None
        n_logical = os.cpu_count() or 1             # macOS / Windows fallback

    # --- Physical core count via Linux sysfs ---
    n_physical = None
    if affinity is not None:
        core_ids = set()
        for cpu in affinity:
            try:
                with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as _f:
                    pkg = _f.read().strip()
                with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id") as _f:
                    core = _f.read().strip()
                core_ids.add((pkg, core))
            except OSError:
                pass
        if core_ids:
            n_physical = len(core_ids)
    if n_physical is None:
        n_physical = n_logical      # sysfs unavailable: assume no SMT
    nthreads_per_core = max(1, n_logical // n_physical)

    if use_logical:
        # Scheduler-specific CPU count env vars (used as a cap to avoid
        # over-subscription when the affinity set is wider than the job's
        # CPU reservation -- observed on some SLURM configurations).
        _SCHEDULER_CPU_VARS = (
            "SLURM_CPUS_PER_TASK",   # SLURM
            "PBS_NUM_PPN",           # PBS (CPUs per node)
            "LSB_DJOB_NUMPROC",      # IBM LSF
            "NSLOTS",                # SGE / Grid Engine
        )
        for var in _SCHEDULER_CPU_VARS:
            val = os.environ.get(var)
            if val is not None:
                n_logical = min(n_logical, int(val))
                break
        return n_logical, 1
    return n_physical, nthreads_per_core


# --------------------------------------------------------------------------
#  worker: stand up one solver, then run case after case on it
# --------------------------------------------------------------------------
def _init_case_worker(worker_id_queue, master_working_dir, config_dict,
                      init_status_queue, hooks=None):
    """Pool initialiser: build this process's :class:`~bouquet.run.Bouquet` once.

    Each spawned worker claims a unique ID from *worker_id_queue*, creates a
    private working directory (so concurrent TokaMaker scratch writes cannot
    collide), copies the mesh locally (a serial HDF5 build deadlocks on
    concurrent opens of one file), redirects its output to a per-worker log,
    and calls ``setup_solver()``.  The resulting ``Bouquet`` lives in the
    module-level ``_worker_state`` and is reused by every :func:`run_case`
    task this process handles -- ``OFT_env`` is a per-process singleton, so it
    can be built exactly once.

    *hooks* is the run-level ``{"before_baseline": fn, "after_baseline": fn}``
    mapping, stashed alongside the solver for :func:`run_case` to call.

    Reports ``(worker_id, None)`` on success or ``(worker_id, traceback_str)``
    on failure to *init_status_queue*, which the parent's barrier drains before
    dispatching any work: a broken initialiser then fails the run immediately
    and legibly instead of hanging in ``imap_unordered``.
    """
    global _worker_state
    import shutil
    import socket
    import sys
    import traceback

    worker_id = -1  # fallback if the queue.get() itself fails
    try:
        config_dict = dict(config_dict)
        nthreads = int(config_dict.pop("_nthreads", 1))
        verbose = bool(config_dict.pop("_verbose", False))

        # Pin the numeric stack before numpy/OFT are touched in this process.
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[_v] = str(nthreads)

        # Timeout so a replacement worker (spawned by the pool after a crash)
        # fails fast instead of blocking forever and deadlocking the parent.
        try:
            worker_id = worker_id_queue.get(timeout=60)
        except Exception:
            raise RuntimeError(
                "[bouquet_cases] Worker ID queue empty -- this is a pool "
                "replacement for a dead worker. Cannot initialise.")

        master_working_dir = os.path.abspath(master_working_dir)  # anchor before chdir
        working_dir = os.path.join(master_working_dir, f"worker_{worker_id}")
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)

        # Redirect this worker's stdout/stderr to a per-worker log file. os.dup2
        # at the fd level also captures output written directly to fd 1/2 by the
        # Fortran/C extensions (OFT), which a sys.stdout swap alone would miss.
        log_path = os.path.join(master_working_dir, f"worker_{worker_id}.log")
        if not verbose:
            _log_fh = open(log_path, "w", buffering=1)   # line-buffered
            os.dup2(_log_fh.fileno(), 1)
            os.dup2(_log_fh.fileno(), 2)
            sys.stdout = _log_fh
            sys.stderr = _log_fh

        from .config import BouquetConfig
        from .run import Bouquet

        config = BouquetConfig.from_dict(config_dict)

        # Copy the mesh into this worker's private directory: several workers
        # opening one HDF5 mesh concurrently trips file locking in a serial
        # HDF5 build. Point the config at the local copy before setup_solver.
        local_mesh = os.path.join(working_dir,
                                  os.path.basename(config.solver.mesh_path))
        shutil.copy2(config.solver.mesh_path, local_mesh)
        config.solver.mesh_path = local_mesh
        config.solver.nthreads = nthreads

        bouquet = Bouquet(config)
        bouquet.setup_solver()

        print(f"[Worker {worker_id}] solver ready -- host={socket.gethostname()}, "
              f"PID={os.getpid()}, cwd={working_dir}, nthreads={nthreads}",
              flush=True)

        _worker_state.update({
            "worker_id":   worker_id,
            "working_dir": working_dir,
            "log_path":    log_path,
            "nthreads":    nthreads,
            "bouquet":     bouquet,
            "hooks":       dict(hooks or {}),
        })
        init_status_queue.put((worker_id, None))        # signal success

    except Exception:
        tb = traceback.format_exc()
        print(f"[Worker {worker_id}] INIT FAILED:\n{tb}", flush=True)
        try:
            init_status_queue.put((worker_id, tb))
        except Exception:
            pass
        raise                                            # kill this worker process


def run_case(run_args):
    """Run ONE case on this worker's standing solver. Returns a result dict.

    Swaps the case onto the worker's :class:`~bouquet.run.Bouquet`
    (:meth:`~bouquet.run.Bouquet.set_case`, which clears the cached baseline so
    the next ``prepare_baseline`` re-points the solver and resets coil
    regularisation), then runs baseline -> generate -> filter, calling the
    run-level ``before_baseline`` / ``after_baseline`` hooks (if any) around
    the baseline solve.

    Deliberately NOT ``Bouquet.run()``: that also calls ``export()``, writing a
    second ``{header}_selected.h5`` per case.  The pass flags are inside the
    archive either way and :func:`merge_cases` produces the deliverable, so the
    per-case export is pure duplication here.

    Never raises for a failed case -- the traceback comes back as data so one
    bad slice cannot take down a sweep of hundreds.
    """
    import traceback

    idx, case, case_dir = run_args
    bouquet = _worker_state["bouquet"]
    worker_id = _worker_state["worker_id"]
    nthreads = _worker_state.get("nthreads", 1)
    hooks = _worker_state.get("hooks") or {}
    before_baseline = hooks.get("before_baseline")
    after_baseline = hooks.get("after_baseline")

    # Per-case archive, absolute so it lands in the shared case directory while
    # cwd stays the worker's private scratch dir. idx keeps it unique even when
    # two inputs share a basename.
    header = os.path.join(case_dir, f"{os.path.basename(case.header)}_idx{idx}")
    tag = f"[Worker {worker_id} | case {idx} | {case.group}/{case.scan_key}]"

    # The worker already pins OMP/BLAS through the environment before numpy
    # and OFT are imported, but env vars only bind libraries that read them at
    # load time; threadpool_limits reaches into the already-loaded pools and is
    # what actually holds a runaway BLAS to `nthreads` inside this case.
    with threadpool_limits(limits=int(nthreads)):
        try:
            print(f"{tag} starting -> {header}.h5", flush=True)
            # set_case takes the source + scan_key from the spec; the header
            # is overridden to the worker-visible absolute path.
            bouquet.set_case(case)
            bouquet.output_header = header
            if before_baseline is not None:
                before_baseline(bouquet, case)
            bouquet.prepare_baseline()
            if after_baseline is not None:
                after_baseline(bouquet, case)
            bouquet.generate()
            bouquet.filter()
            bl = bouquet.baseline
            n_all = len(bouquet.selected_indices("all"))
            n_sel = len(bouquet.selected_indices("selected"))
            print(f"{tag} done -- {n_sel}/{n_all} in spec", flush=True)
            return idx, True, None, {
                "idx":       idx,
                "path":      os.path.abspath(f"{header}.h5"),
                "group":     case.group,
                "scan_key":  case.scan_key,
                "worker_id": worker_id,
                "n_all":     n_all,
                "n_sel":     n_sel,
                "l_i_target": float(getattr(bl, "l_i_target", float("nan"))),
                "Ip_target":  float(getattr(bl, "Ip_target", float("nan"))),
            }
        except Exception:
            tb_str = traceback.format_exc()
            print(f"{tag} FAILED:\n{tb_str}", flush=True)
            return idx, False, tb_str, None


# --------------------------------------------------------------------------
#  orchestration: fan cases across a persistent pool
# --------------------------------------------------------------------------
def parallel_cases(source, config, master_working_dir, *, chunksize="automatic",
                   use_logical_cpus=True, n_cpus_override=None, verbose=False,
                   merge=True, group_by="group", cleanup=False,
                   before_baseline=None, after_baseline=None):
    """Run many independent bouquets in parallel on one node.

    Each case is a full ``prepare_baseline -> generate -> filter`` on a worker's
    standing solver, writing its own archive; nothing is shared between cases,
    so a failure is contained to the case that caused it.

    .. warning::
       Call this from inside an ``if __name__ == "__main__":`` block when you
       run it from a script. Workers are spawned (not forked -- OFT's Fortran
       libraries are not fork-safe), and a spawned worker re-imports the main
       module: unguarded top-level code that calls this function is therefore
       re-executed by every worker, each launching its own pool. In a notebook
       there is nothing to guard.

    Parameters
    ----------
    source : ParallelSource or list of CaseSpec
        The sweep.  A :class:`~bouquet.config.ParallelSource` (e.g.
        :class:`~bouquet.config.IdaTimeslices`) is expanded via ``.expand()``;
        a ready list of :class:`~bouquet.config.CaseSpec` is used as given.
    config : BouquetConfig
        The RUN-level config: solver, uncertainty envelope, generation knobs --
        everything except the baseline source, which each case supplies.
        Its ``source`` is a placeholder and is replaced per case.
    master_working_dir : str
        Root for ``worker_N/`` scratch dirs, ``worker_N.log`` logs, ``cases/``
        archives, ``map_object.pkl`` and ``errors.pkl``.
    chunksize : int or "automatic"
        Tasks handed to a worker at a time.  Cases are minutes-to-hours each, so
        the automatic value is 1 unless the queue is much longer than the pool.
    use_logical_cpus : bool
        ``True`` (default): one single-threaded worker per logical CPU.
        ``False``: one worker per physical core with OFT threading inside
        (see :func:`_get_num_cpus`; note :func:`_warn_multithreaded`).
    n_cpus_override : int, optional
        Force the worker count, bypassing detection (shared machines, tests).
    verbose : bool
        ``False`` (default): each worker's output goes to
        ``<master_working_dir>/worker_N.log`` and the terminal shows only
        parent-side status.  ``True``: everything streams to the terminal,
        interleaved across workers (debugging).
    merge : bool
        Run :func:`merge_cases` on the successful cases afterwards.
    before_baseline, after_baseline : callable, optional
        Per-case hooks, ``f(bouquet, case) -> None``, run on the worker either
        side of ``prepare_baseline()``.  They exist because some per-case setup
        cannot be expressed as a :class:`~bouquet.config.CaseSpec`:

        * ``before_baseline`` sees the case's source but no baseline yet -- the
          place for per-case SOLVER targets, which ``prepare_baseline`` applies
          (e.g. an X-point pin read from THIS slice's g-file).
        * ``after_baseline`` sees the resolved ``bouquet.baseline`` -- the place
          for anything that must live on that slice's ``psi_N_kinetic`` grid,
          notably ``uncertainty.aux_baselines`` / ``aux_sigmas`` (the measured
          E_r / omega_tor switchboard).

        Workers are SPAWNED, so a hook is pickled by reference: it must be a
        module-level function (a lambda, closure, or bound method will not
        pickle).  Anything it reads from module scope is re-imported in the
        worker, so keep it self-contained and cheap.
    group_by : {"group", None}
        Passed to :func:`merge_cases`: ``"group"`` (default) writes one merged
        archive per input group, ``None`` writes a single combined archive.
    cleanup : bool
        Delete the per-case archives once they are merged.

    Returns
    -------
    dict
        ``{"n_runs", "n_success", "cases", "results", "errors", "merged",
        "map_object_path"}``.  ``errors`` maps case index -> traceback string.
    """
    import multiprocessing
    import pickle as pkl
    import queue
    import traceback

    cases = list(source.expand() if hasattr(source, "expand") else source)
    n_runs = len(cases)

    # Hooks travel to the workers as pickled REFERENCES (module + qualname), so
    # check here that they will survive the spawn -- a lambda or closure only
    # fails once the pool is up, as an opaque initialiser error.
    hooks = {}
    for _name, _fn in (("before_baseline", before_baseline),
                       ("after_baseline", after_baseline)):
        if _fn is None:
            continue
        if not callable(_fn):
            raise TypeError(f"{_name} must be callable, got {type(_fn).__name__}")
        try:
            pkl.loads(pkl.dumps(_fn))
        except Exception as exc:
            raise TypeError(
                f"{_name} is not picklable ({exc}); workers are spawned, so a "
                "hook must be a module-level function -- not a lambda, closure, "
                "or bound method.") from exc
        hooks[_name] = _fn

    master_working_dir = os.path.abspath(master_working_dir)
    os.makedirs(master_working_dir, exist_ok=True)
    case_dir = os.path.join(master_working_dir, "cases")
    os.makedirs(case_dir, exist_ok=True)

    if n_runs == 0:
        print("[bouquet_cases] No cases to execute.")
        return dict(n_runs=0, n_success=0, cases=[], results={}, errors={},
                    merged={}, map_object_path=None)

    # ---- worker/thread budget ------------------------------------------
    if n_cpus_override is not None:
        n_cpus, nthreads = int(n_cpus_override), 1
    else:
        n_cpus, nthreads = _get_num_cpus(use_logical=use_logical_cpus)
    n_workers = max(1, min(n_cpus, n_runs))
    _warn_multithreaded(nthreads)
    print(f"[bouquet_cases] Distributing {n_runs} case(s) across {n_workers} "
          f"worker(s) ({n_cpus} CPUs available, {nthreads} thread(s)/worker).")

    if chunksize == "automatic":
        # 10x more tasks than workers, capped at 1000 per chunk. Cases are long
        # and uneven, so this lands on 1 (dynamic scheduling) for normal sweeps.
        chunksize = max(1, min(1000, n_runs // (10 * n_workers)))
        print(f"[bouquet_cases] Using chunksize={chunksize} for dynamic scheduling.")
    else:
        chunksize = int(chunksize)
        print(f"[bouquet_cases] Using user-specified chunksize={chunksize}.")

    # Save the idx -> case map so a finished sweep can be traced back from a run
    # index (completion order is not submission order).
    map_object_path = os.path.join(master_working_dir, "map_object.pkl")
    with open(map_object_path, "wb") as fh:
        pkl.dump(_IndexMap(cases), fh)
    print(f"[bouquet_cases] Saved case map to {map_object_path}")

    # ---- pool setup ------------------------------------------------------
    # 'spawn' avoids fork-safety issues with the Fortran shared libraries in OFT.
    ctx = multiprocessing.get_context("spawn")
    init_status_queue = ctx.Queue()
    worker_id_queue = ctx.Queue()
    for w in range(n_workers):
        worker_id_queue.put(w)

    cfg_dict = config.to_dict()
    cfg_dict["_nthreads"] = nthreads
    cfg_dict["_verbose"] = bool(verbose)

    # Pin BLAS/OpenMP in the PARENT so the spawned workers inherit it at their
    # fresh numpy/OFT import; without this each worker's BLAS grabs every core
    # and N workers thrash the machine. The workers set it again themselves.
    _tvars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    _saved_env = {k: os.environ.get(k) for k in _tvars}
    for k in _tvars:
        os.environ[k] = str(nthreads)

    errors, results = {}, {}
    try:
        _pool = ctx.Pool(
            processes=n_workers,
            initializer=_init_case_worker,
            initargs=(worker_id_queue, master_working_dir, cfg_dict,
                      init_status_queue, hooks),
        )

        # ---- barrier: every worker must report before any task is dispatched.
        # A broken initialiser (bad mesh, missing OFT) otherwise surfaces as a
        # silent hang in imap_unordered rather than an error.
        init_failures = []
        for _ in range(n_workers):
            try:
                wid, tb = init_status_queue.get(timeout=300)   # 5 min per worker
            except queue.Empty:
                init_failures.append((-1, "Worker initialisation timed out (> 300 s)"))
            else:
                if tb is not None:
                    init_failures.append((wid, tb))
                elif verbose:
                    print(f"[bouquet_cases] Worker {wid} ready.", flush=True)
                else:
                    log = os.path.join(master_working_dir, f"worker_{wid}.log")
                    print(f"[bouquet_cases] Worker {wid} ready  (log: {log})",
                          flush=True)

        if init_failures:
            _pool.terminate()
            _pool.join()
            msgs = "\n".join(f"  Worker {wid}:\n{tb}" for wid, tb in init_failures)
            raise RuntimeError(
                f"[bouquet_cases] FATAL: {len(init_failures)} worker(s) failed "
                f"to initialise:\n{msgs}")

        # ---- dispatch ----------------------------------------------------
        per_run_args = [(i, cases[i], case_dir) for i in range(n_runs)]
        try:
            with _pool:
                for idx, ok, err_msg, out in _pool.imap_unordered(
                        run_case, per_run_args, chunksize=chunksize):
                    if ok:
                        results[idx] = out
                        print(f"[bouquet_cases] case {idx} "
                              f"({cases[idx].group}/{cases[idx].scan_key}) done "
                              f"-- {out['n_sel']}/{out['n_all']} in spec",
                              flush=True)
                    else:
                        errors[idx] = err_msg
                        print(f"[bouquet_cases] WARNING: case {idx} "
                              f"({cases[idx].group}/{cases[idx].scan_key}) "
                              f"failed:\n{err_msg}", flush=True)
        except KeyboardInterrupt:
            _pool.join()
            raise
        except Exception as _exc:
            _pool.join()
            raise RuntimeError(
                f"[bouquet_cases] FATAL error during task dispatch:\n"
                f"{traceback.format_exc()}") from _exc
    finally:
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    n_success = n_runs - len(errors)
    print(f"[bouquet_cases] Completed: {n_success}/{n_runs} cases succeeded.",
          flush=True)

    if errors:
        error_path = os.path.join(master_working_dir, "errors.pkl")
        with open(error_path, "wb") as fh:
            pkl.dump(errors, fh)
        print(f"[bouquet_cases] Error details saved to {error_path}")

    merged = {}
    if merge and results:
        out_header = os.path.join(master_working_dir,
                                  os.path.basename(config.output_header))
        merged = merge_cases([results[i] for i in sorted(results)], out_header,
                             group_by=group_by, cleanup=cleanup)

    return dict(n_runs=n_runs, n_success=n_success, cases=cases,
                results=results, errors=errors, merged=merged,
                map_object_path=map_object_path)


# --------------------------------------------------------------------------
#  group-aware merge: reassemble the sweep along the INPUT structure
# --------------------------------------------------------------------------
def merge_cases(results, out_header, *, group_by="group", cleanup=False):
    """Merge per-case archives into one archive per input group.

    Distinct from :func:`merge_archives`, which concatenates the DRAWS of one
    scan produced by the draw-parallel tier.  Here each case archive already
    holds a complete bouquet under its own ``scan/<scan_key>/`` group, so the
    merge copies those groups whole -- each case's own ``config_json``
    provenance and ``_baseline`` ride along inside.

    ``group_by="group"`` (the default) buckets cases by
    :attr:`~bouquet.config.CaseSpec.group` and writes ``{out_header}_{group}.h5``
    per bucket.  That is what preserves the shape of the input: N IDA ``.cdf``
    files in, N archives out, each holding that file's time slices as scan keys
    -- the layout :func:`~bouquet.plotting.plot_bouquet_timeseries` and
    :meth:`~bouquet.archive.BouquetArchive.scan` expect.  ``group_by=None``
    puts every case in one ``{out_header}.h5`` instead.

    Parameters
    ----------
    results : list of dict
        The per-case result dicts from :func:`run_case` (needs ``path``,
        ``group``, ``scan_key``).
    out_header : str
        Stem for the merged archive(s); a group suffix is appended per bucket.
    group_by : {"group", None}
    cleanup : bool
        Delete each per-case archive once it has been copied in.

    Returns
    -------
    dict
        ``{group: merged_path}`` (key ``None`` when ``group_by is None``).
    """
    import h5py
    from .utils import _scan_key, initialize_equilibrium_database, write_provenance

    buckets = {}
    for r in results:
        if not r or not r.get("path"):
            continue
        key = r.get("group") if group_by == "group" else None
        buckets.setdefault(key, []).append(r)

    merged = {}
    for gname, rows in buckets.items():
        # A duplicate scan_key inside a group would silently overwrite one
        # case with another -- the sweep would look complete but hold fewer
        # bouquets than it ran. Refuse rather than lose a case.
        seen = {}
        for r in rows:
            k = _scan_key(r["scan_key"])
            if k in seen:
                raise ValueError(
                    f"duplicate scan_key {r['scan_key']!r} in group {gname!r}: "
                    f"cases {seen[k]} and {r['idx']} would overwrite each other "
                    "in the merged archive. Give each case a unique scan_key.")
            seen[k] = r["idx"]

        stem = f"{out_header}_{gname}" if gname is not None else out_header
        out_path = os.path.abspath(f"{stem}.h5")
        if os.path.exists(out_path):
            os.remove(out_path)                  # fresh archive (init opens append)
        initialize_equilibrium_database(stem)

        with h5py.File(out_path, "a") as dst:
            for r in sorted(rows, key=lambda x: x["idx"]):
                src_path = r["path"]
                if not os.path.exists(src_path):
                    raise FileNotFoundError(
                        f"case archive not found: {src_path} (group {gname!r}, "
                        f"scan_key {r['scan_key']!r}). Merging a partial group "
                        "silently shrinks the sweep -- re-run that case, or "
                        "drop it explicitly from `results`.")
                grp = f"scan/{_scan_key(r['scan_key'])}"
                with h5py.File(src_path, "r") as src:
                    if grp not in src:
                        raise KeyError(
                            f"{src_path} has no '{grp}' group -- the case ran "
                            f"with a different scan_key than its CaseSpec "
                            f"declared ({r['scan_key']!r}).")
                    dst.copy(src[grp], grp)

        # File-level schema/version/updated stamp. The authoritative per-scan
        # config_json came across inside each copied group, so no config here.
        write_provenance(stem)
        merged[gname] = out_path
        print(f"[bouquet_cases] merged {len(rows)} case(s) -> {out_path}",
              flush=True)

        if cleanup:
            for r in rows:
                if os.path.exists(r["path"]):
                    os.remove(r["path"])

    return merged
