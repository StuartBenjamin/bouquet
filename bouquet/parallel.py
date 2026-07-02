"""Process-parallel bouquet generation.

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
each worker uses an independent RNG stream (``seed + worker_id``), so the union
is a statistically-equivalent *different* draw set. The baseline is identical.
"""
from __future__ import annotations

import copy
import os
import pickle

__all__ = [
    "run_shard",
    "merge_archives",
    "parallel_generate",
    "emit_slurm_script",
]


def _shard_size(total, n_workers, worker_id):
    """Draws assigned to *worker_id* when *total* is split over *n_workers*."""
    base, rem = divmod(int(total), int(n_workers))
    return base + (1 if worker_id < rem else 0)


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
        cfg.generation.seed = int(seed_base) + int(worker_id)  # independent stream
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
                   baseline_match_rtol=1e-6):
    """Concatenate per-worker shard archives into ``{out_header}.h5``.

    Draw groups are renumbered to a contiguous running index, and the raw
    ``.eqdsk`` / ``.pfile`` datasets are renamed to the merged header+count form
    so ``load_equilibrium`` (which addresses the eqdsk by reconstructed name)
    keeps working alongside the suffix-based readers. The ``_baseline`` group is
    copied once (the baseline is identical across workers). Returns
    ``(out_path, n_draws)``.

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
                        _group_path, _eqdsk_dataset_name)

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
                    # rename raw eqdsk/pfile to the merged header+count naming so
                    # load_equilibrium(out_header, count=offset) resolves them.
                    new_eqdsk = _eqdsk_dataset_name(out_header, scan_key, offset)
                    new_pfile = new_eqdsk.replace(".eqdsk", ".pfile")
                    for k in list(g.keys()):
                        if k.endswith(".eqdsk") and k != new_eqdsk:
                            g.move(k, new_eqdsk)
                        elif k.endswith(".pfile") and k != new_pfile:
                            g.move(k, new_pfile)
                    offset += 1

    if cleanup:
        for sp in shard_paths:
            if sp and os.path.exists(sp):
                os.remove(sp)
    return out_path, offset


# --------------------------------------------------------------------------
#  orchestration
# --------------------------------------------------------------------------
def parallel_generate(config, *, n_workers, threads_per_worker=1, seed=0,
                      backend="laptop", baseline_match_rtol=1e-6, cleanup=True,
                      verbose=False, progress=True, slurm=None):
    """Fan ``config.generation.n_equils`` draws across worker processes, merge.

    ``backend="laptop"`` runs a ``ProcessPoolExecutor`` (spawn) now;
    ``backend="slurm"`` writes a job-array + merge script via
    :func:`emit_slurm_script` (pass options as the ``slurm`` dict) and returns
    their paths without running.

    The cross-worker **baseline check**: every worker reports its forward-solved
    baseline ``l_i``/``Ip``; if any drifts beyond ``baseline_match_rtol`` the run
    raises (a worker did not converge to the shared baseline -- e.g. a stray
    ``nthreads>1`` or a mismatched source). Returns a summary dict.
    """
    n_total = int(config.generation.n_equils)
    scan_key = config.generation.scan_key
    out_header = config.output_header
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
                                        cleanup=cleanup)
    return dict(out_path=out_path, n_draws=n_merged, n_workers=nw,
                threads_per_worker=threads_per_worker,
                li_target=li0, Ip_target=ip0)


# --------------------------------------------------------------------------
#  cluster: emit a SLURM job-array + dependent merge
# --------------------------------------------------------------------------
def emit_slurm_script(config, *, n_workers, seed, threads_per_worker,
                      out_dir=".", job_name="bouquet", partition=None,
                      time_limit="02:00:00", mem_per_task="16G",
                      python="python"):
    """Write a SLURM job-array (one shard per task) + a dependent merge job.

    Serialises the run into ``{job_name}_bundle.pkl`` and writes two sbatch
    scripts plus a ``{job_name}_submit.sh`` that chains them with
    ``--dependency=afterok``. Nothing is launched. Returns the written paths.
    """
    _warn_multithreaded(threads_per_worker)
    os.makedirs(out_dir, exist_ok=True)
    bundle = dict(
        config=config, n_workers=int(n_workers), seed=int(seed),
        threads_per_worker=int(threads_per_worker),
        n_equils_total=int(config.generation.n_equils),
        scan_key=config.generation.scan_key,
        out_header=config.output_header,
    )
    bpath = os.path.join(out_dir, f"{job_name}_bundle.pkl")
    with open(bpath, "wb") as fh:
        pickle.dump(bundle, fh)

    part = f"#SBATCH --partition={partition}\n" if partition else ""
    # threads pinned to the task's cores; BLAS held to the same to avoid nesting.
    env = (f"export OMP_NUM_THREADS={threads_per_worker}\n"
           "export OMP_PROC_BIND=close OMP_PLACES=cores\n"
           f"export OPENBLAS_NUM_THREADS={threads_per_worker} "
           f"MKL_NUM_THREADS={threads_per_worker}\n")

    array = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}\n"
        f"#SBATCH --array=0-{n_workers - 1}\n"
        f"#SBATCH --cpus-per-task={threads_per_worker}\n"
        f"#SBATCH --time={time_limit}\n#SBATCH --mem={mem_per_task}\n{part}"
        f"{env}{python} -m bouquet.parallel shard {bpath} $SLURM_ARRAY_TASK_ID\n"
    )
    merge = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}_merge\n"
        f"#SBATCH --cpus-per-task=1\n#SBATCH --time=00:20:00\n"
        f"#SBATCH --mem={mem_per_task}\n{part}"
        f"{python} -m bouquet.parallel merge {bpath}\n"
    )
    submit = (
        "#!/bin/bash\n# chain the array + merge with an afterok dependency\n"
        f"aid=$(sbatch --parsable {job_name}_array.sbatch)\n"
        f"sbatch --dependency=afterok:$aid {job_name}_merge.sbatch\n"
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
    with open(bpath, "rb") as fh:
        b = pickle.load(fh)
    if cmd == "shard":
        run_shard(b["config"], int(argv[2]), b["n_workers"],
                  n_equils_total=b["n_equils_total"], seed_base=b["seed"],
                  out_header=b["out_header"], scan_key=b["scan_key"],
                  threads_per_worker=b["threads_per_worker"])
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
                                     scan_key=b["scan_key"], cleanup=True)
        print(f"merged {n} draws -> {out_path}")
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    _cli()
