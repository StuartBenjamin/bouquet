"""Method-agnostic parallel bouquet runner
========================

Distributes ``(input_files, load_files_obj, bouquet_method)`` 
across available CPU cores. Each case (ie. each timeslice, kinetic equilibrium, shot)
has an associated tuple of input file names, which are read into python using the specified 
load method, and then passed to the bouquet method. parallel_runner distributes 
these cases across available CPU cores and runs them in parallel.

'Non atomic' input files with multiple timeslices/kinetic equilibria (eg. IDA files) 
are supported by optional atomic_input_recast and atomic_load_files methods inside load_files_obj.

Basic pfile example: 
    input_files = (eqdsk, pfile)
    load_files_obj.load_files = load_eqdsk_pfile
    bouquet_method = re_generate_bouquet

"""

###########################################################################################################
# General parallel functions
###########################################################################################################

import os
import sys
import queue
import shutil
import socket
import traceback
import pickle as pkl
import multiprocessing
import numpy as np

# Module-level state populated by _init_worker in each spawned worker process.
_worker_state: dict = {}

class _IndexMap:
    """Picklable map_object: ``map_object(idx)`` returns ``flat_list[idx]``.

    ``map_object`` that can be pickled & saved to disk by ``parallel_runner``.
    """
    def __init__(self, flat_list):
        self.flat_list = flat_list

    def __call__(self, idx):
        return self.flat_list[idx]

    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)

class FractionalUncertainty:
    """Picklable callable that returns ``frac * |x|``.

    Use as ``config['jphi_uncertainty_gen']`` when the j_phi uncertainty
    is a fixed fraction of the fitted current profile::

        config["jphi_uncertainty_gen"] = FractionalUncertainty(0.10)  # 10 %

    """
    def __init__(self, frac):
        self.frac = frac

    def __call__(self, x):
        return self.frac * np.abs(x)

def parallel_runner(all_input_files, load_files_obj, bouquet_method, master_working_dir,
                    chunksize='automatic', use_logical_cpus=True, n_cpus_override=None,
                    verbose=False, keep_output=False):
    """Run a bouquet method in parallel across available CPU cores (single node).
    all_input_files must be a list of tuples, where each tuple contains the input files for a single 'case',
    matching the expected input of load_files_obj.load_files.

    Parameters
    ----------
    verbose : bool
        ``False`` (default): each worker's output is redirected to a per-worker
        log file (``<master_working_dir>/worker_N.log``); the terminal only
        shows brief per-worker status lines from the parent process.
        ``True``: no redirection — all worker output streams directly to the
        terminal (asynchronously, used for debugging).
    """

    #===================================================================================
    # Chunking logic
    #===================================================================================

    if n_cpus_override is not None:
        n_cpus, nthreads = n_cpus_override, 1
    else:
        n_cpus, nthreads = _get_num_cpus(use_logical=use_logical_cpus)
    n_runs, map_object = load_files_obj.total_runs(all_input_files)
    if n_runs == 0:
        print("[bouquet_parallel] No runs to execute.")
        return {}, {}
    n_workers = min(n_cpus, n_runs)
    print(
        f"[bouquet_parallel] Distributing {n_runs} runs across "
        f"{n_workers} workers ({n_cpus} CPUs available, {nthreads} thread(s)/worker)."
    )

    if not load_files_obj.is_atomic:
        _all_input_files = load_files_obj.atomic_input_recast(all_input_files)
        load_files = load_files_obj.atomic_load_files
    else:
        _all_input_files = all_input_files
        load_files = load_files_obj.load_files
    assert len(_all_input_files) == n_runs, (
        f"Expected {n_runs} runs from load_files_obj.total_runs, but got "
        f"{len(_all_input_files)} from load_files_obj.atomic_input_recast"
    )

    if chunksize == 'automatic':
        # Heuristic: 10x more tasks than workers, but no more than 1000 tasks per chunk
        chunksize = max(1, min(1000, n_runs // (10 * n_workers)))
        print(f"[bouquet_parallel] Using chunksize={chunksize} for dynamic scheduling.")
    else:
        print(f"[bouquet_parallel] Using user-specified chunksize={chunksize} for dynamic scheduling.")

    # Save map_object so users can look up input files by idx after the run.
    map_object_path = os.path.join(master_working_dir, "map_object.pkl")
    with open(map_object_path, "wb") as f:
        pkl.dump(map_object, f)
    print(f"[bouquet_parallel] Saved input file map to {map_object_path}")

    #===================================================================================
    # Pool setup
    #===================================================================================

    os.makedirs(master_working_dir, exist_ok=True)

    # 'spawn' avoids fork-safety issues with Fortran shared libraries in OFT
    ctx = multiprocessing.get_context("spawn")
    errors = {}
    outputs = {}

    # Each worker reports (worker_id, None) on success or
    # (worker_id, traceback_str) on failure via this queue.  We wait for
    # all n_workers to report before dispatching any tasks so that a
    # broken initializer causes an immediate, clean failure instead of a
    # silent hang in imap_unordered.
    
    init_status_queue = ctx.Queue()

    # Hand each spawned worker a unique ID via a pre-loaded queue.
    worker_id_queue = ctx.Queue()
    for w in range(n_workers):
        worker_id_queue.put(w)

    # Inject nthreads into a config copy so _init_OFT sets thread counts correctly.
    _config = dict(load_files_obj.config)
    _config["_nthreads"] = nthreads
    _config["_verbose"]  = verbose

    _pool = ctx.Pool(
        processes=n_workers,
        initializer=load_files_obj.init_worker,
        initargs=(worker_id_queue, master_working_dir, _config, init_status_queue),
    )

    # Barrier: wait for every worker to finish initialising, terminate if there's a failure
    init_failures = []
    for _ in range(n_workers):
        try:
            wid, tb = init_status_queue.get(timeout=120)  # 2 min per worker
        except queue.Empty:
            init_failures.append((-1, "Worker initialisation timed out (> 120 s)"))
        else:
            if tb is not None:
                init_failures.append((wid, tb))
            else:
                if verbose:
                    print(f"[bouquet_parallel] Worker {wid} ready.", flush=True)
                else:
                    log = os.path.join(master_working_dir, f"worker_{wid}.log")
                    print(f"[bouquet_parallel] Worker {wid} ready  (log: {log})", flush=True)

    if init_failures:
        _pool.terminate()
        _pool.join()
        msgs = "\n".join(
            f"  Worker {wid}:\n{tb}" for wid, tb in init_failures
        )
        raise RuntimeError(
            f"[bouquet_parallel] FATAL: {len(init_failures)} worker(s) failed "
            f"to initialise:\n{msgs}"
        )
    
def _get_num_cpus(use_logical=True):
    """Return ``(n_workers, nthreads_per_worker)`` for spawning OFT workers.

    Puportedly works on Linux HPC cluster (SLURM, PBS, LSF, SGE) and degrades
    gracefully on non-Linux systems (macOS, Windows).

    Parameters
    ----------
    use_logical : bool
        ``True`` (default): one worker per logical CPU (hyperthread),
        ``nthreads=1``.

        ``False``: one worker per physical core, ``nthreads = logical/physical``.
        Uses OFT's OpenMP intra-core parallelism.

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
        # CPU reservation — observed on some SLURM configurations).
        _SCHEDULER_CPU_VARS = (
            "SLURM_CPUS_PER_TASK",   # SLURM
            "PBS_NUM_PPN",            # PBS (CPUs per node)
            "LSB_DJOB_NUMPROC",       # IBM LSF
            "NSLOTS",                 # SGE / Grid Engine
        )
        for var in _SCHEDULER_CPU_VARS:
            val = os.environ.get(var)
            if val is not None:
                n_logical = min(n_logical, int(val))
                break
        return n_logical, 1
    else:
        return n_physical, nthreads_per_core

    
def _init_OFT(worker_id_queue, master_working_dir, config, init_status_queue):
    """Pool initialiser: set up OFT/TokaMaker once per spawned worker process.

    Called automatically by ``Pool`` (via ``load_files_obj.init_worker``) before
    any tasks are dispatched.  Each worker claims a unique ID from
    *worker_id_queue*, creates a private working directory, copies the mesh
    file locally, initialises OFT and TokaMaker, and stores all shared state
    in the module-level ``_worker_state`` dict for use by ``_run_one``.

    On success posts ``(worker_id, None)`` to *init_status_queue*.
    On failure posts ``(worker_id, traceback_str)``.

    Parameters
    ----------
    worker_id_queue : multiprocessing.Queue
        Pre-loaded with integers 0…n_workers−1.  Each worker pops one value
        to claim its unique ID.
    master_working_dir : str
        Root directory under which per-worker subdirectories are created.
    config : dict
        Shared configuration dict (general options, not worker-specific state).
        Must contain ``mesh_file``, ``header``, ``mesh_config_function``, and
        any keys required by that function (e.g. ``oft_order``).
    init_status_queue : multiprocessing.Queue
        Used to signal initialisation success or failure back to the main
        process barrier.
    """
    global _worker_state
    worker_id = -1  # fallback if queue.get() itself fails
    try:
        nthreads = config.get("_nthreads", 1)
        os.environ["OMP_NUM_THREADS"]      = str(nthreads)
        os.environ["MKL_NUM_THREADS"]      = str(nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
        os.environ["NUMEXPR_NUM_THREADS"]  = str(nthreads)
        print(
            f"[Worker {worker_id}] thread env: "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
            f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}",
            flush=True,
        )

        # Use a timeout so replacement workers (spawned after a crash) fail fast
        # rather than blocking forever and deadlocking the parent imap_unordered.
        try:
            worker_id = worker_id_queue.get(timeout=60)
        except Exception:
            raise RuntimeError(
                "[bouquet_parallel] Worker ID queue empty — this is a pool replacement "
                "for a dead worker. Cannot initialise."
            )
        working_dir = os.path.abspath(os.path.join(master_working_dir, f"worker_{worker_id}"))
        os.makedirs(working_dir, exist_ok=True)
        master_working_dir = os.path.abspath(master_working_dir)   # ← anchor before chdir
        os.chdir(working_dir)

        # Redirect this worker's stdout/stderr to a per-worker log file.  
        # os.dup2 at the file-descriptor level also captures
        # output written directly to fd 1/2 by Fortran/C extensions (e.g. OFT).
        # Skipped when config["_verbose"] is True so output goes to the terminal.
        log_path = os.path.join(master_working_dir, f"worker_{worker_id}.log")
        if not config.get("_verbose", False):
            _log_fh = open(log_path, "w", buffering=1)  # line-buffered
            os.dup2(_log_fh.fileno(), 1)
            os.dup2(_log_fh.fileno(), 2)
            sys.stdout = _log_fh
            sys.stderr = _log_fh

        # Add the OFT python directory to sys.path if supplied.
        # This is required when using spawned processes because sys.path
        # modifications in the parent process are not inherited by children.
        oft_python_path = config.get("oft_python_path")
        if oft_python_path and oft_python_path not in sys.path:
            sys.path.insert(0, oft_python_path)

        # Copy the mesh HDF5 into this worker's private directory so that
        # concurrent HDF5 opens by multiple workers do not trigger file-locking
        # conflicts in a serial HDF5 build.  working_dir is already absolute so
        # local_mesh_file is absolute regardless of what os.getcwd() is now.
        local_mesh_file = os.path.join(working_dir, os.path.basename(config["mesh_file"]))
        shutil.copy2(config["mesh_file"], local_mesh_file)

        from OpenFUSIONToolkit import OFT_env
        from OpenFUSIONToolkit.TokaMaker import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun

        from bouquet import (
            read_geqdsk,
            reconstruct_equilibrium,
            generate_bouquet,
            initialize_equilibrium_database,
            store_equilibrium,
        )

        myOFT = OFT_env(nthreads=nthreads)
        mygs = TokaMaker(myOFT)

        config['mesh_config_function'](mygs, config, local_mesh_file)

        print(
            f"[Worker {worker_id}] OFT initialised — "
            f"host={socket.gethostname()}, PID={os.getpid()}, "
            f"cwd={working_dir}",
            flush=True,
        )

        _worker_state.update({
            "worker_id":                     worker_id,
            "working_dir":                   working_dir,
            "log_path":                      log_path,
            "config":                        config,
            "mygs":                          mygs,
            "read_geqdsk":                   read_geqdsk,
            "reconstruct_equilibrium":       reconstruct_equilibrium,
            "generate_bouquet":              generate_bouquet,
            "store_equilibrium":             store_equilibrium,
            "initialize_equilibrium_database": initialize_equilibrium_database,
            "create_power_flux_fun":         create_power_flux_fun,
        })

        init_status_queue.put((worker_id, None))  # signal success to main process

    except Exception:
        tb = traceback.format_exc()
        print(f"[Worker {worker_id}] INIT FAILED:\n{tb}", flush=True)
        try:
            init_status_queue.put((worker_id, tb))
        except Exception:
            pass
        raise  # kill this worker process

