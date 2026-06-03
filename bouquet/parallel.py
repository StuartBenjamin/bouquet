"""Method-agnostic parallel bouquet runner
========================

Distributes ``(input_files, load_files_obj, bouquet_method)`` 
across available CPU cores. Each case (ie. each timeslice, kinetic equilibrium, shot)
has an associated tuple of input file names, which are read into python using the specified 
load method, and then passed to the bouquet method. parallel_runner distributes 
these cases across available CPU cores and runs them in parallel.

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
