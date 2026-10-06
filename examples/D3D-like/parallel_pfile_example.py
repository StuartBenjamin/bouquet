"""
Case-parallel bouquet sweep -- DIII-D-like (geqdsk, p-file) pairs
================================================================

Runs one complete bouquet per ``(geqdsk, p-file)`` pair, all pairs at once
across the machine's cores.  Each worker stands up TokaMaker ONCE and then runs
case after case on it (``OFT_env`` is a per-process singleton), so the solver
setup cost is paid per worker rather than per case.

The shape of the run is:

    one BouquetConfig      -- solver, uncertainties, sampling knobs (shared)
  + one ParallelSource     -- the raw inputs, expanded into atomic cases
  -> bq.parallel_cases(...) -- N independent bouquets, one archive each

Compare ``bouquet.parallel.parallel_generate``, which splits the DRAWS of a
single bouquet across workers instead.  Use that for one case; use this
whenever you have at least as many cases as cores.

Run it as a script (the ``if __name__ == "__main__"`` guard is required --
workers are spawned and re-import this module).
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')   # headless -- remove for interactive use
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# OFT / TokaMaker path -- adjust to your installation
# ---------------------------------------------------------------------------
OFT_PATH = ''    # e.g. '/home/you/src/OpenFUSIONToolkit/builds/install_release/python'
if OFT_PATH:
    sys.path.append(OFT_PATH)

# Add bouquet root so the package is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import bouquet as bq

# file options
PLOT_ONLY = False
remake_dir = True         # delete a pre-existing working directory on re-runs
verbose = False           # False: per-worker log files; True: everything to the terminal
use_logical_cpus = True   # one single-threaded worker per logical CPU

# ============================================================================
# 1. Input files
# ============================================================================
# Paths are relative to this script's directory so the example works from
# anywhere.

HERE = os.path.dirname(os.path.realpath(__file__))

# One (geqdsk, p-file) pair per case.  Point these at your own slices; the
# baseline shipped with the repo is repeated here only so the example runs.
pairs = [
    (os.path.join(HERE, 'D3Dlike_Hmode_baseline.geqdsk'),
     os.path.join(HERE, 'D3Dlike_Hmode_baseline.peqdsk')),
]

MESH_FILE = os.path.join(HERE, 'DIIID_mesh.h5')

# Working directory: per-worker scratch dirs + logs, per-case archives under
# cases/, and the merged per-input archives.
OUTPUT_DIR = os.path.join(HERE, 'output_pfile_parallel')

# Archive base name.  Each case writes cases/<HEADER>_<stem>_idx<i>.h5, and the
# merge collects them into <HEADER>_<stem>.h5 (one per input pair).
HEADER = 'TkMkr_D3Dlike_Hmode_parallel'

# ============================================================================
# 2. The shared config -- everything except the baseline source
# ============================================================================
# Each case supplies its own source (its geqdsk + p-file); everything below is
# common to the whole sweep.  The p-file reader, the profile uncertainties and
# the reconstruction are all part of the standard pipeline now -- there is no
# reader/uncertainty-generator plumbing to wire up.

config = bq.BouquetConfig(
    # Placeholder source: replaced per case by parallel_cases.
    source=bq.ReconstructionSource(geqdsk_path=pairs[0][0],
                                   profiles_path=pairs[0][1],
                                   cocos=1,
                                   impurity_Z=6.0,     # carbon wall (DIII-D)
                                   psi_pad=1e-4,
                                   n_k=5,
                                   psi_bridge=0.99),
    solver=bq.SolverConfig(mesh_path=MESH_FILE, order=3, nthreads=1),
    output_header=HEADER,
)

# --- uncertainty envelope (fractional 1-sigma + GPR correlation lengths) ----
unc = config.uncertainty
unc.ne_scalar_sigma = 0.05      # 5% on electron density
unc.te_scalar_sigma = 0.05      # 5% on electron temperature
unc.ni_scalar_sigma = 0.05      # 5% on ion density
unc.ti_scalar_sigma = 0.05      # 5% on ion temperature
unc.jphi_scalar_sigma = 0.10    # 10% on j_phi
unc.n_ls = 0.5                  # GPR correlation length -- density  (psi_N units)
unc.t_ls = 0.4                  # GPR correlation length -- temperature
unc.j_ls = 0.25                 # GPR correlation length -- current density

# --- sampling / acceptance -------------------------------------------------
gen = config.generation
gen.n_equils = 5                # perturbed equilibria per baseline
gen.l_i_tolerance = 0.05
gen.constrain_sawteeth = True
gen.recalculate_j_BS = True
gen.coil_drift = 0.01           # +/-1% hard coil-current bound
gen.homotopy_passes = [(0.1, 0.10), (0.02, 0.05), (0.015, 0.03)]

# --- in-spec filtering (DIII-D +/-2% coil measurement spec) ----------------
config.filtering.inspec_F_max = 0.02
config.filtering.inspec_VSC_max = 0.02
config.filtering.rms_max_mm = 5.0

# ============================================================================
# 3. The parallel source -- raw inputs expanded into atomic cases
# ============================================================================
# One case per pair; each pair is its own group, so the merge leaves one
# archive per input pair.  Anything in source_kwargs is applied to every case.

parallel_source = bq.GeqdskProfilePairs(
    header=HEADER,
    pairs=pairs,
    source_kwargs=dict(cocos=1, impurity_Z=6.0, psi_pad=1e-4,
                       n_k=5, psi_bridge=0.99),
)


# ============================================================================
# 4. Pre-flight checks
# ============================================================================
def _preflight():
    from bouquet.parallel import _get_num_cpus
    n_cpus, nthreads = _get_num_cpus(use_logical=use_logical_cpus)
    print(f'Running with {n_cpus} CPU core(s), {nthreads} thread(s) per worker.')

    if os.path.exists(OUTPUT_DIR) and remake_dir:
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nChecking required input files...')
    needed = [f for pair in pairs for f in pair] + [MESH_FILE]
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print('ERROR: the following files were not found:')
        for f in missing:
            print(f'  {f}')
        print('\nAdjust the paths above and retry.')
        sys.exit(1)
    print(f'  All {len(pairs)} geqdsk + p-file pair(s) found.')
    print(f'  Mesh: {MESH_FILE}')


# ============================================================================
# 5. Compute in parallel
# ============================================================================
def _compute():
    _preflight()
    cases = parallel_source.expand()
    print(f'\nLaunching {len(cases)} case(s) into: {OUTPUT_DIR}')
    print(f'  {config.generation.n_equils} perturbed samples each\n')

    summary = bq.parallel_cases(
        parallel_source,
        config,
        OUTPUT_DIR,
        use_logical_cpus=use_logical_cpus,
        verbose=verbose,
        merge=True,          # one merged archive per input pair
    )

    # ---- error report -----------------------------------------------------
    if summary['errors']:
        print(f"\nWARNING: {len(summary['errors'])} case(s) failed:")
        for idx, tb in summary['errors'].items():
            print(f"  [{idx}] {summary['cases'][idx].source.geqdsk_path}")
            print(f"    {tb.splitlines()[-1]}")
    else:
        print(f"\nAll {summary['n_runs']} case(s) completed successfully.")
    return summary


# ============================================================================
# 6. Visualise results
# ============================================================================
def _plot(merged):
    """One overview figure per merged archive."""
    for group, path in sorted(merged.items()):
        print(f'\nPlotting {group}: {path}')
        arch = bq.BouquetArchive(path[:-3])
        for key in arch.scan_keys:
            try:
                bq.plot_bouquet(path, scan_key=key)
                out = os.path.join(OUTPUT_DIR, f'bouquet_{group}_{key}.png')
                plt.savefig(out, dpi=130, bbox_inches='tight')
                plt.close('all')
                print(f'  wrote {out}')
            except Exception as exc:      # plotting must not sink a good run
                print(f'  WARNING: plot failed for {group}/{key}: {exc}')


if __name__ == '__main__':
    if PLOT_ONLY:
        print(f'\nPLOT_ONLY: loading results from {OUTPUT_DIR}')
        import glob
        merged = {os.path.basename(p)[:-3]: p
                  for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, f'{HEADER}_*.h5')))}
    else:
        merged = _compute()['merged']
    _plot(merged)
