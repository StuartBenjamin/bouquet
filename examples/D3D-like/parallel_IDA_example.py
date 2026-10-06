"""
Case-parallel bouquet sweep -- DIII-D IDA files (one bouquet per time slice)
===========================================================================

Runs one complete bouquet per TIME SLICE of each IDA ``.cdf``, all slices at
once across the machine's cores.  A single ``.cdf`` holds many slices, so the
inputs are *non-atomic*: :class:`bouquet.IdaTimeslices` expands each file into
one atomic case per slice (pairing it with that slice's g-file), and the pool
treats every slice as an independent bouquet.

The grouping survives to the output.  Cases expanded from one ``.cdf`` share a
``group``, so the merge step writes **one archive per input file**, with the
slices stored inside as ``scan/<time_ms>/`` groups -- the layout
``bq.plot_bouquet_timeseries`` reads.  Many IDA files in, one timeseries
archive each out:

    IDA_shot1.cdf  (5 slices)  ->  <HEADER>_IDA_shot1.h5   scan/2000, 2500, ...
    IDA_shot2.cdf  (3 slices)  ->  <HEADER>_IDA_shot2.h5   scan/4000, 4500, ...

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
HERE = os.path.dirname(os.path.realpath(__file__))

# One entry per IDA file: (cdf_path, geqdsk_paths).  geqdsk_paths must hold
# EXACTLY one g-file per time slice in that .cdf, in time order -- the expander
# checks this against the file and raises if they disagree, rather than
# silently pairing the wrong equilibrium with a slice.
ida_inputs = [
    # ('/path/to/IDA_194123.cdf', ['/path/to/g194123.02000',
    #                              '/path/to/g194123.02500',
    #                              '/path/to/g194123.03000']),
    # ('/path/to/IDA_194124.cdf', [...]),
]

MESH_FILE = os.path.join(HERE, 'DIIID_mesh.h5')

# Working directory: per-worker scratch dirs + logs, per-case archives under
# cases/, and the merged per-input-file archives.
OUTPUT_DIR = os.path.join(HERE, 'output_parallel_IDA')

# Archive base name.  The merge writes <HEADER>_<cdf stem>.h5 per input file.
HEADER = 'TkMkr_D3Dlike_Hmode_parallel_IDA'

# ============================================================================
# 2. The shared config -- everything except the baseline source
# ============================================================================
# Each case supplies its own source (its g-file + the .cdf at one time); the
# rest is common to the sweep.  The IDA reader is picked automatically from the
# .cdf extension, and the same file supplies the sigma envelopes.

_placeholder = (ida_inputs[0][0] if ida_inputs else 'IDA.cdf')
_placeholder_g = (ida_inputs[0][1][0] if ida_inputs else 'g.geqdsk')

config = bq.BouquetConfig(
    # Placeholder source: replaced per case by parallel_cases.
    source=bq.ReconstructionSource(geqdsk_path=_placeholder_g,
                                   profiles_path=_placeholder,
                                   cocos=1,
                                   impurity_Z=6.0,     # carbon wall (DIII-D)
                                   psi_pad=1e-4),
    solver=bq.SolverConfig(mesh_path=MESH_FILE, order=3, nthreads=1),
    output_header=HEADER,
)

# --- uncertainty envelope --------------------------------------------------
# The IDA file carries its own measured *_err sigmas, so the kinetic envelopes
# are read from it rather than set as flat fractions; only j_phi (which IDA has
# no uncertainty for) needs a scalar here.
unc = config.uncertainty
unc.sigma_mode = 'auto'         # 'direct' (*_err) vs 'ensemble', inferred
unc.jphi_scalar_sigma = 0.15    # 15% on j_phi
unc.n_ls = 0.5                  # GPR correlation length -- density  (psi_N units)
unc.t_ls = 0.4                  # GPR correlation length -- temperature
unc.j_ls = 0.25                 # GPR correlation length -- current density

# --- sampling / acceptance -------------------------------------------------
gen = config.generation
gen.n_equils = 5                # perturbed equilibria per time slice
gen.l_i_tolerance = 0.05
gen.constrain_sawteeth = True
gen.recalculate_j_BS = True
gen.coil_drift = 0.01
gen.homotopy_passes = [(0.1, 0.10), (0.02, 0.05), (0.015, 0.03)]

# --- in-spec filtering (DIII-D +/-2% coil measurement spec) ----------------
config.filtering.inspec_F_max = 0.02
config.filtering.inspec_VSC_max = 0.02
config.filtering.rms_max_mm = 5.0

# ============================================================================
# 3. The parallel source -- one case per time slice, grouped by input file
# ============================================================================
parallel_source = bq.IdaTimeslices(
    header=HEADER,
    inputs=ida_inputs,
    source_kwargs=dict(cocos=1, impurity_Z=6.0, psi_pad=1e-4),
)


# ============================================================================
# 4. Pre-flight checks
# ============================================================================
def _preflight():
    from bouquet.parallel import _get_num_cpus
    n_cpus, nthreads = _get_num_cpus(use_logical=use_logical_cpus)
    print(f'Running with {n_cpus} CPU core(s), {nthreads} thread(s) per worker.')

    if not ida_inputs:
        print('ERROR: `ida_inputs` is empty -- add (cdf_path, geqdsk_paths) '
              'entries at the top of this script.')
        sys.exit(1)

    if os.path.exists(OUTPUT_DIR) and remake_dir:
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nChecking required input files...')
    needed = [MESH_FILE]
    for cdf, gfiles in ida_inputs:
        needed.append(cdf)
        needed.extend(gfiles)
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print('ERROR: the following files were not found:')
        for f in missing:
            print(f'  {f}')
        sys.exit(1)
    print(f'  {len(ida_inputs)} IDA file(s) + all g-files found.')
    print(f'  Mesh: {MESH_FILE}')


# ============================================================================
# 5. Compute in parallel
# ============================================================================
def _compute():
    _preflight()

    # expand() reads each .cdf's time axis; do it up front so the slice count
    # and the g-file pairing are validated before any worker is spawned.
    cases = parallel_source.expand()
    n_files = len({c.group for c in cases})
    print(f'\nLaunching {len(cases)} case(s) from {n_files} IDA file(s) '
          f'into: {OUTPUT_DIR}')
    for c in cases:
        print(f'  {c.group}  t={c.source.time:.4f}s  (scan_key {c.scan_key})'
              if c.source.time is not None else f'  {c.group}  slice {c.scan_key}')
    print(f'\n  {config.generation.n_equils} perturbed samples each\n')

    summary = bq.parallel_cases(
        parallel_source,
        config,
        OUTPUT_DIR,
        use_logical_cpus=use_logical_cpus,
        verbose=verbose,
        merge=True,          # one merged archive per IDA input file
    )

    # ---- error report -----------------------------------------------------
    if summary['errors']:
        print(f"\nWARNING: {len(summary['errors'])} case(s) failed:")
        for idx, tb in summary['errors'].items():
            c = summary['cases'][idx]
            print(f"  [{idx}] {c.group} slice {c.scan_key}")
            print(f"    {tb.splitlines()[-1]}")
        print('  (full tracebacks in errors.pkl; the surviving slices were '
              'still merged)')
    else:
        print(f"\nAll {summary['n_runs']} case(s) completed successfully.")
    return summary


# ============================================================================
# 6. Visualise results -- one timeseries per IDA file
# ============================================================================
def _plot(merged):
    for group, path in sorted(merged.items()):
        print(f'\nPlotting {group}: {path}')
        try:
            bq.plot_bouquet_timeseries(path)
            out = os.path.join(OUTPUT_DIR, f'timeseries_{group}.png')
            plt.savefig(out, dpi=130, bbox_inches='tight')
            plt.close('all')
            print(f'  wrote {out}')
        except Exception as exc:          # plotting must not sink a good run
            print(f'  WARNING: timeseries plot failed for {group}: {exc}')


if __name__ == '__main__':
    if PLOT_ONLY:
        print(f'\nPLOT_ONLY: loading results from {OUTPUT_DIR}')
        import glob
        merged = {os.path.basename(p)[len(HEADER) + 1:-3]: p
                  for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, f'{HEADER}_*.h5')))}
    else:
        merged = _compute()['merged']
    _plot(merged)
