"""
Parallel bouquet runner

Runs `re_generate_bouquet` on a D3D IDA file and eqdsk list.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — remove for interactive use
import matplotlib.pyplot as plt
import xarray as xr


# test options
PLOT_ONLY=True
remake_dir=False        # If true, deletes pre-existing working directory on re-runs
verbose=False             # If false, worker outputs are printed to individual log files

# compute options
use_logical_cpus=True    # Multi-thread based on hardware (use with caution if you're not on linux)

#General stuff
OFT_HOME = '/home/benjamins/src/OpenFUSIONToolkit'
OFT_PATH = '/home/benjamins/src/OpenFUSIONToolkit/builds/install_release/python'
if OFT_PATH: sys.path.append(OFT_PATH)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TMDB_v0_dir = "/cscratch/benjamins/TMDB_v0"
IDA_dir  = os.path.join(TMDB_v0_dir,"files/Laszlo_v0_IDA")
BOUQUET_DIR = '/home/benjamins/src/bouquet'
OFT_INSTALL_DIR = '/home/benjamins/src/OpenFUSIONToolkit/builds/install_release'
OFT_PYTHON_PATH = '/home/benjamins/src/OpenFUSIONToolkit/builds/install_release/python'

# Set up Python path for this process and subprocesses
_python_path_entries = [BOUQUET_DIR, OFT_INSTALL_DIR, OFT_PYTHON_PATH]
_existing_pythonpath = os.environ.get('PYTHONPATH', '')
_new_pythonpath = ':'.join(_python_path_entries + ([_existing_pythonpath] if _existing_pythonpath else []))
os.environ['PYTHONPATH'] = _new_pythonpath

# Also add to sys.path for immediate use in this script
for path in _python_path_entries:
    if path and path not in sys.path:
        sys.path.insert(0, path)

#Input files 
shot_nums = [174956,174990,175028]
# Get shot from command-line argument or use first in list
shot = int(sys.argv[1]) if len(sys.argv) > 1 else 174956
assert shot in shot_nums, f"shot {shot} must be in shot_nums"

IDA_filename = os.path.join(IDA_dir,f"IDA_{shot}_downsampled_final.xr")
xri = xr.open_dataset(IDA_filename)
geqdsks = xri.geqdsk_path.values
xri = None

MESH_FILE = os.path.join(
    '/home/benjamins/src/OpenFUSIONToolkit/builds/install_release/examples/TokaMaker/DIIID',
    'DIIID_mesh.h5',
)
target_currents = {
    'ECOILA': -0.977888676757812 / 61.0,
    'ECOILB': -0.962711173828125 / 61.0,
    'F1A':  0.115971984375,       'F1B':  0.128368578125,
    'F2A':  0.05980789453125,     'F2B':  0.0763701328125,
    'F3A': -0.03076001171875,     'F3B': -0.02234583203125,
    'F4A': -0.0401077421875,      'F4B': -0.13314096875,
    'F5A':  0.000723009033203125, 'F5B':  0.000399709045410156,
    'F6A': -0.1178582578125,      'F6B': -0.15356990625,
    'F7A':  0.0341264296875,      'F7B':  0.0415082109375,
    'F8A': -0.05660116015625,     'F8B': -0.05138975390625,
    'F9A':  0.236625375,          'F9B':  0.252380265625,
}

# ============================================================================
# ============================================================================
#DEPENDENT ENV STUFF stuff
OUTPUT_DIR = os.path.join(TMDB_v0_dir, 'files/Laszlo_v0/test4')
HEADER = str('D3D_V0_test'+str(shot))
# ============================================================================

# bouquet_method is the per-run worker function called by parallel_runner
from bouquet.parallel import (
    re_generate_bouquet, load_IDA_file_obj, _mesh_config_simp,
    FractionalUncertainty,
)

# bouquet_method is the per-run worker function called by parallel_runner
bouquet_method = re_generate_bouquet

# ============================================================================
# ============================================================================
# 3. SUS STUFF           
# perturbed equilibria per baseline
n_equils = int(sys.argv[2]) if len(sys.argv) > 2 else 5
oft_maxits = 150

frac_jphi = 0.10   # 10 % on j_phi
n_ls            = 0.5           # GPR correlation length — density  (psi_N units)
t_ls            = 0.4           # GPR correlation length — temperature
j_ls            = 0.25          # GPR correlation length — current density
jBS_scale_range = [0.9, 1.1]    # uniform random scale on j_BS per sample
n_k                  = 5
psi_bridge           = 0.99
l_i_tolerance        = 5.0 # percent
constrain_sawteeth   = True
recalculate_j_BS     = True
jphi_baseline        = True
# ============================================================================

# ============================================================================
# ============================================================================
# ---- coil-drift / homotopy / in-spec (DIII-D +/-2% measurement spec) ----
# Everything is a FRACTION (decimal), never a percentage.
coil_drift      = 0.02                   # +/-1% hard coil-current bound
homotopy_passes = [(0.1, 0.10), (0.03, 0.05)] #, (0.02, 0.03)]  # (F, VSC) limits
inspec_F_max    = 0.03                   # in-spec non-VSC F-coil drift
inspec_VSC_max  = 0.03                   # in-spec VSC (F9A/F9B) drift
vsc_soft_reg_weight = 1.0
p_thresh        = 0.04                   # pressure-match tolerance
isoflux_weight = 500.0    # uniform weight on all isoflux boundary points
# ============================================================================

# OTHER OPTIONS
KEEP_GEQDSK = False
use_python_solve = False
ida_reader_kwargs = {
    'carbon_quasi_neutrality': True,  # use n_12C6 CDF variable for ni
}
ida_uncertainty_kwargs = {}
pad_psi         = 1e-4    # LCFS psi padding for TokaMaker queries
djBS_tol        = 1e-4

config = {
    # --- TokaMaker / mesh ---
    "mesh_file":               MESH_FILE,
    "header":                  HEADER,
    "mesh_config_function":    _mesh_config_simp,
    "oft_order":               3,
    "oft_maxits":              oft_maxits,
    "oft_python_path":         OFT_PATH,
    # --- Profile I/O (IDA) ---
    "profile_reader_kwargs":         ida_reader_kwargs,
    "uncertainty_generator_kwargs":  ida_uncertainty_kwargs,
    # --- Bouquet sampling ---
    "n_equils":       n_equils,
    "n_ls":           n_ls,
    "t_ls":           t_ls,
    "j_ls":           j_ls,
    "psi_pad":        pad_psi,
    "isoflux_weight": isoflux_weight,
    "jphi_uncertainty_gen": FractionalUncertainty(frac_jphi),
    "keep_geqdsk":    KEEP_GEQDSK,
    # --- Optional coil regularisation ---
    "target_currents": target_currents,
    # --- reconstruct_equilibrium keyword overrides ---
    "reconstruct_equilibrium_kwargs": {
        "n_k":                 n_k,
        "psi_bridge":          psi_bridge,
        #"taper_edge_jBS":      False,
        "use_python_solve":    use_python_solve,
        "djBS_tol":            djBS_tol
    },
    "generate_bouquet_kwargs": {
        "l_i_tolerance":       l_i_tolerance,
        "psi_pad":             pad_psi,
        "constrain_sawteeth":  constrain_sawteeth,
        "jBS_scale_range":     jBS_scale_range,
        "recalculate_j_BS":    recalculate_j_BS,
        #"taper_edge_jBS":      False,
        "use_python_solve":    use_python_solve,
        "jphi_baseline":       jphi_baseline,
        "coil_drift":          coil_drift,
        "homotopy_passes":     homotopy_passes,
        "inspec_F_max":        inspec_F_max,
        "inspec_VSC_max":      inspec_VSC_max,
        "vsc_soft_reg_weight": vsc_soft_reg_weight,
        "p_thresh":            p_thresh,
        "djBS_tol":            djBS_tol
    }
}
load_files_obj = load_IDA_file_obj(config)

# ============================================================================
# 4. Pre-flight checks 
# ============================================================================

def _compute():
    from bouquet.parallel import _get_num_cpus as _parallel_get_num_cpus
    n_cpus, _ = _parallel_get_num_cpus()
    if n_cpus < 2:
        print(f'Warning: only {n_cpus} CPU core(s) available.  '
              'Parallel run requires at least 2 cores.')
        sys.exit(0)
    print(f'Running parallel run with {n_cpus} CPU core(s) available.')

    # Clean output directory for a fresh run
    if os.path.exists(OUTPUT_DIR) and remake_dir:
        import shutil as _shutil
        _shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nChecking required input files...')
    missing = [f for f in list(geqdsks) + [IDA_filename, MESH_FILE] if not os.path.exists(f)]
    if missing:
        print('ERROR: the following files were not found:')
        for f in missing:
            print(f'  {f}')
        print('\nAdjust OFT_PATH / MESH_FILE and retry.')
        sys.exit(1)
    print(f'  {len(geqdsks)} geqdsk file(s) and IDA CDF found.')
    print(f'  Mesh: {MESH_FILE}')

# ============================================================================
# 5. Compute in parallel
# ============================================================================
    print(f'\nLaunching parallel run into: {OUTPUT_DIR}')
    print(f'  {len(geqdsks)} equilibria, {n_equils} perturbed samples each')
    print()

    from bouquet.parallel import parallel_runner
    errors, outputs = parallel_runner(
        [(IDA_filename, geqdsks)],
        load_files_obj,
        bouquet_method,
        OUTPUT_DIR,
        use_logical_cpus=use_logical_cpus,
        verbose=verbose,  
    )

# ============================================================================
# 6. Error report
# ============================================================================
    if errors:
        print(f'\nWARNING: {len(errors)} run(s) failed:')
        for idx, tb in errors.items():
            print(f'  [run {idx}] {tb.splitlines()[-1]}')
    else:
        print(f'\nAll runs completed successfully.')
    return errors

def _plot():
    # ============================================================================
    # 7. Visualize results
    # ============================================================================
    # parallel_runner saves one HDF5 database per run, named:
    #   {HEADER}_idx{idx}.h5
    # located in worker subdirectories under OUTPUT_DIR.
    PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR,'bouquet_plots')
    #if os.path.exists(PLOT_OUTPUT_DIR):
    #    import shutil as _shutil
    #    _shutil.rmtree(PLOT_OUTPUT_DIR)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    import glob
    import pickle as pkl
    from collections import defaultdict
    from bouquet import read_geqdsk, plot_bouquet, plot_geqdsk_bouquet

    # Collect all HDF5 databases
    print(f'Searching for HDF5 files')
    # Directories to search have the format OUTPUT_DIR/worker_*/
    h5_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'worker_*', f'{HEADER}_idx*.h5')))
    
    if not h5_files:
        print('\nNo HDF5 result files found; skipping plots.')
        sys.exit(0 if not errors else 1)

    print(f'\n{"="*60}')
    print(f'Visualizing results ({len(h5_files)} equilibrium/database file(s))')
    print('=' * 60)

    print(geqdsks)
    #print(h5_files)

    h5_files = sorted(h5_files, key=lambda x: int(x.split('idx')[1].split('.')[0]))

    print(h5_files)
    #sys.exit()

    # Baseline g-file shapes
    print('\nPlotting baseline g-files...')
    geqdsks_loaded = []
    for i,path in enumerate(geqdsks):
        print("Loading eqdsk num ",i," of ",len(geqdsks))
        geqdsks_loaded.append(read_geqdsk(path))
    #plot_geqdsk_bouquet(geqdsks_loaded, x_coord='rho')
    #out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_baseline_geqdsk.png')
    #plt.savefig(out, dpi=150, bbox_inches='tight')
    #plt.close()
    #print(f'  Saved: {out}')

    # Per-database plots (one per run/equilibrium)
    for i, h5_path in enumerate(h5_files):
        print("Running timeslice",i)
        tag = f'idx{i}'

        try:
            print(f'\nPlotting bouquet — {tag}...')
            plot_bouquet(h5_path, scan_value=None, mode='all')
            out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_bouquet_{tag}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {out}')
        except KeyError as e:
            print(f'  Skipped (empty file): {e}')

        try:
            print(f'Plotting perturbed g-files — {tag}...')
            plot_geqdsk_bouquet(h5path=h5_path, x_coord='rho')
            out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_{tag}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {out}')
        except (KeyError, ValueError) as e:
            print(f'  Skipped (empty file): {e}')


if __name__ == '__main__':
    if PLOT_ONLY:
        errors = {}
        print(f'\nPLOT_ONLY mode: skipping compute, loading results from:\n  {OUTPUT_DIR}')
        _plot()
    else: 
        errors = _compute()
