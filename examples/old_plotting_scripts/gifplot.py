"""
Parallel bouquet runner

Runs `re_generate_bouquet` on a D3D IDA file and eqdsk list.
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — remove for interactive use
import matplotlib.pyplot as plt
import xarray as xr

# Set DIFF_BS true so that we'll read it via os.environ.get('DIFF_BS', '0') == '1'
#os.environ['DIFF_BS'] = '1'

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

# ---- helper to build the per-case paths ---------------------------------
def build_paths(outloc, shot=None):
    """Return (OUTPUT_DIR, HEADER, PLOT_OUTPUT_DIR) for a given outloc name."""
    # always extract shot from the first 6 digits of outloc
    m = re.match(r'(\d{6})', outloc)
    shot = int(m.group(1)) if m else None
    output_dir      = os.path.join(TMDB_v0_dir, 'files/Laszlo_v0', outloc)
    header          = f'D3D_V0_test{shot}' if shot is not None else f'D3D_V0_{outloc}'
    plot_output_dir = os.path.join(output_dir, 'bouquet_plots_gif')
    return output_dir, header, plot_output_dir, shot

# ============================================================================
# ============================================================================
# DEPENDENT ENV STUFF
#
# Two calling conventions:
#   1) Shot mode:  uv run gifplot.py 174956 250 bigT250
#   2) Dirs mode:  uv run gifplot.py --dirs 174956bigT250 174990bigT250 175028bigT125
# ============================================================================
if len(sys.argv) > 1 and sys.argv[1] == '--dirs':
    # -------- alternative: explicit list of OUTPUT_DIR names --------------
    OUTPUT_DIR_LIST = sys.argv[2:]
    assert OUTPUT_DIR_LIST, "provide at least one OUTPUT_DIR name after --dirs"
    RUN_CASES = [build_paths(outloc) for outloc in OUTPUT_DIR_LIST]
else:
    # -------- original: shot + time + suffix ------------------------------
    shot = int(sys.argv[1]) if len(sys.argv) > 1 else 174956
    assert shot in shot_nums, f"shot {shot} must be in shot_nums"
    assert len(sys.argv) > 3
    outloc = str(shot) + sys.argv[3]
    RUN_CASES = [build_paths(outloc, shot=shot)]

# Backward-compatible single-case names (first case), so existing code
# that references these directly still works:
OUTPUT_DIR, HEADER, PLOT_OUTPUT_DIR, shot = RUN_CASES[0]

def _plot(n=None, remove_previous=False, ranges=None, tags=None):
    # ============================================================================
    # 7. Visualize results
    # ============================================================================
    import glob as _glob
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    # Remove any pre-existing SVGs so old frames don't mix with new ones
    for _f in _glob.glob(os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_idx*.svg')):
        if remove_previous: os.remove(_f)

    import glob
    import pickle as pkl
    from collections import defaultdict
    from bouquet import read_geqdsk, plot_bouquet, plot_geqdsk_bouquet

    # Collect all HDF5 databases
    print(f'Searching for HDF5 files')
    h5_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'worker_*', f'{HEADER}_idx*.h5')))

    if not h5_files:
        print('\nNo HDF5 result files found; skipping plots.')
        sys.exit(0 if not errors else 1)

    h5_files = sorted(h5_files, key=lambda x: int(x.split('idx')[1].split('.')[0]))
    #h5_indices = [int(f.split('idx')[1].split('.')[0]) for f in h5_files]


    # --- limit to first n files if requested ---
    if n is not None:
        h5_files = h5_files[:n]
        print(f'[debug] Limiting to first {n} HDF5 file(s).')

    print(f'\n{"="*60}')
    print(f'Visualizing results ({len(h5_files)} equilibrium/database file(s))')
    print('=' * 60)

    print(h5_files)

    # ------------------------------------------------------------------
    # Normalize `tags` into a set for fast membership checks (if provided).
    # Accept either bare indices (e.g. 3) or full tags (e.g. 'idx3').
    # ------------------------------------------------------------------
    tag_filter = None
    if tags is not None:
        tag_filter = set()
        for t in tags:
            if isinstance(t, int):
                tag_filter.add(f'idx{t}')
            else:
                tag_filter.add(t if str(t).startswith('idx') else f'idx{t}')
        print(f'[debug] Restricting Pass 2 to tags: {sorted(tag_filter)}')

    # ------------------------------------------------------------------
    # FIRST PASS (cheap): derive global axis ranges via return_data_only,
    # accumulating across all time slices — no figures are drawn here.
    #
    # Skipped entirely if `ranges` is supplied by the caller.
    # ------------------------------------------------------------------
    def _expand(cur, new):
        """Accumulate a running [min, max] against a new (lo, hi) pair."""
        if new is None:
            return cur
        lo, hi = float(new[0]), float(new[1])
        if cur is None:
            return [lo, hi]
        return [min(cur[0], lo), max(cur[1], hi)]

    if ranges is not None:
        print('\n[Pass 1/2] Skipped — using caller-supplied ranges.')
        for k, v in ranges.items():
            print(f'    {k}: {v}')
    else:
        print('\n[Pass 1/2] Computing global axis ranges (data-only)...')
        ranges = {k: None for k in
                  ('lcfs_xlim', 'lcfs_ylim', 'xlim',
                   'p_ylim', 'q_ylim', 'j_ylim', 'ff_ylim')}

        for i, h5_path in enumerate(h5_files):
            print("Ranges from ",i)
            try:
                d = plot_geqdsk_bouquet(h5path=h5_path, x_coord='rho',
                                        return_data_only=True)
                if not isinstance(d, dict):
                    continue
                for k in ranges:
                    ranges[k] = _expand(ranges[k], d[k])
            except (KeyError, ValueError) as e:
                print(f'  [Pass 1] Skipped idx{i} (empty file): {e}')

        print('  Global ranges:')
        for k, v in ranges.items():
            print(f'    {k}: {v}')

        # Pickle dump the ranges
        ranges_pkl = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_ranges.pkl')
        with open(ranges_pkl, 'wb') as f:
            pkl.dump(ranges, f)
        print(f'Saved ranges to: {ranges_pkl}')

    # ------------------------------------------------------------------
    # SECOND PASS: render each slice with the shared ranges applied.
    # If `tags` is supplied, only those slices are rendered.
    # ------------------------------------------------------------------
    print('\n[Pass 2/2] Rendering frames with fixed ranges...')
    for i, h5_path in enumerate(h5_files):
        tag = f'idx{i}'
        if tag_filter is not None and tag not in tag_filter:
            continue
        print("Running timeslice", i)
        try:
            print(f'Plotting perturbed g-files — {tag}...')
            fig, axes = plot_geqdsk_bouquet(h5path=h5_path, x_coord='rho',
                                            ranges=ranges)
            out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_{tag}.svg')
            plt.savefig(out, bbox_inches='tight', format="svg")
            plt.close()
            print(f'  Saved: {out}')
        except (KeyError, ValueError) as e:
            print(f'  Skipped (empty file): {e}')

def _add_time_label(svg_path, label):
    """
    Add `label` to the top-left corner of the SVG at `svg_path`, using the
    normal matplotlib font (DejaVu Sans) at a size equal to ~5% of the total
    figure height, then save the result alongside the original with '.svg'
    replaced by '_timed.svg'.

    Returns the path to the newly-written '_timed.svg' file.

    The text is added into the SVG's own coordinate system (its viewBox),
    so it scales correctly no matter what output_width/height cairosvg uses.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    SVG_NS = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', SVG_NS)   # keep default (unprefixed) namespace

    tree = ET.parse(svg_path)
    root = tree.getroot()

    def _num(v):
        if v is None:
            return None
        m = _re.match(r'([-0-9.]+)', v)
        return float(m.group(1)) if m else None

    # Prefer the viewBox for coordinate sizing (matplotlib emits one);
    # fall back to width/height attributes if it's missing.
    vb = root.get('viewBox')
    if vb:
        _, _, width, height = (float(x) for x in vb.replace(',', ' ').split())
    else:
        width = _num(root.get('width'))
        height = _num(root.get('height'))

    fontsize = 0.05 * height          # text height = 5% of figure height
    margin   = 0.00 * height
    x        = 0.01 * height          # keep left inset as before
    y        = margin + fontsize      # y is the text baseline -> shift down

    text = ET.SubElement(root, f'{{{SVG_NS}}}text')
    text.set('x', f'{x:.2f}')
    text.set('y', f'{y:.2f}')
    text.set('font-family', 'DejaVu Sans')   # matplotlib's default font
    text.set('font-size', f'{fontsize:.2f}')
    text.set('fill', 'black')
    text.text = label

    # Save alongside the original: '<name>.svg' -> '<name>_timed.svg'
    timed_path = _re.sub(r'\.svg$', '_timed.svg', svg_path)
    tree.write(timed_path)
    return timed_path

def _gif():
    # ============================================================================
    # 7. Read existing SVG figures and create a GIF
    # ============================================================================
    import glob
    import io
    import cairosvg
    import imageio.v2 as imageio
    IDA_filename = os.path.join(IDA_dir,f"IDA_{shot}_downsampled_final.xr")
    xri = xr.open_dataset(IDA_filename)
    geqdsks = xri.geqdsk_path.values
    times = xri.time.values
    # Collect all previously-generated perturbed g-file SVGs
    print('Searching for SVG files')
    svg_pattern = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_idx*.svg')
    svg_files = glob.glob(svg_pattern)
    # Exclude any already-timed frames from the base list; we handle those
    # separately (a base 'foo_idx3.svg' maps to a timed 'foo_idx3_timed.svg').
    svg_files = [f for f in svg_files if not f.endswith('_timed.svg')]
    if not svg_files:
        print('\nNo SVG files found; nothing to animate.')
        sys.exit(1)
    # Sort by the integer idx to keep frames in time order
    svg_files = sorted(svg_files, key=lambda x: int(x.split('idx')[1].split('.')[0]))
    svg_indices = [int(f.split('idx')[1].split('.')[0]) for f in svg_files]
    svg_times = times[svg_indices]
    print(times)
    print("total idxs = 0 to ",range(len(times))[-1])
    print(svg_indices)
    print(svg_times)

    print(f'\n{"="*60}')
    print(f'Animating results ({len(svg_files)} frame(s))')
    print('=' * 60)
    print(svg_files)
    # Rasterize each SVG to a high-quality PNG (in memory) and collect frames
    frames = []
    target_width = None
    target_height = None
    for i, svg_path in enumerate(svg_files):
        print(f'Reading frame {i} of {len(svg_files)}: {svg_path}')
        # Smart logic: if a timed version already exists, reuse it;
        # otherwise create it by adding the "time = {..}ms" label.
        timed_svg_path = re.sub(r'\.svg$', '_timed.svg', svg_path)
        if os.path.exists(timed_svg_path):
            print(f'  Using existing timed SVG: {timed_svg_path}')
        else:
            label = f"time = {svg_times[i]:.0f}ms"
            timed_svg_path = _add_time_label(svg_path, label)
            print(f'  Saved timed SVG: {timed_svg_path}')
        try:
            png_bytes = cairosvg.svg2png(
                url=timed_svg_path,        # rasterize the timed SVG
                scale=2.0,                 # 2x nominal size for crisp output
                output_width=target_width,
                output_height=target_height,
            )
            frame = imageio.imread(io.BytesIO(png_bytes), pilmode='RGBA')
            if target_width is None:
                target_height, target_width = frame.shape[0], frame.shape[1]
            frames.append(frame)
        except Exception as e:
            print(f'  Skipped (could not rasterize): {e}')
    if not frames:
        print('\nNo frames could be rasterized; aborting.')
        sys.exit(1)
    # Write the GIF (high quality: quality=95)
    out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk.mp4')
    imageio.mimsave(out, frames, fps=5, quality=9)
    print(f'\n  Saved GIF: {out}')

def _gif_debug(n=10):
    # ============================================================================
    # 7. DEBUG: read only the first `n` SVG figures and create a short GIF/MP4
    # ============================================================================
    import glob
    import io
    import cairosvg
    import imageio.v2 as imageio
    print(f'[DEBUG] Searching for SVG files (using first {n})')
    svg_pattern = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_idx*.svg')
    svg_files = glob.glob(svg_pattern)
    if not svg_files:
        print('\n[DEBUG] No SVG files found; nothing to animate.')
        sys.exit(1)
    svg_files = sorted(svg_files, key=lambda x: int(x.split('idx')[1].split('.')[0]))
    svg_files = svg_files[:n]   # <-- only the first n frames
    print(f'\n{"="*60}')
    print(f'[DEBUG] Animating {len(svg_files)} frame(s)')
    print('=' * 60)
    print(svg_files)
    frames = []
    target_width = None
    target_height = None
    for i, svg_path in enumerate(svg_files):
        print(f'[DEBUG] Reading frame {i} of {len(svg_files)}: {svg_path}')
        try:
            png_bytes = cairosvg.svg2png(
                url=svg_path,
                scale=2.0,
                output_width=target_width,
                output_height=target_height,
            )
            frame = imageio.imread(io.BytesIO(png_bytes), pilmode='RGBA')

            if target_width is None:
                target_height, target_width = frame.shape[0], frame.shape[1]

            frames.append(frame)
        except Exception as e:
            print(f'  [DEBUG] Skipped (could not rasterize): {e}')
    if not frames:
        print('\n[DEBUG] No frames could be rasterized; aborting.')
        sys.exit(1)
    # GIF (quality=95)
    gif_out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_debug.gif')
    imageio.mimsave(gif_out, frames, format='GIF', fps=5, quality=95)
    print(f'\n[DEBUG] Saved GIF: {gif_out}')
    # MP4 (drop alpha for H.264)
    mp4_out = os.path.join(PLOT_OUTPUT_DIR, f'{HEADER}_perturbed_geqdsk_debug.mp4')
    imageio.mimsave(mp4_out, [f[..., :3] for f in frames], fps=5, quality=9)
    print(f'[DEBUG] Saved MP4: {mp4_out}')


if __name__ == '__main__':
    for OUTPUT_DIR, HEADER, PLOT_OUTPUT_DIR, shot in RUN_CASES:
        errors = {}
        print(f'\nPLOT_ONLY mode: skipping compute, loading results from:\n  {OUTPUT_DIR}')
        #_plot(remove_previous=False,tags=[2,3,5,11,31,32,48,110],
        #    ranges={"p_ylim": [-9,183],
        #            "q_ylim": [0.49,6.63],
        #            "j_ylim": [-0.04,2.59],
        #    }
        #,tags=[0,8,39],
        #    ranges={"p_ylim": [-2.7,55],
        #            "q_ylim": [0.9,9.72],
        #            "j_ylim": [-0.025,1.33],
        #    }
        #)
        #_gif_debug(n=1)
        _gif()
