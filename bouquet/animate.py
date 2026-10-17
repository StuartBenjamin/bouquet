"""Per-shot mp4s of bouquet's plots across a shot's time slices (one archive per scan), one per plot kind,
on axis limits shared by all frames, plus the time-coloured profile overlay (pdf).

    python -m bouquet.animate --study DIR [--study DIR ...] --shot N --out DIR [--kinds geqdsk,kinetic,...]

Needs the ``anim`` extra (imageio-ffmpeg, pillow).
"""
import argparse
import glob
import io
import os
import re
import subprocess
import warnings
from collections import Counter

import numpy as np

from .plotting import plot_aux_profiles, plot_bouquet, plot_bouquet_timeseries, plot_geqdsk_bouquet, plot_jphi

_ARCHIVE_RE = re.compile(r"^(\d+)_(\d+)_(\w+)\.h5$")   # <shot>_<t_ms>_<variant>.h5 (campaign layout)
KINDS = {
    "geqdsk": lambda h5, key: plot_geqdsk_bouquet(h5path=h5, scan_key=key),
    "kinetic": lambda h5, key: plot_bouquet(h5, scan_key=key, mode="kinetic"),
    "pressure": lambda h5, key: plot_bouquet(h5, scan_key=key, mode="pressure"),
    "j-phi": lambda h5, key: plot_bouquet(h5, scan_key=key, mode="j-phi"),
    "boundary": lambda h5, key: plot_bouquet(h5, scan_key=key, mode="boundary"),
    "jphi": lambda h5, key: plot_jphi(h5, scan_key=key),
    "aux": lambda h5, key: plot_aux_profiles(h5, scan_key=key),
}


def shot_archives(study_dirs, shot):
    """[(t_ms, variant, path, study)] of every <study>/[cNNN/]<shot>/bouquet/<shot>_<t_ms>_<variant>.h5, by
    t_ms (numeric), then study order. A scan stored in several studies is listed for each, with a warning."""
    out = []
    for study in study_dirs:
        name = os.path.basename(os.path.normpath(study))
        for d in sorted(glob.glob(os.path.join(study, str(shot), "bouquet"))
                        + glob.glob(os.path.join(study, "c[0-9]*", str(shot), "bouquet"))):
            for f in sorted(os.listdir(d)):
                m = _ARCHIVE_RE.match(f)
                if m and int(m[1]) == shot:
                    out.append((int(m[2]), m[3], os.path.join(d, f), name))
    dup = sorted(k for k, n in Counter((t, v) for t, v, _, _ in out).items() if n > 1)
    if dup:
        warnings.warn(f"shot {shot}: scans in more than one study (each listed): {dup}")
    return sorted(out, key=lambda a: a[0])


def _figures(kind, path, t_ms):
    """The figure(s) of one plot kind for one archive ([] if it has nothing to show)."""
    ret = KINDS[kind](path, str(t_ms))
    return [f for f in (ret[0] if isinstance(ret[0], (list, tuple)) else [ret[0]]) if f is not None]


def _share_limits(frames):
    """Set every frame's k-th figure's j-th axes to the union of their limits (frames: [[fig, ...], ...])."""
    lims = {}
    for figs in frames:
        for k, fig in enumerate(figs):
            for j, ax in enumerate(fig.axes):
                lo = lims.setdefault((k, j), [np.inf, -np.inf, np.inf, -np.inf])
                (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
                lo[:] = min(lo[0], x0), max(lo[1], x1), min(lo[2], y0), max(lo[3], y1)
    for figs in frames:
        for k, fig in enumerate(figs):
            for j, ax in enumerate(fig.axes):
                x0, x1, y0, y1 = lims[(k, j)]
                ax.set_xlim(*((x1, x0) if ax.xaxis_inverted() else (x0, x1)))
                ax.set_ylim(*((y1, y0) if ax.yaxis_inverted() else (y0, y1)))


def _rasterize(fig, dpi=110, max_width=1920):
    """RGB image of fig, at most max_width px wide; closes it."""
    import matplotlib.pyplot as plt
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=min(dpi, max_width / fig.get_figwidth()))
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_mp4(images, path, fps):
    """H.264 mp4 through imageio-ffmpeg's ffmpeg (even dimensions for yuv420p)."""
    import imageio_ffmpeg
    w, h = images[0].size
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for im in images:
        proc.stdin.write(im.resize((w, h)).tobytes())
    proc.stdin.close()
    proc.wait()


def animate_shot(archives, out_dir, kinds=tuple(KINDS), fps=2):
    """<out_dir>/<shot>_<kind>[_<k>].mp4 per plot kind (k: figure index when a kind makes several), one frame
    per archive (shot_archives), labelled with t_ms, study and variant; and <shot>_timeseries.pdf
    (plot_bouquet_timeseries, first archive per t_ms). A frame that fails is skipped. Returns the paths."""
    import matplotlib.pyplot as plt
    shot = int(os.path.basename(archives[0][2]).split("_")[0])
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for kind in kinds:
        frames, labels = [], []
        for t, variant, path, study in archives:
            try:
                figs = _figures(kind, path, t)
            except Exception as e:
                warnings.warn(f"shot {shot} {kind} t={t} ms ({study}): skipped: {e}")
                plt.close("all")
                continue
            if figs:
                frames.append(figs)
                labels.append(f"shot {shot}  t = {t} ms  ({study}, {variant})")
        if not frames:
            continue
        _share_limits(frames)
        images = {}
        for figs, label in zip(frames, labels):
            for k, fig in enumerate(figs):
                fig.text(0.005, 0.995, label, ha="left", va="top", fontsize=9)
                images.setdefault(k, []).append(_rasterize(fig))
        for k, ims in images.items():
            paths.append(os.path.join(out_dir, f"{shot}_{kind}" + (f"_{k}" if len(images) > 1 else "") + ".mp4"))
            _save_mp4(ims, paths[-1], fps)
    first = {}
    for t, _, path, _ in archives:
        first.setdefault(t, path)
    fig, _ = plot_bouquet_timeseries({t: p for t, p in first.items()}, time_label="t [ms]")
    paths.append(os.path.join(out_dir, f"{shot}_timeseries.pdf"))
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 0   # a kind's frames stay open until their limits are shared
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", action="append", required=True, help="study dir; repeat, in priority order")
    p.add_argument("--shot", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--fps", type=float, default=2)
    a = p.parse_args(argv)
    archives = shot_archives(a.study, a.shot)
    if not archives:
        raise SystemExit(f"no archives of shot {a.shot} in {a.study}")
    for path in animate_shot(archives, a.out, a.kinds.split(","), a.fps):
        print(path)


if __name__ == "__main__":
    main()
