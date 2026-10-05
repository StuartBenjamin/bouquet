"""Near-axis sub-grid current structure: detect it and flatten it.

A source current can carry structure inside a radius smaller than the
TokaMaker mesh resolves (FUSE's sawteeth current, core_sources 701, on DIII-D
200061: extrema 0.02 apart in rho_tor inside rho ~ 0.09, against ~3 cm cells).
The GS solve then sees an unresolved current near the axis and q there is
non-monotone.  :func:`flatten_axis_subgrid` replaces the profile inside a cut
with an axis-regular quadratic in ``s`` (= Phi_N = rho_tor^2, so even in rho),
C1 onto the original at the cut and keeping the enclosed current ``∫ j ds``.

All radii are in rho_tor; ``s`` = rho_tor^2 is the area-like coordinate
(Phi_N is proportional to the enclosed area to O(epsilon)).
"""

import numpy as np

#: Default ``n_cells`` of :func:`mesh_axis_rho` (GenerationConfig.swb_saw_axis_flatten_cells).
N_CELLS = 20.0


def _trapz_weights(s):
    """Trapezoid weights on the nodes ``s``."""
    w = np.zeros_like(s)
    d = np.diff(s)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def flatten_axis_subgrid(s, j, s_cut):
    """``(j_flat, s_used)``: ``j`` inside ``s_cut`` replaced by a + b s + c s^2.

    ``s`` rising from 0 (rho_tor^2 of the nodes).  The cut snaps to the first
    node at or past ``s_cut`` (``s_used``); that node and everything outside it
    are unchanged.  The quadratic matches ``j`` and ``dj/ds`` (3-point,
    exact for a quadratic) there and keeps ``∫_0^s_used j ds`` (trapezoid on the nodes,
    so exactly).  A quadratic in ``s`` comes back unchanged.
    """
    s = np.asarray(s, dtype=float)
    j = np.asarray(j, dtype=float)
    if s.ndim != 1 or s.size != j.size or abs(s[0]) > 1e-12 or np.any(np.diff(s) <= 0):
        raise ValueError("flatten_axis_subgrid: s must rise from 0, same size as j")
    k = int(np.searchsorted(s, float(s_cut) - 1e-12 * s[-1]))
    if k < 2 or k >= s.size - 1:
        raise ValueError(f"flatten_axis_subgrid: s_cut={s_cut!r} leaves no inner "
                         f"nodes or no outer neighbour")
    sk = s[:k + 1]
    w = _trapz_weights(sk)
    h1, h2 = s[k] - s[k - 1], s[k + 1] - s[k]
    dj = (-h2 / (h1 * (h1 + h2)) * j[k - 1] + (h2 - h1) / (h1 * h2) * j[k]
          + h1 / (h2 * (h1 + h2)) * j[k + 1])
    A = np.array([[1.0, s[k], s[k] ** 2],
                  [0.0, 1.0, 2.0 * s[k]],
                  [w.sum(), w @ sk, w @ sk ** 2]])
    a, b, c = np.linalg.solve(A, [j[k], dj, w @ j[:k + 1]])
    out = j.copy()
    out[:k] = a + b * s[:k] + c * s[:k] ** 2
    return out, float(s[k])


def axis_extrema(s, j, s_max, rtol=1e-4):
    """Indices of the interior local extrema of ``j`` with ``0 < s < s_max``.

    Steps smaller than ``rtol * max|j|`` carry the previous slope sign.
    """
    s = np.asarray(s, dtype=float)
    d = np.diff(np.asarray(j, dtype=float))
    tol = rtol * (float(np.max(np.abs(j))) or 1.0)
    sign = np.zeros(d.size)
    last = 0.0
    for i, v in enumerate(d):
        if abs(v) > tol:
            last = np.sign(v)
        sign[i] = last
    idx = [i for i in range(1, d.size)
           if sign[i] * sign[i - 1] < 0 and 0.0 < s[i] < s_max]
    return np.asarray(idx, dtype=int)


def mesh_axis_rho(r, lc, boundary_RZ, n_cells=N_CELLS, core=0.3):
    """rho_tor of the flux surface enclosing ``n_cells`` mesh cells' area.

    ``sqrt(n_cells * a_cell / A_lcfs)``: ``a_cell`` the median area of the
    cells whose centroid lies inside the LCFS polygon ``boundary_RZ`` shrunk
    by ``core`` about its centroid (the core resolution), ``A_lcfs`` the
    polygon's area.  Phi_N ~ enclosed area to O(epsilon).  ``r, lc`` are the FE
    elements as loaded, not TokaMaker's order-refined ``mygs.r / lc``.
    """
    from matplotlib.path import Path
    r = np.asarray(r, dtype=float)[:, :2]
    p = r[np.asarray(lc, dtype=int)]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    b = np.asarray(boundary_RZ, dtype=float)[:, :2]
    a_lcfs = 0.5 * abs(np.dot(b[:, 0], np.roll(b[:, 1], 1))
                       - np.dot(b[:, 1], np.roll(b[:, 0], 1)))
    c = b.mean(axis=0)
    inner = Path(c + core * (b - c)).contains_points(p.mean(axis=1))
    if not inner.any():
        raise ValueError("mesh_axis_rho: no mesh cell inside the LCFS core")
    return float(np.sqrt(min(1.0, n_cells * np.median(area[inner]) / a_lcfs)))


def axis_cut(s, j, rho_res, snap=1.5):
    """Automatic cut ``s_cut`` (or None: nothing to flatten) for ``j``.

    None when ``j`` has no interior extremum inside ``rho_res`` (resolved).
    Otherwise the first extremum of ``j`` at or past ``rho_res`` and within
    ``snap * rho_res`` (slope ~0 there, so the C1 match adds no bulge), else
    ``rho_res`` itself.
    """
    s = np.asarray(s, dtype=float)
    s_res = float(rho_res) ** 2
    if axis_extrema(s, j, s_res).size == 0:
        return None
    ext = axis_extrema(s, j, (snap * float(rho_res)) ** 2)
    ext = ext[s[ext] >= s_res]
    return float(s[ext[0]]) if ext.size else s_res


def run_grid_rho(bl):
    """rho_tor of the baseline's run-grid nodes: sqrt(Phi_N) in a Phi_N run;
    in a psi_N run through the source's own (psi_norm, rho_tor_norm), else
    None."""
    x = np.asarray(bl.psi_N, dtype=float)
    if getattr(bl, "coord", "psi_n") == "phi_n":
        return np.sqrt(np.clip(x, 0.0, None))
    fc = getattr(bl, "fuse_currents", None) or {}
    if "psi_norm" in fc and "rho_tor_norm" in fc:
        return np.interp(x, np.asarray(fc["psi_norm"], float),
                         np.asarray(fc["rho_tor_norm"], float))
    return None


#: Guards of :func:`flatten_baseline_saw`: warn when the cut exceeds this many
#: mesh radii, or the current moved (Δ > 0 part) this fraction of the total.
MAX_CUT_RATIO, MAX_MOVED_FRAC = 2.0, 0.01


def flatten_baseline_saw(bl, spec, rho_res=None):
    """Flatten near-axis sub-grid structure of the source total ``bl.j_phi``
    into the sawteeth share: ``Δ = flat(j_phi) - j_phi`` is added to
    ``j_phi``, ``j_other`` and ``j_sawteeth`` (j_inductive, j_BS, j_NBI, j_RF
    and so ``j_phi - j_ind - j_BS - j_sawteeth`` unchanged).

    The total, not j_sawteeth alone, is the reference: FUSE's 701 spike on
    axis offsets the axis hole of j_BS / j_inductive, so flattening it alone
    raises q0.  ``spec`` a float (cut in rho_tor) or ``"auto"`` (cut from
    ``rho_res``, the mesh radius, :func:`axis_cut`).  Warns (and flags) a cut
    beyond ``MAX_CUT_RATIO * rho_res`` or a moved current ``∫Δ⁺ ds / ∫j_phi ds``
    (~ fraction of Ip) above ``MAX_MOVED_FRAC``.  Returns the ``ip_closure``
    record.
    """
    j_phi = np.asarray(bl.j_phi, dtype=float)
    rec = dict(saw_axis_flatten=spec if isinstance(spec, str) else float(spec),
               saw_axis_rho_res=None if rho_res is None else float(rho_res),
               saw_axis_rho_cut=None, saw_axis_cut_over_res=None,
               saw_axis_moved_frac=0.0, saw_axis_enclosed_change=0.0,
               saw_axis_n_extrema=0, saw_axis_warn_wide=False,
               saw_axis_warn_moved=False)
    rho = run_grid_rho(bl)
    if rho is None:
        raise RuntimeError("swb_saw_axis_flatten: no rho_tor map for this psi_N grid "
                           "(Baseline.fuse_currents lacks psi_norm / rho_tor_norm)")
    if getattr(bl, "j_sawteeth", None) is None or not np.any(bl.j_sawteeth):
        print("WARN: swb_saw_axis_flatten: the source has no sawteeth current "
              "to carry the change; not applied")
        rec["saw_axis_flatten_skipped"] = "no sawteeth current"
        return rec
    s = rho ** 2
    if spec == "auto":
        s_cut = axis_cut(s, j_phi, rho_res)
        if s_cut is None:
            return rec
    else:
        s_cut = float(spec) ** 2
    rec["saw_axis_n_extrema"] = int(axis_extrema(s, j_phi, s_cut).size)
    j_flat, s_used = flatten_axis_subgrid(s, j_phi, s_cut)
    delta = j_flat - j_phi
    w = _trapz_weights(s)
    norm = float(w @ j_phi) or 1.0
    cut = float(np.sqrt(s_used))
    rec.update(saw_axis_rho_cut=cut,
               saw_axis_moved_frac=float(w @ np.clip(delta, 0.0, None)) / abs(norm),
               saw_axis_enclosed_change=float(w @ delta) / abs(norm),
               saw_axis_delta_peak=float(np.max(np.abs(delta))))
    if rho_res:
        rec["saw_axis_cut_over_res"] = cut / float(rho_res)
        if rec["saw_axis_cut_over_res"] > MAX_CUT_RATIO:
            rec["saw_axis_warn_wide"] = True
            print(f"WARN: swb_saw_axis_flatten: cut rho {cut:.4f} is "
                  f"{rec['saw_axis_cut_over_res']:.2f}x the mesh radius {rho_res:.4f} "
                  f"(> {MAX_CUT_RATIO}): flattening resolved structure")
    if rec["saw_axis_moved_frac"] > MAX_MOVED_FRAC:
        rec["saw_axis_warn_moved"] = True
        print(f"WARN: swb_saw_axis_flatten: moved {rec['saw_axis_moved_frac']:.2%} of "
              f"the current (> {MAX_MOVED_FRAC:.0%})")
    bl.j_phi = j_phi + delta
    bl.j_other = np.asarray(bl.j_other, dtype=float) + delta
    bl.j_sawteeth = np.asarray(bl.j_sawteeth, dtype=float) + delta
    return rec
