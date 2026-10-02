"""The run's radial coordinate, at the OpenFUSIONToolkit boundary.

A run holds every profile and envelope on one grid ``x`` in one coordinate:
``"psi_n"`` (normalised poloidal flux, the default) or ``"phi_n"``
(normalised toroidal flux).  Profiles cross into TokaMaker as-is, tagged
with their coordinate; ψ_N appears in bouquet only as the address at which
a readback is sampled, and in a Φ_N run that address comes from the solver's
own map (:func:`psi_of`).  A ``"psi_n"`` run never sends a ``coord`` key or
argument, so its calls are those of an OFT without toroidal-flux support.
"""

from __future__ import annotations

import numpy as np

PSI, PHI = "psi_n", "phi_n"
COORDS = (PSI, PHI)
#: Input spelling accepted at io: ρ_tor, converted exactly to Φ_N = ρ².
RHO = "rho_tor"
#: Coordinates a user-supplied grid may be given in: the run's, or ψ_N.
RUN = "run"
INPUT_COORDS = (RUN, PSI)


def check_coord(coord):
    """Return ``coord`` if it is a run coordinate, else raise."""
    if coord not in COORDS:
        raise ValueError(f"coord must be one of {COORDS}, got {coord!r}")
    return coord


def resolve_input_coord(coord, x):
    """``(run_coord, x_run)`` for an io grid given in ``coord``.

    ``"rho_tor"`` becomes ``"phi_n"`` on ``x**2``; the others pass through.
    """
    if coord == RHO:
        return PHI, np.asarray(x, dtype=float) ** 2
    return check_coord(coord), x


def check_source_coord(coord):
    """Return ``coord`` if it is a source coordinate (a run one or ``"rho_tor"``)."""
    if coord not in COORDS + (RHO,):
        raise ValueError(f"source coord must be one of {COORDS + (RHO,)}, got {coord!r}")
    return coord


def check_native(name, value):
    """Return ``value`` if it is ``"psi_n"`` or ``"native"`` (window/seed coords)."""
    if value not in (PSI, "native"):
        raise ValueError(f"{name} must be 'psi_n' or 'native', got {value!r}")
    return value


def to_run_grid(x, x_coord=RUN, psi_map=None):
    """A user-supplied grid ``x`` (given in ``x_coord``) in the run coordinate.

    ``"run"`` passes ``x`` through; ``"psi_n"`` maps it through ``psi_map``
    (``(psi_N, x_run)`` pairs; ``None`` in a ψ_N run, where it is the identity).
    """
    if x_coord not in INPUT_COORDS:
        raise ValueError(f"input coord must be one of {INPUT_COORDS}, got {x_coord!r}")
    if x is None or x_coord == RUN or psi_map is None:
        return x
    return np.interp(np.asarray(x, dtype=float), *psi_map)


def phi_n_from_q(psi_N, q, bracket=False):
    """``(inside, Φ_N)``: the nodes with ψ_N ≤ 1 and their Φ_N from ``q``.

    Φ_N(ψ_N) = ∫₀^ψ_N q dψ_N / ∫₀¹ q dψ_N (trapezoid; ``q`` is interpolated
    to ψ_N = 1 when the grid does not hit it).  For a source that tabulates
    its profiles on ψ_N with its own equilibrium's q (IDA).

    ``bracket``: when the grid does not hit ψ_N = 1, also select the first
    node past it (Φ_N > 1, same trapezoid), so profiles interpolated in Φ_N
    reach Φ_N = 1 between data instead of clamping (ida_fuse's ida_phi_n).
    """
    from scipy.integrate import cumulative_trapezoid
    psi_N = np.asarray(psi_N, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    inside = psi_N <= 1.0
    x = psi_N[inside]
    if x.size < 2 or not np.all(np.diff(x) > 0) or abs(x[0]) > 1e-9:
        raise ValueError("phi_n_from_q: psi_N must rise from 0")
    if not np.all(np.isfinite(q[inside])):
        raise ValueError("phi_n_from_q: q is not finite inside the LCFS")
    xe = x if x[-1] == 1.0 else np.append(x, 1.0)
    phi = cumulative_trapezoid(np.interp(xe, psi_N, q), xe, initial=0.0)
    out = phi[:x.size] / phi[-1]
    if bracket and x[-1] < 1.0 and not inside.all():
        b = int(np.argmin(inside))                  # first node past ψ_N = 1
        if np.isfinite(q[b]):
            q_last = q[inside][-1]
            out = np.append(out, (phi[x.size - 1] + 0.5 * (q_last + q[b])
                                  * (psi_N[b] - x[-1])) / phi[-1])
            inside = inside.copy()
            inside[b] = True
    return inside, out


def oft_prof(kind, x, y, coord=PSI):
    """A TokaMaker profile dict on the run grid.

    In a Φ_N run a ``jphi-*`` profile holds values (tag ``phi_n_relabel``)
    and any other profile holds a derivative, y = dY/dΦ_N (tag ``phi_n``).
    """
    d = {"type": kind, "y": np.array(y, dtype=float), "x": x}
    if check_coord(coord) == PHI:
        d["coord"] = "phi_n_relabel" if kind.startswith("jphi") else PHI
    return d


def pp_prof(mygs, x, p, coord=PSI):
    """TokaMaker ``pp_prof`` for the pressure ``p`` on the run grid.

    ``dp/dx / (psi_bounds[1] - psi_bounds[0])``: dp/dψ in a ψ_N run.  In a
    Φ_N run TokaMaker applies dΦ_N/dψ_N itself, recovering the same dp/dψ.
    """
    from .utils import pchip_derivative
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
    return oft_prof("linterp", x, pchip_derivative(x, p) / psi_range, coord)


def psi_at(mygs, x, coord=PSI):
    """ψ_N of the run-grid nodes ``x`` on ``mygs``'s last solve.

    ``x`` itself in a ψ_N run; the solver's toroidal-flux map in a Φ_N run.
    This is the abscissa for anything that integrates or samples in ψ
    (``flux_integral``, the FSA current measure, ``find_optimal_scale``).
    """
    if check_coord(coord) == PSI:
        return x
    x = np.asarray(x, dtype=float)
    return np.asarray(mygs.get_torflux_map(x.copy(), inverse=True)[0], dtype=float)


def psi_of(mygs, x, coord=PSI, psi_pad=1e-3):
    """:func:`psi_at`, clipped to ``[psi_pad, 1 - psi_pad]``: where to sample
    ``get_profiles``/``get_q``/``sauter_fc`` for the run grid ``x``.
    """
    return np.clip(np.asarray(psi_at(mygs, x, coord), dtype=float),
                   psi_pad, 1.0 - psi_pad)


def window_x(mygs, x, coord=PSI, window_coord=PSI):
    """Abscissa for a hard-coded radial window (thresholds such as ψ_N > 0.9).

    ``window_coord="psi_n"`` compares thresholds in ψ_N (:func:`psi_at`);
    ``"native"`` compares them in the run coordinate.  The two agree in a
    ψ_N run, where ``x`` is returned unchanged.
    """
    if check_native("window_coord", window_coord) == "native":
        return np.asarray(x, dtype=float)
    return np.asarray(psi_at(mygs, x, coord), dtype=float)


def _swb_params():
    """Argument names of the installed ``solve_with_bootstrap`` (cached)."""
    global _SWB_PARAMS
    try:
        return _SWB_PARAMS
    except NameError:
        pass
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
        _SWB_PARAMS = frozenset(inspect.signature(solve_with_bootstrap).parameters)
    except Exception:
        _SWB_PARAMS = frozenset()
    return _SWB_PARAMS


def _swb_grid_arg():
    """``solve_with_bootstrap``'s grid keyword: ``"x"``, the ψ_N-only
    ``"psi_N"`` of older toolkits, or ``None`` (a uniform grid is assumed).
    """
    p = _swb_params()
    return "x" if "x" in p else ("psi_N" if "psi_N" in p else None)


def swb_grid(x):
    """The grid ``solve_with_bootstrap`` places its arrays on.

    ``x`` where the toolkit takes a grid argument; otherwise the uniform grid
    it assumes, which is also the grid of its outputs.
    """
    x = np.asarray(x, dtype=float)
    return x if _swb_grid_arg() else np.linspace(0.0, 1.0, x.size)


def swb_grid_kwargs(x, coord=PSI):
    """Grid arguments for ``solve_with_bootstrap``: the grid (:func:`_swb_grid_arg`),
    plus ``coord`` in a Φ_N run.  Empty on a toolkit without a grid argument
    (ψ_N runs only).
    """
    kw = {}
    arg = _swb_grid_arg()
    if arg:
        kw[arg] = np.asarray(x, dtype=float)
    if check_coord(coord) == PHI:
        kw["coord"] = PHI
    return kw


def swb_seed(x, psi=None):
    """Inductive seed ``(1 - s^1.5)^1.5`` (OFT's ``create_power_flux_fun(n,
    1.5, 1.5)``) at the nodes of :func:`swb_grid`, with ``s`` their ψ_N.

    ``psi`` is the ψ_N of the nodes ``x`` (:func:`seed_psi`): the shape is then
    the ψ_N one in any run; ``None`` writes it in the run coordinate.
    """
    s = swb_grid(x)
    if psi is not None and _swb_grid_arg():
        s = np.asarray(psi, dtype=float)
    return np.power(1.0 - np.power(s, 1.5), 1.5)


def seed_psi(mygs, x, coord=PSI, seed_coord=PSI):
    """The ``psi`` argument of :func:`swb_seed` for ``seed_coord``: the nodes'
    ψ_N (``"psi_n"``) or ``None``, the run coordinate (``"native"``).
    """
    check_native("seed_coord", seed_coord)
    return psi_at(mygs, x, coord) if seed_coord == PSI else None


def check_backend(coord):
    """Raise unless the installed toolkit supports ``coord``.

    A Φ_N run needs ``solve_bootstrap(coord=)``, ``get_torflux_map`` and
    ``solve_with_bootstrap(x, coord)``: an older toolkit ignores a ``coord``
    key in a profile dict, so it would solve on ψ_N without complaint.
    """
    if check_coord(coord) == PSI:
        return
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker
        ok = ("coord" in inspect.signature(TokaMaker.solve_bootstrap).parameters
              and hasattr(TokaMaker, "get_torflux_map")
              and _swb_grid_arg() is not None
              and "coord" in _swb_params())
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError(
            "coord='phi_n' needs an OpenFUSIONToolkit with toroidal-flux profile "
            "support (solve_bootstrap(coord=), TokaMaker.get_torflux_map and "
            "solve_with_bootstrap(x, coord)).")


def check_run(config, coord=None):
    """Refuse, before any solve, a run coordinate this setup cannot run.

    ``coord`` defaults to the source's.  Checks the window/seed coords, and in
    a Φ_N run the toolkit (:func:`check_backend`) and that the internal
    bootstrap solve is used.  Returns the run coordinate.
    """
    if coord is None:
        coord = resolve_input_coord(getattr(config.source, "coord", PSI), [])[0]
    gc = config.generation
    check_native("window_coord", gc.window_coord)
    check_native("seed_coord", gc.seed_coord)
    if check_coord(coord) == PHI:
        check_backend(coord)
        if gc.bootstrap_kwargs.get("use_python_solve"):
            raise ValueError("coord='phi_n' needs the internal bootstrap "
                             "solve: drop use_python_solve from bootstrap_kwargs")
    return coord
