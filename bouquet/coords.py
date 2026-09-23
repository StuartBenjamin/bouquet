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


def psi_of(mygs, x, coord=PSI, psi_pad=1e-3):
    """ψ_N at which to sample a readback for the run grid ``x``.

    A ψ_N run clips ``x`` to ``[psi_pad, 1 - psi_pad]``; a Φ_N run maps it
    through the last solve's toroidal-flux map, then clips.
    """
    x = np.asarray(x, dtype=float)
    if check_coord(coord) == PHI:
        x = np.asarray(mygs.get_torflux_map(x.copy(), inverse=True)[0], dtype=float)
    return np.clip(x, psi_pad, 1.0 - psi_pad)


def window_x(mygs, x, coord=PSI, window_coord=PSI, psi_pad=1e-3):
    """Abscissa for a hard-coded radial window (thresholds such as ψ_N > 0.9).

    ``window_coord="psi_n"`` compares thresholds in ψ_N (via :func:`psi_of`,
    unclipped at the ends); ``"native"`` compares them in the run coordinate.
    The two agree in a ψ_N run, where ``x`` is returned unchanged.
    """
    if window_coord not in (PSI, "native"):
        raise ValueError(f"window_coord must be 'psi_n' or 'native', got {window_coord!r}")
    x = np.asarray(x, dtype=float)
    if check_coord(coord) == PSI or window_coord == "native":
        return x
    return np.asarray(mygs.get_torflux_map(x.copy(), inverse=True)[0], dtype=float)


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


def swb_grid(x):
    """The grid ``solve_with_bootstrap`` places its arrays on.

    ``x`` where the toolkit takes ``psi_N=``; otherwise the uniform grid it
    assumes, which is also the grid of its outputs.
    """
    x = np.asarray(x, dtype=float)
    return x if "psi_N" in _swb_params() else np.linspace(0.0, 1.0, x.size)


def swb_grid_kwargs(x, coord=PSI):
    """Grid arguments for ``solve_with_bootstrap``: ``psi_N=x``, plus ``coord``
    in a Φ_N run.  Empty on a toolkit without ``psi_N=`` (ψ_N runs only).
    """
    kw = {}
    if "psi_N" in _swb_params():
        kw["psi_N"] = np.asarray(x, dtype=float)
    if check_coord(coord) == PHI:
        kw["coord"] = PHI
    return kw


def swb_seed(x):
    """Inductive seed ``(1 - s^1.5)^1.5`` on :func:`swb_grid` (the profile of
    OFT's ``create_power_flux_fun(n, 1.5, 1.5)``, on the grid SWB uses).
    """
    s = swb_grid(x)
    return np.power(1.0 - np.power(s, 1.5), 1.5)


def check_backend(coord):
    """Raise unless the installed toolkit supports ``coord``.

    A Φ_N run needs ``solve_bootstrap(coord=)`` and ``get_torflux_map``: an
    older toolkit ignores a ``coord`` key in a profile dict, so it would
    solve on ψ_N without complaint.
    """
    if check_coord(coord) == PSI:
        return
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker, TokaMaker_equilibrium
        ok = ("coord" in inspect.signature(TokaMaker_equilibrium.solve_bootstrap).parameters
              and hasattr(TokaMaker, "get_torflux_map")
              and "psi_N" in _swb_params())
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError(
            "coord='phi_n' needs an OpenFUSIONToolkit with toroidal-flux profile "
            "support (solve_bootstrap(coord=) and TokaMaker.get_torflux_map).")
