"""The pressure handed to the Grad-Shafranov solver: ONE place.

Every solve of this package hands the solver a ``P'`` profile
(``{"type": "linterp", ...}``) and an axis-pressure target (``pax``).  The
solver builds the pressure by integrating ``P'`` INWARD from the plasma
boundary starting at ZERO, then rescales ``P'`` so the axis value equals
``pax``.  Two consequences, each behind one setting here:

``edge_pprime_pin`` (default ``True``: the behaviour before the setting)
    ``True`` sets the LAST node of ``P'`` (``psi_N = 1``) to zero, so ``P'``
    ramps linearly to zero across the final grid interval.  ``False`` leaves
    the profile's own derivative there (``P'`` then jumps to zero outside
    the plasma, which the solver's piecewise-linear flux function
    represents).  What changes with ``False``: the pressure-driven part of
    the edge current ``P' (<R> - F^2 <1/R>/<B^2>)`` is no longer forced to
    zero at the boundary, so for the same requested total ``<j_phi>`` the
    split between the ``P'`` and ``FF'`` terms in the last interval moves
    (``FF'`` carries less there), and the edge current and ``q95`` follow.

``separatrix_pressure`` (default ``"legacy"``: the behaviour before the setting)
    ``"legacy"`` passes the FULL axis pressure as the target.  When the
    input pressure is not zero at ``psi_N = 1`` (``p_sep``), the solver's
    pressure -- which is zero there by construction -- reaches that target
    only by inflating ``P'`` everywhere by ``p_axis / (p_axis - p_sep)``;
    the reported ``beta`` and ``W_MHD`` are then those of a different
    pressure profile (``p_axis (p - p_sep) / (p_axis - p_sep)``).
    ``"offset"`` passes ``p_axis - p_sep`` as the target, so ``P'`` is the
    input's own, and ``p_sep`` is added back wherever pressure, ``beta`` or
    stored energy is REPORTED or DELIVERED (:func:`pressure_frames`, the
    ``lcfs_pressure`` of a written g-file).  ``p_sep`` is the TOTAL pressure
    handed to the solver (thermal + impurity + fast, exactly the array the
    solve is built from) at its last node, :func:`separatrix_pressure_of`.

Where the model stops: a pressure that is ``p_sep`` just inside the boundary
and zero just outside is not physical.  The real separatrix pressure
continues into the scrape-off layer, which a vacuum-outside free-boundary
equilibrium cannot represent.  Only ``P'`` enters the Grad-Shafranov
equation, so the equilibrium inside the boundary is the one the input's
``P'`` asks for; the constant ``p_sep`` is bookkeeping for readers of the
pressure, not a force.

With both settings at their defaults every array this module returns is
bit for bit what the scattered ``pp["y"][-1] = 0.0`` / ``pax = p[0]`` sites
returned before it existed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ``GenerationConfig.separatrix_pressure`` values.
SEPARATRIX_PRESSURE_CHOICES = ("legacy", "offset")
#: The two settings and their defaults (today's behaviour).
EDGE_PRESSURE_DEFAULTS = {"edge_pprime_pin": True,
                          "separatrix_pressure": "legacy"}
#: What the two frames of :func:`pressure_frames` are.
FRAME_NOTE = (
    "solver frame: the solver's own pressure (zero at psi_N = 1), what the "
    "equilibrium responds to; full frame: the solver's pressure + p_sep "
    "(W_MHD + 1.5 p_sep V; every beta scaled by (int p dV + p_sep V) / "
    "int p dV, V and int p dV from the solved equilibrium)")

_MU0 = 4.0e-7 * np.pi


def validate_edge_pressure_settings(edge_pprime_pin, separatrix_pressure):
    """Refuse a malformed value of either setting, by name."""
    if not isinstance(edge_pprime_pin, (bool, np.bool_)):
        raise ValueError("generation.edge_pprime_pin must be a bool, got "
                         f"{edge_pprime_pin!r}")
    if (not isinstance(separatrix_pressure, str)
            or separatrix_pressure not in SEPARATRIX_PRESSURE_CHOICES):
        raise ValueError("generation.separatrix_pressure must be one of "
                         f"{SEPARATRIX_PRESSURE_CHOICES}, got "
                         f"{separatrix_pressure!r}")


@dataclass(frozen=True)
class EdgePressure:
    """The two settings, validated (see the module docstring)."""
    edge_pprime_pin: bool = True
    separatrix_pressure: str = "legacy"

    def __post_init__(self):
        validate_edge_pressure_settings(self.edge_pprime_pin,
                                        self.separatrix_pressure)

    @property
    def offset(self) -> bool:
        """``separatrix_pressure == "offset"``."""
        return self.separatrix_pressure == "offset"

    @property
    def is_default(self) -> bool:
        return bool(self.edge_pprime_pin) and not self.offset

    def record(self) -> dict:
        """The settings as a JSON-able dict."""
        return dict(edge_pprime_pin=bool(self.edge_pprime_pin),
                    separatrix_pressure=str(self.separatrix_pressure))

    # the helper, as methods (the module functions below are the same)
    def pprime(self, psi_N, pressure, psi_range):
        return solver_pprime(psi_N, pressure, psi_range, self)

    def pp_profile(self, psi_N, pressure, psi_range):
        return solver_pp_profile(psi_N, pressure, psi_range, self)

    def pax(self, pressure) -> float:
        return solver_pax(pressure, self)

    def p_offset(self, pressure) -> float:
        return applied_offset(pressure, self)


def resolve_edge_pressure(obj=None) -> EdgePressure:
    """An :class:`EdgePressure` from ``None`` (the defaults), an
    :class:`EdgePressure`, a dict with the two keys, or anything carrying
    the two attributes (a ``GenerationConfig``).  Validates loudly."""
    if obj is None:
        return EdgePressure()
    if isinstance(obj, EdgePressure):
        return obj
    d = EDGE_PRESSURE_DEFAULTS
    if isinstance(obj, dict):
        unknown = set(obj) - set(d)
        if unknown:
            raise ValueError("edge-pressure settings: unknown key(s) "
                             f"{sorted(unknown)} (known: {sorted(d)})")
        return EdgePressure(
            edge_pprime_pin=obj.get("edge_pprime_pin", d["edge_pprime_pin"]),
            separatrix_pressure=obj.get("separatrix_pressure",
                                        d["separatrix_pressure"]))
    return EdgePressure(
        edge_pprime_pin=getattr(obj, "edge_pprime_pin", d["edge_pprime_pin"]),
        separatrix_pressure=getattr(obj, "separatrix_pressure",
                                    d["separatrix_pressure"]))


# ---------------------------------------------------------------------------
#  what the solver is handed
# ---------------------------------------------------------------------------
def pressure_gradient(psi_N, pressure):
    """``d p / d psi_N`` on *psi_N* (:func:`bouquet.utils.pchip_derivative`):
    the derivative every ``P'`` of this package is built from, and the one
    the engine's first-pass pressure-driven term of a draw is shifted by."""
    from .utils import pchip_derivative
    return pchip_derivative(psi_N, pressure)


def solver_pprime(psi_N, pressure, psi_range, edge=None):
    """The ``P'`` node values handed to the solver: ``d p / d psi_N`` over
    the flux range, with the last node zeroed when ``edge_pprime_pin``."""
    edge = resolve_edge_pressure(edge)
    y = pressure_gradient(psi_N, pressure) / psi_range
    if edge.edge_pprime_pin:
        y[-1] = 0.0
    return y


def solver_pp_profile(psi_N, pressure, psi_range, edge=None):
    """The ``pp_prof`` dict of :func:`solver_pprime` (``x`` is *psi_N*
    itself, as every site passed it)."""
    return {"type": "linterp",
            "y": solver_pprime(psi_N, pressure, psi_range, edge),
            "x": psi_N}


def separatrix_pressure_of(pressure) -> float:
    """``p_sep``: the pressure handed to the solver at its last node
    (``psi_N = 1``; total: thermal + impurity + fast, whatever the solve's
    pressure array is built from)."""
    return float(np.asarray(pressure, dtype=float)[-1])


def applied_offset(pressure, edge=None) -> float:
    """The pressure removed from the axis target and added back at
    reporting / delivery: ``p_sep`` under ``"offset"``, exactly ``0.0``
    under ``"legacy"``."""
    edge = resolve_edge_pressure(edge)
    if not edge.offset:
        return 0.0
    p_sep = separatrix_pressure_of(pressure)
    if not np.isfinite(p_sep):
        raise ValueError("separatrix_pressure='offset': the pressure at "
                         f"psi_N = 1 is not finite ({p_sep!r})")
    return p_sep


def solver_pax(pressure, edge=None) -> float:
    """The axis-pressure target: ``p[0]`` (``"legacy"``) or
    ``p[0] - p_sep`` (``"offset"``; refused unless positive)."""
    edge = resolve_edge_pressure(edge)
    p0 = float(pressure[0])
    if not edge.offset:
        return p0
    pax = p0 - applied_offset(pressure, edge)
    if not (np.isfinite(pax) and pax > 0.0):
        raise ValueError(
            "separatrix_pressure='offset': the axis target p_axis - p_sep = "
            f"{pax!r} Pa is not positive (p_axis = {p0!r}, p_sep = "
            f"{separatrix_pressure_of(pressure)!r})")
    return pax


def solver_pressure(pressure, edge=None):
    """The pressure array whose FIRST element is the axis target, for the
    solver-side routines that take a pressure and read ``pressure[0]`` as
    ``pax``: *pressure* itself (``"legacy"``; the same object) or
    ``pressure - p_sep`` (``"offset"``)."""
    edge = resolve_edge_pressure(edge)
    if not edge.offset:
        return pressure
    solver_pax(pressure, edge)          # the same refusal
    return np.asarray(pressure, dtype=float) - applied_offset(pressure, edge)


# ---------------------------------------------------------------------------
#  what is reported
# ---------------------------------------------------------------------------
_PVOL_KEYS = ("W_MHD", "beta_pol", "beta_tor", "beta_n")


def pressure_frames(stats, p_sep) -> dict:
    """Both frames of the pressure-integral quantities of a solved
    equilibrium.

    *stats* is the solver's ``get_stats()`` dict (``vol``, ``W_MHD = 1.5 int
    p dV``, ``beta_pol``, ``beta_tor``, ``beta_n``, ``P_ax`` -- all built
    from the solver's own pressure, zero at the boundary).  *p_sep* [Pa] is
    the constant added back.  Returns::

        {"p_sep": p_sep, "volume": V, "p_sep_volume": p_sep V,
         "factor": 1 + p_sep V / int p dV,
         "solver": {W_MHD, beta_pol, beta_tor, beta_n, P_ax},
         "full":   {W_MHD + 1.5 p_sep V, beta_* factor, P_ax + p_sep}}

    ``W_MHD`` and every ``beta`` of the solver are linear in ``int p dV``
    with the same geometry, current and field, so the full-frame value is
    the solver's times ``factor`` -- exactly ``+ 1.5 p_sep V`` for the
    energy and ``+ 2 mu0 p_sep / <B_ref^2>`` for each beta.  With ``p_sep =
    0`` both frames ARE the solver's numbers (``factor`` is exactly 1).
    Keys the solver did not report are omitted.
    """
    p_sep = float(p_sep)
    V = float(stats["vol"])
    W = float(stats["W_MHD"])
    pvol = W / 1.5
    if p_sep == 0.0:
        factor = 1.0
    else:
        if not (np.isfinite(pvol) and pvol > 0.0):
            raise ValueError("pressure_frames: the solver's int p dV = "
                             f"{pvol!r} is not positive; the full-frame "
                             "betas cannot be formed")
        factor = 1.0 + p_sep * V / pvol
    solver = {k: float(stats[k]) for k in _PVOL_KEYS if k in stats}
    full = {k: (v if p_sep == 0.0 else
                (v + 1.5 * p_sep * V if k == "W_MHD" else v * factor))
            for k, v in solver.items()}
    if "P_ax" in stats:
        solver["P_ax"] = float(stats["P_ax"])
        full["P_ax"] = float(stats["P_ax"]) + p_sep
    return dict(p_sep=p_sep, volume=V, p_sep_volume=p_sep * V,
                factor=float(factor), solver=solver, full=full,
                note=FRAME_NOTE)


def input_pressure_frames(volume, pvol, p_edge, betas=None) -> dict:
    """The same two frames for an INPUT equilibrium, from its own numbers:
    *volume* [m^3], *pvol* ``= int p dV`` of its FULL pressure, *p_edge*
    its pressure at ``psi_N = 1`` and (optionally) its ``betas`` (any
    mapping of beta names to values built from the full pressure).  The
    solver-frame quantities are those of ``p - p_edge``."""
    volume, pvol, p_edge = float(volume), float(pvol), float(p_edge)
    pv_s = pvol - p_edge * volume
    ratio = pv_s / pvol if pvol != 0.0 else float("nan")
    full = dict(W_MHD=1.5 * pvol)
    solver = dict(W_MHD=1.5 * pv_s)
    for k, v in (betas or {}).items():
        full[k] = float(v)
        solver[k] = float(v) * ratio
    return dict(p_sep=p_edge, volume=volume, p_sep_volume=p_edge * volume,
                factor=(1.0 / ratio if ratio not in (0.0,) and
                        np.isfinite(ratio) else float("nan")),
                solver=solver, full=full)


def describe(edge=None, pressure=None) -> dict:
    """The record block: the settings, ``p_sep`` of *pressure* (the input
    value, whatever the setting), the offset applied and the axis target."""
    edge = resolve_edge_pressure(edge)
    out = edge.record()
    if pressure is not None:
        out["p_sep"] = separatrix_pressure_of(pressure)
        out["p_axis"] = float(pressure[0])
        out["p_sep_applied"] = applied_offset(pressure, edge)
        out["pax_target"] = solver_pax(pressure, edge)
    return out


# ---------------------------------------------------------------------------
#  delivery (written g-files) and the archive record
# ---------------------------------------------------------------------------
#: JSON attribute carrying the edge-pressure record of an archived group
#: (``_baseline`` and every draw): the two settings, ``p_sep``, the offset
#: applied, the axis target and -- for a draw -- both pressure frames.
EDGE_PRESSURE_ATTR = "edge_pressure_json"


def lcfs_kwargs(p_sep) -> dict:
    """The extra ``save_eqdsk`` keyword that makes a written g-file carry
    the FULL pressure: ``{"lcfs_pressure": p_sep}`` (the solver adds the
    constant to ``PRES``; ``PPRIME`` is unchanged, so ``PRES`` still
    differentiates to ``PPRIME``), or ``{}`` when nothing is added back --
    the call is then exactly the one made before the setting existed."""
    p_sep = float(p_sep)
    return {} if p_sep == 0.0 else {"lcfs_pressure": p_sep}


def archive_record(edge=None, pressure=None, stats=None, p_sep_applied=None):
    """The record stored under :data:`EDGE_PRESSURE_ATTR`.

    :func:`describe` of *pressure* (when given), ``p_sep_applied``
    overridden by an explicit value (an engine draw reports its own), and
    ``frames`` = :func:`pressure_frames` of *stats* (``None`` without
    stats, or when they cannot be formed -- the reason is recorded)."""
    edge = resolve_edge_pressure(edge)
    out = describe(edge, pressure)
    if p_sep_applied is not None:
        out["p_sep_applied"] = float(p_sep_applied)
    out["frames"] = None
    if stats is not None:
        try:
            out["frames"] = pressure_frames(stats,
                                            out.get("p_sep_applied", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            out["frames_error"] = f"{type(exc).__name__}: {exc}"
    return out


def store_record(header, record, scan_key=None, count=None) -> None:
    """Write *record* as the JSON attribute :data:`EDGE_PRESSURE_ATTR` on
    the archive's ``_baseline`` group (``count=None``) or on draw *count*.
    No-op when the group does not exist or *record* is ``None``."""
    if record is None:
        return
    import json
    import h5py
    from .jbs_loop import jsonable
    from .utils import _baseline_group_path, _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "a") as hf:
        gp = (_baseline_group_path(scan_key) if count is None
              else _group_path(scan_key, count))
        if gp in hf:
            hf[gp].attrs[EDGE_PRESSURE_ATTR] = json.dumps(
                jsonable(record), allow_nan=True)


def load_record(header, count=None, scan_key=None):
    """The record :func:`store_record` wrote, or ``None``."""
    import json
    import h5py
    from .utils import _baseline_group_path, _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "r") as hf:
        gp = (_baseline_group_path(scan_key) if count is None
              else _group_path(scan_key, count))
        if gp not in hf or EDGE_PRESSURE_ATTR not in hf[gp].attrs:
            return None
        return json.loads(hf[gp].attrs[EDGE_PRESSURE_ATTR])
