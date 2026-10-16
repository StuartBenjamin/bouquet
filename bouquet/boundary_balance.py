"""The boundary (isoflux) constraint's partition, measured and reported
(docs/engine.md, "The boundary constraint: a constant total inside a measured
window").

The unified engine's boundary constraint is a least-squares penalty on the
flux differences along the boundary, solved together with the coil
regularisation (a least-squares penalty on the coil currents' departure from
their targets).  Only the RATIO of the two penalties matters to the solution:
a perturbation of the plasma is split between moving the boundary and moving
the coils in proportion to it.  The total of the boundary set is a constant,
:data:`bouquet.solver_geometry.ISOFLUX_WEIGHT_TOTAL` (or the explicit
override ``SolverConfig.isoflux_weight_total``), chosen inside the window
where the two cuts of the in-spec rule cost the solver comparably.

This module MEASURES that split on every reconstruction and reports it; it
never changes the total.  The PARTITION PROBE (:func:`measure_partition`),
run once after the reconstruction converged: the delivered request re-solved
from the delivered equilibrium with the pressure scaled by
``1 + PARTITION_PRESSURE_STEP`` (a beta_p perturbation at fixed current
request and I_p; deterministic), under the draws' strong coil
regularisation toward the delivered currents and the installed boundary set.
The partition is

    rho = (boundary RMS / boundary cut) / (coil drift / coil cut)

with the coil fraction the binding one of the coil test's criteria, each 1
exactly at its cut (``sqrt(chi2_nu / chi2_max)`` and ``max|z| / z_max``, or
the legacy drift test's F-coil percentage over its bound), taken over the
REGULARISED coils only: the anti-series vertical-control pair is excluded.
That pair and the boundary move together as one rigid vertical displacement
(its ``#VSC`` channel is nearly unregularised on purpose), so with it in the
coil fraction the ratio is flat near 1 over most of the window and says
nothing about the strength.  Outside :data:`PARTITION_BAND` the record is
flagged and a ``UserWarning`` names it.  One extra GS solve (capped at the
draws' iteration cap); the delivered equilibrium and the reconstruction's
coil regularisation are restored afterwards, so the baseline is unchanged.
"""
from __future__ import annotations

import math
import time

import numpy as np

#: Version of the probe and its record.
PARTITION_VERSION = "boundary_partition/1"

#: The probe's perturbation: the delivered pressure scaled by (1 + this) at
#: fixed current request and I_p.  2 %: the amplitude at which the probe's
#: re-solve was feasible on every case of the 2026-10 investigation (at 10 %
#: the solve stopped at the iteration cap on some experimental cases), and
#: still far above the re-solve's own reproduction (sub-0.1 mm, 1e-5).  The
#: partition is a ratio of two linear responses, so to first order it does
#: not depend on the amplitude (measured: the balance crossing moved by at
#: most a factor 1.6 between 2 % and 10 %).
PARTITION_PRESSURE_STEP = 0.02

#: The band the reported partition is expected in: a ratio outside it (one
#: cut costs the solver more than three times the other) is flagged and
#: warned, never acted on.  Rationale: at the default total the six cases of
#: the 2026-10 investigation measured 0.8-2.6 (vertical-control pair
#: excluded); a factor of three either side holds all of them and flags a
#: case where one cut alone would decide acceptance.
PARTITION_BAND = (1.0 / 3.0, 3.0)

#: How the coil fraction is formed, stamped.
COIL_FRACTION_RULE = {
    "chi2": ("coil fraction = max(sqrt(chi2_nu / chi2_max), max|z| / z_max) "
             "over the regularised coils (the vertical-control pair "
             "excluded)"),
    "legacy": ("coil fraction = max F-coil drift / F_max over the "
               "regularised coils (the vertical-control pair excluded)"),
}

PROBE_RULE = ("partition probe (report only): after the reconstruction, the "
              "delivered request re-solved once from the delivered equilibrium "
              "with the pressure x (1 + PARTITION_PRESSURE_STEP), under the "
              "draws' strong coil regularisation toward the delivered "
              "currents and the installed boundary set; rho = (boundary RMS / "
              "cut) / coil fraction (vertical-control pair excluded); the "
              "delivered equilibrium and the coil regularisation are restored")


def coil_fraction(kind, info, chi2_max=None, z_max=None, inspec_F_max=None,
                  inspec_VSC_max=None):
    """The coil test's binding fraction of its cut (1 = exactly at the cut;
    :data:`COIL_FRACTION_RULE`); NaN when unmeasured.  *info* is measured
    over the regularised coils only (the caller leaves the vertical-control
    pair out); ``inspec_VSC_max`` is accepted for a uniform call and unused."""
    if kind == "chi2":
        f = [math.sqrt(max(float(info.get("chi2_nu", np.nan)), 0.0)
                       / float(chi2_max))]
        if z_max is not None:
            f.append(float(info.get("max_abs_z", np.nan)) / float(z_max))
    else:
        f = [float(info.get("max_F_drift_pct", np.nan))
             / (100.0 * float(inspec_F_max))]
    f = np.asarray(f, dtype=float)
    return float(np.max(f)) if np.all(np.isfinite(f)) else float("nan")


def drift_pcts(draw, base, vsc_coils=()):
    """The legacy drift test's F-coil number [%] of a coil-current dict, the
    vertical-control pair *vsc_coils* left out."""
    F = []
    for c, i0 in base.items():
        if c in vsc_coils or c not in draw or abs(float(i0)) <= 0:
            continue
        F.append(100.0 * abs(float(draw[c]) - float(i0)) / abs(float(i0)))
    return dict(max_F_drift_pct=float(max(F)) if F else float("nan"))


def partition_cuts(bq, base):
    """The in-spec rule's two cuts as ``generate()`` will resolve them, for
    the coil currents *base*: ``dict(delta_cut_mm, cut_source, kind, sigma,
    frac_kw, vsc_coils, coil_fraction_rule)``.  A disabled boundary cut is
    replaced by the generic one (:data:`bouquet.devices.
    GENERIC_BOUNDARY_RMS_MM`) and stamped as such: the ratio needs a
    length."""
    import warnings
    from .devices import GENERIC_BOUNDARY_RMS_MM
    fc = bq.config.filtering
    try:
        mm, src = bq._boundary_cut(quiet=True)
    except Exception as e:                # pragma: no cover - defensive
        mm, src = None, f"unresolvable ({type(e).__name__})"
    if mm is None:
        mm, src = GENERIC_BOUNDARY_RMS_MM, (f"{src}: the boundary cut is off; "
                                            "the partition uses the generic "
                                            f"{GENERIC_BOUNDARY_RMS_MM:g} mm")
    vsc = tuple(getattr(bq.config.solver, "coil_vsc", {}) or {})
    kind, sig, cm, zm = "legacy", None, None, None
    if fc.coil_filter == "chi2":
        from .coil_spec import (CoilSigmaUnavailable, resolve_coil_acceptance,
                                resolve_coil_sigma)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sig, model = resolve_coil_sigma(
                    dict(base), sigma=fc.coil_sigma, device=bq.config.device,
                    era=bq._coil_daq_era())
                cm, zm, _src, _nu = resolve_coil_acceptance(
                    model, chi2_max=fc.chi2_max, z_max=fc.z_max)
            kind = "chi2"
        except CoilSigmaUnavailable:
            kind = "legacy"
    return dict(delta_cut_mm=float(mm), cut_source=str(src), kind=kind,
                sigma=sig, vsc_coils=vsc,
                frac_kw=dict(chi2_max=cm, z_max=zm,
                             inspec_F_max=fc.inspec_F_max,
                             inspec_VSC_max=fc.inspec_VSC_max),
                coil_fraction_rule=COIL_FRACTION_RULE[kind])


def partition_of(cur, base, rms_mm, cuts):
    """``(rho, boundary_frac, coil_frac, coil_info)`` of a probe state with
    coil currents *cur* and boundary RMS *rms_mm* [mm] against *base* and
    the *cuts* (:func:`partition_cuts`); the vertical-control pair is left
    out of the coil fraction."""
    from .coil_spec import coil_chi2
    vsc = tuple(cuts["vsc_coils"])
    if cuts["kind"] == "chi2":
        coils = [c for c in (cuts["sigma"] or {}) if c not in vsc]
        info = coil_chi2(cur, base, cuts["sigma"], coils=coils)
        info = {k: info[k] for k in ("chi2_nu", "max_abs_z", "nu",
                                     "worst_coil")}
    else:
        info = drift_pcts(cur, base, vsc)
    cf = coil_fraction(cuts["kind"], info, **cuts["frac_kw"])
    bf = float(rms_mm) / float(cuts["delta_cut_mm"])
    rho = bf / cf if (np.isfinite(bf) and np.isfinite(cf) and cf > 0) \
        else float("nan")
    return float(rho), float(bf), float(cf), info


#: What the probe needs of the solver object (a live TokaMaker has all of it).
PROBE_SOLVER_API = ("copy_eq", "replace_eq", "trace_surf",
                    "get_coil_currents", "set_coil_reg", "coil_reg_term",
                    "coil_sets")


def probe_capable(mygs) -> bool:
    """Whether *mygs* offers everything the probe uses
    (:data:`PROBE_SOLVER_API`); a solver double without it (the fast tests'
    stand-ins) is not probed, and the record says why."""
    return mygs is not None and all(hasattr(mygs, a) for a in PROBE_SOLVER_API)


def draw_coil_reg_weights():
    """``(soft_reg_weight, vsc_soft_reg_weight)``: the draws' strong coil
    regularisation, read from ``generate_bouquet``'s own defaults so the
    probe can never drift from what the homotopy installs."""
    import inspect
    from .TokaMaker_interface import generate_bouquet
    p = inspect.signature(generate_bouquet).parameters
    return (float(p["soft_reg_weight"].default),
            float(p["vsc_soft_reg_weight"].default))


def in_band(rho, band=PARTITION_BAND):
    """``True`` / ``False`` for a finite *rho* inside / outside *band*,
    ``None`` when unmeasured."""
    if rho is None or not np.isfinite(rho):
        return None
    return bool(band[0] <= float(rho) <= band[1])


def partition_warning(rec):
    """The warning text for a partition record, or ``None`` when there is
    nothing to warn about (measured and inside the band, or switched off)."""
    lo, hi = rec.get("band") or PARTITION_BAND
    W = rec.get("total_nominal")
    Wtxt = "?" if W is None else f"{float(W):.4g}"
    if rec.get("in_band") is False:
        return (f"boundary partition {float(rec['partition']):.3g} at the "
                f"installed total {Wtxt} is outside [{lo:.3g}, {hi:g}]: one "
                "of the in-spec rule's two cuts (boundary RMS, coil test) "
                "costs the solver more than three times the other on this "
                "case (reported only; the total is unchanged; "
                "Baseline.engine['boundary_balance']['partition'])")
    if rec.get("status") not in ("measured", "off"):
        return (f"boundary partition not measured ({rec.get('status')}"
                + (f": {rec['error']}" if rec.get("error") else "") + ")")
    return None


def _finite(obj):
    """*obj* (already :func:`bouquet.jbs_loop.jsonable`) with every
    non-finite float replaced by ``None``: the record is archived as JSON,
    and a NaN would neither survive that nor compare equal to itself."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def measure_partition(bq, request, pressure, make_backend, *, psi_pad,
                      maxits, coil_reg_terms, geom_record):
    """Measure the partition at the INSTALLED boundary set (module
    docstring) on the reconstruction's delivered state, which the solver
    holds now; restore it afterwards.  *request*: the delivered current
    request; *pressure*: the reconstruction's pressure [Pa];
    *make_backend(maxits)*: an engine backend on the live solver with the
    reconstruction's settings; *coil_reg_terms*: the coil regularisation the
    reconstruction solved under (re-installed afterwards; ``None`` = not on
    record, and the probe is not run, since it could not be undone);
    *geom_record*: the solver geometry's record (the totals).  Returns the
    JSON-safe record (warned per :func:`partition_warning`); a failed probe
    solve is recorded, never raised."""
    import warnings
    from .filtering import boundary_deviation_mm
    from .jbs_loop import jsonable
    from .TokaMaker_interface import strong_coil_reg
    from .utils import safe_trace_surf
    mygs = bq.mygs
    t0 = time.perf_counter()
    rec = dict(version=PARTITION_VERSION, report_only=True,
               probe_rule=PROBE_RULE,
               pressure_step=PARTITION_PRESSURE_STEP,
               band=list(PARTITION_BAND),
               total_nominal=geom_record.get("isoflux_weight_total"),
               total_installed=geom_record.get(
                   "isoflux_weight_installed_total"),
               probe_maxits=maxits, n_solves=0, partition=None,
               in_band=None, status=None)
    if not probe_capable(mygs):
        rec["status"] = ("not measured: the solver object cannot be probed "
                         "(a test stand-in)")
    elif coil_reg_terms is None:
        rec["status"] = ("not measured: the reconstruction's coil "
                         "regularisation is not on record, so the probe "
                         "could not restore it")
    else:
        soft, vsc_w = draw_coil_reg_weights()
        rec.update(coil_reg_weight=soft, vsc_reg_weight=vsc_w)
        snap = mygs.copy_eq()
        try:
            cur0, _ = mygs.get_coil_currents()
            base = {str(k): float(v) for k, v in cur0.items()}
            ref_lcfs = safe_trace_surf(mygs, 1.0 - float(psi_pad))
            cuts = partition_cuts(bq, base)
            rec.update(boundary_cut_mm=cuts["delta_cut_mm"],
                       boundary_cut_source=cuts["cut_source"],
                       coil_test=cuts["kind"],
                       coil_fraction_rule=cuts["coil_fraction_rule"],
                       excluded_coils=list(cuts["vsc_coils"]),
                       coil_acceptance={k: v for k, v in
                                        cuts["frac_kw"].items()
                                        if v is not None})
            pb = make_backend(maxits)
            pb.set_inputs(pressure=(1.0 + PARTITION_PRESSURE_STEP)
                          * np.asarray(pressure, dtype=float))
            mygs.set_coil_reg(reg_terms=strong_coil_reg(mygs, base, soft,
                                                        vsc_w))
            ts = time.perf_counter()
            rec["n_solves"] = 1
            pb.solve(request, n_passes=1)
            rec["solve_s"] = float(time.perf_counter() - ts)
            lcfs = safe_trace_surf(mygs, 1.0 - float(psi_pad))
            rms, mx = boundary_deviation_mm(ref_lcfs, lcfs)
            cur, _ = mygs.get_coil_currents()
            cur = {str(k): float(v) for k, v in cur.items()}
            rho, bf, cf, info = partition_of(cur, base, rms, cuts)
            rec.update(partition=rho, boundary_rms_mm=float(rms),
                       boundary_max_mm=float(mx), boundary_frac=bf,
                       coil_frac=cf, coil_info=info, in_band=in_band(rho),
                       status=("measured" if np.isfinite(rho)
                               else "unmeasured"))
        except Exception as e:
            rec["status"] = "probe_failed"
            rec["error"] = f"{type(e).__name__}: {str(e).strip()[:300]}"
        finally:
            mygs.replace_eq(source_eq=snap)
            mygs.set_coil_reg(reg_terms=list(coil_reg_terms))
    rec["wall_s"] = float(time.perf_counter() - t0)
    rec = _finite(jsonable(rec))
    msg = partition_warning(rec)
    if msg is not None:
        print("[engine] WARNING " + msg, flush=True)
        warnings.warn(msg, UserWarning, stacklevel=3)
    if rec["status"] == "measured":
        print(f"[engine] boundary partition {rec['partition']:.3g} at W = "
              f"{rec['total_nominal']} (boundary {rec['boundary_frac']:.3g} "
              f"/ coils {rec['coil_frac']:.3g} of their cuts; "
              f"{rec['n_solves']} probe solve, {rec['wall_s']:.1f} s; "
              "report only)", flush=True)
    return rec
