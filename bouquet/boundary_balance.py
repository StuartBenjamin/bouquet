"""The boundary (isoflux) constraint's strength: a calibrated balance between
the boundary cut and the coil cut (docs/engine.md, "The boundary constraint:
a calibrated balance between the boundary cut and the coil cut").

The unified engine's boundary constraint is a least-squares penalty on the
flux differences along the boundary, solved together with the coil
regularisation (a least-squares penalty on the coil currents' departure from
their targets).  Only the RATIO of the two penalties matters to the solution:
a perturbation of the plasma is split between moving the boundary and moving
the coils in proportion to it.  The in-spec rule judges a draw on both
(the boundary RMS cut and the coil cut), so the ratio is chosen to split a
perturbation evenly between the two cuts: a boundary miss equal to the
boundary cut costs the solver the same as a coil drift equal to the coil cut.

Rule (TokaMaker scales every constraint ROW by its weight, so the penalty is
the sum of the squared weighted residuals; the coil rows act on mu0 x the
current in A, the isoflux rows on psi in Wb/rad, and d psi = R B_p d n)::

    sum_j (W u_j (R B_p)_j delta_cut)^2 = sum_c (r_c mu0 dI_c,cut)^2

    W_a = mu0 sqrt(sum_c (r_c dI_c,cut)^2) / (delta_cut sqrt(sum_j (u_j (R B_p)_j)^2))

with ``u_j`` the per-length weight fractions of the standard set (summing to
1; the reference point ``j = 0`` is not applied by TokaMaker and is left out),
``delta_cut`` the boundary RMS cut [m], ``dI_c,cut`` the coil cut per coil
(``sqrt(chi2_max) x sigma_c`` on the chi^2 test: a drift of that size on every
coil sits exactly at ``chi2/nu = chi2_max``; ``F_max x |I_c|`` / ``VSC_max x
|I_c|`` on the legacy drift test) and ``r_c`` the draws' soft coil
regularisation weight (``generate_bouquet``'s ``soft_reg_weight``).

``W_a`` is only the start: TokaMaker's internal normalisation and the
plasma's own response are measured, not derived, by the PARTITION PROBE
(:func:`calibrate_log_root` over :func:`probe_point`), on the reconstruction's
anchor equilibrium: the anchor request solved with the pressure scaled by
``1 + PROBE_PRESSURE_SCALE`` (a beta_p perturbation at fixed current profile
and I_p, deterministic, no random numbers), under the draws' strong coil
regularisation toward the anchor's own currents and the boundary set at a
trial total ``W``.  The partition ratio is

    rho(W) = (boundary RMS / boundary cut) / (coil drift / coil cut)

(the coil fraction is the binding one of the coil test's criteria, each 1
exactly at its cut: ``sqrt(chi2_nu / chi2_max)`` and ``max|z| / z_max``, or
the legacy drift test's two percentages over their bounds).  A bracketed
root find on ``log W`` (regula falsi with the Illinois step) brings ``rho``
to 1 within :data:`BALANCE_TOLERANCE`.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

#: Version of the balance rule and probe, stamped with every calibration.
BALANCE_VERSION = "boundary_balance/1"

#: mu0 [H/m]: TokaMaker's coil rows act on mu0 x the current in A.
MU0 = 4.0e-7 * math.pi

#: The probe's perturbation: the anchor's pressure scaled by (1 + this) at
#: fixed current-profile request and I_p (a beta_p perturbation).  The
#: partition ratio is a ratio of two linear responses, so its value does not
#: depend on this amplitude to first order; 10 % moves the boundary and the
#: coils well above the re-solve's own reproduction (sub-0.1 mm, 1e-5).
PROBE_PRESSURE_SCALE = 0.10

#: The calibration's tolerance on the partition ratio: converged when
#: 1/BALANCE_TOLERANCE <= rho <= BALANCE_TOLERANCE.  20 %: the coil and
#: boundary fractions of a draw scatter by far more than that from draw to
#: draw, so a tighter balance buys nothing, and each further bracket step
#: costs one GS solve.
BALANCE_TOLERANCE = 1.2

#: The probe's solve budget (each a capped GS solve from the anchor state).
PROBE_MAX_SOLVES = 8

#: The first bracket step and the bracket's reach from the analytic estimate,
#: in decades of W.  The reach is wider than the one-decade test bar on
#: purpose: a calibrated W more than a decade from the analytic estimate is
#: still found (and reported), so the test can flag it.
PROBE_STEP_DECADES = 1.0
PROBE_MAX_DECADES = 3.0

#: How the coil cut per coil and the coil fraction are formed, stamped.
COIL_CUT_RULE = {
    "chi2": ("dI_c,cut = sqrt(chi2_max) x sigma_c; coil fraction = max("
             "sqrt(chi2_nu / chi2_max), max|z| / z_max)"),
    "legacy": ("dI_c,cut = F_max x |I_c| (VSC pair: VSC_max x |I_c|); coil "
               "fraction = max(F drift / F_max, VSC drift / VSC_max)"),
}

PROBE_RULE = ("partition probe: the anchor request re-solved from the anchor "
              "equilibrium with the pressure x (1 + PROBE_PRESSURE_SCALE), "
              "under the draws' strong coil regularisation toward the "
              "anchor's currents, the boundary set at the trial total; "
              "rho = (boundary RMS / cut) / coil fraction; bracketed root "
              "find on log W (Illinois), from the analytic estimate")


def analytic_weight_total(unit_w, rbp, delta_cut_m, coil_cuts, reg_weights):
    """The analytic balance estimate ``W_a`` (module docstring).

    *unit_w*: the set's per-length weights normalised to sum 1 (the
    reference point ``unit_w[0]`` is dropped, as TokaMaker drops it);
    *rbp*: ``R B_p`` [T m] at the same points; *delta_cut_m*: the boundary
    cut [m]; *coil_cuts*: ``{coil: dI_c,cut}`` [A]; *reg_weights*:
    ``{coil: r_c}`` (a coil missing from it is not regularised and is left
    out).  Returns ``(W_a, parts)``; ``W_a`` is NaN when it cannot be formed
    (no regularised coil with a cut, no boundary field, a non-positive
    cut)."""
    u = np.asarray(unit_w, dtype=float).ravel()
    b = np.asarray(rbp, dtype=float).ravel()
    if u.shape != b.shape:
        raise ValueError("unit_w and rbp must have the same length")
    ok = np.isfinite(b)
    ok[0] = False                       # the reference point: not applied
    bnd = float(np.sqrt(np.sum((u[ok] * b[ok]) ** 2))) if ok.any() else 0.0
    coil = float(np.sqrt(sum((float(reg_weights[c]) * float(d)) ** 2
                             for c, d in coil_cuts.items()
                             if c in reg_weights and np.isfinite(d))))
    parts = dict(boundary_rms_field=bnd, coil_penalty_root=coil,
                 n_points_applied=int(ok.sum()),
                 n_coils=int(sum(1 for c in coil_cuts if c in reg_weights)),
                 delta_cut_m=float(delta_cut_m))
    if not (bnd > 0 and coil > 0 and delta_cut_m > 0):
        return float("nan"), parts
    return float(MU0 * coil / (float(delta_cut_m) * bnd)), parts


def coil_cuts_for(kind, base, sigma=None, chi2_max=None, inspec_F_max=None,
                  inspec_VSC_max=None, vsc_coils=()):
    """``{coil: dI_c,cut}`` [A] of the in-spec rule's coil test
    (:data:`COIL_CUT_RULE`)."""
    if kind == "chi2":
        s = math.sqrt(float(chi2_max))
        return {c: s * float(v) for c, v in (sigma or {}).items() if c in base}
    return {c: (float(inspec_VSC_max) if c in vsc_coils
                else float(inspec_F_max)) * abs(float(i))
            for c, i in base.items()}


def coil_fraction(kind, info, chi2_max=None, z_max=None, inspec_F_max=None,
                  inspec_VSC_max=None):
    """The coil test's binding fraction of its cut (1 = exactly at the cut;
    :data:`COIL_CUT_RULE`); NaN when unmeasured."""
    if kind == "chi2":
        f = [math.sqrt(max(float(info.get("chi2_nu", np.nan)), 0.0)
                       / float(chi2_max))]
        if z_max is not None:
            f.append(float(info.get("max_abs_z", np.nan)) / float(z_max))
    else:
        f = [float(info.get("max_F_drift_pct", np.nan))
             / (100.0 * float(inspec_F_max)),
             float(info.get("max_VSC_drift_pct", np.nan))
             / (100.0 * float(inspec_VSC_max))]
    f = np.asarray(f, dtype=float)
    return float(np.max(f)) if np.all(np.isfinite(f)) else float("nan")


def drift_pcts(draw, base, vsc_coils=()):
    """The legacy drift test's two numbers [%] of a coil-current dict."""
    F, V = [], []
    for c, i0 in base.items():
        if c not in draw or abs(float(i0)) <= 0:
            continue
        d = 100.0 * abs(float(draw[c]) - float(i0)) / abs(float(i0))
        (V if c in vsc_coils else F).append(d)
    return dict(max_F_drift_pct=float(max(F)) if F else float("nan"),
                max_VSC_drift_pct=float(max(V)) if V else float("nan"))


def calibrate_log_root(evaluate: Callable[[float], dict], W0: float, *,
                       tol=BALANCE_TOLERANCE, max_solves=PROBE_MAX_SOLVES,
                       step_decades=PROBE_STEP_DECADES,
                       max_decades=PROBE_MAX_DECADES) -> dict:
    """Bring ``rho(W)`` to 1 within *tol* by a bracketed root find on
    ``f(u) = log rho(exp u)`` from ``W0``.  *evaluate(W)* returns a dict
    with ``rho`` (a non-finite or missing ``rho``, or an ``error``, is a
    failed point).  ``rho`` falls as ``W`` grows (a stiffer boundary moves
    the boundary less and the coils more).

    Bracket: from ``W0`` step by *step_decades* toward the root (doubling
    the step each time) until ``f`` changes sign, within *max_decades* of
    ``W0``; a failed point halves the step back toward the last good point.
    Then regula falsi on ``u`` with the Illinois modification.  Returns a
    record: ``W`` (the calibrated total, or ``None``), ``status``
    (``"converged"``: |log rho| <= log tol; ``"budget"``: bracketed but not
    within *tol* after *max_solves* -- ``W`` is the best measured point;
    ``"no_bracket"``: no sign change found -- ``W`` is ``None``, the caller
    falls back), ``points`` (every evaluation), ``n_solves``."""
    lt = math.log(float(tol))
    pts = []
    u0 = math.log(float(W0))

    def ev(u):
        # the start is evaluated at W0 exactly (not exp(log W0))
        W = float(W0) if u == u0 else float(math.exp(u))
        try:
            r = dict(evaluate(W))
        except Exception as e:           # a failed probe solve
            r = dict(error=f"{type(e).__name__}: {str(e).strip()[:200]}")
        r["W"] = W
        rho = r.get("rho")
        r["f"] = (float(math.log(rho)) if rho is not None and np.isfinite(rho)
                  and rho > 0 else None)
        pts.append(r)
        return r["f"]

    def done(status, W):
        good = [p for p in pts if p["f"] is not None]
        best = min(good, key=lambda p: abs(p["f"])) if good else None
        if status != "no_bracket" and W is None and best is not None:
            W = best["W"]
        return dict(W=W, status=status, points=pts, n_solves=len(pts),
                    tolerance=float(tol),
                    rho=(None if W is None else
                         next((p.get("rho") for p in pts if p["W"] == W),
                              None)))

    f0 = ev(u0)
    if f0 is None:
        return done("no_bracket", None)
    if abs(f0) <= lt:
        return done("converged", float(W0))
    s = 1.0 if f0 > 0 else -1.0          # f > 0: the boundary moves too much
    ua, fa = u0, f0
    step = float(step_decades) * math.log(10.0)
    reach = float(max_decades) * math.log(10.0)
    ub = fb = None
    while len(pts) < max_solves:
        u = ua + s * step
        if abs(u - u0) > reach + 1e-12:
            u = u0 + s * reach
            if abs(u - ua) < 1e-12:
                break
        f = ev(u)
        if f is None:
            step *= 0.5                   # failed: back toward the good side
            continue
        if abs(f) <= lt:
            return done("converged", pts[-1]["W"])
        if (f > 0) != (fa > 0):
            ub, fb = u, f
            break
        ua, fa = u, f
        step *= 2.0
    if ub is None:
        return done("no_bracket", None)
    side = 0
    while len(pts) < max_solves:
        u = (ua * fb - ub * fa) / (fb - fa)
        f = ev(u)
        if f is None:
            break
        if abs(f) <= lt:
            return done("converged", pts[-1]["W"])
        if (f > 0) == (fa > 0):
            ua, fa = u, f
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            ub, fb = u, f
            if side == 1:
                fa *= 0.5
            side = 1
    return done("budget", None)


def boundary_rbp(mygs, pts):
    """``R B_p`` [T m] of the CURRENT equilibrium at *pts* (NaN where the
    field cannot be placed on the mesh)."""
    from .mse import mse_field_at
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    B, found = mse_field_at(mygs, P[:, 0], P[:, 1])
    out = P[:, 0] * np.hypot(B[:, 0], B[:, 2])
    out[~np.asarray(found, dtype=bool)] = np.nan
    return out


def balance_cuts(bq, base):
    """The in-spec rule's two cuts as ``generate()`` will resolve them, for
    the coil currents *base*: ``dict(delta_cut_m, cut_source, kind,
    coil_cuts, frac_kw, model)`` (``frac_kw``: :func:`coil_fraction`'s
    keyword arguments).  A disabled boundary cut is replaced by the generic
    one (:data:`bouquet.devices.GENERIC_BOUNDARY_RMS_MM`) and stamped as
    such: the balance needs a length."""
    import warnings
    from .devices import GENERIC_BOUNDARY_RMS_MM
    fc = bq.config.filtering
    try:
        mm, src = bq._boundary_cut(quiet=True)
    except Exception as e:                # pragma: no cover - defensive
        mm, src = None, f"unresolvable ({type(e).__name__})"
    if mm is None:
        mm, src = GENERIC_BOUNDARY_RMS_MM, (f"{src}: the boundary cut is off; "
                                            "the balance uses the generic "
                                            f"{GENERIC_BOUNDARY_RMS_MM:g} mm")
    vsc = tuple(getattr(bq.config.solver, "coil_vsc", {}) or {})
    out = dict(delta_cut_m=float(mm) * 1e-3, delta_cut_mm=float(mm),
               cut_source=str(src), vsc_coils=vsc)
    kind, model, sig, cm, zm = "legacy", None, None, None, None
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
    out.update(kind=kind, sigma=sig, model=model,
               coil_cuts=coil_cuts_for(kind, base, sig, cm, fc.inspec_F_max,
                                       fc.inspec_VSC_max, vsc),
               frac_kw=dict(chi2_max=cm, z_max=zm,
                            inspec_F_max=fc.inspec_F_max,
                            inspec_VSC_max=fc.inspec_VSC_max),
               coil_cut_rule=COIL_CUT_RULE["chi2" if kind == "chi2"
                                           else "legacy"])
    return out


def probe_point(mygs, anchor_eq, backend, request, pts, unit_w, W, *,
                base, ref_lcfs, cuts, reg_terms, psi_pad):
    """ONE partition-probe evaluation at total *W* (module docstring):
    restores *anchor_eq*, installs the set at *W* and the strong coil
    regularisation *reg_terms*, solves *request* with the probe pressure and
    measures.  Returns ``dict(rho, boundary_rms_mm, boundary_frac,
    coil_frac, coil_info, solve_s)``; raises what the solve raises."""
    from .coil_spec import coil_chi2
    from .filtering import boundary_deviation_mm
    from .utils import safe_trace_surf
    mygs.replace_eq(source_eq=anchor_eq)
    mygs.set_isoflux(np.asarray(pts, dtype=float),
                     weights=np.asarray(unit_w, dtype=float) * float(W))
    mygs.set_coil_reg(reg_terms=list(reg_terms))
    t0 = time.perf_counter()
    backend.solve(request, n_passes=1)
    dt = time.perf_counter() - t0
    lcfs = safe_trace_surf(mygs, 1.0 - float(psi_pad))
    rms, mx = boundary_deviation_mm(ref_lcfs, lcfs)
    cur, _ = mygs.get_coil_currents()
    cur = {str(k): float(v) for k, v in cur.items()}
    if cuts["kind"] == "chi2":
        info = coil_chi2(cur, base, cuts["sigma"])
        info = {k: info[k] for k in ("chi2_nu", "max_abs_z", "nu",
                                     "worst_coil")}
    else:
        info = drift_pcts(cur, base, cuts["vsc_coils"])
    cf = coil_fraction(cuts["kind"], info, **cuts["frac_kw"])
    bf = float(rms) / cuts["delta_cut_mm"]
    rho = bf / cf if (np.isfinite(bf) and np.isfinite(cf) and cf > 0) \
        else float("nan")
    return dict(rho=float(rho), boundary_rms_mm=float(rms),
                boundary_max_mm=float(mx), boundary_frac=float(bf),
                coil_frac=float(cf), coil_info=info, solve_s=float(dt))


#: What the probe needs of the solver object (a live TokaMaker has all of it).
PROBE_SOLVER_API = ("copy_eq", "replace_eq", "get_field_eval", "trace_surf",
                    "get_coil_currents", "set_coil_reg", "coil_reg_term",
                    "set_isoflux", "coil_sets")


def probe_capable(mygs) -> bool:
    """Whether *mygs* offers everything the probe uses
    (:data:`PROBE_SOLVER_API`).  A solver double without it (the fast
    tests' stand-ins) is not calibrated: its set keeps the provisional total
    and the record says why."""
    return mygs is not None and all(hasattr(mygs, a) for a in PROBE_SOLVER_API)


def unit_weights(w):
    """The per-length weights of a set normalised to sum 1."""
    w = np.asarray(w, dtype=float)
    return w / float(np.sum(w))


def draw_coil_reg_weights():
    """``(soft_reg_weight, vsc_soft_reg_weight)``: the draws' strong coil
    regularisation, read from ``generate_bouquet``'s own defaults so the
    probe can never drift from what the homotopy installs."""
    import inspect
    from .TokaMaker_interface import generate_bouquet
    p = inspect.signature(generate_bouquet).parameters
    return (float(p["soft_reg_weight"].default),
            float(p["vsc_soft_reg_weight"].default))


def calibrate_boundary_constraint(bq, contract, make_backend, *, psi_pad,
                                  maxits):
    """Calibrate the total of the solver geometry's standard boundary set on
    this case (module docstring).  Runs from the solver's CURRENT state (the
    reconstruction's start) and restores it exactly afterwards: the anchor
    request solved twice under the geometry's provisional set and the
    reconstruction's coil regularisation (the reconstruction's own anchor
    solve), then the probe.  *make_backend(maxits)* builds an engine backend
    on the live solver with the reconstruction's settings; the probe's
    solves are capped at *maxits* (``engine_draw_solve_maxits``, as every
    draw solve).  Returns ``(W, record)``: ``W`` the total to install (the
    calibrated one; the analytic estimate when the probe found no bracket;
    the provisional total when not even that could be formed) and the
    JSON-safe record stamped in the solver geometry and the baseline."""
    import warnings
    from .jbs_loop import jsonable
    from .TokaMaker_interface import strong_coil_reg
    from .utils import safe_trace_surf
    mygs, geom = bq.mygs, bq._geom
    t0 = time.perf_counter()
    unit_w = unit_weights(geom.iso_w)
    W_prov = float(np.sum(geom.iso_w))
    soft, vsc_w = draw_coil_reg_weights()
    rec = dict(version=BALANCE_VERSION, mode="calibrated",
               provisional_total=W_prov, probe_rule=PROBE_RULE,
               probe_pressure_scale=PROBE_PRESSURE_SCALE,
               tolerance=BALANCE_TOLERANCE, max_solves=PROBE_MAX_SOLVES,
               probe_maxits=maxits, coil_reg_weight=soft,
               vsc_reg_weight=vsc_w)
    start = mygs.copy_eq()
    try:
        ab = make_backend(None)
        ab.solve(contract.anchor_request, n_passes=2)
        anchor_eq = mygs.copy_eq()
        cur, _ = mygs.get_coil_currents()
        base = {str(k): float(v) for k, v in cur.items()}
        ref_lcfs = safe_trace_surf(mygs, 1.0 - float(psi_pad))
        rbp = boundary_rbp(mygs, geom.iso_pts)
        cuts = balance_cuts(bq, base)
        reg_w = {str(c): soft for c in mygs.coil_sets}
        W_a, parts = analytic_weight_total(unit_w, rbp, cuts["delta_cut_m"],
                                           cuts["coil_cuts"], reg_w)
        rec.update(analytic_total=W_a, analytic_parts=parts,
                   boundary_cut_mm=cuts["delta_cut_mm"],
                   boundary_cut_source=cuts["cut_source"],
                   coil_test=cuts["kind"], coil_cut_rule=cuts["coil_cut_rule"],
                   coil_acceptance={k: v for k, v in cuts["frac_kw"].items()
                                    if v is not None},
                   rbp_boundary=dict(min=float(np.nanmin(rbp)),
                                     max=float(np.nanmax(rbp)),
                                     n_nan=int(np.sum(~np.isfinite(rbp)))))
        if not np.isfinite(W_a):
            raise ValueError("the analytic estimate could not be formed "
                             f"({parts})")
        reg_terms = strong_coil_reg(mygs, base, soft, vsc_w)
        pb = make_backend(maxits)
        pb.set_inputs(pressure=(1.0 + PROBE_PRESSURE_SCALE)
                      * np.asarray(contract.pressure, dtype=float))
        res = calibrate_log_root(
            lambda W: probe_point(mygs, anchor_eq, pb, contract.anchor_request,
                                  geom.iso_pts, unit_w, W, base=base,
                                  ref_lcfs=ref_lcfs, cuts=cuts,
                                  reg_terms=reg_terms, psi_pad=psi_pad),
            W_a)
        rec.update(status=res["status"], points=res["points"],
                   n_probe_solves=res["n_solves"])
        if res["W"] is None:
            W, rec["fallback"] = W_a, ("no bracket: the analytic estimate is "
                                       "installed")
        else:
            W = float(res["W"])
            rec["partition_ratio"] = res["rho"]
    except Exception as e:
        W_a = rec.get("analytic_total")
        if W_a is not None and np.isfinite(W_a):
            W = float(W_a)
            rec["fallback"] = ("probe failed: the analytic estimate is "
                               "installed")
        else:
            W = W_prov
            rec["fallback"] = ("probe failed before the analytic estimate: "
                               "the provisional total is installed")
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {str(e).strip()[:300]}"
    finally:
        mygs.replace_eq(source_eq=start)
    rec["total"] = float(W)
    rec["installed_total"] = float(W * (1.0 - unit_w[0]))
    rec["wall_s"] = float(time.perf_counter() - t0)
    if rec.get("fallback"):
        msg = (f"boundary balance: {rec['fallback']} (W = {W:.4g}; status "
               f"{rec.get('status')}{'; ' + rec['error'] if 'error' in rec else ''})")
        print("[engine] WARNING " + msg, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    else:
        print(f"[engine] boundary balance: W = {W:.4g} (analytic "
              f"{rec['analytic_total']:.4g}, rho {rec.get('partition_ratio')}"
              f", {rec['n_probe_solves']} probe solves, "
              f"{rec['wall_s']:.1f} s)", flush=True)
    return float(W), jsonable(rec)
