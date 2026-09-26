"""Self-consistent bootstrap current: the outer fixed-point loop.

With ``GenerationConfig.jbs_self_consistent=True`` the bootstrap current is no
longer computed once and frozen.  Every path that builds a current profile
containing a bootstrap runs the same relaxed outer iteration::

    E_0      = anchor equilibrium (already solved by the caller)
    jBS_0    = evaluate_jBS(E_0)            (or the legacy SWB result)
    for k = 0 .. K-1:
        E_k+1   = step(jBS_k)               closure on E_k's geometry + GS solve
        J       = evaluate(E_k+1)           Redl on the NEW equilibrium
        r_k     = residuals(J, jBS_k, E_k+1, E_k)
        jBS_k+1 = (1 - omega) jBS_k + omega J
        converged when every active criterion holds on two consecutive passes

``step`` and ``evaluate`` are supplied by the caller (the IMAS baseline, the
structured / MSE closures, a perturbation draw, the geqdsk reconstruction);
this module owns only what is common to all of them: the residual
definitions, the convergence rule, the relaxation schedule, the failure
policy and the record.

Residuals (all logged every pass)
---------------------------------
``r_j``  current-weighted L2 residual of the profile,
         ``||J - jBS_k||_w / ||J||_w`` with ``||f||_w^2 = int |w| f^2 dpsi_N``
         and ``w`` the pass's own Ip weights (so the norm measures current, not
         raw density).  It is the UNRELAXED fixed-point residual -- the
         distance between the bootstrap the equilibrium was solved with and the
         Redl bootstrap of that equilibrium -- which is ``1/omega`` times the
         relaxed step ``||jBS_k+1 - jBS_k||``, i.e. never looser than it.
``r_I``  ``|int w (J - jBS_k) dpsi_N| / Ip``: the same residual as a fraction
         of the plasma current, on the linear part of the closure's measure.
``dl_i`` ``|l_i(E_k+1) - l_i(E_k)|`` (only where the caller measures l_i).
``dq0``  ``|q0(E_k+1) - q0(E_k)|`` (only where an axis row / q0 target is
         active).

The delivered equilibrium is the last solve; the bootstrap it was solved with
is ``jBS_used``; ``J_final`` (Redl on that equilibrium) differs from it by at
most the tolerances and is recorded alongside.

Failure is never silent: the library raises :class:`JBSNotConverged` carrying
the full residual history; with ``jbs_loop_on_fail="flag"`` the last iterate is
returned with ``converged=False`` and the caller records a closure-limited
reason.  A residual that grows on ``JBS_GROWTH_ABORT_PASSES`` consecutive
passes at the relaxation floor aborts early with the same error.

Nothing here touches an existing solver tolerance: the GS solver's own
``nl_tol``/``maxits``, the closure tolerances and the correctors' acceptance
bands are unchanged; the numbers below define what "j_BS converged" means.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

#: Floor of the under-relaxation factor (``jbs_relax`` is halved toward it
#: whenever ``r_j`` grows).
JBS_RELAX_FLOOR = 0.25
#: Consecutive passes that must meet every active criterion.
JBS_REQUIRED_CONSECUTIVE = 2
#: Growing-``r_j`` passes AT the relaxation floor that abort the loop.
JBS_GROWTH_ABORT_PASSES = 3
#: Extra passes a draw may take at the tight coil stage after the homotopy.
JBS_POST_HOMOTOPY_PASSES = 2
#: MSE-constrained structured closure: ceiling on chord steps (closure with
#: the linearised MSE term -> solve -> evaluate_jBS) after the Jacobian.
MSE_CHORD_MAX_STEPS = 4
#: MSE chord iteration: the linearisation point has stopped moving when the
#: synthetic tan(gamma) changes by less than this many sigma_eff on every chord
#: between consecutive chord steps.  A NEW criterion introduced with the loop
#: (the plan leaves its value open) -- recorded with every result so it can be
#: reviewed; it gates only when the chord iteration stops, never acceptance.
MSE_CHORD_OFFSET_TOL_SIGMA = 0.1
#: Prefix of every closure_limited reason this module's callers add.
JBS_FLAG_PREFIX = "j_BS loop: "

_INIT_CHOICES = ("anchor", "swb")
_ON_FAIL_CHOICES = ("raise", "flag")


class JBSNotConverged(RuntimeError):
    """The self-consistent bootstrap loop did not converge.

    ``record`` (also ``history``) is the loop record of
    :func:`run_jbs_loop` -- every pass's residuals, relaxation factor and the
    reason it stopped -- so a failure can be diagnosed from the exception
    alone.
    """

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = dict(record or {})
        self.history = self.record


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def validate_jbs_settings(gc) -> None:
    """Refuse malformed ``jbs_*`` values in a :class:`GenerationConfig`.

    Values only (type/range/choice); the workflow-level refusals (the loop
    with ``single_profile_jphi``, or without a bootstrap recompute) live in
    ``Bouquet._validate_workflow``.  Defaults always pass, so this is inert
    for a config that never sets the fields.
    """
    def _get(name, default):
        return getattr(gc, name, default)

    on = _get("jbs_self_consistent", False)
    if not isinstance(on, (bool, np.bool_)):
        raise ValueError(f"generation.jbs_self_consistent must be a bool, got "
                         f"{on!r}")
    init = _get("jbs_init", "anchor")
    if init not in _INIT_CHOICES:
        raise ValueError(f"generation.jbs_init must be one of {_INIT_CHOICES},"
                         f" got {init!r}")
    fail = _get("jbs_loop_on_fail", "raise")
    if fail not in _ON_FAIL_CHOICES:
        raise ValueError(f"generation.jbs_loop_on_fail must be one of "
                         f"{_ON_FAIL_CHOICES}, got {fail!r}")
    for name, default in (("jbs_rtol_j", 1e-3), ("jbs_rtol_Ip", 1e-4),
                          ("jbs_tol_li", 1e-3), ("jbs_tol_q0", 2e-3)):
        v = _get(name, default)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"generation.{name} must be a positive number, "
                             f"got {v!r}") from None
        if not (np.isfinite(fv) and fv > 0.0):
            raise ValueError(f"generation.{name} must be a positive finite "
                             f"number, got {v!r}")
    for name, default in (("jbs_max_passes", 8), ("jbs_max_passes_draw", 6)):
        v = _get(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"generation.{name} must be an integer, got "
                             f"{v!r}")
        if int(v) < JBS_REQUIRED_CONSECUTIVE:
            raise ValueError(
                f"generation.{name}={int(v)} cannot converge: convergence "
                f"needs {JBS_REQUIRED_CONSECUTIVE} consecutive passing passes")
    w = _get("jbs_relax", 0.7)
    try:
        fw = float(w)
    except (TypeError, ValueError):
        raise ValueError(f"generation.jbs_relax must be a number in "
                         f"[{JBS_RELAX_FLOOR}, 1], got {w!r}") from None
    if not (JBS_RELAX_FLOOR <= fw <= 1.0):
        raise ValueError(f"generation.jbs_relax must lie in "
                         f"[{JBS_RELAX_FLOOR}, 1] (the floor is "
                         f"JBS_RELAX_FLOOR), got {w!r}")


def jbs_settings(gc, *, draw: bool = False) -> dict:
    """The loop settings of a :class:`GenerationConfig`, validated.

    ``draw=True`` selects ``jbs_max_passes_draw`` as the pass ceiling.
    ``enabled`` is ``False`` for a config without the fields (an old archive's
    provenance), so every caller can gate on it.
    """
    validate_jbs_settings(gc)
    return dict(
        enabled=bool(getattr(gc, "jbs_self_consistent", False)),
        init=str(getattr(gc, "jbs_init", "anchor")),
        rtol_j=float(getattr(gc, "jbs_rtol_j", 1e-3)),
        rtol_Ip=float(getattr(gc, "jbs_rtol_Ip", 1e-4)),
        tol_li=float(getattr(gc, "jbs_tol_li", 1e-3)),
        tol_q0=float(getattr(gc, "jbs_tol_q0", 2e-3)),
        max_passes=int(getattr(gc, "jbs_max_passes_draw", 6) if draw
                       else getattr(gc, "jbs_max_passes", 8)),
        relax=float(getattr(gc, "jbs_relax", 0.7)),
        on_fail=str(getattr(gc, "jbs_loop_on_fail", "raise")),
        relax_floor=float(JBS_RELAX_FLOOR),
        required_consecutive=int(JBS_REQUIRED_CONSECUTIVE),
        growth_abort_passes=int(JBS_GROWTH_ABORT_PASSES),
        post_homotopy_passes=int(JBS_POST_HOMOTOPY_PASSES),
    )


def tolerances_record(settings: dict) -> dict:
    """The tolerance block every loop record carries."""
    return dict(rtol_j=settings["rtol_j"], rtol_Ip=settings["rtol_Ip"],
                tol_li=settings["tol_li"], tol_q0=settings["tol_q0"],
                max_passes=settings["max_passes"],
                required_consecutive=settings["required_consecutive"],
                relax_start=settings["relax"],
                relax_floor=settings["relax_floor"],
                growth_abort_passes=settings["growth_abort_passes"])


# ---------------------------------------------------------------------------
#  provenance
# ---------------------------------------------------------------------------
_OFT_BUILD_CACHE = {}


def oft_build_info() -> dict:
    """``{path, git_hash}`` of the imported OpenFUSIONToolkit (cached).

    The git hash is read from the checkout the package lives in when there is
    one; an install tree without ``.git`` records ``None``.  Never raises.
    """
    if "info" in _OFT_BUILD_CACHE:
        return dict(_OFT_BUILD_CACHE["info"])
    info = {"path": None, "git_hash": None}
    try:
        import os
        import subprocess
        import OpenFUSIONToolkit as _oft
        path = os.path.dirname(os.path.abspath(_oft.__file__))
        info["path"] = path
        try:
            out = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                info["git_hash"] = out.stdout.strip()
        except Exception:
            pass
    except Exception:
        pass
    _OFT_BUILD_CACHE["info"] = dict(info)
    return info


# ---------------------------------------------------------------------------
#  residuals
# ---------------------------------------------------------------------------
def _trap(y, x):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(y, dtype=float),
                           np.asarray(x, dtype=float)))


def weighted_norm(f, w, x) -> float:
    """``sqrt(int |w| f^2 dx)`` -- the current-weighted L2 norm."""
    f = np.asarray(f, dtype=float)
    w = np.abs(np.asarray(w, dtype=float))
    return float(np.sqrt(max(_trap(w * f * f, x), 0.0)))


def profile_residuals(J, jbs, w, x, Ip) -> dict:
    """``r_j`` and ``r_I`` of a Redl profile ``J`` against the profile ``jbs``
    the equilibrium was solved with, plus the logged pedestal diagnostics."""
    J = np.asarray(J, dtype=float)
    jbs = np.asarray(jbs, dtype=float)
    x = np.asarray(x, dtype=float)
    d = J - jbs
    nJ = weighted_norm(J, w, x)
    r_j = (weighted_norm(d, w, x) / nJ) if nJ > 0.0 else (
        0.0 if weighted_norm(d, w, x) == 0.0 else float("inf"))
    Ip = abs(float(Ip))
    I_J = _trap(np.asarray(w, dtype=float) * J, x)
    I_used = _trap(np.asarray(w, dtype=float) * jbs, x)
    r_I = abs(I_J - I_used) / Ip if Ip > 0.0 else float("inf")
    i_pk = int(np.argmax(np.abs(J))) if J.size else 0
    return dict(r_j=float(r_j), r_I=float(r_I), I_BS=float(I_J),
                I_BS_used=float(I_used),
                jBS_peak=float(J[i_pk]) if J.size else float("nan"),
                jBS_peak_psiN=float(x[i_pk]) if J.size else float("nan"))


def residual_weights(eq, psi_N, psi_pad=1e-3):
    """``(w, x, kind)``: the per-surface Ip weights of equilibrium *eq* on
    ``psi_N``, for the residual norms.

    The linear part of the closure's ``jphi-linterp`` measure,
    ``(V'/2pi) |dpsi/dpsi_N| <1/R^2>/<1/R>``, when ``get_q`` returns ``<1/R^2>``;
    otherwise (legacy OFT layout) the ``fsa`` weights
    ``(V'/2pi) |dpsi/dpsi_N| <1/R>`` -- the two differ by the tiny
    ``<R><1/R^2>/<1/R>^2 - 1`` shaping factor, far below anything the norm is
    used to resolve.  ``kind`` names which one was used.  One ``get_q`` call,
    no trace.
    """
    from .utils import fsa_current_geometry
    x = np.asarray(psi_N, dtype=float)
    g = fsa_current_geometry(eq, x, psi_pad=psi_pad, want_pprime=False)
    base = g["dV_dpsi"] / (2.0 * np.pi) * g["dpsi_dpsiN"]
    if g["inv_R2"] is not None:
        return base * g["inv_R2"] / g["inv_R"], x, "jphi-linterp"
    return base * g["inv_R"], x, "fsa"


def _finite_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ---------------------------------------------------------------------------
#  the kernel
# ---------------------------------------------------------------------------
def run_jbs_loop(jbs0, step: Callable, evaluate: Callable, settings: dict, *,
                 Ip: float, meas0: Optional[dict] = None,
                 gate_li: bool = True, gate_q0: bool = False,
                 label: str = "", init: Optional[str] = None,
                 grid: str = "psi_N native",
                 on_pass: Optional[Callable] = None,
                 max_passes: Optional[int] = None,
                 raise_on_fail: Optional[bool] = None) -> dict:
    """Iterate closure <-> GS <-> Redl to the fixed point.

    Parameters
    ----------
    jbs0 : array
        Initial bootstrap profile (the one the FIRST solve is assembled with).
    step : callable ``step(jbs, k) -> meas``
        Closure + assembly + GS solve with bootstrap ``jbs``; returns the
        measurement dict of the NEW equilibrium: ``w`` and ``x`` (the Ip weights
        and the grid ``jbs`` lives on), optionally ``li`` and ``q0``.  Any other
        keys are carried through untouched (``meas_final``).
    evaluate : callable ``evaluate(meas) -> J``
        Redl bootstrap on the equilibrium ``step`` just produced, composed
        exactly the way ``jbs`` is (same smoothing / isolation / scale).
    settings : dict
        :func:`jbs_settings`.
    Ip : float
        Plasma current the ``r_I`` residual is normalised by.
    meas0 : dict or None
        Measurements of the starting equilibrium E_0 (``li``/``q0``), so the
        first pass has a ``dl_i``/``dq0``.
    gate_li, gate_q0 : bool
        Whether ``dl_i`` / ``dq0`` are convergence criteria here (``dq0`` only
        where an axis row / q0 target is active); they are logged either way
        whenever measured.
    on_pass : callable or None
        ``on_pass(k, meas, J, entry)`` after every pass (callers use it to
        refresh per-pass state, e.g. an axis-row target).  ``entry`` carries
        the pass's residuals, ``omega_used`` and ``omega_next`` (the
        relaxation the next iterate is built with; ``None`` when no further
        pass follows), so a caller's own per-pass update can be relaxed by the
        same factor.
    max_passes : int or None
        Override of ``settings["max_passes"]`` (the post-homotopy stage).
    raise_on_fail : bool or None
        Override of the ``settings["on_fail"]`` policy.

    Returns
    -------
    dict with ``jbs_used`` (the bootstrap of the delivered solve), ``J_final``
    (Redl on the delivered equilibrium), ``meas_final``, ``converged`` and
    ``record`` (the logged block, JSON-safe).  On non-convergence raises
    :class:`JBSNotConverged` unless the policy is ``"flag"``.
    """
    t0 = time.perf_counter()
    K = int(settings["max_passes"] if max_passes is None else max_passes)
    need = int(settings.get("required_consecutive", JBS_REQUIRED_CONSECUTIVE))
    floor = float(settings.get("relax_floor", JBS_RELAX_FLOOR))
    n_abort = int(settings.get("growth_abort_passes", JBS_GROWTH_ABORT_PASSES))
    omega = float(settings["relax"])
    if raise_on_fail is None:
        raise_on_fail = (settings.get("on_fail", "raise") == "raise")

    rec = dict(
        enabled=True, label=str(label),
        init=str(init if init is not None else settings.get("init", "anchor")),
        grid=str(grid),
        tolerances=tolerances_record(dict(settings, max_passes=K)),
        criteria=dict(r_j=True, r_I=True, dl_i=bool(gate_li),
                      dq0=bool(gate_q0)),
        n_passes=0, converged=False, stop_reason=None,
        omega=[], r_j=[], r_I=[], dl_i=[], dq0=[], I_BS=[], I_BS_used=[],
        jBS_peak=[], jBS_peak_psiN=[], li=[], q0=[], pass_ok=[],
        wall_s=None,
    )
    try:
        from .physics import EVALUATE_JBS_VERSION
        rec["evaluate_jBS_version"] = EVALUATE_JBS_VERSION
    except Exception:
        rec["evaluate_jBS_version"] = None
    rec["oft_build"] = oft_build_info()

    jbs = np.asarray(jbs0, dtype=float).copy()
    prev = dict(meas0 or {})
    streak = 0
    growth_at_floor = 0
    omega_used_for_current = None     # omega that produced `jbs` (None: init)
    r_j_prev = None
    meas = None
    J = None
    jbs_used = jbs

    for k in range(K):
        jbs_used = jbs
        meas = step(jbs, k)
        J = np.asarray(evaluate(meas), dtype=float)
        if J.shape != jbs.shape:
            raise ValueError(f"run_jbs_loop[{label}]: evaluate returned shape "
                             f"{J.shape}, the iterate has {jbs.shape}")
        res = profile_residuals(J, jbs, meas["w"], meas["x"], Ip)
        li_new = _finite_or_none(meas.get("li"))
        q0_new = _finite_or_none(meas.get("q0"))
        li_old = _finite_or_none(prev.get("li"))
        q0_old = _finite_or_none(prev.get("q0"))
        dl_i = (abs(li_new - li_old) if (li_new is not None
                                         and li_old is not None) else None)
        dq0 = (abs(q0_new - q0_old) if (q0_new is not None
                                        and q0_old is not None) else None)
        ok = (np.isfinite(res["r_j"]) and res["r_j"] <= settings["rtol_j"]
              and np.isfinite(res["r_I"])
              and res["r_I"] <= settings["rtol_Ip"])
        if gate_li:
            ok = ok and (dl_i is not None and dl_i <= settings["tol_li"])
        if gate_q0:
            ok = ok and (dq0 is not None and dq0 <= settings["tol_q0"])
        ok = bool(ok)
        rec["omega"].append(None if omega_used_for_current is None
                            else float(omega_used_for_current))
        rec["r_j"].append(res["r_j"])
        rec["r_I"].append(res["r_I"])
        rec["dl_i"].append(dl_i)
        rec["dq0"].append(dq0)
        rec["I_BS"].append(res["I_BS"])
        rec["I_BS_used"].append(res["I_BS_used"])
        rec["jBS_peak"].append(res["jBS_peak"])
        rec["jBS_peak_psiN"].append(res["jBS_peak_psiN"])
        rec["li"].append(li_new)
        rec["q0"].append(q0_new)
        rec["pass_ok"].append(ok)
        rec["n_passes"] = k + 1
        entry = dict(k=k, ok=ok, dl_i=dl_i, dq0=dq0,
                     omega_used=omega_used_for_current, **res)
        print(f"  [jbs-loop{(' ' + label) if label else ''}] pass {k + 1}/{K}: "
              f"r_j={res['r_j']:.3e} (tol {settings['rtol_j']:.0e}) "
              f"r_I={res['r_I']:.3e} (tol {settings['rtol_Ip']:.0e})"
              + ("" if dl_i is None else f" dl_i={dl_i:.2e}")
              + ("" if dq0 is None else f" dq0={dq0:.2e}")
              + f" I_BS={res['I_BS'] / 1e3:.2f} kA"
              + ("" if omega_used_for_current is None
                 else f" omega={omega_used_for_current:.3f}")
              + (" ok" if ok else ""), flush=True)
        streak = streak + 1 if ok else 0
        if streak >= need:
            rec["converged"] = True
            rec["stop_reason"] = (f"all active criteria met on {need} "
                                  "consecutive passes")
            if on_pass is not None:
                on_pass(k, meas, J, dict(entry, omega_next=None))
            break
        # ---- relaxation schedule: halve on growth, abort at the floor ----
        if r_j_prev is not None and res["r_j"] > r_j_prev:
            if (omega_used_for_current is not None
                    and omega_used_for_current <= floor + 1e-15):
                growth_at_floor += 1
            else:
                growth_at_floor = 0
            omega = max(0.5 * omega, floor)
        else:
            growth_at_floor = 0
        r_j_prev = res["r_j"]
        stop = (growth_at_floor >= n_abort) or (k == K - 1)
        if on_pass is not None:
            # omega_next: the relaxation the NEXT iterate is built with (the
            # caller relaxes any per-pass update of its own -- e.g. a moved
            # constraint row -- by the same factor)
            on_pass(k, meas, J, dict(entry, omega_next=(None if stop
                                                        else omega)))
        if growth_at_floor >= n_abort:
            rec["stop_reason"] = (f"r_j grew on {n_abort} consecutive passes "
                                  f"at the relaxation floor omega={floor:g}")
            break
        if k == K - 1:
            break                       # no further solve: keep jbs as used
        jbs = (1.0 - omega) * jbs + omega * J
        omega_used_for_current = omega
        prev = meas

    rec["wall_s"] = float(time.perf_counter() - t0)
    rec["jbs_converged"] = bool(rec["converged"])
    if not rec["converged"] and rec["stop_reason"] is None:
        rec["stop_reason"] = (f"pass ceiling {K} reached without {need} "
                              "consecutive passing passes")
    rec["final"] = dict(
        r_j=rec["r_j"][-1] if rec["r_j"] else None,
        r_I=rec["r_I"][-1] if rec["r_I"] else None,
        dl_i=rec["dl_i"][-1] if rec["dl_i"] else None,
        dq0=rec["dq0"][-1] if rec["dq0"] else None,
        I_BS=rec["I_BS"][-1] if rec["I_BS"] else None)
    out = dict(jbs_used=np.asarray(jbs_used, dtype=float),
               J_final=(None if J is None else np.asarray(J, dtype=float)),
               meas_final=meas, converged=bool(rec["converged"]), record=rec)
    if not rec["converged"]:
        msg = (f"self-consistent j_BS loop{(' [' + label + ']') if label else ''}"
               f" did not converge: {rec['stop_reason']}; residual history "
               f"r_j={_fmt_hist(rec['r_j'])} r_I={_fmt_hist(rec['r_I'])}"
               + ("" if not gate_li else f" dl_i={_fmt_hist(rec['dl_i'])}")
               + ("" if not gate_q0 else f" dq0={_fmt_hist(rec['dq0'])}"))
        rec["fail_message"] = msg
        print("  [jbs-loop] " + msg, flush=True)
        if raise_on_fail:
            raise JBSNotConverged(msg, rec)
    return out


def _fmt_hist(vals):
    return "[" + ", ".join("n/a" if v is None else f"{float(v):.2e}"
                           for v in vals) + "]"


def flag_reason(record: dict) -> str:
    """The closure_limited reason a ``"flag"``-mode caller records."""
    return (JBS_FLAG_PREFIX + "did not converge ("
            + str(record.get("stop_reason")) + "; final r_j="
            + _fmt_one((record.get("final") or {}).get("r_j")) + ", r_I="
            + _fmt_one((record.get("final") or {}).get("r_I")) + ")")


def _fmt_one(v):
    return "n/a" if v is None else f"{float(v):.2e}"


def check_delivered(J, jbs_used, w, x, Ip, settings: dict) -> dict:
    """One residual check of an equilibrium that was moved after the loop
    converged (the draw's post-perturb homotopy): does the bootstrap it carries
    still match its own Redl bootstrap to the loop's tolerances?"""
    res = profile_residuals(J, jbs_used, w, x, Ip)
    res["ok"] = bool(np.isfinite(res["r_j"])
                     and res["r_j"] <= settings["rtol_j"]
                     and np.isfinite(res["r_I"])
                     and res["r_I"] <= settings["rtol_Ip"])
    return res


def jsonable(obj):
    """Deep-convert a loop record to JSON/HDF5-attr-safe builtins."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
