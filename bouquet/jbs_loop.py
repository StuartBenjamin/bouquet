"""Self-consistent bootstrap current: the outer fixed-point loop.

With ``GenerationConfig.jbs_self_consistent=True`` the bootstrap current is no
longer computed once and frozen.  Every path that builds a current profile
containing a bootstrap runs the same relaxed outer iteration::

    E_0      = anchor equilibrium (already solved by the caller)
    jBS_0    = evaluate_jBS(E_0)            (or the legacy SWB result)
    for k = 0 .. K-1:
        jc_k    = closure(E_k, jBS_k)       closure on E_k's geometry
        js_k    = (1 - beta) js_k-1 + beta jc_k     (k >= 1; js_0 = jc_0)
        E_k+1   = GS solve of js_k          (step(jBS_k) does both)
        J       = evaluate(E_k+1)           Redl on the NEW equilibrium
        r_k     = residuals(J, jBS_k, E_k+1, E_k)
        jBS_k+1 = (1 - omega) jBS_k + omega J
        converged when every active criterion holds on two consecutive passes

Two relaxations, both of the PATH only (at the fixed point ``js = jc`` and
``jBS = J``, so neither moves it):

* ``omega`` (``jbs_relax``) on the bootstrap.  Held fixed; it is halved (floor
  :data:`JBS_RELAX_FLOOR`) only on SUSTAINED growth of ``r_j`` -- growth on
  ``jbs_relax_halve_on`` consecutive passes -- because a single growth event
  is the forced response of the oscillating geometry mode below, not
  divergence.
* ``beta`` (``jbs_relax_current``) on the SOLVED current: the equilibrium of
  pass k is solved with a blend of the previous pass's solved current and the
  closure's new one.  Each pass closes on the PREVIOUS equilibrium's
  geometry, so the closure's current and the geometry it produces form an
  oscillating two-state mode (l_i swings back by a fraction g < 0 per pass)
  that ``omega`` does not act on; ``beta ~ 1/(1 - g)`` damps it.  A step opts
  in by accepting a ``relax`` keyword (:class:`CurrentRelaxer`); the record
  carries ``beta``, the per-pass gap ``||js - jc|| / ||jc||`` and, next to
  it, the UNRELAXED closure-half residual ``||jc_k - js_k-1|| / ||jc_k||``
  (``= gap / (1 - beta)`` on a blended pass; the gap understates it by
  ``1 - beta``).  Both are recorded only, never gated.

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
passes at the relaxation floor aborts early with the same error, and so does a
pass that can never count (a gated ``l_i``/``q0`` the step did not return, or
an identically zero ``J`` against a non-zero iterate) -- at that pass, not at
the ceiling.  A non-finite initial guess or evaluated ``J`` raises
:class:`JBSNonFinite` at once, whatever the policy, before it can reach the
next solve.

Nothing here touches an existing solver tolerance: the GS solver's own
``nl_tol``/``maxits``, the closure tolerances and the correctors' acceptance
bands are unchanged; the numbers below define what "j_BS converged" means.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

#: Floor of the under-relaxation factor (``jbs_relax`` is halved toward it
#: on sustained growth of ``r_j``, see ``jbs_relax_halve_on``).
JBS_RELAX_FLOOR = 0.25
#: Consecutive passes that must meet every active criterion.
JBS_REQUIRED_CONSECUTIVE = 2
#: Growing-``r_j`` passes AT the relaxation floor that abort the loop.
JBS_GROWTH_ABORT_PASSES = 3
#: Default of ``GenerationConfig.jbs_max_passes_post_homotopy``: the passes
#: a draw may take at the tight coil stage after the post-perturb homotopy
#: (a ceiling, not a tolerance: the two-consecutive-pass rule applies there
#: too, so a post-homotopy stage whose first pass misses needs at least 3).
JBS_POST_HOMOTOPY_PASSES = 4
#: (The MSE chord stage and the geqdsk post-corrective stage are passes of
#: the baseline loop and take ``jbs_max_passes`` as their ceiling.)
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


class JBSNonFinite(JBSNotConverged):
    """The loop was handed, or evaluated, a non-finite bootstrap.

    Raised at once -- before the value can be blended into the next iterate
    and handed to a GS solve -- and REGARDLESS of the ``"flag"`` policy: a
    non-finite iterate is not a result that can be flagged and kept.
    ``pass_number`` is 0 for the initial guess, ``index`` / ``psi_N`` locate
    the first non-finite node (``psi_N`` is ``None`` when the grid is not
    known yet).  A subclass of :class:`JBSNotConverged`, so every caller that
    treats a failed loop as a failed slice or draw does so here too.
    """

    def __init__(self, message, record=None, *, pass_number=None,
                 index=None, psi_N=None):
        super().__init__(message, record)
        self.pass_number = pass_number
        self.index = index
        self.psi_N = psi_N


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
    for name, default in (("jbs_max_passes", 8), ("jbs_max_passes_draw", 12),
                          ("jbs_max_passes_post_homotopy",
                           JBS_POST_HOMOTOPY_PASSES)):
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
    b = _get("jbs_relax_current", 0.7)
    try:
        fb = float(b)
    except (TypeError, ValueError):
        raise ValueError(f"generation.jbs_relax_current must be a number in "
                         f"(0, 1], got {b!r}") from None
    if isinstance(b, bool) or not (np.isfinite(fb) and 0.0 < fb <= 1.0):
        raise ValueError(f"generation.jbs_relax_current must lie in (0, 1] "
                         f"(1 = no relaxation of the solved current), got "
                         f"{b!r}")
    h = _get("jbs_relax_halve_on", 3)
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) \
            or int(h) < 1:
        raise ValueError(f"generation.jbs_relax_halve_on must be an integer "
                         f">= 1 (consecutive growing passes that halve "
                         f"omega; 1 = halve on every growth), got {h!r}")


def jbs_settings(gc, *, draw: bool = False) -> dict:
    """The loop settings of a :class:`GenerationConfig`, validated.

    ``draw=True`` selects ``jbs_max_passes_draw`` as the pass ceiling;
    ``post_homotopy_passes`` is ``jbs_max_passes_post_homotopy`` either way.
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
        max_passes=int(getattr(gc, "jbs_max_passes_draw", 12) if draw
                       else getattr(gc, "jbs_max_passes", 8)),
        relax=float(getattr(gc, "jbs_relax", 0.7)),
        relax_current=float(getattr(gc, "jbs_relax_current", 0.7)),
        relax_halve_on=int(getattr(gc, "jbs_relax_halve_on", 3)),
        on_fail=str(getattr(gc, "jbs_loop_on_fail", "raise")),
        relax_floor=float(JBS_RELAX_FLOOR),
        required_consecutive=int(JBS_REQUIRED_CONSECUTIVE),
        growth_abort_passes=int(JBS_GROWTH_ABORT_PASSES),
        post_homotopy_passes=int(getattr(gc, "jbs_max_passes_post_homotopy",
                                         JBS_POST_HOMOTOPY_PASSES)),
    )


def tolerances_record(settings: dict) -> dict:
    """The tolerance block every loop record carries."""
    return dict(rtol_j=settings["rtol_j"], rtol_Ip=settings["rtol_Ip"],
                tol_li=settings["tol_li"], tol_q0=settings["tol_q0"],
                max_passes=settings["max_passes"],
                required_consecutive=settings["required_consecutive"],
                relax_start=settings["relax"],
                relax_floor=settings["relax_floor"],
                relax_halve_on=int(settings.get("relax_halve_on", 1)),
                relax_current=float(settings.get("relax_current", 1.0)),
                growth_abort_passes=settings["growth_abort_passes"],
                post_homotopy_passes=int(settings.get(
                    "post_homotopy_passes", JBS_POST_HOMOTOPY_PASSES)))


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
#  relaxation of the solved current
# ---------------------------------------------------------------------------
class CurrentRelaxer:
    """``beta``-relaxation of the current a loop pass SOLVES.

    A step that accepts it (``step(jbs, k, relax=...)``) passes the current
    its closure assembled -- the ``jphi-linterp`` input it would otherwise
    hand the solver -- through ``relax(j_closure)`` and solves what comes
    back::

        js_0 = jc_0,    js_k = (1 - beta) js_{k-1} + beta jc_k   (k >= 1)

    ``js_{k-1}`` is the current the PREVIOUS pass solved (committed by the
    kernel after each pass), so calling ``relax`` more than once inside one
    pass (a step that re-solves) blends against the same previous current.
    ``beta = 1`` returns ``j_closure`` unchanged.  At the fixed point
    ``jc_k = js_{k-1}`` and the blend is the identity: the path changes, the
    fixed point does not.
    """

    def __init__(self, beta: float):
        self.beta = float(beta)
        self._prev = None            # js of the previous (committed) pass
        self._last = None            # (jc, js) of the current pass
        self.n_calls = 0

    def __call__(self, j_closure):
        jc = np.asarray(j_closure, dtype=float)
        if (self.beta >= 1.0 or self._prev is None
                or self._prev.shape != jc.shape):
            js = jc.copy()
        else:
            js = (1.0 - self.beta) * self._prev + self.beta * jc
        self._last = (jc.copy(), js.copy())
        self.n_calls += 1
        return js

    def begin_pass(self):
        self._last = None

    def commit(self):
        """End of a pass: the current it solved becomes the blend base."""
        if self._last is not None:
            self._prev = self._last[1].copy()

    def unrelaxed_residual(self, w=None, x=None):
        """``||jc_k - js_{k-1}|| / ||jc_k||`` for the pass just taken, or
        ``None`` on a pass without a previous solved current.

        The UNRELAXED residual of the closure half of the fixed point: the
        distance between the current the closure asks for on this pass's
        geometry and the current the previous pass solved.  For a blended
        pass it equals ``gap / (1 - beta)`` exactly (the recorded ``gap`` is
        scaled down by ``1 - beta``); unlike that quotient it is also defined
        at ``beta = 1``.  RECORD ONLY -- never gated.  Current-weighted with
        the pass's ``w``/``x`` when the shapes match, plain L2 otherwise.
        """
        if self._last is None or self._prev is None:
            return None
        jc = self._last[0]
        if self._prev.shape != jc.shape:
            return None
        d = jc - self._prev
        if (w is not None and x is not None
                and np.shape(w) == jc.shape == np.shape(x)):
            n = weighted_norm(jc, w, x)
            dn = weighted_norm(d, w, x)
        else:
            n = float(np.linalg.norm(jc))
            dn = float(np.linalg.norm(d))
        return (dn / n) if n > 0.0 else (0.0 if dn == 0.0 else float("inf"))

    def gap(self, w=None, x=None):
        """``(rel, blended)`` for the pass just taken: ``||js - jc|| / ||jc||``
        (current-weighted with the pass's ``w``/``x`` when the shapes match,
        plain L2 otherwise) and whether a blend was applied at all."""
        if self._last is None:
            return None, False
        jc, js = self._last
        d = js - jc
        blended = bool(np.any(d != 0.0))
        if (w is not None and x is not None
                and np.shape(w) == jc.shape == np.shape(x)):
            n = weighted_norm(jc, w, x)
            return ((weighted_norm(d, w, x) / n) if n > 0.0 else
                    (0.0 if not blended else float("inf"))), blended
        n = float(np.linalg.norm(jc))
        return ((float(np.linalg.norm(d)) / n) if n > 0.0 else
                (0.0 if not blended else float("inf"))), blended


def _step_takes_relax(step) -> bool:
    import inspect
    try:
        return "relax" in inspect.signature(step).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
#  the kernel
# ---------------------------------------------------------------------------
def run_jbs_loop(jbs0, step: Callable, evaluate: Callable, settings: dict, *,
                 Ip: float, meas0: Optional[dict] = None,
                 gate_li: bool = True, gate_q0: bool = False,
                 label: str = "", init: Optional[str] = None,
                 init_source: Optional[str] = None,
                 grid: str = "psi_N native",
                 on_pass: Optional[Callable] = None,
                 max_passes: Optional[int] = None,
                 raise_on_fail: Optional[bool] = None) -> dict:
    """Iterate closure <-> GS <-> Redl to the fixed point.

    Parameters
    ----------
    jbs0 : array
        Initial bootstrap profile (the one the FIRST solve is assembled with).
    step : callable ``step(jbs, k) -> meas`` or ``step(jbs, k, relax=...)``
        Closure + assembly + GS solve with bootstrap ``jbs``; returns the
        measurement dict of the NEW equilibrium: ``w`` and ``x`` (the Ip weights
        and the grid ``jbs`` lives on), optionally ``li`` and ``q0``.  Any other
        keys are carried through untouched (``meas_final``).  A step that
        accepts a ``relax`` keyword receives the pass's
        :class:`CurrentRelaxer` and solves ``relax(j_closure)`` instead of
        ``j_closure`` (the ``jbs_relax_current`` relaxation of the solved
        current); a two-argument step is called as before and the record says
        the current was not relaxed.
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
    init : str or None
        The ``jbs_init`` setting recorded as ``record["init"]`` (default: the
        settings' own).
    init_source : str or None
        What ``jbs0`` actually is, in words, recorded as
        ``record["init_source"]`` (e.g. a draw's "evaluate_jBS on the draw's
        anchor equilibrium with the draw's own perturbed kinetics").  Record
        only: the fixed point does not depend on the initial iterate.
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
    halve_on = int(settings.get("relax_halve_on", 1))
    beta = float(settings.get("relax_current", 1.0))
    takes_relax = _step_takes_relax(step)
    relaxer = CurrentRelaxer(beta) if takes_relax else None
    if raise_on_fail is None:
        raise_on_fail = (settings.get("on_fail", "raise") == "raise")

    rec = dict(
        enabled=True, label=str(label),
        init=str(init if init is not None else settings.get("init", "anchor")),
        init_source=(None if init_source is None else str(init_source)),
        grid=str(grid),
        tolerances=tolerances_record(dict(settings, max_passes=K)),
        criteria=dict(r_j=True, r_I=True, dl_i=bool(gate_li),
                      dq0=bool(gate_q0)),
        n_passes=0, converged=False, stop_reason=None,
        omega=[], r_j=[], r_I=[], dl_i=[], dq0=[], I_BS=[], I_BS_used=[],
        jBS_peak=[], jBS_peak_psiN=[], li=[], q0=[], pass_ok=[],
        wall_s=None,
        relax_current=(beta if takes_relax else None),
        current_relaxation=(
            ("solved current relaxed: js_k = (1-beta) js_k-1 + beta jc_k "
             "(k >= 1); path only, the fixed point is unchanged")
            if takes_relax else
            "not applied: this caller's step does not take the relaxer"),
        current_gap=[], current_blended=[],
        # RECORD ONLY, never gated: the unrelaxed residual of the closure
        # half, ||jc_k - js_{k-1}|| / ||jc_k|| (= current_gap / (1 - beta)
        # on a blended pass).  current_gap understates it by (1 - beta).
        current_residual_unrelaxed=[],
        current_residual_unrelaxed_definition=(
            "||jc_k - js_{k-1}||_w / ||jc_k||_w = current_gap / (1 - beta) "
            "on a blended pass: the unrelaxed closure-half residual; "
            "recorded only, NOT a convergence criterion"),
        relax_halve_on=int(halve_on), omega_halved_at_pass=[],
    )
    try:
        from .physics import EVALUATE_JBS_VERSION
        rec["evaluate_jBS_version"] = EVALUATE_JBS_VERSION
    except Exception:
        rec["evaluate_jBS_version"] = None
    rec["oft_build"] = oft_build_info()

    def _nonfinite(k, what, arr, grid):
        """Raise :class:`JBSNonFinite` at the first non-finite node of
        *arr* (pass ``k + 1``; ``k = -1`` is the initial guess)."""
        bad = ~np.isfinite(arr)
        i = int(np.argmax(bad))
        psi = None
        if grid is not None and np.shape(grid) == np.shape(arr):
            psi = float(np.asarray(grid, dtype=float)[i])
        where = (f"index {i}" + ("" if psi is None else f", psi_N={psi:.6g}")
                 + f"; {int(bad.sum())} of {bad.size} node(s)")
        when = "before pass 1" if k < 0 else f"on pass {k + 1}/{K}"
        msg = (f"self-consistent j_BS loop"
               f"{(' [' + label + ']') if label else ''}: {what} is "
               f"non-finite {when} ({where}) -- refusing to blend it into "
               "the next iterate or hand it to a GS solve")
        rec["stop_reason"] = msg
        rec["wall_s"] = float(time.perf_counter() - t0)
        rec["jbs_converged"] = False
        rec["fail_message"] = msg
        print("  [jbs-loop] " + msg, flush=True)
        raise JBSNonFinite(msg, rec, pass_number=max(k + 1, 0), index=i,
                           psi_N=psi)

    jbs = np.asarray(jbs0, dtype=float).copy()
    if not np.all(np.isfinite(jbs)):
        _nonfinite(-1, "the initial bootstrap guess jbs0", jbs,
                   (meas0 or {}).get("x"))
    prev = dict(meas0 or {})
    streak = 0
    growth_at_floor = 0
    grow_streak = 0                   # consecutive passes on which r_j grew
    omega_used_for_current = None     # omega that produced `jbs` (None: init)
    r_j_prev = None
    meas = None
    J = None
    jbs_used = jbs

    for k in range(K):
        jbs_used = jbs
        if relaxer is not None:
            relaxer.begin_pass()
            meas = step(jbs, k, relax=relaxer)
        else:
            meas = step(jbs, k)
        J = np.asarray(evaluate(meas), dtype=float)
        if J.shape != jbs.shape:
            raise ValueError(f"run_jbs_loop[{label}]: evaluate returned shape "
                             f"{J.shape}, the iterate has {jbs.shape}")
        if not np.all(np.isfinite(J)):
            rec["n_passes"] = k + 1
            _nonfinite(k, "the evaluated bootstrap J", J, meas.get("x"))
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
        if relaxer is not None:
            _gap, _bl = relaxer.gap(meas.get("w"), meas.get("x"))
            _unrel = relaxer.unrelaxed_residual(meas.get("w"), meas.get("x"))
            relaxer.commit()
        else:
            _gap, _bl, _unrel = None, False, None
        rec["current_gap"].append(_gap)
        rec["current_blended"].append(bool(_bl))
        rec["current_residual_unrelaxed"].append(_unrel)
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
              + ("" if not _bl else f" |js-jc|/|jc|={_gap:.2e}")
              + ("" if (not _bl or _unrel is None)
                 else f" (unrelaxed {_unrel:.2e}, record only)")
              + (" ok" if ok else ""), flush=True)
        # ---- a pass that can never count: stop now, not at the ceiling ----
        _never = None
        if gate_li and li_new is None:
            _never = ("the step returned no finite l_i although l_i is a "
                      "convergence criterion here (gate_li=True)")
        elif gate_q0 and q0_new is None:
            _never = ("the step returned no finite q0 although q0 is a "
                      "convergence criterion here (gate_q0=True)")
        elif (not np.any(J != 0.0)) and np.any(jbs != 0.0):
            _never = ("the evaluated bootstrap J is identically zero while "
                      "the iterate is not (r_j is infinite)")
        if _never is not None:
            rec["stop_reason"] = (f"pass {k + 1}/{K}: {_never} -- "
                                  "convergence is impossible, stopped at once "
                                  "instead of running to the pass ceiling")
            break
        streak = streak + 1 if ok else 0
        if streak >= need:
            rec["converged"] = True
            rec["stop_reason"] = (f"all active criteria met on {need} "
                                  "consecutive passes")
            if on_pass is not None:
                on_pass(k, meas, J, dict(entry, omega_next=None))
            break
        # ---- relaxation schedule: halve on SUSTAINED growth (halve_on
        # consecutive growing passes; 1 = every growth, the earlier
        # schedule), abort after n_abort growing passes at the floor --------
        if r_j_prev is not None and res["r_j"] > r_j_prev:
            if (omega_used_for_current is not None
                    and omega_used_for_current <= floor + 1e-15):
                growth_at_floor += 1
            else:
                growth_at_floor = 0
            grow_streak += 1
            if grow_streak >= halve_on:
                _om = max(0.5 * omega, floor)
                if _om < omega:
                    rec["omega_halved_at_pass"].append(k + 1)
                omega = _om
                grow_streak = 0
        else:
            growth_at_floor = 0
            grow_streak = 0
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
        I_BS=rec["I_BS"][-1] if rec["I_BS"] else None,
        current_gap=rec["current_gap"][-1] if rec["current_gap"] else None,
        current_residual_unrelaxed=(rec["current_residual_unrelaxed"][-1]
                                    if rec["current_residual_unrelaxed"]
                                    else None))
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
