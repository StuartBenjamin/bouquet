"""Draws on the unified reconstruction engine (``reconstruction_engine=
"unified"``; docs/engine.md, "Draws").

A draw starts from the reconstruction's DELIVERED state
(:class:`bouquet.engine.EngineState`): its geometry snapshot ``G*``, the
closure coefficients ``x*``, the relaxed bootstrap ``lambda_BS*`` and (when
on) the delivery correction ``Delta*``.  It

* perturbs the kinetics exactly as today's sampler does (the pressure-matched
  kinetic channels, the auxiliary channels) and the PARALLEL inductive
  component ``lambda_ind`` with the existing Gaussian-process sample (the
  toroidal sigma ``sigma_jphi`` and length scale ``j_ls``), and takes the
  bootstrap scale the run drew for it;
* composes the current with ``x*`` HELD::

      J = s_ind(x*) F<1/R>/<B^2> lambda_ind + s_bs(x*) F<1/R>/<B^2> lambda_BS
          + F<1/R>/<B^2> lambda_fix + p'(<R> - F^2<1/R>/<B^2>)

  on the latest solved geometry, the pressure term from the DRAW's own
  ``p'`` every pass, and closes ONE row: Ip, as a scalar amplitude on the
  inductive term in the exact (``jphi-linterp``) measure -- route R2's
  logic, zero extra solves.  With ``engine_draw_q0_row=True`` the
  reconstruction's q0 row is kept too, acting through
  :class:`bouquet.jbs_loop.AxisRowPin` on a second scalar (the bootstrap
  amplitude): the sawtooth two-scalar closure in increment form;
* runs ONE bootstrap loop (:func:`bouquet.jbs_loop.run_jbs_loop`, the draw
  ceiling ``jbs_max_passes_draw``, the current gate as a standing
  criterion, :class:`~bouquet.jbs_loop.JBSNonFinite` at once), then -- in
  ``generate_bouquet`` -- the optional coil homotopy and the existing
  post-homotopy check (:func:`post_homotopy`, dispatched through
  :func:`bouquet.TokaMaker_interface._post_homotopy_jbs`) with its
  saturation guard;
* records l_i(3), l_i(1), beta_N, q0 at its labelled radius, q95, the Ip
  amplitude, the loop record, the delivery check, the l_i ATTRIBUTION and
  the cost by stage; the post-hoc filters (the l_i band ``l_i_tolerance``
  around the reconstruction's l_i, ``constrain_sawteeth``) are applied to
  the archived draw -- a draw outside a band is ARCHIVED with
  ``in_spec=False``, never dropped.

**Zero-perturbation identity by construction.**  Every perturbed quantity is
formed as ``base + (drawn - base)`` (kinetics, pressure, inductive, aux) so
a zero perturbation is exactly the base; the first pass composes on ``G*``
with ``x*`` and ``lambda_BS*`` (the pressure term's ``p'`` shifted by the
draw's pressure change, zero at identity) and the Ip amplitude's increment
is formed from differences that vanish exactly -- so the first request IS
the stored request, bit for bit (:class:`EngineDrawContext` refuses to run
otherwise).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Version stamp of the engine draw (recorded with every draw).
ENGINE_DRAW_VERSION = "unified-engine-draw/1"

#: The random stream of an engine draw, as recorded.
RNG_STREAM = (
    "one Generator per run (sampling.make_rng(seed)); per draw, in order: "
    "[the bootstrap scale, from the block generate_bouquet draws once before "
    "the loop]; the kinetic channels ne, Te, (Zeff -> ni | ni), Ti, redrawn "
    "together until the flux-surface integral of the thermal pressure "
    "matches the baseline's within p_thresh (perturb_kinetic_equilibrium's "
    "rejection loop, same calls, same order); the auxiliary channels in "
    "their dict order; ONE inductive candidate, redrawn only while it is "
    "negative where its mean exceeds its sigma (up to max_proxy_draws).  "
    "Identical to the legacy draw stream through the first inductive "
    "candidate; the legacy routes then consume further candidates (Fix C "
    "band resampling, the standard route's l_i pre-screen and band loop), "
    "the engine draw none -- so from the first legacy draw that resampled, "
    "the two runs' draws start at different stream offsets.")

#: What an engine draw's loop restarts from after its first solve
#: (``engine_draw_bootstrap_refresh=True``).
REFRESH_SOURCE = (
    "scale x [lambda_BS* + Redl(the draw's kinetics) on the FIRST SOLVED "
    "equilibrium (the draw's pressure) - Redl(the reconstruction's kinetics) "
    "on the starting equilibrium]: the anchor with its draw-kinetics Redl "
    "moved from G* to the geometry the first solve produced; pass 1's own "
    "Redl, so 0 extra solves and 0 extra Redl evaluations")

#: What an engine draw's post-homotopy loop starts from.
INIT_POST_HOMOTOPY = (
    "relaxed: (1 - omega) x the bootstrap the draw carries + omega x Redl on "
    "the delivered post-homotopy equilibrium (the draw's own kinetics, times "
    "its bootstrap scale)")


class EngineDrawRefused(ValueError):
    """An engine draw cannot be run honestly (the stored state does not
    reproduce the stored request, an input the engine draw cannot use, no
    physical inductive candidate).  Raised with the reason."""


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def draw_loop_settings(gc) -> dict:
    """The loop settings of an engine draw: :func:`bouquet.jbs_loop.
    jbs_settings` with ``draw=True`` (the ceiling ``jbs_max_passes_draw``,
    ``post_homotopy_passes``) and the closure-half current gate ON (decision
    9: a standing criterion, as in the engine's reconstruction)."""
    from .jbs_loop import jbs_settings
    s = dict(jbs_settings(gc, draw=True))
    s["gate_current_residual"] = True
    return s


# ---------------------------------------------------------------------------
#  the inputs of one draw
# ---------------------------------------------------------------------------
@dataclass
class EngineDrawInputs:
    """What one draw perturbs.  ``kinetics`` / ``pressure`` /
    ``pressure_thermal`` / ``jB_ind`` on the engine grid ``psi_N``;
    ``kinetics_native`` (``ne, te, ni, ti``) on the kinetic grid (archived);
    ``scale`` the bootstrap scale (1.0 = the reconstruction's)."""

    kinetics: dict
    kinetics_native: dict
    pressure: np.ndarray
    pressure_thermal: np.ndarray
    jB_ind: np.ndarray
    scale: float = 1.0
    aux: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)


def _pchip(x, y, xn):
    from .utils import pchip_interp
    return pchip_interp(np.asarray(x, float), np.asarray(y, float),
                        np.asarray(xn, float))


# ---------------------------------------------------------------------------
#  the reconstruction a draw inherits
# ---------------------------------------------------------------------------
class EngineDrawContext:
    """Everything a draw inherits from ONE engine reconstruction (built once
    per run from the live engine and its result).

    *native* is the kinetic-grid base the sampler perturbs:
    ``dict(psi_N, ne, te, ni, ti)`` (+ optional ``z_fast``), exactly the
    arrays ``generate()`` hands the legacy sampler (the Baseline's).  On
    construction the reconstruction's composition is rebuilt from the state
    and must reproduce the stored request BIT FOR BIT, else
    :class:`EngineDrawRefused`."""

    def __init__(self, eng, res, *, loop, native, q0_row=False,
                 label="engine draw", bootstrap_refresh=False):
        from .engine import _lin, complete_geometry  # noqa: F401
        from .utils import (closure_sign_convention, pchip_derivative,
                            structured_basis_eval)
        st = res["state"]
        self.eng = eng
        self.c = eng.c
        self.psi = np.asarray(eng.psi, dtype=float)
        self.label = str(label)
        self.loop = dict(loop)
        #: engine_draw_bootstrap_refresh: restart the loop's bootstrap after
        #: its first solve (:func:`run_draw`)
        self.bootstrap_refresh = bool(bootstrap_refresh)
        self.reconstruction_converged = bool(res.get("converged", False))
        self.Phi = structured_basis_eval(eng.basis, self.psi)
        K = self.Phi.shape[0]
        self.x = np.asarray(st.x, dtype=float).copy()
        self.s_ind = 1.0 + self.x[:K] @ self.Phi
        self.s_bs = 1.0 + self.x[K:] @ self.Phi
        self.lam = np.asarray(st.lambda_bs, dtype=float).copy()
        self.geom = st.geom
        self.request = np.asarray(st.request, dtype=float).copy()
        self.delta = (None if st.delivery_correction is None else
                      np.asarray(st.delivery_correction, dtype=float).copy())
        self.delivery_correction = self.delta is not None
        # ---- the identity: the stored state composes the stored request
        J, parts = self.compose(self.geom, self.c.jB_ind, self.lam)
        R = J if self.delta is None else J + self.delta
        if not np.array_equal(R, self.request):
            d = float(np.max(np.abs(R - self.request)))
            raise EngineDrawRefused(
                f"{self.label}: composing on the stored geometry with x* and "
                f"lambda_BS* does not reproduce the stored request (max "
                f"|diff| {d:.3e}); the zero-perturbation identity would not "
                "hold -- refusing to draw from this state")
        self.J_star = J
        self.parts_star = parts
        g = self.geom
        lin = [_lin(g, parts[k]) for k in ("ind", "bs", "fix")]
        _sgn, _ips, c_s = closure_sign_convention(*lin, g["c_affine"],
                                                  float(self.c.Ip))
        #: the Ip the reconstruction's delivered composition carries in the
        #: exact measure on G*: its linear part and its affine part
        self.lin_star = _lin(g, J)
        self.c_star = float(c_s)
        self.Ip_star = float(self.lin_star + self.c_star)
        # ---- the pressure term's p' on G* for a draw's pressure (pass 1)
        psi_q = np.asarray(g["psi_q"], dtype=float)
        self._psi_q = psi_q
        self.dPq_star = np.interp(psi_q, self.psi, pchip_derivative(
            self.psi, np.asarray(self.c.pressure, dtype=float)))
        pp = np.asarray(g["pprime"], dtype=float)
        nn = float(self.dPq_star @ self.dPq_star)
        self.sigma_p = float(pp @ self.dPq_star) / nn if nn > 0.0 else 0.0
        if not np.isfinite(self.sigma_p):
            self.sigma_p = 0.0
        # ---- the reconstruction's delivered measurements (the reference)
        m = eng.delivered_meas
        stats = m.get("stats") or {}
        from .physics import SOLVER_Q0_PSI_N
        self.ref = dict(
            l_i=float(m["li"]), l_i_1=_f(m.get("li_1")),
            q_row=float(m["q_row"]), q_row_psi_N=float(psi_q[0]),
            q0_stats=_f(stats.get("q_0")),
            q0_stats_psi_N=float(SOLVER_Q0_PSI_N),
            q95=_f(stats.get("q_95")), beta_n=_f(stats.get("beta_n")),
            Ip=_f(m.get("Ip")))
        # ---- the kinetic-grid base of the sampler
        nat = dict(native)
        self.native = {k: np.asarray(v, dtype=float)
                       for k, v in nat.items() if v is not None}
        self.psi_kin = self.native["psi_N"]
        self._kin_eq_native = {k: _pchip(self.psi_kin, self.native[k],
                                         self.psi)
                               for k in ("ne", "te", "ni", "ti")}
        self.z_fast_eq = (None if self.native.get("z_fast") is None else
                          _pchip(self.psi_kin, self.native["z_fast"],
                                 self.psi))
        self.pressure_thermal_base = np.asarray(
            self.c.pressure_parts["thermal"], dtype=float)
        # ---- the q0 row (engine_draw_q0_row)
        self.q0_row = bool(q0_row)
        if self.q0_row:
            if "q0" not in eng.rows or st.q0_target is None:
                raise EngineDrawRefused(
                    f"{self.label}: engine_draw_q0_row=True keeps the "
                    "reconstruction's q0 row, but the reconstruction had no "
                    "active q0 row (not requested, or the sawtooth gate "
                    "rejected it)")
            self.q0_target = float(st.q0_target)
            self.row0 = float(np.interp(float(psi_q[0]), self.psi, J))
        self._li_grad = None

    # ---- composition -----------------------------------------------------
    def compose(self, geom, jB_ind, jB_bs):
        """``(J, parts)``: the composition with ``x*`` held -- the SAME
        operations, in the same order, as the reconstruction's closure
        assembles its current (``s_ind*ind + s_bs*bs + (fix + pressure)``)."""
        from .engine import compose
        _, p = compose(geom, jB_ind, jB_bs, self.c.jB_fix)
        j_ind = self.s_ind * p["ind"]
        j_bs = self.s_bs * p["bs"]
        j_fix = p["fix"] + p["pressure"]
        J = j_ind + j_bs + j_fix
        return J, dict(ind=j_ind, bs=j_bs, fix=j_fix, driven=p["fix"],
                       pressure=p["pressure"], kappa=p["kappa"])

    def geom_for_pressure(self, pressure):
        """``G*`` with its ``p'`` shifted by the draw's pressure change (the
        first pass composes on it): ``p'* + sigma_p (dP_draw - dP*)`` on the
        row grid, ``sigma_p`` the least-squares factor between the solver's
        ``p'`` on ``G*`` and ``d p / d psi_N``.  Exactly ``G*`` at zero
        perturbation."""
        from .utils import pchip_derivative
        dPq = np.interp(self._psi_q, self.psi, pchip_derivative(
            self.psi, np.asarray(pressure, dtype=float)))
        g = dict(self.geom)
        g["pprime"] = np.asarray(self.geom["pprime"], dtype=float) \
            + self.sigma_p * (dPq - self.dPq_star)
        return g

    # ---- inputs -----------------------------------------------------------
    def eq_kinetics(self, native_draw, zeff_eq_delta=None):
        """The engine-grid kinetics of a draw: ``base + (drawn - base)``,
        each on the grid the adapter used (exactly the base at zero
        perturbation)."""
        out = {}
        for k in ("ne", "te", "ni", "ti"):
            out[k] = np.asarray(self.c.kinetics[k], dtype=float) + (
                _pchip(self.psi_kin, native_draw[k], self.psi)
                - self._kin_eq_native[k])
        z = np.asarray(self.c.kinetics["zeff"], dtype=float)
        out["zeff"] = z if zeff_eq_delta is None else z + zeff_eq_delta
        return out

    def _thermal(self, k):
        from .physics import ELEMENTARY_CHARGE as EC
        return EC * (k["ne"] * k["te"] + k["ni"] * k["ti"])

    def _impurity(self, k):
        Z = self.c.pressure_parts.get("Z_imp")
        if not Z:
            return None
        from .physics import impurity_pressure
        ne = k["ne"]
        if self.z_fast_eq is not None:
            ne = np.maximum(ne - self.z_fast_eq, 0.0)
        return impurity_pressure(ne, k["ni"], k["ti"], Z)

    def pressures(self, kin_eq):
        """``(total, thermal)`` solve pressure of a draw's kinetics: the
        adapter's own assembly (thermal + impurity + fast) in increment
        form, ``base + (drawn - base)`` per part (exactly the contract's
        pressure at zero perturbation)."""
        k0 = self.c.kinetics
        dth = self._thermal(kin_eq) - self._thermal(k0)
        th = self.pressure_thermal_base + dth
        p = np.asarray(self.c.pressure, dtype=float) + dth
        i1, i0 = self._impurity(kin_eq), self._impurity(k0)
        if i1 is not None:
            p = p + (i1 - i0)
        return p, th

    def zero_inputs(self, scale=1.0):
        """The inputs of a draw with every perturbation zero."""
        nat = {k: self.native[k].copy() for k in ("ne", "te", "ni", "ti")}
        kin = self.eq_kinetics(nat)
        p, th = self.pressures(kin)
        return EngineDrawInputs(
            kinetics=kin, kinetics_native=nat, pressure=p,
            pressure_thermal=th,
            jB_ind=np.asarray(self.c.jB_ind, dtype=float).copy(),
            scale=float(scale), sampler=dict(zero_perturbation=True))

    # ---- the l_i attribution (docs/engine.md, "l_i controllability") ----
    def attribution(self, inputs, jbs_used, amp):
        """The l_i change of a draw split linearly with the closure's own
        l_i gradient (:func:`bouquet.utils.structured_li_model` /
        :func:`~bouquet.utils.structured_li_gradient` along each direction)
        on the reconstruction geometry ``G*``.  Directions (toroidal, on
        ``G*``): inductive shape ``s_ind kappa (lambda_ind' - lambda_ind)``,
        bootstrap ``s_bs kappa (lambda_BS' - lambda_BS*)``, pressure term
        ``P(p'_draw) - P(p'*)``, Ip amplitude ``d_ind s_ind kappa
        lambda_ind'`` (and the q0 row's ``d_bs s_bs kappa lambda_BS'``).
        Returns the per-part linear l_i changes; the caller adds the
        remainder against the delivered l_i."""
        from .engine import li_of_current, pressure_term
        from .utils import structured_li_gradient, structured_li_model
        g = self.geom
        kap = np.asarray(self.parts_star["kappa"], dtype=float)
        lam_d = np.asarray(inputs.jB_ind, dtype=float)
        jb = np.asarray(jbs_used, dtype=float)
        d = dict(
            inductive=self.s_ind * kap * (lam_d - np.asarray(self.c.jB_ind,
                                                             float)),
            bootstrap=self.s_bs * kap * (jb - self.lam),
            pressure=(pressure_term(self.geom_for_pressure(inputs.pressure))
                      - pressure_term(g)),
            amplitude=float(amp.get("d_ind", 0.0)) * self.s_ind * kap * lam_d)
        if self.q0_row:
            d["q0_row"] = float(amp.get("d_bs", 0.0)) * self.s_bs * kap * jb
        names = list(d)
        D = np.vstack([d[n] for n in names])
        lg = g["li_geom"]
        model = structured_li_model(
            self.psi, g["w_lin"], D, np.ones_like(self.psi),
            np.zeros_like(self.psi), self.J_star - 1.0, lg, self.Ip_star,
            li_kind="li_3")
        grad = structured_li_gradient(model)[:len(names)]
        li0 = li_of_current(self.J_star, g, "li_3")
        li_full = li_of_current(self.J_star + D.sum(axis=0), g, "li_3")
        return dict(
            parts={n: float(v) for n, v in zip(names, grad)},
            linear_total=float(np.sum(grad)),
            model_on_reconstruction_geometry=float(li_full - li0),
            gradient=("bouquet.utils.structured_li_gradient of "
                      "structured_li_model along each direction, on the "
                      "reconstruction geometry G* (li_3)"),
            directions=("toroidal currents on G*: inductive s_ind kappa "
                        "(lambda_ind' - lambda_ind); bootstrap s_bs kappa "
                        "(lambda_BS' - lambda_BS*); pressure P(p'_draw) - "
                        "P(p'*); amplitude d_ind s_ind kappa lambda_ind'"
                        + ("; q0_row d_bs s_bs kappa lambda_BS'"
                           if self.q0_row else "")))


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
#  the sampler (perturb_kinetic_equilibrium's, in increment form)
# ---------------------------------------------------------------------------
def sample_draw_inputs(ctx, rng, unc, flux_integral, *, scale=1.0,
                       p_thresh=0.05, max_pressure_iter=None,
                       max_proxy_draws=500):
    """One draw's perturbations, drawn from *rng* exactly as
    :func:`bouquet.TokaMaker_interface.perturb_kinetic_equilibrium` draws
    them (same calls, same arguments, same order; :data:`RNG_STREAM`):
    :func:`sample_kinetics` (the pressure-matched kinetic channels, then the
    auxiliary channels), then :func:`sample_inductive` (one inductive
    candidate).  Returns :class:`EngineDrawInputs`."""
    kin = sample_kinetics(ctx, rng, unc, flux_integral, p_thresh=p_thresh,
                          max_pressure_iter=max_pressure_iter)
    cand, tries, n_floor = sample_inductive(ctx, rng, unc,
                                            max_proxy_draws=max_proxy_draws)
    kin.jB_ind = cand
    kin.scale = float(scale)
    kin.sampler.update(inductive_tries=int(tries),
                       inductive_floor_nodes=int(n_floor))
    return kin


def sample_kinetics(ctx, rng, unc, flux_integral, *, p_thresh=0.05,
                    max_pressure_iter=None):
    """The kinetic half of a draw (``jB_ind`` left at the reconstruction's):

    * ne, Te, (Zeff -> ni via quasineutrality | ni), Ti on the kinetic grid,
      redrawn together until ``mean |<p>_th - <p>_th,draw| / <p>_th`` is
      within *p_thresh* (a FRACTION), ``<.>`` the solver's flux integral;
    * the auxiliary channels (``unc["aux_sigmas"]``) in their dict order.

    Every drawn profile is formed as ``base + (sample - mean) * norm`` so a
    zero sigma is exactly the base.  *unc*: ``sigma_ne, sigma_te,
    sigma_ni, sigma_ti`` (kinetic grid), ``n_ls, t_ls``, optional
    ``aux_sigmas, aux_baselines, aux_length_scales``."""
    from .sampling import _draw_monotonic_perturbation, generate_perturbed_GPR
    from .TokaMaker_interface import _MAX_PRESSURE_ITER
    if max_pressure_iter is None:
        max_pressure_iter = _MAX_PRESSURE_ITER
    psi_kin, psi = ctx.psi_kin, ctx.psi
    nat = ctx.native
    ne, te, ni, ti = (nat[k] for k in ("ne", "te", "ni", "ti"))
    z_fast = nat.get("z_fast")
    s_ne, s_te, s_ni, s_ti = (np.asarray(unc[k], dtype=float) for k in
                              ("sigma_ne", "sigma_te", "sigma_ni",
                               "sigma_ti"))
    n_ls, t_ls, j_ls = unc["n_ls"], unc["t_ls"], unc["j_ls"]
    aux_sigmas = unc.get("aux_sigmas") or None
    aux_baselines = unc.get("aux_baselines") or {}
    aux_ls = unc.get("aux_length_scales") or {}

    def _mono(base, sig, ls):
        mn = base / base[0]
        smp = _draw_monotonic_perturbation(psi_kin, mn, sig / base[0], ls,
                                           rng=rng)
        return base + (smp - mn) * base[0]

    zeff_active = bool(aux_sigmas) and ("zeff" in aux_sigmas) \
        and aux_baselines.get("zeff") is not None
    Z_imp = None
    if zeff_active:
        from .physics import impurity_charge_with_fast_ions
        Z_imp, _ = impurity_charge_with_fast_ions(
            ne, ni, np.asarray(aux_baselines["zeff"], dtype=float),
            np.zeros_like(ne) if z_fast is None else z_fast)

    def _zclip(z, ne_):
        if z_fast is None:
            return np.clip(z, 1.0, Z_imp * (1.0 - 1e-9))
        fth = np.clip((ne_ - np.asarray(z_fast, dtype=float))
                      / np.clip(ne_, 1e10, None), 0.0, 1.0)
        return np.clip(z, np.maximum(fth, 1e-9), Z_imp * fth * (1.0 - 1e-9))

    inp = float(flux_integral(psi, ctx.pressure_thermal_base))
    thr = float(p_thresh) * 100.0
    p_err, n_iter = np.inf, 0
    zeff_draw = None
    while p_err > thr:
        n_iter += 1
        if n_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} "
                f"iterations (last error {p_err:.2f}% vs threshold "
                f"{thr:.2f}%)")
        ne_d = _mono(ne, s_ne, n_ls)
        te_d = _mono(te, s_te, t_ls)
        if zeff_active and Z_imp is not None:
            from .physics import main_ion_density_from_zeff
            zb = np.asarray(aux_baselines["zeff"], dtype=float)
            zs = np.asarray(aux_sigmas["zeff"], dtype=float)
            z0 = float(np.max(np.abs(zb))) or 1.0
            zmn = zb / z0
            smp = np.atleast_1d(np.asarray(np.squeeze(generate_perturbed_GPR(
                psi_kin, zmn, zs / z0, length_scale=aux_ls.get("zeff", 0.4),
                n_samples=1, rng=rng)), dtype=float))
            zeff_draw = _zclip(zb + (smp - zmn) * z0, ne_d)
            ni_d = ni + (main_ion_density_from_zeff(ne_d, zeff_draw, Z_imp,
                                                    z_fast=z_fast)
                         - main_ion_density_from_zeff(ne, _zclip(zb, ne),
                                                      Z_imp, z_fast=z_fast))
        else:
            ni_d = _mono(ni, s_ni, n_ls)
        ti_d = _mono(ti, s_ti, t_ls)
        nat_d = dict(ne=ne_d, te=te_d, ni=ni_d, ti=ti_d)
        kin = ctx.eq_kinetics(nat_d)
        _p, th = ctx.pressures(kin)
        tmp = float(flux_integral(psi, th))
        p_err = float(np.mean(np.abs(inp - tmp) / inp) * 100.0)
    # ---- the auxiliary channels (the switchboard), in dict order
    aux_out = {}
    if aux_sigmas:
        for name, es in aux_sigmas.items():
            if name == "zeff" and zeff_draw is not None:
                continue
            eb = aux_baselines.get(name)
            if eb is None:
                continue
            eb = np.asarray(eb, dtype=float)
            es = np.asarray(es, dtype=float)
            e0 = float(np.max(np.abs(eb)))
            if not (e0 > 0):
                e0 = 1.0
            emn = eb / e0
            smp = np.squeeze(generate_perturbed_GPR(
                psi_kin, emn, es / e0, length_scale=aux_ls.get(name, 0.4),
                n_samples=1, rng=rng))
            aux_out[name] = np.atleast_1d(np.asarray(eb + (smp - emn) * e0,
                                                     dtype=float))
        if zeff_draw is not None:
            aux_out["zeff"] = zeff_draw
    dz = None
    if "zeff" in aux_out and aux_baselines.get("zeff") is not None:
        zb = np.asarray(aux_baselines["zeff"], dtype=float)
        dz = (np.clip(_pchip(psi_kin, aux_out["zeff"], psi), 1.0, None)
              - np.clip(_pchip(psi_kin, zb, psi), 1.0, None))
    kin = ctx.eq_kinetics(nat_d, zeff_eq_delta=dz)
    p, th = ctx.pressures(kin)
    return EngineDrawInputs(
        kinetics=kin, kinetics_native=nat_d, pressure=p, pressure_thermal=th,
        jB_ind=np.asarray(ctx.c.jB_ind, dtype=float).copy(), aux=aux_out,
        sampler=dict(pressure_match_iterations=int(n_iter),
                     pressure_match_err_pct=float(p_err),
                     p_thresh=float(p_thresh),
                     zeff_primary=bool(zeff_draw is not None),
                     rng_stream=RNG_STREAM))


def sample_inductive(ctx, rng, unc, *, max_proxy_draws=500):
    """``(candidate, tries, n_floor_nodes)``: ONE Gaussian-process sample of
    the PARALLEL inductive ``lambda_ind``, drawn IN TOROIDAL UNITS with the
    legacy call -- ``generate_perturbed_GPR(psi, j/j[0], sigma_jphi/j[0],
    j_ls)`` on ``j = s_ind kappa* lambda_ind``, the inductive term of the
    composition on ``G*`` -- and mapped back, ``delta lambda = delta j /
    (s_ind kappa*)``.  The toroidal perturbation is therefore the legacy
    draw's for the same normals (the factorised covariance ``sigma_i sigma_j
    K_ij`` + jitter does not depend on the mean), with today's marginal sigma
    ``sigma_jphi`` and length scale ``j_ls``.  Redrawn only while it is
    negative where its mean exceeds its sigma; where the mean is within
    sigma of zero (the floor zone) a negative excursion is clipped to zero
    (the standard route's half-Gaussian rule; nodes whose mean is itself
    negative are left alone, so a zero sigma is exactly the mean)."""
    from .sampling import generate_perturbed_GPR
    psi = ctx.psi
    lam = np.asarray(ctx.c.jB_ind, dtype=float)
    kap = np.asarray(ctx.parts_star["kappa"], dtype=float)
    fac = ctx.s_ind * kap
    jt = fac * lam
    sig = np.asarray(unc["sigma_jphi"], dtype=float)
    j0 = float(jt[0])
    if not (np.isfinite(j0) and j0 != 0.0):
        raise EngineDrawRefused(
            f"{ctx.label}: the inductive component is zero on axis; the "
            "sampler normalises by it (as the legacy sampler does)")
    mn, sn = jt / j0, sig / j0
    floor = jt <= sig
    for tries in range(1, int(max_proxy_draws) + 1):
        smp = generate_perturbed_GPR(psi, mn, sigma_profile=sn,
                                     length_scale=unc["j_ls"], n_samples=1,
                                     rng=rng, diag_plot=False)
        c = lam + (smp - mn) * j0 / fac
        neg = (c < 0.0) & (lam >= 0.0)
        if np.any(neg & ~floor):
            continue
        return (np.where(neg, 0.0, c), tries,
                int(np.sum(floor & (lam >= 0.0))))
    raise EngineDrawRefused(
        f"{ctx.label}: no inductive candidate without a negative excursion "
        f"where its mean exceeds its sigma in {int(max_proxy_draws)} tries")


# ---------------------------------------------------------------------------
#  one draw's loop
# ---------------------------------------------------------------------------
class _DrawPasses:
    """The kernel's ``step`` for an engine draw: compose with ``x*`` held
    on the latest solved geometry, close the Ip row (and the q0 row) with
    scalar increments, relax, ONE solve, measure."""

    def __init__(self, ctx, backend, inputs, *, geom, pin=None,
                 delta=None, coil_guard=None, phase="loop", on_solve=None):
        self.ctx = ctx
        self.b = backend
        self.inp = inputs
        self.geom = geom
        self.pin = pin
        self.delta = (None if delta is None
                      else np.asarray(delta, dtype=float).copy())
        self.coil_guard = coil_guard
        self.phase = str(phase)
        self.passes = []
        self.last = None
        self.first_request = None
        self.on_solve = on_solve

    def close(self, geom, jbs):
        """``(jc, parts, amp)`` on *geom*: the composition with ``x*`` held
        plus the scalar increments that close the Ip row (and the q0 row) in
        the exact measure -- increments formed from differences that are
        exactly zero at the reconstruction state."""
        from .engine import EngineClosureRefused, _lin
        from .utils import closure_sign_convention
        ctx = self.ctx
        J0, p = ctx.compose(geom, self.inp.jB_ind, jbs)
        li_ = _lin(geom, p["ind"])
        lb_ = _lin(geom, p["bs"])
        lf_ = _lin(geom, p["fix"])
        _s, _i, c_k = closure_sign_convention(li_, lb_, lf_,
                                              geom["c_affine"],
                                              float(ctx.c.Ip))
        dIp = (ctx.lin_star - _lin(geom, J0)) + (ctx.c_star - float(c_k))
        amp = dict(dIp_A=float(dIp))
        if self.pin is None:
            if not (np.isfinite(li_) and li_ != 0.0 and np.isfinite(dIp)):
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip amplitude is undefined (inductive "
                    f"Ip {li_!r}, increment {dIp!r})")
            d_ind = dIp / li_
            jc = J0 + d_ind * p["ind"]
            amp.update(d_ind=float(d_ind), a_ind=float(1.0 + d_ind))
        else:
            psi0 = float(geom["psi_q"][0])
            A = np.array([[li_, lb_],
                          [float(np.interp(psi0, ctx.psi, p["ind"])),
                           float(np.interp(psi0, ctx.psi, p["bs"]))]])
            rhs = np.array([dIp, float(self.pin.row)
                            - float(np.interp(psi0, ctx.psi, J0))])
            try:
                dd = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError as e:
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip + q0 two-scalar closure is "
                    f"singular ({e})") from e
            if not np.all(np.isfinite(dd)):
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip + q0 two-scalar closure returned "
                    f"{dd!r}")
            jc = J0 + dd[0] * p["ind"] + dd[1] * p["bs"]
            amp.update(d_ind=float(dd[0]), d_bs=float(dd[1]),
                       a_ind=float(1.0 + dd[0]), a_bs=float(1.0 + dd[1]),
                       axis_row=float(self.pin.row), axis_psi_N=psi0)
        amp["Ip_exact_measure_A"] = float(_lin(geom, jc) + float(c_k))
        amp["Ip_target_A"] = float(ctx.Ip_star)
        return jc, p, amp

    def step(self, jbs, k, relax=None):
        from .engine import _delivery_stats, complete_geometry, \
            conversion_factor
        ctx = self.ctx
        geom = self.geom
        jc, p, amp = self.close(geom, jbs)
        jint = np.asarray(relax(jc) if relax is not None else jc,
                          dtype=float)
        req = jint if self.delta is None else jint + self.delta
        if self.first_request is None:
            self.first_request = req.copy()
        n0 = int(self.b.n_solves)
        try:
            self.b.solve(req, n_passes=1)
        except Exception as e:
            if k == 0 and self.phase == "loop" and (
                    type(e) is Exception
                    or isinstance(e, (ValueError, RuntimeError))):
                from .TokaMaker_interface import DrawAnchorSolveFailed
                raise DrawAnchorSolveFailed(
                    f"{ctx.label}: the first pass (the reconstruction's "
                    "stored state composed with the draw's components, at "
                    f"the draw's pressure) failed to solve "
                    f"({type(e).__name__}: {str(e).strip()[:300]}); the "
                    "draw is REJECTED") from e
            raise
        if self.on_solve is not None:
            self.on_solve(int(self.b.n_solves) - n0)
        if self.coil_guard is not None:
            self.coil_guard(f"engine draw {self.phase} pass {k + 1}")
        m = self.b.measure()
        g1 = complete_geometry(m["geom"])
        dstat, dvec = _delivery_stats(req, m["achieved"], g1)
        self.last = dict(k=k, jc=jc, jint=jint, req=req, parts=p, amp=amp,
                         geom_composed=geom, geom=g1, m=m, dvec=dvec,
                         jbs=np.asarray(jbs, dtype=float))
        self.passes.append(dict(
            k=int(k), phase=self.phase, **amp, li=float(m["li"]),
            q_row=float(m["q_row"]), Ip=_f(m.get("Ip")), delivery=dstat,
            n_solves=int(self.b.n_solves)))
        self.geom = g1
        meas = dict(w=g1["w_lin"] * conversion_factor(g1), x=ctx.psi,
                    li=m["li"], redl=m["redl"])
        if self.pin is not None:
            meas["q0"] = m["q_row"]
            meas["axis_current_solved"] = float(np.interp(
                float(g1["psi_q"][0]), ctx.psi, jint))
        return meas

    def on_pass(self, k, meas, J, entry):
        if entry.get("omega_next") is None or self.delta is None:
            return
        # the delivery correction (engine_delivery_correction): one Newton
        # step per pass, as the reconstruction takes it
        self.delta = np.asarray(self.last["dvec"], dtype=float).copy()
        self.passes[-1]["delivery_correction_max"] = float(np.max(np.abs(
            self.delta)))


class _Clock:
    """Solves, passes and wall time by stage (anchor, loop, homotopy,
    post_homotopy, filters, archive)."""

    STAGES = ("anchor", "loop", "homotopy", "post_homotopy", "filters",
              "archive")

    def __init__(self, solve_counter=None):
        self.counter = solve_counter
        self.guarded = False
        self.t0 = time.perf_counter()
        self.stages = {s: dict(solves=0, passes=0, wall_s=0.0)
                       for s in self.STAGES}
        self.cur = None
        self._t = None
        self._n = None

    def _count(self):
        try:
            return None if self.counter is None else int(self.counter())
        except Exception:
            return None

    def start(self, stage):
        self.stop()
        self.cur = stage
        self._t = time.perf_counter()
        self._n = self._count()

    def stop(self):
        if self.cur is None:
            return
        st = self.stages[self.cur]
        st["wall_s"] += float(time.perf_counter() - self._t)
        n1 = self._count()
        if n1 is not None and self._n is not None:
            st["solves"] += n1 - self._n
        elif st.get("solves_counted") is None:
            st["solves_counted"] = False
        self.cur = None

    def add(self, stage, solves=0, passes=0):
        self.stages[stage]["solves"] += int(solves)
        self.stages[stage]["passes"] += int(passes)

    def record(self):
        out = {k: dict(v) for k, v in self.stages.items()}
        out["total"] = dict(
            solves=int(sum(v["solves"] for v in self.stages.values())),
            passes=int(sum(v["passes"] for v in self.stages.values())),
            wall_s=float(sum(v["wall_s"] for v in self.stages.values())))
        out["solve_counter"] = ("DrawSolveGuard.n_solves (every GS solve of "
                                "the draw)" if self.guarded else
                                "the engine backend's own solves (the "
                                "homotopy's direct solves are not counted "
                                "without a DrawSolveGuard)")
        return out


def run_draw(ctx, backend, inputs, *, label=None, coil_guard=None,
             clock=None, bnd_diag=None, bootstrap_refresh=None):
    """ONE engine draw from the reconstruction state (module docstring).

    *bootstrap_refresh* (default: ``ctx.bootstrap_refresh``, i.e.
    ``GenerationConfig.engine_draw_bootstrap_refresh``): after the loop's
    FIRST solve the anchor's kinetic increment is re-evaluated on that
    solved geometry and the loop RESTARTS from it (:data:`REFRESH_SOURCE`,
    ``run_jbs_loop(start_refresh=...)``) instead of blending toward the
    start computed on ``G*``.  Zero extra solves and zero extra Redl
    evaluations (pass 1's Redl is the loop's own); the first request, every
    criterion and the fixed point are unchanged -- the path only.  At zero
    perturbation the refreshed bootstrap is ``lambda_BS*`` plus the change
    of Redl between the stored and the RE-SOLVED equilibrium -- rounding on
    the toy stand-in; on the live solver the re-solve reproduces the stored
    state only to its own convergence (measured on the g-file example:
    the refresh step r_j = 1.9e-5, against jbs_rtol_j = 1e-3).

    Returns a dict: ``record`` (the JSON-safe draw record), ``jbs_used``,
    ``passes`` (the :class:`_DrawPasses` of the loop, the post-homotopy stage
    continues it), ``inputs``.  Raises what the loop raises
    (:class:`~bouquet.jbs_loop.JBSNotConverged` /
    :class:`~bouquet.jbs_loop.JBSNonFinite`, a closure refusal, a failed
    first solve as :class:`~bouquet.TokaMaker_interface.
    DrawAnchorSolveFailed`) -- the draw is then rejected."""
    from .jbs_loop import AxisRowPin, jsonable, run_jbs_loop
    label = str(label or ctx.label)
    clock = (clock if clock is not None
             else _Clock(lambda: int(backend.n_solves)))
    backend.set_inputs(pressure=inputs.pressure, kinetics=inputs.kinetics)
    scale = float(inputs.scale)
    # ---- anchor stage (no solve): the kinetic response of Redl on the
    # state the draw starts from, as an increment on lambda_BS*
    clock.start("anchor")
    try:
        r_d = np.asarray(backend.redl(inputs.kinetics), dtype=float)
        r_0 = np.asarray(backend.redl(ctx.c.kinetics), dtype=float)
    except Exception as e:
        clock.stop()
        from .TokaMaker_interface import DrawAnchorSolveFailed
        raise DrawAnchorSolveFailed(
            f"{label}: Redl on the starting equilibrium failed "
            f"({type(e).__name__}: {str(e).strip()[:300]})") from e
    jbs0 = scale * (ctx.lam + (r_d - r_0))
    if bootstrap_refresh is None:
        bootstrap_refresh = bool(getattr(ctx, "bootstrap_refresh", False))
    start_refresh = None
    if bootstrap_refresh:
        def start_refresh(J, meas):
            # the anchor's increment with Redl(draw kinetics) taken on the
            # FIRST SOLVED geometry instead of G* (meas["redl"]: the loop's
            # own pass-1 Redl, the draw's kinetics, solved at the draw's
            # pressure); exactly the anchor's form, so lambda_BS* at zero
            # perturbation up to the re-solve's reproduction of G*
            return scale * (ctx.lam + (np.asarray(meas["redl"], dtype=float)
                                       - r_0))
    clock.start("loop")
    pin = None
    if ctx.q0_row:
        pin = AxisRowPin(ctx.q0_target, float(ctx.eng.s["q0_tol"]),
                         ctx.row0, label=label)
    dp = _DrawPasses(ctx, backend, inputs,
                     geom=ctx.geom_for_pressure(inputs.pressure), pin=pin,
                     delta=ctx.delta, coil_guard=coil_guard,
                     on_solve=lambda n: clock.add("loop", passes=1))
    res = run_jbs_loop(
        jbs0, dp.step, lambda m: scale * np.asarray(m["redl"], dtype=float),
        ctx.loop, Ip=float(ctx.c.Ip),
        meas0=dict(li=ctx.ref["l_i"],
                   q0=(ctx.ref["q_row"] if pin is not None else None)),
        gate_li=True, gate_q0=pin is not None, label=label,
        init_source=("the reconstruction's lambda_BS* + (Redl with the "
                     "draw's kinetics - Redl with the reconstruction's) on "
                     "the starting equilibrium, times the draw's bootstrap "
                     "scale (exactly lambda_BS* at zero perturbation)"),
        on_pass=dp.on_pass, q0_pin=pin, raise_on_fail=True,
        start_refresh=start_refresh)
    if start_refresh is not None:
        res["record"]["bootstrap_refresh"]["source"] = REFRESH_SOURCE
    m_fin = backend.measure(final=True)
    clock.stop()
    if bnd_diag is not None:
        bnd_diag("after engine draw loop")
    out = _finish(ctx, backend, inputs, dp, res, m_fin, pin, label)
    from .jbs_loop import jsonable
    # the anchor and loop stages (generate() completes the clock)
    out["record"]["cost"] = jsonable(clock.record())
    return out


def _finish(ctx, backend, inputs, dp, res, m_fin, pin, label):
    from .engine import conversion_factor
    from .jbs_loop import check_delivered, jsonable
    last = dp.last
    jbs_used = np.asarray(res["jbs_used"], dtype=float)
    meas = res["meas_final"]
    chk = check_delivered(res["J_final"], jbs_used, meas["w"], meas["x"],
                          float(ctx.c.Ip), ctx.loop)
    stats = m_fin.get("stats") or {}
    delivered = dict(
        l_i_3=float(m_fin["li"]), l_i_1=_f(m_fin.get("li_1")),
        beta_n=_f(stats.get("beta_n")),
        q0=float(m_fin["q_row"]), q0_psi_N=float(ctx.ref["q_row_psi_N"]),
        q0_stats=_f(stats.get("q_0")),
        q0_stats_psi_N=float(ctx.ref["q0_stats_psi_N"]),
        q95=_f(stats.get("q_95")), Ip=_f(m_fin.get("Ip")),
        delivery_check=dict(r_j=float(chk["r_j"]), r_I=float(chk["r_I"]),
                            ok=bool(chk["ok"])),
        request_minus_achieved=dp.passes[-1]["delivery"])
    amp = last["amp"]
    att = ctx.attribution(inputs, jbs_used, amp)
    dli = float(m_fin["li"]) - float(ctx.ref["l_i"])
    att["delta_l_i"] = dli
    att["remainder"] = float(dli - att["linear_total"])
    att["remainder_split"] = dict(
        nonlinear_frozen_geometry=float(att["model_on_reconstruction_geometry"]
                                        - att["linear_total"]),
        geometry_and_delivery=float(dli
                                    - att["model_on_reconstruction_geometry"]))
    ident = dict(
        pass1_request_bit_identical=bool(np.array_equal(dp.first_request,
                                                        ctx.request)),
        pass1_request_max_abs_diff=float(np.max(np.abs(dp.first_request
                                                       - ctx.request))),
        zero_perturbation=bool(inputs.sampler.get("zero_perturbation",
                                                  False)))
    kap = conversion_factor(last["geom_composed"])
    j_bs_tor = np.asarray(last["parts"]["bs"], dtype=float) * (
        1.0 + float(amp.get("d_bs", 0.0)))
    rec = dict(
        version=ENGINE_DRAW_VERSION, label=label,
        route=("engine draw: x* held, the Ip row as a scalar amplitude on "
               "the inductive in the exact measure"
               + (" + the q0 row (AxisRowPin on the bootstrap amplitude)"
                  if pin is not None else "")),
        reconstruction_converged=ctx.reconstruction_converged,
        identity=ident,
        inputs=dict(scale=float(inputs.scale), **{
            k: v for k, v in inputs.sampler.items() if k != "rng_stream"}),
        rng_stream=RNG_STREAM,
        amplitude=dict(final=amp, per_pass=[
            dict(k=p["k"], a_ind=p.get("a_ind"), a_bs=p.get("a_bs"),
                 dIp_A=p.get("dIp_A"),
                 Ip_exact_measure_A=p.get("Ip_exact_measure_A"))
            for p in dp.passes]),
        loop=res["record"], passes=dp.passes,
        q0_row=(None if pin is None else dict(
            target=float(ctx.q0_target), record=pin.record())),
        delivered=delivered, reference=dict(ctx.ref),
        deltas=dict(
            l_i_3=dli,
            l_i_1=(None if (delivered["l_i_1"] is None
                            or ctx.ref["l_i_1"] is None)
                   else delivered["l_i_1"] - ctx.ref["l_i_1"]),
            beta_n=(None if (delivered["beta_n"] is None
                             or ctx.ref["beta_n"] is None)
                    else delivered["beta_n"] - ctx.ref["beta_n"]),
            q0=float(delivered["q0"] - ctx.ref["q_row"]),
            q95=(None if (delivered["q95"] is None or ctx.ref["q95"] is None)
                 else delivered["q95"] - ctx.ref["q95"])),
        attribution=att,
        solves=dict(loop=int(backend.n_solves)))
    return dict(record=jsonable(rec), jbs_used=jbs_used, passes=dp,
                inputs=inputs, pin=pin, measure=m_fin,
                split=dict(j_phi=np.asarray(last["jint"], dtype=float),
                           j_BS=j_bs_tor,
                           j_NBI=kap * np.asarray(
                               ctx.c.jB_fix_parts["nbi"], dtype=float),
                           j_RF=kap * np.asarray(
                               ctx.c.jB_fix_parts["rf"], dtype=float)))


def post_homotopy(ctx, backend, draw, settings, *, coil_guard=None,
                  label=None, clock=None):
    """The post-homotopy bootstrap check of an engine draw (the homotopy
    moved coils and boundary after the loop converged) -- the existing
    check (:func:`bouquet.jbs_loop.check_delivered` on Redl of the delivered
    equilibrium against the bootstrap the draw carries, the loop
    tolerances), then at most ``settings["post_homotopy_passes"]`` further
    passes AT the current (tight) coil stage with the engine draw's own step
    (``x*`` held, the Ip amplitude, the draw's pressure), each checked by
    *coil_guard* (the homotopy's saturation criterion at those bounds).
    Raises :class:`~bouquet.jbs_loop.JBSNotConverged` when that fails.
    Returns ``(record, jbs_used, j_BS_toroidal, j_phi)`` -- the last two
    ``None`` when the draw was kept as is."""
    from .engine import complete_geometry, conversion_factor
    from .jbs_loop import (JBS_POST_HOMOTOPY_PASSES, check_delivered,
                           jsonable, run_jbs_loop)
    from .TokaMaker_interface import COIL_SATURATION_FRACTION
    label = str(label or ctx.label) + " post-homotopy"
    inputs = draw["inputs"]
    scale = float(inputs.scale)
    jbs_used = np.asarray(draw["jbs_used"], dtype=float)
    n_ph = int(settings.get("post_homotopy_passes", JBS_POST_HOMOTOPY_PASSES))
    backend.set_inputs(pressure=inputs.pressure, kinetics=inputs.kinetics)
    m = backend.measure()
    g = complete_geometry(m["geom"])
    w = g["w_lin"] * conversion_factor(g)
    J = scale * np.asarray(m["redl"], dtype=float)
    chk = check_delivered(J, jbs_used, w, ctx.psi, float(ctx.c.Ip), settings)
    rec = dict(check=jsonable(dict(chk)),
               accepted_without_passes=bool(chk["ok"]),
               solve="engine draw step (x* held, Ip amplitude), one "
                     "jphi-linterp solve per pass, beta-relaxed")
    rec["coil_saturation_guard"] = (
        dict(active=False, note="no guard supplied: re-solves unchecked")
        if coil_guard is None else
        dict(active=True, stage=getattr(coil_guard, "stage", None),
             F_lim=(getattr(coil_guard, "limits", (None, None))[0]),
             VSC_lim=(getattr(coil_guard, "limits", (None, None))[1]),
             fraction=float(COIL_SATURATION_FRACTION),
             checks=getattr(coil_guard, "log", None)))
    print(f"  [engine draw post-homotopy] r_j={chk['r_j']:.3e} "
          f"r_I={chk['r_I']:.3e} -> "
          + ("inside tolerance, draw kept" if chk["ok"] else
             f"outside tolerance, up to {n_ph} passes at the tight coil "
             "stage"), flush=True)
    if chk["ok"]:
        return rec, jbs_used, None, None
    omega = float(settings["relax"])
    jbs0 = (1.0 - omega) * jbs_used + omega * J
    prev = draw["passes"]
    dp = _DrawPasses(ctx, backend, inputs, geom=g, pin=draw.get("pin"),
                     delta=prev.delta, coil_guard=coil_guard,
                     phase="post_homotopy",
                     on_solve=(None if clock is None else
                               (lambda n: clock.add("post_homotopy",
                                                    passes=1))))
    res = run_jbs_loop(
        jbs0, dp.step, lambda mm: scale * np.asarray(mm["redl"], dtype=float),
        settings, Ip=float(ctx.c.Ip), meas0=dict(
            li=m["li"], q0=(m["q_row"] if dp.pin is not None else None)),
        gate_li=True, gate_q0=dp.pin is not None, label=label,
        init_source=INIT_POST_HOMOTOPY, on_pass=dp.on_pass, q0_pin=dp.pin,
        max_passes=n_ph, raise_on_fail=True)
    rec["passes"] = res["record"]
    rec["amplitude"] = dp.last["amp"]
    last = dp.last
    j_bs = np.asarray(last["parts"]["bs"], dtype=float) * (
        1.0 + float(last["amp"].get("d_bs", 0.0)))
    draw["jbs_used"] = np.asarray(res["jbs_used"], dtype=float)
    draw["passes_post_homotopy"] = dp
    return rec, draw["jbs_used"], j_bs, np.asarray(last["jint"], dtype=float)


# ---------------------------------------------------------------------------
#  the post-hoc filters of the archived draw
# ---------------------------------------------------------------------------
def post_hoc_verdicts(ctx, final, *, l_i_tolerance, constrain_sawteeth,
                      l_i_reference=None):
    """The engine draw's post-hoc filters on the ARCHIVED state *final*
    (a backend measurement): the l_i band ``|l_i - l_i*| <= l_i_tolerance *
    l_i*`` (``l_i_tolerance`` a FRACTION, ``l_i*`` the reconstruction's
    delivered l_i unless *l_i_reference*), and, with
    ``constrain_sawteeth``, ``q0 >= 1`` at the q-row radius (the legacy
    gate's radius, psi_N = psi_pad).  Returns the verdict dict
    (``in_band``)."""
    li_ref = float(ctx.ref["l_i"] if l_i_reference is None
                   else l_i_reference)
    li = float(final["li"])
    rel = abs(li - li_ref) / li_ref if li_ref else float("inf")
    li_ok = bool(np.isfinite(rel) and rel <= float(l_i_tolerance))
    q0 = float(final["q_row"])
    saw_ok = True if not constrain_sawteeth else bool(np.isfinite(q0)
                                                      and q0 >= 1.0)
    reasons = []
    if not li_ok:
        reasons.append(f"l_i band: |l_i - l_i*|/l_i* = {rel:.4f} > "
                       f"l_i_tolerance {float(l_i_tolerance):g}")
    if not saw_ok:
        reasons.append(f"constrain_sawteeth: q0 = {q0:.4f} < 1 at psi_N "
                       f"{ctx.ref['q_row_psi_N']:g}")
    return dict(l_i_3=li, l_i_reference=li_ref, l_i_rel=float(rel),
                l_i_tolerance=float(l_i_tolerance), l_i_in_band=li_ok,
                constrain_sawteeth=bool(constrain_sawteeth), q0=q0,
                q0_psi_N=float(ctx.ref["q_row_psi_N"]), q0_ok=saw_ok,
                in_band=bool(li_ok and saw_ok), reasons=reasons)


#: Group attribute of an archived engine draw: the post-hoc band verdict (an
#: added filter flag, ANDed into ``selected`` by
#: :func:`bouquet.filtering._recompute_selected`).
DRAW_BAND_FLAG = "passes_draw_band"


#: The solver's own text for a solve that ran out of nonlinear iterations
#: (TokaMaker: ``Exceeded "maxits"``), matched case-insensitively anywhere in
#: the exception chain.
_MAXITS_PATTERN = r'exceeded\s*"?maxits'


def solve_hit_iteration_cap(exc) -> bool:
    """Whether *exc* (or an exception it chains) is a Grad-Shafranov solve
    that stopped at its nonlinear iteration cap -- the solver's own reason
    text, not merely a message that mentions the cap."""
    import re
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if re.search(_MAXITS_PATTERN, str(e), flags=re.IGNORECASE):
            return True
        e = e.__cause__ or e.__context__
    return False


def engine_rejection_reason(exc, stage):
    """The :data:`~bouquet.TokaMaker_interface.DRAW_REJECTION_REASONS` code
    of an engine draw rejected by *exc*: a non-finite bootstrap/current is
    ``jbs_non_finite``, a closure refusal ``engine_closure_refused``, every
    other case the legacy mapping (:func:`~bouquet.TokaMaker_interface.
    _draw_rejection_reason`)."""
    from .engine import EngineClosureRefused
    from .jbs_loop import JBSNonFinite
    from .TokaMaker_interface import _draw_rejection_reason
    if isinstance(exc, JBSNonFinite):
        return "jbs_non_finite"
    if isinstance(exc, EngineClosureRefused):
        return "engine_closure_refused"
    return _draw_rejection_reason(exc, stage)


# ---------------------------------------------------------------------------
#  generate(): the hook generate_bouquet calls
# ---------------------------------------------------------------------------
def tokamaker_backend(mygs, contract, *, psi_pad, q_psi, maxits):
    """The draw's backend on a live solver (monkeypatched by the fast
    tests)."""
    from .engine import TokaMakerBackend
    return TokaMakerBackend(mygs, contract, psi_pad=psi_pad, li_kind="li_3",
                            q_psi=q_psi, maxits=maxits)


class GenerateEngineDraws:
    """What ``generate_bouquet(engine_draw=...)`` runs per draw (built by
    ``Bouquet.generate`` from the live reconstruction)."""

    def __init__(self, ctx, *, unc, psi_pad, q_psi=None, maxits=None,
                 homotopy=True, l_i_tolerance=0.05, p_thresh=0.05,
                 max_proxy_draws=500):
        self.ctx = ctx
        self.unc = dict(unc)
        self.psi_pad = float(psi_pad)
        self.q_psi = q_psi
        self.maxits = maxits
        self.homotopy = bool(homotopy)
        self.l_i_tolerance = float(l_i_tolerance)
        self.p_thresh = float(p_thresh)
        self.max_proxy_draws = int(max_proxy_draws)
        self.loop_settings = dict(ctx.loop)
        self._cur = None
        self._cap_saved = None
        #: capped homotopy / post-homotopy solves that did not converge
        self.cap_events = []

    # ---- the pressure the baseline re-solve and every draw use --------
    def solve_pressure(self, psi_N=None):
        return np.asarray(self.ctx.c.pressure, dtype=float).copy()

    def backend(self, mygs):
        from types import SimpleNamespace
        c = self.ctx.c
        dc = SimpleNamespace(psi_N=c.psi_N, pressure=c.pressure, Ip=c.Ip,
                             kinetics=c.kinetics)
        return tokamaker_backend(mygs, dc, psi_pad=self.psi_pad,
                                 q_psi=self.q_psi, maxits=self.maxits)

    def validate(self, *, pin_jphi, jbs_delta_mode, l_i_uncertainty,
                 recalculate_j_BS, jbs_loop):
        import os
        bad = []
        if bool(pin_jphi) or os.environ.get("PIN_JPHI", "0") == "1":
            bad.append("PIN_JPHI (pin_jphi=True or env PIN_JPHI=1)")
        if os.environ.get("DIFF_BS", "0") == "1":
            bad.append("DIFF_BS=1 (env)")
        if bool(jbs_delta_mode):
            bad.append("jbs_delta_mode=True")
        if float(l_i_uncertainty or 0.0) > 0.0:
            bad.append("l_i_uncertainty > 0 (engine draws match no l_i)")
        if not recalculate_j_BS:
            bad.append("recalculate_j_BS=False")
        if not (jbs_loop and jbs_loop.get("enabled")):
            bad.append("no self-consistent loop settings")
        if bad:
            raise EngineDrawRefused(
                "engine draws (reconstruction_engine='unified') refuse: "
                + "; ".join(bad) + " -- legacy draw modes that would be "
                "silently ignored")

    def rejection_reason(self, exc, stage):
        """The rejection code; with ``draw_solve_maxits`` set, a
        post-homotopy pass whose solve stopped at the cap is
        ``post_homotopy_maxits`` (its own code, never folded into
        ``jbs_post_homotopy_error``)."""
        if stage == "post_homotopy" and self.hit_cap(exc):
            self.announce_cap("post-homotopy pass", exc)
            return "post_homotopy_maxits"
        return engine_rejection_reason(exc, stage)

    # ---- draw_solve_maxits on the homotopy and post-homotopy solves ------
    def hit_cap(self, exc) -> bool:
        """A solve that stopped at ``draw_solve_maxits`` (only when a cap is
        set: with the default ``None`` nothing is re-classified)."""
        return self.maxits is not None and solve_hit_iteration_cap(exc)

    def announce_cap(self, where, exc):
        """Print and record a capped solve that did not converge (the draw
        is then REJECTED by the caller)."""
        ev = dict(draw=(None if self._cur is None else self._cur["count"]),
                  where=str(where), maxits=self.maxits,
                  error=f"{type(exc).__name__}: {str(exc).strip()[:300]}")
        self.cap_events.append(ev)
        print(f"  [engine draw] {where}: the GS solve stopped at "
              f"draw_solve_maxits={self.maxits} without converging -> draw "
              "REJECTED (not rolled back, not archived)", flush=True)

    def cap_solver(self, mygs):
        """Set ``draw_solve_maxits`` on the solver for the homotopy stage
        (the post-homotopy passes run through the engine backend, which
        applies it per solve).  A no-op without a cap, or when the solver
        already carries it (``Bouquet.generate``'s DrawSolveGuard); the
        value found is put back by :meth:`uncap_solver`."""
        if self.maxits is None or self._cap_saved is not None:
            return
        st = getattr(mygs, "settings", None)
        if st is None:
            return
        cur = getattr(st, "maxits", None)
        if cur == self.maxits:
            return
        self._cap_saved = (cur,)
        st.maxits = self.maxits
        mygs.update_settings()

    def uncap_solver(self, mygs):
        """Undo :meth:`cap_solver` (idempotent)."""
        if self._cap_saved is None:
            return
        (cur,), self._cap_saved = self._cap_saved, None
        mygs.settings.maxits = cur
        mygs.update_settings()

    # ---- one draw ---------------------------------------------------------
    def draw(self, mygs, rng, scale, count, *, coil_guard=None,
             bnd_diag=None, solve_guard=None):
        """The legacy 7-tuple ``(ne, te, ni, ti, w_ExB, j_phi,
        diagnostics)`` of one engine draw (kinetic-grid profiles)."""
        ctx = self.ctx
        self.uncap_solver(mygs)             # (a previous draw's, if any)
        b = self.backend(mygs)
        clock = _Clock((lambda: int(b.n_solves)) if solve_guard is None
                       else (lambda: int(solve_guard.n_solves)))
        clock.guarded = solve_guard is not None
        self._cur = dict(count=int(count), clock=clock, backend=b)
        # the weak exploratory coil regularisation for the loop, as the
        # legacy self-consistent draw installs it (generate_bouquet puts the
        # strong one back after the draw)
        _stashed_reg = getattr(mygs, "_strong_coil_reg", None)
        if _stashed_reg is not None:
            try:
                _weak = getattr(mygs, "_weak_coil_reg", None)
                if _weak is None:
                    _weak = [mygs.coil_reg_term({_n: 1.0}, target=0.0,
                                                weight=1.0)
                             for _n in mygs.coil_sets]
                    _weak.append(mygs.coil_reg_term({"#VSC": 1.0},
                                                    target=0.0, weight=1e-2))
                mygs.set_coil_reg(reg_terms=_weak)
            except Exception as _e:
                print(f"  [engine draw hygiene] weak-reg install failed "
                      f"({_e}); the loop runs under the strong reg")
        inputs = sample_draw_inputs(
            ctx, rng, self.unc, b.flux_integral, scale=float(scale),
            p_thresh=self.p_thresh, max_proxy_draws=self.max_proxy_draws)
        _hard = getattr(mygs, "_coil_drift_bounds", None)
        out = run_draw(ctx, b, inputs,
                       label=f"engine draw {int(count)}",
                       coil_guard=(coil_guard if _hard is not None
                                   else None),
                       clock=clock, bnd_diag=bnd_diag)
        clock.start("homotopy")
        self._cur["draw"] = out
        sp = out["split"]
        jfix = sp["j_NBI"] + sp["j_RF"]
        rec = out["record"]
        diag = dict(
            j0_scales=[], Ip_scales=[],
            iteration_l_is=[rec["delivered"]["l_i_3"]],
            iteration_Ips=[rec["delivered"]["Ip"]],
            j_inductive=sp["j_phi"] - sp["j_BS"] - jfix, j_BS=sp["j_BS"],
            j_BS_edge=None, proxy_bias_observed=None, r2_ip_scale=None,
            r2_f_ind=None, aux=dict(inputs.aux),
            jbs_loop=_jbs_block(rec["loop"]),
            engine=rec,
            _jbs_ctx=dict(kind="engine", engine=self,
                          isolate_edge_jBS=False))
        nat = inputs.kinetics_native
        return (nat["ne"], nat["te"], nat["ni"], nat["ti"],
                np.zeros_like(ctx.psi), sp["j_phi"], diag)

    def post_homotopy(self, settings, coil_guard=None):
        """``_post_homotopy_jbs``'s engine branch."""
        cur = self._cur
        clock = cur["clock"]
        clock.start("post_homotopy")
        try:
            rec, jbs, jb_tor, jphi = post_homotopy(
                self.ctx, cur["backend"], cur["draw"], settings,
                coil_guard=coil_guard, clock=clock)
        finally:
            clock.start("homotopy")
        return rec, jb_tor, jb_tor, jphi

    def mark(self, stage):
        if self._cur is not None:
            self._cur["clock"].start(stage)

    def measured_drifts(self, mygs, baseline_coils, coil_drift):
        """``engine_draw_homotopy=False``: no homotopy stage; the drift of
        the loop's own delivered draw, judged at the spec (``coil_drift``)."""
        from .TokaMaker_interface import _coil_drift_pct
        cur, _ = mygs.get_coil_currents()
        return (_coil_drift_pct(cur, baseline_coils), -1, float(coil_drift),
                float(coil_drift))

    def post_hoc(self, mygs, diagnostics, in_spec, *, constrain_sawteeth,
                 l_i_target=None):
        """The post-hoc filters on the archived state; returns the draw's
        ``in_spec`` (the coil verdict AND the band)."""
        cur = self._cur
        clock = cur["clock"]
        clock.start("filters")
        b = cur["backend"]
        fin = b.measure(final=True)
        v = post_hoc_verdicts(self.ctx, fin, l_i_tolerance=self.l_i_tolerance,
                              constrain_sawteeth=constrain_sawteeth,
                              l_i_reference=l_i_target)
        v["coil_in_spec"] = bool(in_spec)
        v["in_spec"] = bool(in_spec and v["in_band"])
        stats = fin.get("stats") or {}
        rec = diagnostics["engine"]
        rec["archived"] = dict(
            l_i_3=float(fin["li"]), l_i_1=_f(fin.get("li_1")),
            beta_n=_f(stats.get("beta_n")), q0=float(fin["q_row"]),
            q0_psi_N=float(self.ctx.ref["q_row_psi_N"]),
            q95=_f(stats.get("q_95")),
            note=("the archived (post-homotopy) state; 'delivered' is the "
                  "loop's"))
        rec["post_hoc"] = v
        _jl = diagnostics.get("jbs_loop") or {}
        if _jl.get("post_homotopy") is not None:
            rec["post_homotopy"] = _jl["post_homotopy"]
        diagnostics[DRAW_BAND_FLAG] = bool(v["in_band"])
        if v["reasons"]:
            print("  [engine draw post-hoc] OUT OF BAND (archived, "
                  "in_spec=False): " + "; ".join(v["reasons"]), flush=True)
        clock.start("archive")
        return v["in_spec"]

    def stored_pressures(self):
        inp = self._cur["draw"]["inputs"]
        return (np.asarray(inp.pressure_thermal, dtype=float).copy(),
                np.asarray(inp.pressure, dtype=float).copy())

    def store_draw(self, header, count, scan_key, diagnostics):
        """Write the draw's ``engine`` block (JSON attribute/dataset
        :data:`bouquet.engine.ENGINE_ATTR`) and the band flag onto its
        archived group."""
        cur = self._cur
        clock = cur["clock"]
        clock.stop()
        rec = diagnostics.get("engine")
        if rec is None:
            return
        rec["homotopy"] = dict(
            enabled=bool(self.homotopy),
            solve_maxits=self.maxits,
            homotopy_pass=diagnostics.get("homotopy_pass"),
            homotopy_F_lim=diagnostics.get("homotopy_F_lim"),
            homotopy_VSC_lim=diagnostics.get("homotopy_VSC_lim"),
            max_F_drift_pct=diagnostics.get("max_F_drift_pct"),
            max_VSC_drift_pct=diagnostics.get("max_VSC_drift_pct"),
            note=("generate()'s coil homotopy (homotopy_passes) and the "
                  "post-homotopy bootstrap check" if self.homotopy else
                  "engine_draw_homotopy=False: no homotopy stage; the "
                  "drift of the loop's delivered draw"))
        rec["cost"] = clock.record()
        _write_draw_block(header, count, scan_key, rec,
                          diagnostics.get(DRAW_BAND_FLAG))
        clock.start("filters")

    def until_n(self, ok, reasons, diagnostics, header=None, count=None,
                scan_key=None):
        """The until-N verdict of an engine draw: the configured coil +
        boundary verdict AND the post-hoc band (so the ledger counts what
        ``.filter()`` marks ``selected``)."""
        band = diagnostics.get(DRAW_BAND_FLAG)
        reasons = list(reasons)
        if band is not None and not band:
            reasons.append("engine post-hoc band")
        cur = self._cur
        if cur is not None and header is not None:
            cur["clock"].stop()
            rec = diagnostics.get("engine")
            if rec is not None:
                rec["cost"] = cur["clock"].record()
                _write_draw_block(header, count, scan_key, rec, band)
        return bool(ok and (band is None or band)), reasons

    def store_baseline(self, header, scan_key, baseline):
        """The baseline's ``engine`` block (the reconstruction record, with
        the draws' settings) on ``_baseline``."""
        from .engine import store_baseline_engine
        rec = dict(getattr(baseline, "engine", None) or {})
        if not rec:
            return
        rec["draws"] = dict(
            version=ENGINE_DRAW_VERSION, rng_stream=RNG_STREAM,
            loop=self.loop_settings, q0_row=self.ctx.q0_row,
            bootstrap_refresh=bool(self.ctx.bootstrap_refresh),
            solve_maxits=self.maxits,
            homotopy=self.homotopy, l_i_tolerance=self.l_i_tolerance,
            ip_row=("the Ip the delivered composition carries in the exact "
                    "measure on G*"), Ip_target_A=self.ctx.Ip_star,
            reference=self.ctx.ref)
        store_baseline_engine(header, rec, scan_key=scan_key)


def _jbs_block(loop_rec):
    """The draw's ``jbs_loop`` archive block, shaped like a legacy draw's."""
    from .jbs_loop import jsonable
    return jsonable(dict(
        enabled=True, converged=bool(loop_rec.get("converged")),
        kind="engine", n_loops=1,
        n_passes_total=int(loop_rec.get("n_passes", 0)),
        init_source=loop_rec.get("init_source"),
        loops=[dict(label=loop_rec.get("label"),
                    init_source=loop_rec.get("init_source"),
                    n_passes=int(loop_rec.get("n_passes", 0)),
                    converged=bool(loop_rec.get("converged")))],
        final=loop_rec))


def _write_draw_block(header, count, scan_key, record, band):
    import h5py
    from .engine import write_engine_json
    from .utils import _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "a") as hf:
        gp = _group_path(scan_key, count)
        if gp not in hf:
            return
        grp = hf[gp]
        write_engine_json(grp, record)
        if band is not None:
            grp.attrs[DRAW_BAND_FLAG] = bool(band)


def read_draw_engine(header, count, scan_key=None):
    """The ``engine`` block of one archived draw, or ``None`` (a legacy
    draw)."""
    import h5py
    from .engine import read_engine_json
    from .utils import _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "r") as hf:
        gp = _group_path(scan_key, count)
        if gp not in hf:
            return None
        return read_engine_json(hf[gp])


def build_generate_context(bq, env):
    """The :class:`GenerateEngineDraws` of ``Bouquet.generate`` (the live
    reconstruction of THIS session is required)."""
    run = getattr(bq, "_engine_run", None)
    if not run or run.get("baseline") is not bq.baseline:
        raise NotImplementedError(
            "engine draws (Stage 3) need the unified engine's reconstruction "
            "from prepare_baseline() in this session (its live state is what "
            "a draw inherits); call prepare_baseline() first")
    gc = bq.config.generation
    bl = bq.baseline
    ctx = context_from_run(run, gc, bl)
    unc = dict(env)
    return GenerateEngineDraws(
        ctx, unc=unc, psi_pad=run["psi_pad"], q_psi=run.get("q_psi"),
        maxits=getattr(gc, "draw_solve_maxits", None),
        homotopy=bool(getattr(gc, "engine_draw_homotopy", True)),
        l_i_tolerance=float(gc.l_i_tolerance))


def context_from_run(run, gc, bl):
    """The :class:`EngineDrawContext` of a live reconstruction and its
    Baseline (the kinetic-grid base the sampler perturbs)."""
    native = dict(psi_N=bl.psi_N_kinetic, ne=bl.ne, te=bl.te, ni=bl.ni,
                  ti=bl.ti, z_fast=getattr(bl, "z_fast", None))
    return EngineDrawContext(
        run["engine"], run["result"], loop=draw_loop_settings(gc),
        native=native, q0_row=bool(getattr(gc, "engine_draw_q0_row", False)),
        bootstrap_refresh=bool(getattr(gc, "engine_draw_bootstrap_refresh",
                                       False)))


# ---------------------------------------------------------------------------
#  verify_sigma0_consistency under the engine
# ---------------------------------------------------------------------------
def verify_zero_perturbation(ctx, backend, *, label="sigma=0 engine draw"):
    """The engine draw at zero perturbation (bootstrap scale 1.0) from the
    backend's current state: the request identity, and the delivered draw
    against the reconstruction -- ``r_j`` / ``r_I`` of the draw's bootstrap
    against ``lambda_BS*`` (the draw's final parallel weights), ``dl_i``,
    ``dq0`` (at the row radius and at the solver's q0 radius), ``dq95``.
    ``passed``: the request is bit-identical, the loop converged,
    ``r_j <= rtol_j``, ``r_I <= rtol_Ip`` and ``|dl_i| <= tol_li`` -- the
    draw-route rule of the legacy check, at the unchanged loop
    tolerances."""
    from .jbs_loop import JBSNotConverged, jsonable, profile_residuals
    s = ctx.loop
    out = dict(invariant="engine-draw", tolerances=dict(
        rtol_j=s["rtol_j"], rtol_Ip=s["rtol_Ip"], tol_li=s["tol_li"]),
        criterion=("pass-1 request bit-identical to the stored request, "
                   "loop converged, r_j <= rtol_j, r_I <= rtol_Ip, "
                   "|l_i(draw) - l_i*| <= tol_li; q0/q95 reported"))
    try:
        d = run_draw(ctx, backend, ctx.zero_inputs(), label=label)
    except JBSNotConverged as e:
        out.update(passed=False, loop_converged=False,
                   error=f"{type(e).__name__}: {str(e)[:300]}",
                   record=jsonable(getattr(e, "record", None)))
        return out
    rec = d["record"]
    meas = d["passes"].last
    from .engine import conversion_factor
    w = meas["geom"]["w_lin"] * conversion_factor(meas["geom"])
    cmp_ = profile_residuals(d["jbs_used"], ctx.lam, w, ctx.psi,
                             float(ctx.c.Ip))
    dl = rec["deltas"]
    conv = bool(rec["loop"]["converged"])
    ident = rec["identity"]
    ok = bool(ident["pass1_request_bit_identical"] and conv
              and cmp_["r_j"] <= s["rtol_j"] and cmp_["r_I"] <= s["rtol_Ip"]
              and abs(dl["l_i_3"]) <= s["tol_li"])
    out.update(
        passed=ok, request_bit_identical=ident["pass1_request_bit_identical"],
        request_max_abs_diff=ident["pass1_request_max_abs_diff"],
        loop_converged=conv, n_passes=int(rec["loop"]["n_passes"]),
        r_j=float(cmp_["r_j"]), r_I=float(cmp_["r_I"]),
        dl_i=float(dl["l_i_3"]), dq0=float(dl["q0"]),
        dq0_psi_N=float(ctx.ref["q_row_psi_N"]),
        dq0_stats=(None if (rec["delivered"]["q0_stats"] is None
                            or ctx.ref["q0_stats"] is None)
                   else rec["delivered"]["q0_stats"] - ctx.ref["q0_stats"]),
        dq0_stats_psi_N=float(ctx.ref["q0_stats_psi_N"]),
        dq95=dl["q95"], amplitude=rec["amplitude"]["final"].get("a_ind"),
        solves=rec["solves"], record=rec)
    print(f"[sigma0-check engine] {'PASS' if ok else 'FAIL'}: request "
          f"{'bit-identical' if ident['pass1_request_bit_identical'] else 'DIFFERS'}"
          f"; loop {'converged' if conv else 'NOT converged'} in "
          f"{out['n_passes']} pass(es); r_j={cmp_['r_j']:.3e} (tol "
          f"{s['rtol_j']:.0e}) r_I={cmp_['r_I']:.3e} (tol {s['rtol_Ip']:.0e})"
          f" |dl_i|={abs(dl['l_i_3']):.2e} (tol {s['tol_li']:.0e}); dq0="
          f"{dl['q0']:+.2e} (psi_N {ctx.ref['q_row_psi_N']:g})", flush=True)
    return out
