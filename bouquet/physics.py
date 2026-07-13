"""Physics reductions shared by the baseline resolvers.

Two conversions are needed to bring heterogeneous inputs into bouquet's internal
conventions:

  * :func:`isotropize_fast_pressure` -- collapse an anisotropic (gyrotropic)
    fast-ion pressure to a single scalar, since TokaMaker solves a scalar-pressure
    Grad-Shafranov equation.

  * :func:`parallel_to_toroidal` -- convert a flux-surface-averaged *parallel*
    current density <j.B>/B0 (the IMAS / neoclassical convention for j_ohmic,
    j_bootstrap, and the driven currents) to the *toroidal* current density
    <j_phi/R>/<1/R>. bouquet stores every current component as toroidal j_phi.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Map the legacy positional index of ``TokaMaker.get_q``'s ``ravgs`` array to the
# key used by newer OFT builds, which return ``ravgs`` as a dict.
_GET_Q_RAVG_INDEX = {"<R>": 0, "<1/R>": 1, "dV/dPsi": 2}


def q_ravg(ravgs, which: str):
    """Extract a flux-surface average from the ``ravgs`` element of ``TokaMaker.get_q``.

    Supports both the legacy positional array ``[<R>, <1/R>, dV/dPsi]`` and the
    newer dict form ``{'<R>', '<1/R>', '<1/R^2>', 'dV/dPsi'}`` returned by recent
    OpenFUSIONToolkit releases. ``which`` is one of ``'<R>'``, ``'<1/R>'``,
    ``'dV/dPsi'`` (note the dict inserted ``'<1/R^2>'``, so the positional index
    of ``dV/dPsi`` is unchanged at 2 in the legacy array).
    """
    if isinstance(ravgs, dict):
        return ravgs[which]
    return ravgs[_GET_Q_RAVG_INDEX[which]]


def isotropize_fast_pressure(p_perp, p_par, method: str = "trace"):
    """Reduce anisotropic fast-ion pressure to a scalar for the scalar-p GS solve.

    For a gyrotropic pressure tensor ``P = p_par b b + p_perp (I - b b)`` the
    standard scalar pressure is one-third of the trace:

        method="trace"  ->  (2 * p_perp + p_par) / 3        [DEFAULT]
        method="mean"   ->  (p_perp + p_par) / 2
        method="perp"   ->  p_perp

    ``"trace"`` is recommended: it is the textbook scalar pressure p = tr(P)/3 of
    a gyrotropic distribution and it preserves the fast-ion energy density
    (w = (1/2)(p_par + 2 p_perp) = (3/2) p_scalar), consistent with how
    kinetic-EFIT constrains the total stored pressure
    (p_tot = p_e + p_i + p_Z + p_fast). Use ``"perp"`` only if matching the
    diamagnetic magnetic response specifically; the rigorous alternative is a
    modified anisotropic Grad-Shafranov solve (out of scope for a scalar solver).

    Inputs are per-species arrays on a common grid; the caller sums species.

    References
    ----------
    - Scalar pressure as tr(P)/3 of a gyrotropic tensor (standard kinetic theory).
    - Anisotropic Grad-Shafranov treatment: arXiv:1301.4714; J. Plasma Phys.,
      "Analysis of the isotropic and anisotropic Grad-Shafranov equation".
    - Kinetic-EFIT total-pressure constraint p_tot = p_e + p_i + p_Z + p_fast.
    """
    p_perp = np.asarray(p_perp, dtype=float)
    p_par = np.asarray(p_par, dtype=float)
    if p_perp.shape != p_par.shape:
        raise ValueError(
            f"p_perp and p_par must have the same shape; got {p_perp.shape} vs {p_par.shape}"
        )
    if method == "trace":
        return (2.0 * p_perp + p_par) / 3.0
    if method == "mean":
        return (p_perp + p_par) / 2.0
    if method == "perp":
        return p_perp
    raise ValueError(
        f"unknown p_fast reduction method {method!r}; expected 'trace', 'mean', or 'perp'"
    )


def parallel_to_toroidal(
    j_parallel,
    *,
    j_parallel_total=None,
    j_tor_total=None,
    geom: Optional[dict] = None,
):
    """Convert FSA parallel current density <j.B>/B0 to toroidal <j_phi/R>/<1/R>.

    bouquet stores all current components (inductive/ohmic, bootstrap, NBI, RF)
    as toroidal j_phi. IMAS/neoclassical outputs are parallel; this applies the
    per-flux-surface geometric conversion. The correction is typically modest but
    grows toward the edge / at low aspect ratio.

    This is needed in TWO places, not just on IMAS input:
      * IMAS-input bootstrap/ohmic/driven currents (use the *ratio* method).
      * bouquet's OWN recomputed bootstrap -- TokaMaker ``solve_with_bootstrap``
        returns a *parallel* j_BS, which must be converted here before it
        replaces the baseline j_BS (when ``recalculate_j_BS`` is True). Use the
        *analytic* method with the TokaMaker equilibrium's FSA metrics.

    Two methods:

    * **ratio** (preferred, used for FUSE input) -- when the source provides both
      the total parallel current and the total toroidal current (FUSE
      ``core_profiles`` carries ``j_total`` *and* ``j_tor``), form the per-surface
      factor ``c(psi) = j_tor_total / j_parallel_total`` and apply it to the
      component. Exact to the geometric mapping shared by field-aligned
      components; self-consistent with the source equilibrium.

    * **analytic** -- compute from equilibrium FSA metrics in ``geom`` when
      totals are unavailable (the reconstruction path, and bouquet's own
      per-draw ``solve_with_bootstrap`` output). Models the component as
      field-aligned, ``j = lambda(psi) B`` with ``lambda = <j.B>/<B^2>``
      (the standard treatment of the neoclassical banana-plateau /
      driven currents; the Pfirsch-Schlueter return current, which has
      ``<j_PS.B> = 0``, is by construction not part of the component), so

          j_tor = <j_phi/R>/<1/R> = lambda * F * <1/R^2> / <1/R>
                = <j.B> * F * <1/R^2> / (<B^2> <1/R>)
                = <j.B> / (F <1/R>) * [<B_phi^2>/<B^2>]

      using ``<B_phi^2> = F^2 <1/R^2>`` (exact, since ``B_phi = F/R``).
      ``geom`` keys:

        ``F``          flux function ``R*B_phi`` [T m]
        ``avg_inv_R``  ``<1/R>`` [1/m]
        ``avg_B2``     ``<B^2>`` [T^2]
        ``avg_inv_R2`` ``<1/R^2>`` [1/m^2], OPTIONAL -- when absent the
                       bracket ``<B_phi^2>/<B^2>`` is taken as 1,
                       neglecting ``<B_p^2>/<B^2> ~ (eps/q)^2`` (sub-1%%
                       at a DIII-D edge); the retained ``1/(F<1/R>)``
                       projection carries the O(eps^2) geometry.
        ``B0``         normalisation of the input, OPTIONAL (default 1):
                       pass the IMAS ``vacuum_toroidal_field`` B0 when
                       ``j_parallel`` is the IMAS convention ``<j.B>/B0``;
                       leave at 1 when passing raw ``<j.B>`` [T A/m^2].

    Pass either (``j_parallel_total``, ``j_tor_total``) for the ratio method or
    ``geom`` for the analytic method.
    """
    j_parallel = np.asarray(j_parallel, dtype=float)

    if j_parallel_total is not None and j_tor_total is not None:
        j_parallel_total = np.asarray(j_parallel_total, dtype=float)
        j_tor_total = np.asarray(j_tor_total, dtype=float)
        # Per-surface geometric factor c(psi) = j_tor_total / j_parallel_total,
        # shared by all field-aligned components. Guard the on-axis / low-current
        # surfaces where the total parallel current passes through zero: there the
        # ratio is ill-defined, so fall back to the nearest well-defined factor.
        eps = 1e-12 * np.nanmax(np.abs(j_parallel_total)) if j_parallel_total.size else 0.0
        good = np.abs(j_parallel_total) > eps
        if not np.any(good):
            raise ValueError("j_parallel_total is ~0 everywhere; cannot form ratio")
        c = np.ones_like(j_parallel_total)
        c[good] = j_tor_total[good] / j_parallel_total[good]
        if not np.all(good):
            # nearest-neighbour fill for the masked (near-zero) surfaces
            idx = np.arange(c.size)
            c[~good] = np.interp(idx[~good], idx[good], c[good])
        return j_parallel * c

    if geom is not None:
        try:
            F = np.asarray(geom["F"], dtype=float)
            avg_inv_R = np.asarray(geom["avg_inv_R"], dtype=float)
            avg_B2 = np.asarray(geom["avg_B2"], dtype=float)
        except KeyError as missing:
            raise ValueError(
                f"geom is missing required key {missing} "
                "(need 'F', 'avg_inv_R', 'avg_B2'; optional 'avg_inv_R2', 'B0')"
            ) from None
        j_dot_B = j_parallel * float(geom.get("B0", 1.0))
        # field-aligned component: j_tor = <j.B> F <1/R^2> / (<B^2> <1/R>);
        # F^2 <1/R^2> == <B_phi^2>, ~= <B^2> when <1/R^2> is unavailable
        # (neglects <B_p^2>/<B^2> ~ (eps/q)^2).
        if "avg_inv_R2" in geom and geom["avg_inv_R2"] is not None:
            bphi2_over_B2 = F**2 * np.asarray(geom["avg_inv_R2"], dtype=float) / avg_B2
        else:
            bphi2_over_B2 = 1.0
        return j_dot_B * bphi2_over_B2 / (F * avg_inv_R)

    raise ValueError(
        "provide either (j_parallel_total, j_tor_total) for the ratio method "
        "or geom for the analytic method"
    )


def effective_impurity_charge(ne, ni, zeff, min_dilution=1e-3):
    """Effective single-impurity charge Z_imp from a baseline (ne, ni, Zeff).

    Quasineutrality with one impurity species (main ion Z=1) gives, per flux
    surface::

        Z_imp = 1 + ne (Zeff - 1) / (ne - ni)

    Returns the median over surfaces with meaningful dilution
    (``(ne - ni)/ne > min_dilution`` and ``Zeff > 1``), which is robust to
    edge noise and to the axis where dilution can vanish. Returns ``None``
    when the baseline carries no dilution information at all (``ni ~= ne``
    everywhere, e.g. the IDA ``ni = ne`` workflow) -- in that case a Zeff
    draw cannot be mapped onto a main-ion density.
    """
    ne = np.asarray(ne, dtype=float)
    ni = np.asarray(ni, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dil = (ne - ni) / ne
        z = 1.0 + ne * (zeff - 1.0) / (ne - ni)
    ok = np.isfinite(z) & (dil > min_dilution) & (zeff > 1.0)
    if not np.any(ok):
        return None
    return float(np.median(z[ok]))


def main_ion_density_from_zeff(ne, zeff, Z_imp):
    """Main-ion density from (ne, Zeff) under single-impurity quasineutrality.

    ::

        ni  = ne (Z_imp - Zeff) / (Z_imp - 1)
        nz  = (ne - ni) / Z_imp            (the implied impurity density)

    For ``1 <= Zeff <= Z_imp`` this guarantees ``0 <= ni <= ne`` and
    ``nz >= 0`` -- the consistent (ne, ni, Zeff, nz) set that the independent
    per-channel draws cannot provide. Returns ``ni``.
    """
    ne = np.asarray(ne, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    Z_imp = float(Z_imp)
    if not Z_imp > 1.0:
        raise ValueError(f"Z_imp must exceed 1 (got {Z_imp})")
    return ne * (Z_imp - zeff) / (Z_imp - 1.0)


# Elementary charge [C] -- thermal pressure p = e * sum_s(n_s * T_s) with n in
# m^-3 and T in eV.
_EC = 1.602176634e-19


def impurity_pressure(ne, ni, ti, Z_imp):
    """Thermal pressure of the (single, effective) impurity species [Pa].

    One-Zeff single-impurity model: the impurity density follows from the SAME
    ``(ne, ni, Z_imp)`` set that derives the main ion, ``nz = (ne - ni)/Z_imp``,
    assumed thermalized at the main-ion ``ti``. This is the carbon (impurity)
    pressure term that single-ion ``e*(ne*Te + ni*Ti)`` omits. Returns zeros if
    ``Z_imp`` is falsy/None (no measured dilution -> no impurity to add), so the
    same call is safe on impurity-free sources.
    """
    if not Z_imp:
        return np.zeros_like(np.asarray(ni, dtype=float))
    nz = np.clip((np.asarray(ne, dtype=float) - np.asarray(ni, dtype=float))
                 / float(Z_imp), 0.0, None)
    return _EC * nz * np.asarray(ti, dtype=float)


def fast_pressure_residual(psi_N, ne, te, ni, ti, Z_imp, psi_N_gfile, p_gfile):
    """Fast-ion pressure inferred as the g-file total minus the thermal pressure.

    The reconstruction path takes thermal kinetics from the IDA ``.cdf`` (no fast
    channel), but the g-file's ``equilibrium.pressure`` is the *total* (thermal +
    impurity + fast) pressure that constrained the original EFIT. The difference
    is the fast (beam) pressure plus any small GS / numerical residual::

        p_fast(psi) = max( p_gfile(psi)
                           - e (ne Te + ni Ti)        # main e + i thermal
                           - p_impurity(ne, ni, Ti)   # carbon thermal
                         , 0 )

    ``p_gfile`` (on its own ``psi_N_gfile`` grid) is linearly interpolated onto
    the kinetic ``psi_N`` grid before differencing, so the result is ready to feed
    to ``FixedComponentsConfig.p_fast`` (with ``psi_N``). It is clipped at zero to
    stay physical. Subtracting :func:`impurity_pressure` here avoids double-counting
    the carbon term, which bouquet re-adds per draw. An explicit TRANSP/ONETWO
    ``p_fast`` should be preferred when available; this residual is the
    no-extra-data fallback for NBI-heated shots.

    Returns ``p_fast`` [Pa] on ``psi_N``.
    """
    psi_N = np.asarray(psi_N, dtype=float)
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te, dtype=float)
    ni = np.asarray(ni, dtype=float)
    ti = np.asarray(ti, dtype=float)

    p_gfile_kin = np.interp(psi_N, np.asarray(psi_N_gfile, dtype=float),
                            np.asarray(p_gfile, dtype=float))
    p_thermal = _EC * (ne * te + ni * ti)
    p_imp = impurity_pressure(ne, ni, ti, Z_imp)
    return np.clip(p_gfile_kin - p_thermal - p_imp, 0.0, None)


def infer_fast_pressure(psi_N, ne, te, ni, ti, Z_imp, psi_N_gfile, p_gfile,
                        axis_psi_N: float = 0.15):
    """Validated fast-ion pressure from the g-file/thermal residual.

    Wraps :func:`fast_pressure_residual` with a physical-validity gate. The
    residual ``p_gfile - p_thermal - p_imp`` is only a meaningful fast pressure
    when the g-file is a *kinetic*-EFIT whose total pressure was constrained to
    ``p_thermal + p_fast``; for a magnetics/standard EFIT the total can fall
    *below* the kinetic thermal pressure near the axis, and the clipped residual
    is then an artifact (zero core, spurious mid-radius bump) rather than a real
    beam profile -- a true NBI fast pressure peaks *at* the axis.

    Gate: a real core-peaked fast profile must be non-negative on axis, so if the
    (unclipped) residual is negative on average over the near-axis region
    (``psi_N <= axis_psi_N``) -- thermal already exceeds the g-file total there --
    the g-file is not usable for fast-pressure extraction, and this returns
    **zeros** (thermal-only, matching the reference behaviour) with a diagnostic
    message. Otherwise it returns the clipped residual.

    Returns ``(p_fast, info)`` where ``info`` is a dict with ``valid`` (bool),
    ``axis_residual_Pa``, ``peak_Pa``, ``peak_psi_N``, and a human-readable
    ``message`` the caller can print.
    """
    psi_N = np.asarray(psi_N, dtype=float)
    ne = np.asarray(ne, dtype=float); te = np.asarray(te, dtype=float)
    ni = np.asarray(ni, dtype=float); ti = np.asarray(ti, dtype=float)

    p_gfile_kin = np.interp(psi_N, np.asarray(psi_N_gfile, dtype=float),
                            np.asarray(p_gfile, dtype=float))
    raw = p_gfile_kin - _EC * (ne * te + ni * ti) - impurity_pressure(ne, ni, ti, Z_imp)

    axis = psi_N <= axis_psi_N
    axis_resid = float(np.mean(raw[axis])) if axis.any() else float(raw[0])
    valid = axis_resid >= 0.0

    if valid:
        p_fast = np.clip(raw, 0.0, None)
        ipk = int(np.argmax(p_fast))
        info = dict(valid=True,
                    axis_residual_Pa=axis_resid,
                    peak_Pa=float(p_fast[ipk]),
                    peak_psi_N=float(psi_N[ipk]),
                    message=(f"[p_fast] kinetic-EFIT residual OK: peak "
                             f"{p_fast[ipk]:.0f} Pa at psi_N={psi_N[ipk]:.2f}"))
    else:
        p_fast = np.zeros_like(psi_N)
        info = dict(valid=False,
                    axis_residual_Pa=axis_resid,
                    peak_Pa=0.0,
                    peak_psi_N=float("nan"),
                    message=(f"[p_fast] g-file total < thermal near the axis "
                             f"(mean near-axis residual {axis_resid:.0f} Pa < 0): "
                             f"not a kinetic-EFIT -> p_fast=0 (thermal-only). Supply "
                             f"a kinetic-EFIT g-file or a TRANSP/ONETWO p_fast to "
                             f"include fast pressure."))
    return p_fast, info


def radial_field_from_cer(psi_N, n_carbon, t_carbon, omega_tor, v_pol,
                          Bpol, Rmaj, dpsiN_dR, B_phi, Z_C=6.0,
                          sigma_n_carbon=None, sigma_t_carbon=None,
                          sigma_omega_tor=None, sigma_v_pol=None):
    r"""Radial electric field from the impurity (carbon CER) radial force balance.

    The steady-state radial force balance of the measured impurity species
    (charge ``e_a = Z_C e``), with inertia and viscosity neglected, is

    .. math::

        E_r = \frac{1}{Z_C e\, n_C}\frac{dp_C}{dR}
              \;-\; v_\theta B_\phi \;+\; v_\phi B_\theta ,

    with :math:`p_C = e\,n_C T_C`, :math:`v_\phi = \omega_{tor} R`, and
    :math:`v_\theta` the measured poloidal velocity. This is the standard CER
    relation (Wesson, *Tokamaks* 4th ed., momentum eq. 2.23.2 and the radial
    force-balance passage; Burrell, *Phys. Plasmas* 4, 1499 (1997)). All three
    terms are of comparable size at low rotation.

    Inputs are SI on the common ``psi_N`` grid: ``n_carbon`` m^-3, ``t_carbon``
    eV, ``omega_tor`` rad/s, ``v_pol`` m/s, ``Bpol``/``B_phi`` T, ``Rmaj`` m,
    ``dpsiN_dR`` (= d psi_N / dR) 1/m. Signs are taken from the inputs as given
    (the caller must supply ``B_phi``/``Bpol``/``v_pol``/``omega_tor`` in a
    consistent convention); inspect the returned component terms to check.

    ``dp_C/dR`` is formed as ``d p_C/d psi_N * dpsiN_dR``. If the measured
    ``sigma_*`` envelopes are supplied, a 1-sigma ``sigma`` on E_r is propagated
    (diamagnetic term via the fractional (n_C, T_C) errors -- approximate through
    the gradient -- plus the exact rotation terms), suitable as a switchboard
    ``aux_sigmas['e_r']``.

    Returns ``(E_r, info)`` with ``info`` holding ``diamagnetic``, ``toroidal``
    (:math:`v_\phi B_\theta`), ``poloidal`` (:math:`-v_\theta B_\phi`), and
    ``sigma`` (zeros if no errors given), all [V/m].
    """
    psi_N = np.asarray(psi_N, dtype=float)
    n_C = np.asarray(n_carbon, dtype=float)
    t_C = np.asarray(t_carbon, dtype=float)
    Z_C = float(Z_C)

    p_C = _EC * n_C * t_C                                   # Pa
    dpC_dR = np.gradient(p_C, psi_N) * np.asarray(dpsiN_dR, dtype=float)   # Pa/m
    with np.errstate(divide="ignore", invalid="ignore"):
        diamag = np.where(n_C > 0, dpC_dR / (Z_C * _EC * n_C), 0.0)   # V/m
    v_phi = np.asarray(omega_tor, dtype=float) * np.asarray(Rmaj, dtype=float)
    toroidal = v_phi * np.asarray(Bpol, dtype=float)       # v_phi B_theta
    poloidal = -np.asarray(v_pol, dtype=float) * np.asarray(B_phi, dtype=float)  # -v_theta B_phi
    E_r = diamag + toroidal + poloidal

    sigma = np.zeros_like(E_r)
    if any(s is not None for s in (sigma_n_carbon, sigma_t_carbon,
                                   sigma_omega_tor, sigma_v_pol)):
        def _frac(sig, val):
            if sig is None:
                return np.zeros_like(val)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(val) > 0, np.asarray(sig, float) / np.abs(val), 0.0)
        # diamagnetic: approximate its relative error by the (n_C, T_C) fractional
        # errors added in quadrature (the gradient's relative error ~ profile's).
        rel_dia = np.sqrt(_frac(sigma_n_carbon, n_C) ** 2 + _frac(sigma_t_carbon, t_C) ** 2)
        s_dia = np.abs(diamag) * rel_dia
        s_tor = (np.abs(np.asarray(Rmaj, float) * np.asarray(Bpol, float))
                 * np.asarray(sigma_omega_tor, float)) if sigma_omega_tor is not None \
            else np.zeros_like(E_r)
        s_pol = (np.abs(np.asarray(B_phi, float)) * np.asarray(sigma_v_pol, float)) \
            if sigma_v_pol is not None else np.zeros_like(E_r)
        sigma = np.sqrt(s_dia ** 2 + s_tor ** 2 + s_pol ** 2)

    info = dict(diamagnetic=diamag, toroidal=toroidal, poloidal=poloidal, sigma=sigma)
    return E_r, info
