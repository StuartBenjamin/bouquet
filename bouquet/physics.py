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
