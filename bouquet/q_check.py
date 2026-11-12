"""Checks of an equilibrium's q against MSE pitch angles.

Blocks are in :mod:`bouquet.mse`'s schema; the reference equilibrium is a
:class:`bouquet.io.geqdsk.GEQDSKEquilibrium`.
"""

import numpy as np

from .mse import (MSE_MIN_CHORDS, MSEDataUnusable, mse_chords,
                  mse_equilibrium_orientation, mse_er_terms, mse_exclude,
                  mse_orientation, mse_tan_gamma)


def _outboard_R(eq, psi_N):
    """R [m] where the surfaces *psi_N* of *eq* cross Z = Z_axis outboard."""
    r = np.linspace(eq.R_mag, eq.R_grid[-1], 2001)
    p = eq.psi_N_at(r, np.full_like(r, eq.Z_mag))
    i0 = int(np.argmin(p))
    stop = np.nonzero(np.diff(p[i0:]) <= 0)[0]
    i1 = i0 + (stop[0] + 1 if stop.size else p.size - i0)
    # sqrt(psi_N) is close to linear in R near the axis
    return np.interp(np.sqrt(psi_N), np.sqrt(np.clip(p[i0:i1], 0, None)),
                     r[i0:i1], right=np.nan)


def mse_er_at_chords(eq, block, phi_N, E_r, E_r_err=None):
    """A copy of MSE *block* with ``Er`` [V/m] at its chords.

    *E_r* (and *E_r_err*) is the outboard-midplane radial field on the
    normalised toroidal flux *phi_N*.  E_r is the field of a flux-function
    potential, so on each surface E_R is proportional to R B_Z:
    ``E_R = E_r(Phi_N) (R B_Z) / (R B_Z)_out``, with the surfaces, Phi_N
    (from its q) and B of the reference *eq*, and ``out`` the surface's
    outboard point at Z_axis.  Chords outside the LCFS or the profile get
    NaN, which :func:`bouquet.mse.mse_chords` excludes with its reason.

    With *E_r_err*, ``sigma`` gains ``|d tan(gamma) / d E_R| sigma_ER`` in
    quadrature (the forward model on *eq*'s field, in the block's stated
    orientation).
    """
    def rbz(R, Z):
        return R * eq.field_at(R, Z)[:, 2]

    R, Z = (np.asarray(block[k], dtype=float) for k in ("R", "Z"))
    psi_c = eq.psi_N_at(R, Z)
    phi_c = np.interp(psi_c, np.linspace(0.0, 1.0, eq.rhovn.size), eq.rhovn ** 2,
                      right=np.nan)
    R_out = _outboard_R(eq, psi_c)
    ratio = rbz(R, Z) / rbz(R_out, np.full_like(R_out, eq.Z_mag))

    def at_chords(y):
        return np.interp(phi_c, phi_N, y, left=np.nan, right=np.nan) * ratio

    out = dict(block, Er=at_chords(E_r))
    if E_r_err is not None:
        ch = mse_chords(out, min_chords=1)
        B = eq.field_at(ch["R"], ch["Z"])
        sp, st = mse_orientation(ch, mse_equilibrium_orientation(
            B, ch["R"], ch["Z"], (eq.R_mag, eq.Z_mag)))
        dtg = (mse_tan_gamma(B, dict(ch, Er=np.ones(ch["n_active"])), sp, st)
               - mse_tan_gamma(B, dict(ch, Er=np.zeros(ch["n_active"])), sp, st))
        i = ch["index"]
        sigma = np.array(block["sigma"], dtype=float)
        sigma[i] = np.hypot(sigma[i], dtg * np.abs(at_chords(E_r_err))[i])
        out["sigma"] = sigma
    return out


def q_points(eq, block, axis_dR=0.0, min_chords=MSE_MIN_CHORDS, sigma_sys=0.0):
    """MSE pitch angles as iota = 1/q points on the surfaces of the reference *eq*.

    Relabelled-flux model: the true surfaces are *eq*'s with psi relabelled,
    so on each surface the poloidal field scales by ``lam``, while B_phi,
    Phi_N and rho_tor are unchanged and ``iota = lam / |q_ref|``.  Per chord

    ``lam = (t a - e) / (s_p (b - t c))``, with ``a = A2 s_t B_phi``,
    ``b = A1 B_Z``, ``c = A3 B_R + A4 B_Z`` and ``e = A5 E_R``,

    on *eq*'s field in the block's stated orientation ``(s_p, s_t)``.
    ``sigma_lam`` adds two terms in quadrature: ``|d lam / d t| sigma_eff``
    and half the change of ``lam`` when the field moves by +-*axis_dR* [m]
    (the error in the reference axis position, dominant near the axis).

    Chords at psi_N >= 1 are excluded.  Raises :class:`MSEDataUnusable`
    when fewer than *min_chords* remain, or when the median ``lam`` is
    <= 0 (the stated orientation contradicts the data).

    Returns ``dict(rho, psi_N, iota, sigma_iota, q, sigma_q, lam,
    sigma_lam, R, Z, index, excluded, orientation=(s_p, s_t), er_terms)``,
    one entry per used chord.
    """
    ch = mse_chords(block, min_chords=min_chords, sigma_sys=sigma_sys)
    psi = eq.psi_N_at(ch["R"], ch["Z"])
    ch = mse_exclude(ch, psi >= 1.0, "outside the reference LCFS (psi_N >= 1)")
    psi = psi[psi < 1.0]
    if ch["n_active"] < ch["min_chords"]:
        raise MSEDataUnusable(f"only {ch['n_active']} MSE chords lie inside the "
                              f"reference LCFS; at least {ch['min_chords']} "
                              "are required")
    R, Z, t = ch["R"], ch["Z"], ch["tgamma"]
    B = eq.field_at(R, Z)
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(
        B, R, Z, (eq.R_mag, eq.Z_mag)))
    e = ch["A5"] * ch["Er"]

    def lam(B):
        a = ch["A2"] * st * B[:, 1]
        b = ch["A1"] * B[:, 2]
        c = ch["A3"] * B[:, 0] + ch["A4"] * B[:, 2]
        return (t * a - e) / (sp * (b - t * c)), (a * b - c * e) / (sp * (b - t * c) ** 2)

    lm, dlm = lam(B)
    if not np.median(lm) > 0:
        raise MSEDataUnusable(
            f"median lam = {np.median(lm):.3g} <= 0: the stated ip_sign/bt_sign "
            "contradict the pitch angles on the reference equilibrium")
    s_geom = 0.5 * np.abs(lam(eq.field_at(R + axis_dR, Z))[0]
                          - lam(eq.field_at(R - axis_dR, Z))[0])
    s_lm = np.hypot(np.abs(dlm) * ch["sigma_eff"], s_geom)
    x = np.linspace(0.0, 1.0, eq.qpsi.size)
    q_ref = np.abs(np.interp(psi, x, eq.qpsi))
    iota, s_iota = lm / q_ref, s_lm / q_ref
    return dict(rho=np.interp(psi, x, eq.rhovn), psi_N=psi, iota=iota,
                sigma_iota=s_iota, q=1.0 / iota, sigma_q=s_iota / iota ** 2,
                lam=lm, sigma_lam=s_lm, R=R, Z=Z, index=ch["index"],
                excluded=ch["excluded"], orientation=(sp, st),
                er_terms=mse_er_terms(ch))
