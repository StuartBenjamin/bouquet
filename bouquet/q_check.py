"""Checks of an equilibrium's q against MSE pitch angles.

Blocks are in :mod:`bouquet.mse`'s schema; the reference equilibrium is a
:class:`bouquet.io.geqdsk.GEQDSKEquilibrium`.
"""

import numpy as np

from .mse import (mse_chords, mse_equilibrium_orientation, mse_orientation,
                  mse_tan_gamma)


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
