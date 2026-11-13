"""Checks of an equilibrium's q against MSE pitch angles.

Blocks are in :mod:`bouquet.mse`'s schema; the reference equilibrium is a
:class:`bouquet.io.geqdsk.GEQDSKEquilibrium`.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .coords import phi_n_from_q
from .mse import (MSE_MIN_CHORDS, MSEDataUnusable, mse_chords,
                  mse_equilibrium_orientation, mse_er_terms, mse_exclude,
                  mse_orientation, mse_tan_gamma)
from .sampling import _CHOL_JITTER


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


def _chol(K):
    """Lower Cholesky factor of *K* with sampling's jitter rule."""
    n = K.shape[0]
    return np.linalg.cholesky(K + _CHOL_JITTER * n * np.max(np.diag(K)) * np.eye(n))


def _matern52(x1, x2, ell):
    s = np.sqrt(5.0) * np.abs(x1[:, None] - x2[None, :]) / ell
    return (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def _basis(x):
    return np.column_stack([np.ones_like(x), x ** 2])


def _gp_factor(x, y, s, amp, ell):
    """``(L, alpha, Hl, A, beta)`` of the GP with a flat-prior basis (R&W eq. 2.42)."""
    L = _chol(amp ** 2 * _matern52(x, x, ell) + np.diag(s ** 2))
    alpha = solve_triangular(L, y, lower=True)
    Hl = solve_triangular(L, _basis(x), lower=True)
    A = Hl.T @ Hl
    return L, alpha, Hl, A, np.linalg.solve(A, Hl.T @ alpha)


def _neg_log_ml(th, x, y, s):
    """Negative log marginal likelihood, basis coefficients integrated out (R&W eq. 2.45)."""
    L, alpha, Hl, A, beta = _gp_factor(x, y, s, *np.exp(th))
    return (0.5 * (alpha @ alpha - alpha @ Hl @ beta) + np.log(np.diag(L)).sum()
            + 0.5 * np.linalg.slogdet(A)[1])


class QCheck:
    """GP posterior of iota(rho_tor) given q points, and candidate q-profiles' likelihood.

    ``iota = b0 + b1 rho^2 + r(rho)``: a flat prior on (b0, b1), so the slope
    at the axis is zero without a derivative observation; ``r ~ GP(0, amp^2
    Matern-5/2(ell))``; noise ``diag(sigma_iota^2)`` (Rasmussen & Williams
    2006, sec. 2.7).  Candidates are scored on the posterior at ``m`` evenly
    spaced rho over the points' range, ``m = clip(ceil(range / ell) + 1, 2,
    n)``: about one per correlation length, where the posterior covariance is
    well conditioned.
    """

    def __init__(self, rho, iota, sigma_iota, amp, ell):
        self.rho, self.iota, self.sigma_iota = (
            np.asarray(a, dtype=float) for a in (rho, iota, sigma_iota))
        self.amp, self.ell = float(amp), float(ell)
        self._L, self._alpha, self._Hl, self._A, self._beta = _gp_factor(
            self.rho, self.iota, self.sigma_iota, self.amp, self.ell)
        m = int(np.clip(np.ceil(np.ptp(self.rho) / self.ell) + 1, 2, self.rho.size))
        self.rho_test = np.linspace(self.rho.min(), self.rho.max(), m)
        self._mean, cov = self.predict(self.rho_test)
        self._Lt = _chol(cov)
        self._logdet = 2.0 * np.log(np.diag(self._Lt)).sum() + m * np.log(2.0 * np.pi)

    def predict(self, rho):
        """Posterior mean and covariance of iota at *rho*."""
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        V = solve_triangular(self._L, self.amp ** 2 * _matern52(self.rho, rho, self.ell),
                             lower=True)
        R = _basis(rho) - V.T @ self._Hl
        return (V.T @ self._alpha + R @ self._beta,
                self.amp ** 2 * _matern52(rho, rho, self.ell) - V.T @ V
                + R @ np.linalg.solve(self._A, R.T))

    def loglike(self, psi_N, q):
        """``dict(logL, d2, m)`` of the candidate profiles q(psi_N), one per row of *q*.

        Each row's iota is placed on rho_tor by its own q.  ``d2`` is its
        Mahalanobis distance from the posterior at ``rho_test`` (chi^2 with
        ``m`` degrees of freedom for a draw from it) and ``logL = -(d2 +
        log|2 pi Sigma|) / 2``.  A row with non-finite q gets NaN.
        """
        q = np.abs(np.atleast_2d(np.asarray(q, dtype=float)))
        psi = np.broadcast_to(np.atleast_2d(np.asarray(psi_N, dtype=float)), q.shape)
        iota = np.full((q.shape[0], self.rho_test.size), np.nan)
        for i, (x, y) in enumerate(zip(psi, q)):
            if np.all(np.isfinite(y)):
                if x[0] > 0:   # held at the innermost node down to the axis
                    x, y = np.r_[0.0, x], np.r_[y[0], y]
                inside, phi = phi_n_from_q(x, y)
                iota[i] = np.interp(self.rho_test, np.sqrt(phi), 1.0 / y[inside])
        z = solve_triangular(self._Lt, (iota - self._mean).T, lower=True, check_finite=False)
        d2 = np.sum(z ** 2, axis=0)
        return dict(logL=-0.5 * (d2 + self._logdet), d2=d2, m=self.rho_test.size)

    def save(self, path):
        np.savez(path, rho=self.rho, iota=self.iota, sigma_iota=self.sigma_iota,
                 amp=self.amp, ell=self.ell)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["rho"], f["iota"], f["sigma_iota"], float(f["amp"]), float(f["ell"]))


def fit_q_check(points):
    """:class:`QCheck` of *points* (``rho``, ``iota``, ``sigma_iota``; e.g. :func:`q_points`).

    ``(amp, ell)`` maximise the marginal likelihood, by bounded L-BFGS-B from
    a fixed grid of starts (deterministic).  ``amp`` is at least the median
    ``sigma_iota``: when a quadratic fits within the errors, it would
    otherwise collapse, and the test would claim structure the data cannot
    resolve.  ``ell`` lies in [0.05, 2 x the rho range].  Non-finite points
    are dropped; fewer than 3 left raises ``ValueError``.
    """
    x, y, s = (np.asarray(points[k], dtype=float) for k in ("rho", "iota", "sigma_iota"))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    x, y, s = x[ok], y[ok], s[ok]
    if x.size < 3:
        raise ValueError(f"fit_q_check: {x.size} finite points; at least 3 are required")
    bounds = np.log([(np.median(s), 10.0), (0.05, 2.0 * np.ptp(x))])
    starts = [np.clip(np.log([a, ell]), bounds[:, 0], bounds[:, 1])
              for a in np.median(s) * np.array([1.0, 10.0]) for ell in (0.15, 0.4, 1.0)]
    best = min((minimize(_neg_log_ml, th, args=(x, y, s), method="L-BFGS-B", bounds=bounds)
                for th in starts), key=lambda r: r.fun)
    return QCheck(x, y, s, *np.exp(best.x))
