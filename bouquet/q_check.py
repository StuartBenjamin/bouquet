"""Checks of an equilibrium's q against MSE pitch angles.

Blocks are in :mod:`bouquet.mse`'s schema; the reference equilibrium is a
:class:`bouquet.io.geqdsk.GEQDSKEquilibrium`.  Three checks of a candidate:

* GP: :func:`q_points` turns the angles into iota = 1/q points on a reference
  equilibrium (E_r first mapped onto it by :func:`mse_er_at_chords`),
  :func:`fit_q_check` fits a GP to them, and :meth:`QCheck.gp_loglike` scores
  candidate q profiles against its posterior;
* points alone: :meth:`QCheck.points_loglike` scores them against the points;
* direct: :func:`direct_loglike` compares the angles with a candidate's own field
  (no mapping; needs the candidate's equilibrium, not just its q).  For
  candidates with equilibria this is the likelihood; the q points serve the
  q plot and candidates known only by their q profile.

:func:`floor_angles` takes a known q floor of the candidates out of the data.

:func:`plot_q_points` draws the points, the GP band and candidate profiles.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .coords import phi_n_from_q
from .mse import (MSE_MIN_CHORDS, MSEDataUnusable, mse_chi2, mse_chords,
                  mse_equilibrium_orientation, mse_er_terms, mse_exclude,
                  mse_orientation, mse_tan_gamma)
from .sampling import _CHOL_JITTER


#: A chord is left out when its tan(gamma) lies within this many sigma_eff of
#: the pole of lam(tan(gamma)) (:func:`q_points`): lam is far from Gaussian there.
POLE_SIGMAS = 3.0


def _field(eq, ch):
    """``(B, s_p, s_t)``: *eq*'s field at *ch*'s chords and the block's stated orientation."""
    B = eq.field_at(ch["R"], ch["Z"])
    return (B, *mse_orientation(ch, mse_equilibrium_orientation(
        B, ch["R"], ch["Z"], (eq.R_mag, eq.Z_mag))))


def _chords(block, **kw):
    """:func:`bouquet.mse.mse_chords` of *block*, and its ``Er_tor`` (see
    :func:`mse_er_at_chords`) at the active chords, 0 without it."""
    blk = dict(block)
    tor = blk.pop("Er_tor", None)
    ch = mse_chords(blk, **kw)
    return ch, (np.zeros(ch["n_active"]) if tor is None
                else np.asarray(tor, dtype=float)[ch["index"]])


def mse_er_at_chords(eq, block, phi_N, E_r, E_r_err=None, E_tor=None, B_theta=None):
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

    *E_tor* is the part of *E_r* that is v_phi B_theta, computed with the
    outboard-midplane poloidal field *B_theta* [T] (both on *phi_N*).  It is
    moved onto *eq*'s field, ``E_tor |B_Z(out)| / |B_theta|`` (``E_tor /
    |B_theta|`` is interpolated: smooth at the axis), and also given at the
    chords as ``Er_tor``, which :func:`q_points` scales with the poloidal
    field (bouquet's MSE solvers refuse that key).
    """
    def rbz(R, Z):
        return R * eq.field_at(R, Z)[:, 2]

    R, Z = (np.asarray(block[k], dtype=float) for k in ("R", "Z"))
    psi_c = eq.psi_N_at(R, Z)
    phi_c = np.interp(psi_c, np.linspace(0.0, 1.0, eq.rhovn.size), eq.rhovn ** 2,
                      right=np.nan)
    R_out = eq.outboard_R(psi_c)
    rbz_out = rbz(R_out, np.full_like(R_out, eq.Z_mag))
    ratio = rbz(R, Z) / rbz_out

    def at_chords(y):
        return np.interp(phi_c, phi_N, y, left=np.nan, right=np.nan) * ratio

    out = dict(block, Er=at_chords(E_r))
    if E_tor is not None:
        b_theta = np.abs(np.asarray(B_theta, dtype=float))
        v_tor = np.asarray(E_tor, dtype=float) / np.where(b_theta > 0, b_theta, np.nan)
        ok = np.isfinite(v_tor)
        out["Er_tor"] = (np.interp(phi_c, np.asarray(phi_N)[ok], v_tor[ok], right=np.nan)
                         * np.abs(rbz_out / R_out) * ratio)
        out["Er"] = at_chords(np.asarray(E_r) - E_tor) + out["Er_tor"]
    if E_r_err is not None:
        ch, _ = _chords(out, min_chords=1)
        B, sp, st = _field(eq, ch)
        dtg = (mse_tan_gamma(B, dict(ch, Er=np.ones(ch["n_active"])), sp, st)
               - mse_tan_gamma(B, dict(ch, Er=np.zeros(ch["n_active"])), sp, st))
        i = ch["index"]
        sigma = np.array(block["sigma"], dtype=float)
        sigma[i] = np.hypot(sigma[i], dtg * np.abs(at_chords(E_r_err))[i])
        out["sigma"] = sigma
    return out


def _lam(ch, er_tor, B, sp, st):
    """``(lam, d lam / d t, distance of t from the pole in sigma_eff)`` per chord
    of *ch* on the field *B* (:func:`q_points`)."""
    t = ch["tgamma"]
    e, e_t = ch["A5"] * (ch["Er"] - er_tor), ch["A5"] * er_tor
    a = ch["A2"] * st * B[:, 1]
    b = ch["A1"] * B[:, 2]
    c = ch["A3"] * B[:, 0] + ch["A4"] * B[:, 2]
    den = sp * (b - t * c) + e_t
    return ((t * a - e) / den, (sp * (a * b - c * e) + a * e_t) / den ** 2,
            np.abs(den / c) / ch["sigma_eff"])


def _require_chords(ch, where):
    if ch["n_active"] < ch["min_chords"]:
        raise MSEDataUnusable(f"only {ch['n_active']} MSE chords {where}; at least "
                              f"{ch['min_chords']} are required")


def q_points(eq, block, min_chords=MSE_MIN_CHORDS, sigma_sys=0.0):
    """MSE pitch angles as iota = 1/q points on the surfaces of the reference *eq*.

    Relabelled-flux model: the true surfaces are *eq*'s with psi relabelled,
    so on each surface the poloidal field scales by ``lam``, while B_phi,
    Phi_N and rho_tor are unchanged and ``iota = lam / |q_ref|``.  Per chord

    ``lam = (t a - e) / (s_p (b - t c) + e_tor)``, with ``a = A2 s_t B_phi``,
    ``b = A1 B_Z``, ``c = A3 B_R + A4 B_Z``, ``e = A5 (E_R - E_tor)`` and
    ``e_tor = A5 E_tor``,

    on *eq*'s field in the block's stated orientation ``(s_p, s_t)``; E_tor is
    the block's ``Er_tor`` (:func:`mse_er_at_chords`), the v_phi B_theta part
    of E_R, which scales with the poloidal field like B_Z;
    ``sigma_lam = |d lam / d t| sigma_eff``.  ``diota_dR_axis = (d lam / d R) /
    |q_ref|`` [1/m], with the reference field moved along R and q_ref held: the
    iota error per metre of error in the reference axis position, dominant near
    the axis, signed, since one axis offset moves every point together
    (:class:`QCheck` adds it for an axis error).

    Chords at psi_N >= 1 are excluded, and so are chords whose t lies within
    :data:`POLE_SIGMAS` sigma_eff of the pole of lam(t), where ``s_p (b - t c)
    + e_tor = 0`` (B_Z ~ 0 at the chord).  Raises :class:`MSEDataUnusable`
    when fewer than *min_chords* remain, or when the median ``lam`` is
    <= 0 (the stated orientation contradicts the data).

    Returns ``dict(rho, psi_N, iota, sigma_iota, diota_dR_axis, q, sigma_q,
    lam, sigma_lam, R, Z, index, excluded, orientation=(s_p, s_t), er_terms,
    R_axis)``, one entry per used chord (``R_axis``: *eq*'s axis).
    """
    ch, er_tor = _chords(block, min_chords=min_chords, sigma_sys=sigma_sys)
    psi = eq.psi_N_at(ch["R"], ch["Z"])
    ch = mse_exclude(ch, psi >= 1.0, "outside the reference LCFS (psi_N >= 1)")
    psi, er_tor = psi[psi < 1.0], er_tor[psi < 1.0]
    _require_chords(ch, "lie inside the reference LCFS")
    B, sp, st = _field(eq, ch)
    near = _lam(ch, er_tor, B, sp, st)[2] < POLE_SIGMAS
    ch = mse_exclude(ch, near, f"tan(gamma) within {POLE_SIGMAS:g} sigma of the pole of "
                     "lam(tan(gamma)) (B_Z ~ 0 at the chord)")
    psi, er_tor, B = psi[~near], er_tor[~near], B[~near]
    _require_chords(ch, "lie inside the reference LCFS, away from the pole of lam")
    R, Z = ch["R"], ch["Z"]
    lam, dlam_dt, _ = _lam(ch, er_tor, B, sp, st)
    if not np.median(lam) > 0:
        raise MSEDataUnusable(
            f"median lam = {np.median(lam):.3g} <= 0: the stated ip_sign/bt_sign "
            "contradict the pitch angles on the reference equilibrium")
    h = 1e-3   # [m]
    dlam_dR = (_lam(ch, er_tor, eq.field_at(R + h, Z), sp, st)[0]
               - _lam(ch, er_tor, eq.field_at(R - h, Z), sp, st)[0]) / (2 * h)
    sigma_lam = np.abs(dlam_dt) * ch["sigma_eff"]
    x = np.linspace(0.0, 1.0, eq.qpsi.size)
    q_ref = np.abs(np.interp(psi, x, eq.qpsi))
    iota, sigma_iota = lam / q_ref, sigma_lam / q_ref
    return dict(rho=np.interp(psi, x, eq.rhovn), psi_N=psi, iota=iota,
                sigma_iota=sigma_iota, diota_dR_axis=dlam_dR / q_ref, q=1.0 / iota,
                sigma_q=sigma_iota / iota ** 2, lam=lam, sigma_lam=sigma_lam, R=R, Z=Z,
                index=ch["index"], excluded=ch["excluded"], orientation=(sp, st),
                er_terms=mse_er_terms(ch), R_axis=float(eq.R_mag))


def direct_loglike(eq, block, min_chords=MSE_MIN_CHORDS, sigma_sys=0.0):
    """Pitch angles of *block* against *eq*'s own field: no mapping assumption.

    bouquet's forward model on ``eq.field_at``, in the block's stated
    orientation.  ``Er`` is used as given, so map it on *eq* itself
    (:func:`mse_er_at_chords`, which also moves E_tor onto *eq*'s field).
    Returns ``dict(chi2, logL, nu, z, excluded)``, where ``logL = -(chi2 +
    sum log(2 pi sigma_eff^2)) / 2`` and ``nu`` is the number of chords used.
    """
    ch, _ = _chords(block, min_chords=min_chords, sigma_sys=sigma_sys)
    B, sp, st = _field(eq, ch)
    chi2, z = mse_chi2(mse_tan_gamma(B, ch, sp, st), ch)
    return dict(chi2=chi2, logL=-0.5 * (chi2 + np.log(2 * np.pi * ch["sigma_eff"] ** 2).sum()),
                nu=ch["n_active"], z=z, excluded=ch["excluded"])


def floor_angles(eq, block, q_floor, profile=()):
    """A copy of MSE *block* with the angles of a plasma whose q is clipped at *q_floor*.

    Chords whose q point on *eq* (:func:`q_points`; E_r mapped by
    ``mse_er_at_chords(eq, block, *profile)`` when *profile* is given) lies
    below *q_floor* get the tan(gamma) of *eq*'s field scaled to ``lam = q_ref /
    q_floor`` (the relabelled-flux model, E_tor scaled with it), so that their q
    point is the floor; the other angles are kept.  For candidates that cannot
    go below that q by construction: the floor deficit is taken out of the data
    on *eq*'s geometry (best: an MSE-constrained reconstruction), while the
    direct check still tests the candidate's own geometry, e.g. its axis.
    """
    mapped = mse_er_at_chords(eq, block, *profile) if profile else block
    pts = q_points(eq, mapped, min_chords=1)
    low = pts["q"] < q_floor
    ch, er_tor = _chords(mapped, min_chords=1)
    B, sp, st = _field(eq, ch)
    pos = np.searchsorted(ch["index"], pts["index"][low])
    lam = np.ones(ch["n_active"])
    lam[pos] = (pts["lam"] / pts["iota"])[low] / q_floor
    tg = mse_tan_gamma(B * np.c_[lam, np.ones_like(lam), lam],
                       dict(ch, Er=ch["Er"] + (lam - 1.0) * er_tor), sp, st)
    t = np.array(block["tgamma"], dtype=float)
    t[ch["index"][pos]] = tg[pos]
    return dict(block, tgamma=t)


def _chol(K):
    """Lower Cholesky factor of *K* with sampling's jitter rule."""
    n = K.shape[0]
    return np.linalg.cholesky(K + _CHOL_JITTER * n * np.max(np.diag(K)) * np.eye(n))


def _matern52(x1, x2, ell):
    s = np.sqrt(5.0) * np.abs(x1[:, None] - x2[None, :]) / ell
    return (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def _basis(x):
    return np.column_stack([np.ones_like(x), x ** 2])


def _gp_factor(x, y, s, amp, ell, gd):
    """``(L, alpha, Hl, A, beta)`` of the GP with a flat-prior basis (R&W eq. 2.42) and
    noise ``diag(s^2) + gd gd^T``."""
    L = _chol(amp ** 2 * _matern52(x, x, ell) + np.diag(s ** 2) + np.outer(gd, gd))
    alpha = solve_triangular(L, y, lower=True)
    Hl = solve_triangular(L, _basis(x), lower=True)
    A = Hl.T @ Hl
    return L, alpha, Hl, A, np.linalg.solve(A, Hl.T @ alpha)


def _neg_log_ml(th, x, y, s, gd):
    """Negative log marginal likelihood, basis coefficients integrated out (R&W eq. 2.45)."""
    L, alpha, Hl, A, beta = _gp_factor(x, y, s, *np.exp(th), gd)
    return (0.5 * (alpha @ alpha - alpha @ Hl @ beta) + np.log(np.diag(L)).sum()
            + 0.5 * np.linalg.slogdet(A)[1])


def _q_on_rho(psi_N, q):
    """``(rho_tor, |q|)`` of q(psi_N) inside the LCFS, rho_tor from that q (q held at the
    innermost node down to the axis)."""
    x, y = np.asarray(psi_N, dtype=float), np.abs(np.asarray(q, dtype=float))
    if x[0] > 0:
        x, y = np.r_[0.0, x], np.r_[y[0], y]
    inside, phi = phi_n_from_q(x, y)
    return np.sqrt(phi), y[inside]


def _axis_error(R_ref, R_alt, R_cand=None):
    """Largest |R - R_ref| over the finite *R_alt* and *R_cand*; 0 when nothing is known."""
    other = np.atleast_1d(np.asarray(R_alt, dtype=float))
    if R_cand is not None:
        other = np.r_[other, R_cand]
    other = other[np.isfinite(other)]
    if not (np.isfinite(R_ref) and other.size):
        return 0.0
    return float(np.max(np.abs(other - R_ref)))


class QCheck:
    """GP posterior of iota(rho_tor) given q points, and candidate q-profiles' likelihood.

    ``iota = b0 + b1 rho^2 + r(rho)``: a flat prior on (b0, b1), so the slope
    at the axis is zero without a derivative observation; ``r ~ GP(0, amp^2
    Matern-5/2(ell))``; noise ``diag(sigma_iota^2) + dR^2 g g^T`` with ``g =
    diota_dR_axis``, as one axis offset moves every point together (Rasmussen &
    Williams 2006, sec. 2.7).  ``dR`` is the axis error
    (:meth:`axis_error`): the largest distance in R of *R_axis_alt* (other
    reconstructions' axes) and of the candidate's axis from the points'
    reference axis *R_axis*.  Candidates are scored on the posterior at ``m``
    evenly spaced rho over the points' range, ``m = clip(ceil(range / ell) +
    1, 2, n)``: about one per correlation length, where the posterior
    covariance is well conditioned.  *q_floor* records the floor the points were
    clipped to (:func:`fit_q_check`), or NaN.
    """

    def __init__(self, rho, iota, sigma_iota, amp, ell, diota_dR_axis=None,
                 R_axis=np.nan, R_axis_alt=(), q_floor=np.nan):
        self.rho, self.iota, self.sigma_iota = (
            np.asarray(a, dtype=float) for a in (rho, iota, sigma_iota))
        self.diota_dR_axis = (np.zeros_like(self.rho) if diota_dR_axis is None
                              else np.asarray(diota_dR_axis, dtype=float))
        self.amp, self.ell = float(amp), float(ell)
        self.R_axis = float(R_axis)
        self.R_axis_alt = np.atleast_1d(np.asarray(R_axis_alt, dtype=float))
        self.q_floor = float(q_floor)
        m = int(np.clip(np.ceil(np.ptp(self.rho) / self.ell) + 1, 2, self.rho.size))
        self.rho_test = np.linspace(self.rho.min(), self.rho.max(), m)
        self._cache = {}

    def axis_error(self, R_axis=None):
        """Axis error [m] for a candidate whose axis is at *R_axis* (None: unknown)."""
        return _axis_error(self.R_axis, self.R_axis_alt, R_axis)

    def _gp(self, dR):
        """GP factor, posterior at ``rho_test`` (mean, Cholesky) and log|2 pi Sigma| for *dR*."""
        key = round(float(dR), 4)   # one posterior per 0.1 mm of axis error
        if key not in self._cache:
            f = _gp_factor(self.rho, self.iota, self.sigma_iota, self.amp, self.ell,
                           self.diota_dR_axis * key)
            mean, cov = self._predict(f, self.rho_test)
            Lt = _chol(cov)
            self._cache[key] = (f, mean, Lt, 2.0 * np.log(np.diag(Lt)).sum()
                                + mean.size * np.log(2.0 * np.pi))
        return self._cache[key]

    def _predict(self, f, rho):
        L, alpha, Hl, A, beta = f
        V = solve_triangular(L, self.amp ** 2 * _matern52(self.rho, rho, self.ell), lower=True)
        R = _basis(rho) - V.T @ Hl
        return (V.T @ alpha + R @ beta, self.amp ** 2 * _matern52(rho, rho, self.ell)
                - V.T @ V + R @ np.linalg.solve(A, R.T))

    def predict(self, rho, R_axis=None):
        """Posterior mean and covariance of iota at *rho* (axis error as in :meth:`gp_loglike`)."""
        return self._predict(self._gp(self.axis_error(R_axis))[0],
                             np.atleast_1d(np.asarray(rho, dtype=float)))

    def gp_loglike(self, psi_N, q, R_axis=None):
        """``dict(logL, d2, m, axis_error)`` of the candidate profiles q(psi_N), one per row of *q*.

        Each row's iota is placed on rho_tor by its own q.  *R_axis* is the
        candidates' magnetic-axis R (scalar or one per row; None: not known),
        which sets each row's axis error.  ``d2`` is the row's Mahalanobis
        distance from the posterior at ``rho_test`` (chi^2 with ``m`` degrees
        of freedom for a draw from it) and ``logL = -(d2 + log|2 pi Sigma|) /
        2``.  A row with non-finite q gets NaN.
        """
        iota, dR = self._candidates(psi_N, q, self.rho_test, R_axis)
        keys = np.round(dR, 4)
        d2, logdet = np.empty(dR.size), np.empty(dR.size)
        for key in np.unique(keys):
            rows = keys == key
            _, mean, Lt, logdet[rows] = self._gp(key)
            z = solve_triangular(Lt, (iota[rows] - mean).T, lower=True, check_finite=False)
            d2[rows] = np.sum(z ** 2, axis=0)
        return dict(logL=-0.5 * (d2 + logdet), d2=d2, m=self.rho_test.size, axis_error=dR)

    def points_loglike(self, psi_N, q, R_axis=None):
        """``dict(logL, chi2, nu, axis_error)`` of the candidates against the q points alone.

        No GP: each row's iota at the points' rho_tor against the points, with
        covariance ``diag(sigma_iota^2) + dR^2 g g^T`` (axis error as in
        :meth:`gp_loglike`); ``chi2`` over the ``nu`` points and ``logL = -(chi2 +
        log|2 pi Sigma|) / 2``.  A cross-check of :meth:`gp_loglike`, free of the GP
        prior.
        """
        iota, dR = self._candidates(psi_N, q, self.rho, R_axis)
        w, g = self.sigma_iota ** -2.0, self.diota_dR_axis
        r = iota - self.iota
        k = dR ** 2 * np.sum(w * g ** 2)
        chi2 = r ** 2 @ w - dR ** 2 * (r @ (w * g)) ** 2 / (1.0 + k)   # Sherman-Morrison
        return dict(logL=-0.5 * (chi2 + np.sum(np.log(2 * np.pi / w)) + np.log1p(k)), chi2=chi2,
                    nu=self.rho.size, axis_error=dR)

    def _candidates(self, psi_N, q, rho, R_axis):
        """Rows of *q*: iota at *rho* (each placed on rho_tor by its own q; NaN for a
        non-finite row) and the axis error."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        n = q.shape[0]
        psi = np.broadcast_to(np.atleast_2d(np.asarray(psi_N, dtype=float)), q.shape)
        iota = np.full((n, rho.size), np.nan)
        for i, (x, y) in enumerate(zip(psi, q)):
            if np.all(np.isfinite(y)):
                r, qq = _q_on_rho(x, y)
                iota[i] = np.interp(rho, r, 1.0 / qq)
        R_c = [None] * n if R_axis is None else np.broadcast_to(np.asarray(R_axis, float), (n,))
        return iota, np.array([self.axis_error(r) for r in R_c])

    def save(self, path):
        np.savez(path, rho=self.rho, iota=self.iota, sigma_iota=self.sigma_iota,
                 diota_dR_axis=self.diota_dR_axis, amp=self.amp, ell=self.ell,
                 R_axis=self.R_axis, R_axis_alt=self.R_axis_alt, q_floor=self.q_floor)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["rho"], f["iota"], f["sigma_iota"], float(f["amp"]), float(f["ell"]),
                       f["diota_dR_axis"], float(f["R_axis"]), f["R_axis_alt"], float(f["q_floor"]))


def fit_q_check(points, R_axis_alt=(), q_floor=None):
    """:class:`QCheck` of *points* (``rho``, ``iota``, ``sigma_iota``, and
    optionally ``diota_dR_axis``, ``R_axis``; e.g. :func:`q_points`), with
    other reconstructions' axes *R_axis_alt* [m].

    With *q_floor*, points below it are clipped to it (iota <= 1 / q_floor) for
    candidates that cannot go below that q by construction: they are then
    judged only on how far they sit above what the floor forces.

    ``(amp, ell)`` maximise the marginal likelihood, with the axis error of
    *R_axis_alt* alone, by bounded L-BFGS-B from a fixed grid of starts
    (deterministic).  ``amp`` is at least the median point error: when a
    quadratic fits within the errors, it would otherwise collapse, and the
    test would claim structure the data cannot resolve.  ``ell`` lies in
    [0.05, 2 x the rho range].  Non-finite points are dropped; fewer than 3
    left raises ``ValueError``.
    """
    x, y, s = (np.asarray(points[k], dtype=float) for k in ("rho", "iota", "sigma_iota"))
    g = np.asarray(points.get("diota_dR_axis", np.zeros_like(x)), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(g)
    x, y, s, g = x[ok], y[ok], s[ok], g[ok]
    if q_floor is not None:
        y = np.minimum(y, 1.0 / q_floor)
    if x.size < 3:
        raise ValueError(f"fit_q_check: {x.size} finite points; at least 3 are required")
    R_axis = float(points.get("R_axis", np.nan))
    dR = _axis_error(R_axis, R_axis_alt)
    se = np.hypot(s, g * dR)
    bounds = np.log([(np.median(se), 10.0), (0.05, 2.0 * np.ptp(x))])
    starts = [np.clip(np.log([a, ell]), bounds[:, 0], bounds[:, 1])
              for a in np.median(se) * np.array([1.0, 10.0]) for ell in (0.15, 0.4, 1.0)]
    best = min((minimize(_neg_log_ml, th, args=(x, y, s, g * dR), method="L-BFGS-B", bounds=bounds)
                for th in starts), key=lambda r: r.fun)
    return QCheck(x, y, s, *np.exp(best.x), g, R_axis, R_axis_alt,
                  np.nan if q_floor is None else q_floor)


def plot_q_points(points, qc=None, profiles=(), ax=None):
    """q of the MSE points (+-1 sigma) on rho_tor, with *qc*'s posterior band, the points
    where :meth:`QCheck.gp_loglike` compares, and *profiles*.

    *points* as from :func:`q_points`; with *qc* the error bars include its axis error for a
    candidate of unknown axis.  *profiles*: ``(label, psi_N, q)`` each, placed on rho_tor by
    its own q.  Returns the axes.
    """
    import matplotlib.pyplot as plt

    ax = plt.subplots(figsize=(5.0, 3.5))[1] if ax is None else ax
    iota = np.asarray(points["iota"], dtype=float)
    dR = 0.0 if qc is None else qc.axis_error()
    s = np.hypot(points["sigma_iota"], np.asarray(points.get("diota_dR_axis", 0.0)) * dR)
    handles = []                                 # legend: the MSE side first, then the profiles
    if qc is not None:
        rho = np.linspace(qc.rho_test[0], qc.rho_test[-1], 100)
        mean, cov = qc.predict(rho)
        sd = np.sqrt(np.diag(cov))
        handles.append(ax.fill_between(rho, 1.0 / (mean + sd), 1.0 / np.maximum(mean - sd, 1e-3),
                                       alpha=0.3, label=r"GP fit to MSE q ($\pm1\sigma$)"))
    handles.append(ax.errorbar(points["rho"], 1.0 / iota, yerr=s / iota ** 2, fmt="o", ms=3,
                               capsize=2, color="k", label=r"q from MSE ($\pm1\sigma$)", zorder=3))
    if qc is not None:
        handles += ax.plot(qc.rho_test, 1.0 / qc.predict(qc.rho_test)[0], "kd", ms=5, zorder=4,
                           label="where the GP check compares")
    if qc is not None and np.isfinite(qc.q_floor):
        handles.append(ax.axhline(qc.q_floor, color="0.4", ls="--", lw=0.8,
                                  label=f"q floor {qc.q_floor:g}: points below it clipped"))
    for label, psi, q in profiles:
        handles += ax.plot(*_q_on_rho(psi, q), label=label)
    known = s / iota < 0.5   # q known to within half its value
    ax.set(xlabel=r"$\rho_{tor} = \sqrt{\Phi / \Phi_{edge}}$", ylabel="safety factor q",
           xlim=(0.0, 1.0), ylim=(0.0, 1.2 * np.max(1.0 / iota[known], initial=1.0)))
    ax.legend(handles=handles, fontsize="small")
    return ax
