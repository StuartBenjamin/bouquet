"""bouquet.q_check on an analytic Solov'ev equilibrium (test_geqdsk's)."""

import numpy as np
import pytest

from bouquet.io.geqdsk import GEQDSKEquilibrium, _read_geqdsk, read_geqdsk
from bouquet.mse import (MSEDataUnusable, mse_chords,
                         mse_equilibrium_orientation, mse_orientation,
                         mse_tan_gamma)
from bouquet.q_check import (QCheck, _matern52, fit_q_check, mse_er_at_chords, mse_loglike,
                             plot_q_points, q_points)
from test_geqdsk import generate_solovev_gfile

R0, A, KAPPA, K = 1.7, 0.5, 1.4, 6e3


@pytest.fixture
def solovev(tmp_path):
    """psi_N = ((R - R0)^2 + kappa^2 Z^2) / a^2, q = q0 (1 + psi_N / 2)."""
    path = str(tmp_path / "g.geqdsk")
    generate_solovev_gfile(path, R0=R0, a=A, kappa=KAPPA, NW=129, NH=129)
    return read_geqdsk(path)


def _profile(err=False):
    """Phi_e' = -K (1 + psi_N): E_r,out = 2 K (1 + psi) sqrt(psi) / a on Phi_N."""
    psi = np.linspace(0, 1, 401) ** 2
    phi = (psi + psi ** 2 / 4) / 1.25
    er = 2 * K * (1 + psi) * np.sqrt(psi) / A
    return (phi, er, 0.2 * er + 2e3) if err else (phi, er)


def _block():
    R = np.array([1.35, 1.52, 1.83, 2.04, 1.72, 2.30])
    Z = np.array([0.21, -0.07, 0.15, -0.18, 0.01, 0.00])
    one = np.ones(R.size)
    return dict(R=R, Z=Z, tgamma=0.1 * one, sigma=0.004 * one, weight=one,
                A1=-0.63 * one, A2=0.75 * one, A3=-0.4 * one, A4=0.05 * one,
                A5=-3.5e-7 * one, ip_sign=1, bt_sign=1)


def test_er_is_flux_function_field(solovev):
    blk = _block()
    out = mse_er_at_chords(solovev, blk, *_profile())
    psi = ((blk["R"] - R0) ** 2 + KAPPA ** 2 * blk["Z"] ** 2) / A ** 2
    expected = K * (1 + psi) * 2 * (blk["R"] - R0) / A ** 2   # E_R = -Phi_e' dpsi_N/dR
    np.testing.assert_allclose(out["Er"][:5], expected[:5], rtol=1e-3, atol=1.0)
    assert np.isnan(out["Er"][5])                             # outside the LCFS
    assert out["sigma"] is blk["sigma"] and "Er" not in blk
    ch = mse_chords(out)
    assert ch["excluded"][0][0] == 5 and "E_r" in ch["excluded"][0][1]


def test_er_err_inflates_sigma_by_forward_model(solovev):
    blk = _block()
    out = mse_er_at_chords(solovev, blk, *_profile(err=True))
    ch = mse_chords(out)
    B = solovev.field_at(ch["R"], ch["Z"])
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(
        B, ch["R"], ch["Z"], (solovev.R_mag, solovev.Z_mag)))
    dtg = mse_tan_gamma(B, dict(ch, Er=ch["Er"] + 1.0), sp, st) - mse_tan_gamma(B, ch, sp, st)
    R, Z = blk["R"][:5], blk["Z"][:5]
    psi = ((R - R0) ** 2 + KAPPA ** 2 * Z ** 2) / A ** 2
    s_er = (0.2 * 2 * K * (1 + psi) * np.sqrt(psi) / A + 2e3) * np.abs(R - R0) / (A * np.sqrt(psi))
    np.testing.assert_allclose(out["sigma"][:5], np.hypot(0.004, dtg * s_er), rtol=1e-3)
    assert out["sigma"][5] == 0.004 and np.all(blk["sigma"] == 0.004)


def _b_out(eq):
    """B_Z at the outboard midplane of *eq*'s surfaces on _profile's grid."""
    R = eq.outboard_R(np.linspace(0, 1, 401) ** 2)
    return eq.field_at(R, np.full_like(R, eq.Z_mag))[:, 2]


def test_er_tor_moves_to_the_equilibrium_field(solovev):
    phi, er = _profile()
    blk = _block()
    out = mse_er_at_chords(solovev, blk, phi, er, None, 0.6 * er, 2 * _b_out(solovev))
    full = mse_er_at_chords(solovev, blk, phi, er)["Er"]     # E_tor was made with twice the field
    np.testing.assert_allclose(out["Er_tor"][:5], 0.3 * full[:5], rtol=1e-3)
    np.testing.assert_allclose(out["Er"][:5], 0.7 * full[:5], rtol=1e-3)
    assert np.isnan(out["Er_tor"][5]) and mse_loglike(solovev, out)["nu"] == 5
    with pytest.raises(MSEDataUnusable, match="unknown key"):
        mse_chords(out)                                      # bouquet's MSE solvers refuse Er_tor


def _scaled_eq(path, c):
    """The g-file at *path* with psi scaled by *c* (B_pol x c, q / c, same surfaces)."""
    raw = _read_geqdsk(path)
    for k in ("PSIRZ", "SIMAG", "SIBRY"):
        raw[k] = raw[k] * c
    raw["QPSI"] = raw["QPSI"] / c
    return GEQDSKEquilibrium.from_raw(raw)


@pytest.fixture
def scaled(tmp_path):
    """(reference, true, block): angles from the reference with psi scaled by 1.15."""
    path = str(tmp_path / "g.geqdsk")
    generate_solovev_gfile(path, R0=R0, a=A, kappa=KAPPA, NW=129, NH=129)
    true = _scaled_eq(path, 1.15)
    blk = _block()
    blk["Er"] = np.array([-3e4, -1e4, 1e4, 3e4, 1e3, 0.0])
    B = true.field_at(blk["R"], blk["Z"])
    ch = mse_chords(blk)
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(B, blk["R"], blk["Z"], (R0, 0.0)))
    blk["tgamma"] = mse_tan_gamma(B, ch, sp, st)
    return read_geqdsk(path), true, blk, path


def test_q_points_recover_scaled_flux(scaled):
    ref, true, blk, _ = scaled
    pts = q_points(ref, blk)
    np.testing.assert_allclose(pts["lam"], 1.15, rtol=1e-9)
    psi = true.psi_N_at(pts["R"], pts["Z"])
    x = np.linspace(0, 1, true.qpsi.size)
    np.testing.assert_allclose(pts["q"], np.interp(psi, x, true.qpsi), rtol=1e-9)
    np.testing.assert_allclose(pts["rho"], np.interp(psi, x, true.rhovn), rtol=1e-9)
    assert [i for i, _ in pts["excluded"]] == [5]           # outside the LCFS


def test_q_points_scale_the_toroidal_er(scaled):
    ref, true, blk, _ = scaled
    prof = (*_profile(), None, 0.8 * _profile()[1], _b_out(ref))   # E_tor made with ref's field
    base = {k: v for k, v in blk.items() if k != "Er"}
    on_true = mse_er_at_chords(true, base, *prof)                   # E_tor x 1.15 on the true field
    ch = mse_chords({k: v for k, v in on_true.items() if k != "Er_tor"})
    B = true.field_at(ch["R"], ch["Z"])
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(B, ch["R"], ch["Z"], (R0, 0.0)))
    tg = blk["tgamma"].copy()
    tg[ch["index"]] = mse_tan_gamma(B, ch, sp, st)
    pts = q_points(ref, mse_er_at_chords(ref, dict(base, tgamma=tg), *prof))
    np.testing.assert_allclose(pts["lam"], 1.15, rtol=1e-9)
    held = q_points(ref, mse_er_at_chords(ref, dict(base, tgamma=tg), *prof[:2]))
    assert np.max(np.abs(held["lam"] - 1.15)) > 1e-3                # E_r held fixed misses it


def _with_pole_chord(true, blk):
    """*blk* plus chord 6 almost straight above the axis (B_Z ~ 0), angles from *true*."""
    extra = dict(R=R0 + 0.001, Z=0.15, Er=0.0)
    blk = {k: np.r_[v, extra.get(k, v[0])] if np.ndim(v) else v for k, v in blk.items()}
    ch = mse_chords(blk)
    B = true.field_at(ch["R"], ch["Z"])
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(B, ch["R"], ch["Z"], (R0, 0.0)))
    blk["tgamma"][ch["index"]] = mse_tan_gamma(B, ch, sp, st)
    return blk


def test_q_points_drop_chords_near_the_pole(scaled):
    ref, true, blk, _ = scaled
    blk = _with_pole_chord(true, blk)
    pts = q_points(ref, blk)
    assert 6 not in pts["index"] and "pole" in dict(pts["excluded"])[6]
    np.testing.assert_allclose(pts["lam"], 1.15, rtol=1e-9)
    sharp = q_points(ref, dict(blk, sigma=np.r_[blk["sigma"][:6], 1e-6]))
    assert 6 in sharp["index"]                               # a precise angle: the pole is far in sigma
    np.testing.assert_allclose(sharp["lam"], 1.15, rtol=1e-9)


def test_q_points_count_chords_away_from_the_pole(scaled):
    ref, true, blk, _ = scaled
    with pytest.raises(MSEDataUnusable, match="away from the pole"):
        q_points(ref, _with_pole_chord(true, blk), min_chords=6)   # 6 inside the LCFS, 5 usable


def test_q_points_orientation_is_stated(scaled):
    ref, _, blk, _ = scaled
    flipped = q_points(ref.flip_Bt_Ip(copy=True), blk)
    np.testing.assert_allclose(flipped["lam"], q_points(ref, blk)["lam"], rtol=1e-12)
    with pytest.raises(MSEDataUnusable, match="median lam"):
        q_points(ref, dict(blk, ip_sign=-blk["ip_sign"]))


def test_q_points_sigma(scaled):
    ref, _, blk, _ = scaled
    pts = q_points(ref, blk)
    d = 1e-6
    fd = (q_points(ref, dict(blk, tgamma=blk["tgamma"] + d))["lam"]
          - q_points(ref, dict(blk, tgamma=blk["tgamma"] - d))["lam"]) / (2 * d)
    np.testing.assert_allclose(pts["sigma_lam"], np.abs(fd) * 0.004, rtol=1e-5)
    h = 1e-4                                                 # field moved along R
    dlam = (q_points(ref, dict(blk, R=blk["R"] + h))["lam"]
            - q_points(ref, dict(blk, R=blk["R"] - h))["lam"]) / (2 * h)
    np.testing.assert_allclose(pts["diota_daxis"], dlam * pts["iota"] / pts["lam"], rtol=1e-2)
    assert np.ptp(np.sign(pts["diota_daxis"])) == 2          # signed: opposite across the axis
    assert pts["R_axis"] == ref.R_mag


PSI = np.linspace(0.001, 0.999, 300)


def _q(psi):
    return 1.0 + 2.0 * psi ** 2


def _points():
    """iota = 1/q on rho_tor for q = 1 + 2 psi_N^2 (Phi_N from that q), sigma 0.01."""
    psi = np.linspace(0.02, 0.9, 15)
    rho = np.sqrt((psi + 2 * psi ** 3 / 3) / (5 / 3))
    return dict(rho=rho, iota=1 / _q(psi), sigma_iota=np.full(psi.size, 0.01))


def test_gp_prefers_the_true_profile():
    res = fit_q_check(_points()).loglike(PSI, [_q(PSI), 1.05 * _q(PSI), 0.95 * _q(PSI)])
    assert res["logL"][0] > max(res["logL"][1:])
    assert res["d2"][0] < res["m"]


def test_gp_axis_error_per_candidate():
    p = _points()
    g = np.linspace(2.0, 0.1, p["rho"].size)                 # largest near the axis
    qc = fit_q_check(dict(p, diota_daxis=g, R_axis=1.70), R_axis_alt=[1.71])
    rows = [_q(PSI), 1.02 * _q(PSI)]
    res = qc.loglike(PSI, rows, R_axis=[1.69, 1.74])
    np.testing.assert_allclose(res["axis_dR"], [0.01, 0.04])  # max |R - 1.70| over EFIT02-like and candidate
    for i, dR in enumerate((0.01, 0.04)):
        plain = QCheck(qc.rho, qc.iota, qc.sigma_iota, qc.amp, qc.ell, g, 1.70, [1.70 + dR])
        np.testing.assert_allclose(res["logL"][i], plain.loglike(PSI, rows[i])["logL"][0], rtol=1e-12)
    assert qc.loglike(PSI, rows[0])["axis_dR"][0] == pytest.approx(0.01)
    with_nan = fit_q_check(dict(p, diota_daxis=g, R_axis=1.70), R_axis_alt=[1.71, np.nan])
    np.testing.assert_allclose(with_nan.loglike(PSI, rows, R_axis=[1.69, np.nan])["axis_dR"], 0.01)


def test_points_loglike():
    p = _points()
    g = np.linspace(1.0, -0.5, p["rho"].size)
    qc = QCheck(p["rho"], p["iota"], p["sigma_iota"], 0.1, 0.4, g, R_axis=1.70, R_axis_alt=[1.72])
    res = qc.points_loglike(PSI, [_q(PSI), 1.05 * _q(PSI)], R_axis=[1.70, 1.75])
    assert res["chi2"][0] < 1e-3 and res["nu"] == p["rho"].size
    for i, (f, dR) in enumerate(((1.0, 0.02), (1.05, 0.05))):
        cov = np.diag(p["sigma_iota"] ** 2) + dR ** 2 * np.outer(g, g)   # one offset, every point
        r = p["iota"] * (1 / f - 1)
        np.testing.assert_allclose(res["chi2"][i], r @ np.linalg.solve(cov, r), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(res["logL"][i], -0.5 * (res["chi2"][i]
                                   + np.linalg.slogdet(2 * np.pi * cov)[1]), rtol=1e-12)


def test_gp_noise_is_rank_one():
    p = _points()
    g = np.linspace(1.0, -0.5, p["rho"].size)
    qc = QCheck(p["rho"], p["iota"], p["sigma_iota"], 0.05, 0.4, g, R_axis=1.70, R_axis_alt=[1.73])
    L = qc._gp(qc.axis_error())[0][0]
    K = (0.05 ** 2 * _matern52(p["rho"], p["rho"], 0.4) + np.diag(p["sigma_iota"] ** 2)
         + 0.03 ** 2 * np.outer(g, g))
    np.testing.assert_allclose(L @ L.T, K, rtol=1e-6, atol=1e-12)


def test_gp_batch_and_save(tmp_path):
    qc = fit_q_check(_points())
    rows = np.array([_q(PSI) * f for f in (0.97, 1.0, 1.02)])
    rows[2, 7] = np.nan
    res = qc.loglike(PSI, rows)
    for i in range(2):
        np.testing.assert_allclose(res["logL"][i], qc.loglike(PSI, rows[i])["logL"][0], rtol=1e-12)
    assert np.isnan(res["logL"][2])
    qc.save(tmp_path / "qc.npz")
    np.testing.assert_array_equal(QCheck.load(tmp_path / "qc.npz").loglike(PSI, rows)["logL"],
                                  res["logL"])


def test_direct_and_gp_agree(scaled):
    ref, true, blk, path = scaled
    assert mse_loglike(true, blk)["chi2"] < 1e-12
    cands = [_scaled_eq(path, c) for c in (1.10, 1.15, 1.20)]
    chi2 = [mse_loglike(eq, blk)["chi2"] for eq in cands]
    qc = fit_q_check(q_points(ref, blk))
    logL = qc.loglike(np.linspace(0, 1, ref.qpsi.size), [eq.qpsi for eq in cands])["logL"]
    assert np.argmin(chi2) == 1 and np.argmax(logL) == 1


def test_plot_q_points(scaled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref, true, blk, _ = scaled
    pts = q_points(ref, blk)
    qc = fit_q_check(pts, R_axis_alt=[ref.R_mag + 0.01])
    ax = plot_q_points(pts, qc, profiles=[("true", np.linspace(0, 1, true.qpsi.size), true.qpsi)])
    bars = ax.containers[0]
    np.testing.assert_allclose(bars.lines[0].get_ydata(), pts["q"])
    half = [abs(seg[1, 1] - seg[0, 1]) / 2 for seg in bars.lines[2][0].get_segments()]
    s = np.hypot(pts["sigma_iota"], pts["diota_daxis"] * 0.01)   # the axis error is in the bars
    np.testing.assert_allclose(half, s / pts["iota"] ** 2, rtol=1e-6)
    assert {"MSE", "true", r"GP $\pm1\sigma$"} <= set(ax.get_legend_handles_labels()[1])
    plt.close(ax.figure)
