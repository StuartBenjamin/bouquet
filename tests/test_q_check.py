"""bouquet.q_check on an analytic Solov'ev equilibrium (test_geqdsk's)."""

import numpy as np
import pytest

from bouquet.io.geqdsk import GEQDSKEquilibrium, _read_geqdsk, read_geqdsk
from bouquet.mse import (MSEDataUnusable, mse_chords,
                         mse_equilibrium_orientation, mse_orientation,
                         mse_tan_gamma)
from bouquet.q_check import QCheck, fit_q_check, mse_er_at_chords, q_points
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


@pytest.fixture
def scaled(tmp_path):
    """(reference, block): angles from the reference with psi scaled by 1.15."""
    path = str(tmp_path / "g.geqdsk")
    generate_solovev_gfile(path, R0=R0, a=A, kappa=KAPPA, NW=129, NH=129)
    raw = _read_geqdsk(path)
    for k in ("PSIRZ", "SIMAG", "SIBRY"):
        raw[k] = raw[k] * 1.15
    raw["QPSI"] = raw["QPSI"] / 1.15
    true = GEQDSKEquilibrium.from_raw(raw)
    blk = _block()
    blk["Er"] = np.array([-3e4, -1e4, 1e4, 3e4, 1e3, 0.0])
    B = true.field_at(blk["R"], blk["Z"])
    ch = mse_chords(blk)
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(B, blk["R"], blk["Z"], (R0, 0.0)))
    blk["tgamma"] = mse_tan_gamma(B, ch, sp, st)
    return read_geqdsk(path), true, blk


def test_q_points_recover_scaled_flux(scaled):
    ref, true, blk = scaled
    pts = q_points(ref, blk)
    np.testing.assert_allclose(pts["lam"], 1.15, rtol=1e-9)
    psi = true.psi_N_at(pts["R"], pts["Z"])
    x = np.linspace(0, 1, true.qpsi.size)
    np.testing.assert_allclose(pts["q"], np.interp(psi, x, true.qpsi), rtol=1e-9)
    np.testing.assert_allclose(pts["rho"], np.interp(psi, x, true.rhovn), rtol=1e-9)
    assert [i for i, _ in pts["excluded"]] == [5]           # outside the LCFS


def test_q_points_orientation_is_stated(scaled):
    ref, _, blk = scaled
    flipped = q_points(ref.flip_Bt_Ip(copy=True), blk)
    np.testing.assert_allclose(flipped["lam"], q_points(ref, blk)["lam"], rtol=1e-12)
    with pytest.raises(MSEDataUnusable, match="median lam"):
        q_points(ref, dict(blk, ip_sign=-blk["ip_sign"]))


def test_q_points_sigma(scaled):
    ref, _, blk = scaled
    pts = q_points(ref, blk)
    d = 1e-6
    fd = (q_points(ref, dict(blk, tgamma=blk["tgamma"] + d))["lam"]
          - q_points(ref, dict(blk, tgamma=blk["tgamma"] - d))["lam"]) / (2 * d)
    np.testing.assert_allclose(pts["sigma_lam"], np.abs(fd) * 0.004, rtol=1e-5)
    geom = 0.5 * np.abs(q_points(ref, dict(blk, R=blk["R"] + 0.01))["lam"]
                        - q_points(ref, dict(blk, R=blk["R"] - 0.01))["lam"])
    np.testing.assert_allclose(q_points(ref, blk, axis_dR=0.01)["sigma_lam"],
                               np.hypot(pts["sigma_lam"], geom), rtol=1e-9)


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
