"""bouquet.q_check on an analytic Solov'ev equilibrium (test_geqdsk's)."""

import numpy as np
import pytest

from bouquet.io.geqdsk import read_geqdsk
from bouquet.mse import (mse_chords, mse_equilibrium_orientation,
                         mse_orientation, mse_tan_gamma)
from bouquet.q_check import mse_er_at_chords
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
