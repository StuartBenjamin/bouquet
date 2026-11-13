"""bouquet.archive_checks on the golden slim archive (read-only)."""

import os

import h5py
import numpy as np
import pytest

from bouquet.archive_checks import _eqdsk, check_archive, check_archives
from bouquet.coil_check import CoilCheck
from bouquet.mse import mse_chords, mse_equilibrium_orientation, mse_orientation, mse_tan_gamma
from bouquet.q_check import QCheck, direct_loglike, q_points

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "D3Dlike_Hmode_golden_slim.h5")


def _checks():
    with h5py.File(GOLDEN, "r") as hf:
        bl = hf["scan/0/_baseline"]
        names = [n.decode() for n in bl["coil_names"][()]]
        cur = bl["coil_currents"][()]
        fsa = hf["scan/0/0/eq_fsa"]
        psi, q = fsa["psi_N"][()], fsa["q"][()]
    rho = np.sqrt(np.cumsum(q) / np.sum(q))[::40]           # draw 0's own iota on rough rho_tor
    qc = QCheck(rho, 1 / q[::40], np.full(rho.size, 0.02), amp=0.1, ell=0.4,
                diota_dR_axis=np.full(rho.size, 1.0), R_axis=1.70)
    cc = CoilCheck(names, cur, 0.01 * np.abs(cur) + 100.0)
    return qc, cc, names, (psi, q)


def test_rows_match_direct_calls():
    qc, cc, names, (psi, q) = _checks()
    rows = check_archive(GOLDEN, q_check=qc, coil_check=cc, sigma_model={n: 500.0 for n in names})
    with h5py.File(GOLDEN, "r") as hf:
        draws = sorted(int(k) for k in hf["scan/0"] if k.isdigit())
        cur0 = hf["scan/0/0/coil_currents"][()]
        R0 = float(bytes(hf["scan/0/0/eqdsk"][()]).split(b"\n")[2][:16])
    assert [r["draw"] for r in rows] == [-1] + draws and rows[0]["is_baseline"]
    r0 = next(r for r in rows if r["draw"] == 0)
    assert r0["axis_error"] == pytest.approx(abs(R0 - 1.70)) and r0["axis_error"] > 0
    assert r0["chi2_points"] == pytest.approx(qc.points_loglike(psi, q, R_axis=R0)["chi2"][0])
    ref = cc.loglike(names, cur0, {n: 500.0 for n in names})
    assert r0["chi2_coil"] == pytest.approx(ref["chi2"][0]) and r0["nu_coil"] == ref["nu"]
    assert np.isfinite(rows[0]["chi2_points"]) and rows[0]["chi2_coil"] == pytest.approx(0.0)
    assert all(r["status"] == "ok" for r in rows)
    assert np.isnan(rows[1]["logL_gp"])                       # the GP check is optional


def test_parallel_and_status(tmp_path):
    qc, cc, names, _ = _checks()
    qc.save(tmp_path / "qc.npz")
    cc.save(tmp_path / "cc.npz")
    job = dict(path=GOLDEN, q_check=str(tmp_path / "qc.npz"), coil_check=str(tmp_path / "cc.npz"),
               sigma_model=0)
    jobs = [job, dict(job, path=str(tmp_path / "missing.h5"))]
    one = check_archives(jobs, n_workers=1)
    two = check_archives(jobs, n_workers=2)
    assert [r["chi2_points"] for r in one[:-1]] == [r["chi2_points"] for r in two[:-1]]
    assert [r["chi2_coil"] for r in one[:-1]] == [r["chi2_coil"] for r in two[:-1]]
    assert one[-1]["status"] == two[-1]["status"] == "no_archive"
    with pytest.raises(ValueError, match="sigma_model"):
        check_archive(GOLDEN, coil_check=cc)


def _mse_of_baseline():
    """An MSE block with the golden baseline's own angles (E_r = 0), and the baseline."""
    with h5py.File(GOLDEN, "r") as hf:
        eq = _eqdsk(hf["scan/0/_baseline"])
    R = eq.R_mag + np.array([-0.30, -0.12, 0.04, 0.15, 0.27, 0.40])
    Z = eq.Z_mag + np.array([0.05, -0.03, 0.02, -0.04, 0.03, 0.0])
    one = np.ones(R.size)
    blk = dict(R=R, Z=Z, tgamma=0.1 * one, sigma=0.004 * one, weight=one, A1=-0.63 * one,
               A2=0.75 * one, A3=-0.4 * one, A4=0.05 * one, ip_sign=1, bt_sign=1)
    ch = mse_chords(blk)
    B = eq.field_at(R, Z)
    sp, st = mse_orientation(ch, mse_equilibrium_orientation(B, R, Z, (eq.R_mag, eq.Z_mag)))
    blk["tgamma"] = mse_tan_gamma(B, ch, sp, st)
    return blk, eq


def test_direct_axis_offset_and_aligned_diagnostic():
    blk, eq0 = _mse_of_baseline()
    ref = (eq0.R_mag - 0.01, eq0.Z_mag + 0.005)               # an axis 1 cm inboard of the baseline's
    rows = check_archive(GOLDEN, mse=(blk, ()), axis_ref=ref, align_axis=True)
    b = rows[0]
    assert b["chi2_direct"] < 1e-12                            # its own angles
    assert b["axis_offset_R"] == pytest.approx(0.01) and b["axis_offset_Z"] == pytest.approx(-0.005)
    moved = dict(blk, R=blk["R"] + 0.01, Z=blk["Z"] - 0.005)  # the baseline moved onto ref
    assert b["chi2_direct_aligned"] == pytest.approx(direct_loglike(eq0, moved)["chi2"])
    assert b["chi2_direct_aligned"] > 1.0                      # the offset is a misfit, kept apart


def test_direct_alignment_is_optional():
    blk, eq0 = _mse_of_baseline()
    rows = check_archive(GOLDEN, mse=(blk, ()), axis_ref=(eq0.R_mag, eq0.Z_mag))
    assert np.isfinite(rows[1]["axis_offset_R"]) and np.isnan(rows[1]["chi2_direct_aligned"])
    plain = check_archive(GOLDEN, mse=(blk, ()))
    assert np.isnan(plain[1]["axis_offset_R"]) and plain[1]["chi2_direct"] == rows[1]["chi2_direct"]


def test_gp_on_own_surfaces():
    qc, _, _, _ = _checks()
    blk, eq0 = _mse_of_baseline()
    rows = check_archive(GOLDEN, q_check=qc, mse=(blk, ()), gp=True)
    pts = q_points(eq0, blk)
    own = QCheck(pts["rho"], pts["iota"], pts["sigma_iota"], qc.amp, qc.ell)
    ref = own.gp_loglike(np.linspace(0, 1, eq0.qpsi.size), eq0.qpsi)
    assert rows[0]["d2_gp"] == pytest.approx(ref["d2"][0]) and rows[0]["m_gp"] == ref["m"]
    assert rows[0]["d2_gp"] < rows[0]["m_gp"]                  # its own angles: consistent


def test_gp_needs_the_mse_block():
    qc, _, _, _ = _checks()
    alone = check_archive(GOLDEN, q_check=qc, gp=True)
    assert np.isnan(alone[0]["d2_gp"]) and "gp: needs mse" in alone[0]["status"]
