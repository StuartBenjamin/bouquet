"""bouquet.archive_checks on the golden slim archive (read-only)."""

import os

import h5py
import numpy as np
import pytest

from bouquet.archive_checks import check_archive, check_archives
from bouquet.coil_check import CoilCheck
from bouquet.q_check import QCheck

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
    assert r0["logL_gp"] == pytest.approx(qc.gp_loglike(psi, q, R_axis=R0)["logL"][0])
    assert r0["chi2_points"] == pytest.approx(qc.points_loglike(psi, q, R_axis=R0)["chi2"][0])
    ref = cc.loglike(names, cur0, {n: 500.0 for n in names})
    assert r0["chi2_coil"] == pytest.approx(ref["chi2"][0]) and r0["nu_coil"] == ref["nu"]
    assert np.isfinite(rows[0]["logL_gp"]) and rows[0]["chi2_coil"] == pytest.approx(0.0)
    assert all(r["status"] == "ok" for r in rows)


def test_parallel_and_status(tmp_path):
    qc, cc, names, _ = _checks()
    qc.save(tmp_path / "qc.npz")
    cc.save(tmp_path / "cc.npz")
    job = dict(path=GOLDEN, q_check=str(tmp_path / "qc.npz"), coil_check=str(tmp_path / "cc.npz"),
               sigma_model=0)
    jobs = [job, dict(job, path=str(tmp_path / "missing.h5"))]
    one = check_archives(jobs, n_workers=1)
    two = check_archives(jobs, n_workers=2)
    assert [r["logL_gp"] for r in one[:-1]] == [r["logL_gp"] for r in two[:-1]]
    assert [r["chi2_coil"] for r in one[:-1]] == [r["chi2_coil"] for r in two[:-1]]
    assert one[-1]["status"] == two[-1]["status"] == "no_archive"
    with pytest.raises(ValueError, match="sigma_model"):
        check_archive(GOLDEN, coil_check=cc)
