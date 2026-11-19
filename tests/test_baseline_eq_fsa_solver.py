"""The baseline's live-equilibrium FSA block (``_baseline/eq_fsa``), archived as each
draw's: on the warm-start route (coil handling on) and without it (``coil_drift =
None``), it is the block of the archived baseline state (``tokamaker_state``),
with the draws' keys.  One seeded draw per route; synthetic inputs only.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest  # noqa: E402

_DATA = os.path.join(_harness.REPO_ROOT, "tests", "data")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_PROF = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_MESH = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like",
                     "DIIID_mesh.h5")


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PROF, _MESH))
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not (_files_ok and _oft_importable()),
                       reason="needs OFT + the synthetic fixtures"),
]


def child(out, coil_off):
    """Generate one seeded draw; record the baseline block against the block
    recomputed from the archived baseline state."""
    import h5py
    import numpy as np
    import bouquet as bq
    from bouquet.archive import BaselineView, BouquetArchive
    from bouquet.physics import capture_equilibrium_fsa
    _harness.assert_bouquet_is_repo_local()
    rec = {}
    try:
        header = os.path.splitext(out)[0]
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PROF, mesh=_MESH,
                                   nthreads=1, n_draws=1, header=header,
                                   reconstruction_engine="unified")
        gc = b.config.generation
        gc.seed, gc.n_inspec_target = 12345, None
        gc.write_ifile = gc.store_tokamaker_state = True
        if coil_off:
            gc.coil_drift = None
        b.setup_solver()
        b.prepare_baseline()
        b.generate()
        with h5py.File(header + ".h5", "r") as hf:
            key = sorted(hf["scan"].keys())[0]
            grp = hf["scan/" + key]
            draw = sorted(k for k in grp.keys() if k.isdigit())[0]
            rec["draw_keys"] = sorted(grp[draw]["eq_fsa"].keys())
            fsa = grp["_baseline"].get("eq_fsa")
            rec["keys"] = None if fsa is None else sorted(fsa.keys())
            stored = None if fsa is None else {k: fsa[k][()] for k in fsa}
        if stored is not None:
            gen = b.config.generation
            again = capture_equilibrium_fsa(
                BaselineView(BouquetArchive(header + ".h5"), key).load_state(),
                npsi=int(gen.capture_npsi), psi_pad=float(b.config.source.psi_pad),
                exact_inv_R2=bool(gen.capture_exact_inv_R2))
            rec["max_rel_dq"] = float(np.max(np.abs(stored["q"] - again["q"])
                                             / np.abs(again["q"])))
            rec["same_grid"] = bool(np.array_equal(stored["psi_N"], again["psi_N"]))
    except Exception as exc:                       # recorded, never lost
        import traceback
        rec["error"] = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()[-3000:]
    with open(out, "w") as fh:
        json.dump(rec, fh)


def _run(work, coil_off):
    out = str(work / f"bfsa_{int(coil_off)}.json")
    pr = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", out, str(int(coil_off))],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out + ".log", "w") as fh:
        fh.write(pr.stdout)
    assert os.path.isfile(out), pr.stdout[-4000:]
    with open(out) as fh:
        return json.load(fh)


@pytest.mark.parametrize("coil_off", [False, True], ids=["warm_start", "coil_drift_none"])
def test_baseline_eq_fsa_is_its_archived_state(tmp_path, coil_off):
    r = _run(tmp_path, coil_off)
    assert "error" not in r, r.get("error")
    assert r["keys"] == r["draw_keys"], r
    assert r["same_grid"] and r["max_rel_dq"] < 1e-8, r


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1], bool(int(sys.argv[i + 2])))
