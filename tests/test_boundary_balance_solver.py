"""The boundary constraint's calibrated balance on the live solver
(``pytest -m solver``): the partition probe and the draw-yield probe on the
synthetic fixtures (docs/engine.md, "The boundary constraint: a calibrated
balance between the boundary cut and the coil cut").

Three legs; the third, the sigma=0 reproduction under the calibrated total,
is ``tests/test_sigma0_identity_solver.py`` (unchanged: it runs the default,
calibrated, set).

* PARTITION: on each synthetic fixture (the H-mode thermal g-file pair, the
  L-mode pair, the synthetic dd at 1.2 s) the calibration converges, the
  partition ratio measured at the calibrated total is within a factor of 2
  of 1, and the calibrated total lies within one decade of the analytic
  estimate.  What the bars test: the analytic estimate encodes TokaMaker's
  conventions (each constraint row scaled by its weight; coil rows on
  mu0 x the current in A; isoflux rows on psi in Wb/rad).  A change in any of
  them moves the calibrated total away from the estimate by that
  convention's factor (mu0 alone is six decades; a squared-vs-linear weight
  change moves it by orders of magnitude), so the one-decade bar flags a
  convention change while leaving room for the plasma's own response (the
  fixtures sit at 0.3-0.6 of a decade).  The factor-2 bar on the ratio
  checks the stamp is a measurement at the installed total: the calibration
  stops within 20 %, so a ratio outside a factor 2 means the probe and the
  installed set disagree.
* YIELD: a fixed-seed set of 10 draw attempts on each fixture archives at
  least 80 % of the attempts that reach a solve (8 of 10 when none is
  refused for non-physical drawn kinetics, which happens BEFORE any solve
  and says nothing about the constraint; at least 5 must reach a solve).
  The bar is the regression of the symptom that motivated the balance: at a
  fixed total ten times too strong for a reversed-I_p case, 9 of its 10
  draws stopped at the homotopy's iteration cap; the balance archived 8-10
  of 10 on every case it was verified on.  No reversed-I_p fixture ships with
  the repository; that case is verified on the owner's private ladder.

Every solver call runs in a subprocess (``OFT_env`` is a per-process
singleton; tests/_harness.py).
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import traceback

import _harness

_harness.ensure_repo_on_syspath()

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like",
                                     "DIIID_mesh.h5"))
FIXTURES = ("hmode_gfile", "lmode_gfile", "dd_1200")

#: the bars (module docstring): a partition ratio within this factor of 1 ...
PARTITION_FACTOR = 2.0
#: ... a calibrated total within this many decades of the analytic estimate
ANALYTIC_DECADES = 1.0
#: ... and at least this many of YIELD_ATTEMPTS fixed-seed draws archived
YIELD_ATTEMPTS = 10
YIELD_MIN_FRACTION = 0.8
YIELD_MIN_SOLVED = 5
YIELD_SEED = 20261016


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
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


pytestmark = [pytest.mark.solver, pytest.mark.skipif(
    not (os.path.isfile(_MESH) and _oft_importable()),
    reason="needs OFT + the D3D-like mesh")]


def _bouquet(which, work):
    import bouquet as bq
    if which == "hmode_gfile":
        return bq.Bouquet.from_geqdsk(
            os.path.join(_DATA, "g_synthetic.geqdsk"),
            profiles=os.path.join(_DATA, "diiid_profs_synthetic.cdf"),
            mesh=_MESH, header=os.path.join(work, which))
    if which == "lmode_gfile":
        return bq.Bouquet.from_geqdsk(
            os.path.join(_DATA, "g_synthetic_lmode.geqdsk"),
            profiles=os.path.join(_DATA, "ida_synthetic_lmode.cdf"),
            mesh=_MESH, header=os.path.join(work, which))
    dd = os.path.join(work, "dd_synthetic.json")
    with gzip.open(os.path.join(_DATA, "dd_synthetic.json.gz")) as f, \
            open(dd, "wb") as g:
        shutil.copyfileobj(f, g)
    return bq.Bouquet.from_imas(dd, mesh=_MESH, time=1.2,
                                header=os.path.join(work, which))


def _probe(which, work, draws):
    out = dict(which=which)
    path = os.path.join(work, f"balance_{which}.json")
    try:
        b = _bouquet(which, work)
        b.setup_solver()
        b.prepare_baseline()
        out["balance"] = dict(b.baseline.engine["boundary_balance"])
        out["geometry"] = {k: v for k, v in b._geom.record().items()
                           if k.startswith("isoflux_weight")}
        if draws:
            g = b.config.generation
            g.seed = YIELD_SEED
            g.n_inspec_target = None
            b.generate(n=YIELD_ATTEMPTS)
            out["attempts"] = YIELD_ATTEMPTS
            out["archived"] = len(b.diagnostics or [])
            out["rejections"] = [dict(reason=r.get("reason"),
                                      stage=r.get("stage"))
                                 for r in (b.draw_rejections or [])]
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    finally:
        with open(path, "w") as fh:
            json.dump(out, fh, default=float)


def _run(tmp_path_factory, which, draws):
    work = tmp_path_factory.mktemp(f"balance_{which}")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), which, str(work),
         "draws" if draws else "recon"],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    p = os.path.join(str(work), f"balance_{which}.json")
    if not os.path.exists(p):
        pytest.fail(f"{which} balance probe wrote no JSON (rc="
                    f"{proc.returncode}):\n{proc.stdout[-3000:]}\n"
                    f"{proc.stderr[-4000:]}")
    with open(p) as fh:
        d = json.load(fh)
    assert "fatal" not in d, d["fatal"]
    return d


@pytest.mark.parametrize("which", FIXTURES)
def test_the_partition_probe_balances_the_two_cuts(tmp_path_factory, which):
    import math
    d = _run(tmp_path_factory, which, draws=False)
    bal = d["balance"]
    assert bal["mode"] == "calibrated" and not bal.get("fallback"), bal
    assert bal["status"] == "converged", bal
    rho = float(bal["partition_ratio"])
    assert 1.0 / PARTITION_FACTOR <= rho <= PARTITION_FACTOR, bal
    dec = abs(math.log10(float(bal["total"]) / float(bal["analytic_total"])))
    assert dec <= ANALYTIC_DECADES, (dec, bal["total"], bal["analytic_total"])
    # the geometry installs and stamps the calibrated total
    assert d["geometry"]["isoflux_weight_mode"] == "calibrated"
    assert d["geometry"]["isoflux_weight_total"] == pytest.approx(bal["total"])


@pytest.mark.parametrize("which", FIXTURES)
def test_the_balanced_constraint_keeps_the_draw_yield(tmp_path_factory, which):
    d = _run(tmp_path_factory, which, draws=True)
    n_kin = sum(1 for r in d["rejections"]
                if r["reason"] == "kinetics_nonphysical")
    solved = d["attempts"] - n_kin
    assert solved >= YIELD_MIN_SOLVED, (solved, d["rejections"])
    assert d["archived"] >= YIELD_MIN_FRACTION * solved, (
        d["archived"], solved, d["rejections"])


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1], sys.argv[2], sys.argv[3] == "draws")
