"""The boundary constraint on the live solver (``pytest -m solver``): the
report-only partition stamp and the draw yield on the synthetic fixtures at
the default constant total (docs/engine.md, "The boundary constraint: a
constant total inside a measured window").

Three legs; the third, the sigma=0 reproduction at the default total, is
``tests/test_sigma0_identity_solver.py`` (unchanged).

* PARTITION: on each synthetic fixture (the H-mode thermal g-file pair, the
  L-mode pair, the synthetic dd at 1.2 s) the reconstruction installs the
  constant total (mode "constant", 2e5) and stamps the partition it
  measured there, which lies inside the report band [1/3, 3] at a cost of at
  most 2 probe solves.  What the band tests: the partition encodes
  TokaMaker's conventions (each constraint row scaled by its weight; coil
  rows on mu0 x the current in A; isoflux rows on psi in Wb/rad) relative to
  the in-spec rule's two cuts.  A change of any of them moves the partition
  at a fixed total by that convention's factor (mu0 alone is six decades; a
  squared-vs-linear weight change moves it by orders of magnitude), far
  outside the band, while the band holds the measured spread of the cases
  (2026-10: 0.8-2.6 at 2e5 on six equilibria).
* YIELD: a fixed-seed set of 10 draw attempts on each fixture archives at
  least 80 % of the attempts that reach a solve (8 of 10 when none is
  refused for non-physical drawn kinetics, which happens BEFORE any solve
  and says nothing about the constraint; at least 5 must reach a solve).
  The bar is the regression of the symptom the homotopy's #VSC floor fixed:
  at 2e5, a reversed-I_p case with a near-zero vertical-control coil had 9
  of its 10 draws stop at the homotopy's iteration cap, and 9 of 10 archived
  with the floor.  No fixture with a vertical-control coil below the 10 kA
  floor ships with the repository (the synthetic pairs carry more than
  40 kA): the floor's arithmetic on such a coil set is the fast test
  ``tests/test_homotopy_vsc_floor.py::test_the_vsc_bound_uses_the_floor_for_a_near_zero_coil``,
  and the sub-10 kA yield is verified on the owner's private ladder.

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

#: the bars (module docstring): the stamped partition inside the report band
#: at a cost of at most this many probe solves ...
PARTITION_MAX_SOLVES = 2
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
                                      stage=r.get("stage"),
                                      info=r.get("info"))
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
def test_the_stamped_partition_at_the_default_total_is_in_band(
        tmp_path_factory, which):
    from bouquet.boundary_balance import PARTITION_BAND
    from bouquet.solver_geometry import ISOFLUX_WEIGHT_TOTAL
    d = _run(tmp_path_factory, which, draws=False)
    bal = d["balance"]
    assert bal["mode"] == "constant", bal
    assert d["geometry"]["isoflux_weight_mode"] == "constant"
    assert d["geometry"]["isoflux_weight_total"] == pytest.approx(
        ISOFLUX_WEIGHT_TOTAL)
    part = bal["partition"]
    assert part["status"] == "measured", part
    assert part["report_only"] is True
    assert part["total_nominal"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL)
    assert part["n_solves"] <= PARTITION_MAX_SOLVES, part
    lo, hi = PARTITION_BAND
    assert lo <= float(part["partition"]) <= hi, part
    assert part["in_band"] is True


@pytest.mark.parametrize("which", FIXTURES)
def test_the_default_total_keeps_the_draw_yield(tmp_path_factory, which):
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
