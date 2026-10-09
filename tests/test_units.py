"""bouquet.units: unit building, records, failures through the queue (solver-free), and two real units."""
import gzip
import json
import os
import shutil

import h5py
import pytest

from bouquet import units, workqueue as wq
from bouquet.config import BouquetConfig, ImasSource, SolverConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))


def _config(ids_path=_DD):
    return BouquetConfig(source=ImasSource(ids_path=ids_path, time=1.013), solver=SolverConfig(mesh_path=_MESH),
                         output_header="x")


def test_make_units():
    cfg = _config()
    us = units.make_units(cfg, [{"source.time": 1.0, "generation.scan_key": 1000},
                                {"source.time": 1.1, "generation.scan_key": 1100}], ["a_1000", "a_1100"])
    assert [u["id"] for u in us] == ["a_1000", "a_1100"]
    c = [BouquetConfig.from_dict(u["config"]) for u in us]
    assert (c[0].source.time, c[1].generation.scan_key) == (1.0, 1100)
    assert c[0].generation.seed == units.unit_seed(cfg.generation.seed, "a_1000") != c[1].generation.seed
    json.dumps(us)
    with pytest.raises(KeyError):
        units.make_units(cfg, [{"source.no_such_field": 1}], ["b"])


def test_archive_counts(tmp_path):
    h5 = tmp_path / "a.h5"
    with h5py.File(h5, "w") as f:
        for c, sel in ((0, True), (1, False), (2, True)):
            g = f.create_group(f"scan/1000/{c}")
            g.attrs["selected"] = sel
            g.attrs["profile_coord"] = "phi_n"
        f.create_group("scan/1000/_baseline")
    assert units.archive_counts(str(h5)) == ([0, 1, 2], [0, 2], "phi_n")


def test_failing_units_get_failed_records(tmp_path):
    q = str(tmp_path / "q")
    wq.Queue(q)
    wq.write_json(os.path.join(q, "config.json"), dict(MAX_CONSECUTIVE_FAILURES=10, STALE_S=5))
    bad = units.make_units(_config(ids_path=str(tmp_path / "missing.json")), [{}, {}], ["u1", "u2"])
    assert units.add_units(q, bad) == 2
    out = str(tmp_path / "out")
    reasons = units.work(q, out, workers=2, idle_exit=0)
    assert set(reasons.values()) <= {"drained", "idle"}
    for uid in ("u1", "u2"):
        rec = json.load(open(os.path.join(out, f"{uid}.json")))
        assert rec["status"] == "failed" and rec["archive"] is None and rec["error"]["signature"]
        assert not os.path.exists(os.path.join(out, f"{uid}.h5"))
    s = wq.Queue(q).status()
    assert (s["failed"], s["open"]) == (2, 0)


def _oft_ready():
    try:
        from test_phi_solver_draws import _oft_phi_ready
        return _oft_phi_ready()
    except Exception:
        return False


@pytest.mark.solver
@pytest.mark.skipif(not (os.path.isfile(_MESH) and _oft_ready()), reason="needs OFT")
def test_two_units_end_to_end(tmp_path):
    dd = str(tmp_path / "dd.json")
    with gzip.open(_DD, "rb") as fi, open(dd, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    import bouquet as bq
    run = bq.Bouquet.from_imas(dd, mesh=_MESH, time=1.013, n_draws=1, ida_path=_IDA, LCFS_geqdsk=_GEQ,
                               impurity_Z=6.0, solve_method="swb")
    q, out = str(tmp_path / "q"), str(tmp_path / "out")
    units.add_units(q, units.make_units(run.config, [{"generation.scan_key": 1013}] * 2, ["s_a", "s_b"]))
    units.work(q, out, workers=2, idle_exit=0)
    seeds = set()
    for uid in ("s_a", "s_b"):
        rec = json.load(open(os.path.join(out, f"{uid}.json")))
        assert rec["status"] == "done" and rec["n_draws"] == 1 and os.path.isfile(os.path.join(out, rec["archive"]))
        assert rec["threads"] and all(p["num_threads"] == 1 for p in rec["threads"])
        seeds.add(rec["seed"])
        assert bq.BouquetArchive(os.path.join(out, rec["archive"])).scan_keys
    assert len(seeds) == 2
    # each worker ran its units in its own directory, on its own copy of the mesh
    wdirs = [d for d in os.listdir(os.path.join(out, "_work")) if os.path.isfile(
        os.path.join(out, "_work", d, os.path.basename(_MESH)))]
    assert wdirs
