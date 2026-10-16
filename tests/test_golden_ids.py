"""What the IDS-route golden is (``tests/golden/D3Dlike_ids_golden_slim.h5``).

The h5 golden of the production IDS route: the synthetic dd's 1.2 s slice
(``tests/data/dd_synthetic.json.gz``) run IDA-hybrid with its paired
IDA-layout profile file (``tests/data/diiid_profs_synthetic.cdf``) on the
unified engine, every setting at its default (20 attempts, seed 12345, one
thread), made by ``tests/golden/regenerate_golden_run.py --golden ids`` and
slimmed by ``tests/golden/make_golden_fixture.py --golden ids``.

Its value-level checks are the h5 golden tests, run on both goldens
(``tests/test_golden_bouquet.py``: per-draw scalars, coil currents,
X-points, boundary deviations, the retained geqdsks, the filters and
selection replayed on the archive) and the seeded draw stream
(``tests/test_rng_reproducibility.py``, ``ids_rng_stream_manifest.json``).
These fast checks say what the file is, so that a pass of those is read as
a pass of the production IDS route and of nothing else.
"""
import json
import os
import sys

import h5py
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
_SLIM = os.path.join(_GOLDEN_DIR, "D3Dlike_ids_golden_slim.h5")
_MANIFEST = os.path.join(_GOLDEN_DIR, "ids_golden_manifest.json")
_RNG = os.path.join(_GOLDEN_DIR, "ids_rng_stream_manifest.json")

pytestmark = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in (_SLIM, _MANIFEST, _RNG)),
    reason="IDS golden not built; run tests/golden/make_golden_fixture.py "
           "--golden ids")


def _recipe():
    sys.path.insert(0, _GOLDEN_DIR)
    import regenerate_golden_run as rgr
    return rgr


@pytest.fixture(scope="module")
def cfg():
    from bouquet.utils import load_config
    return load_config(_SLIM, scan_key=0)


@pytest.fixture(scope="module")
def manifest():
    with open(_MANIFEST) as fh:
        return json.load(fh)


def test_it_is_the_ida_hybrid_slice_of_the_synthetic_dd(cfg):
    from bouquet.config import ImasSource
    rgr = _recipe()
    src = cfg.source
    assert isinstance(src, ImasSource), type(src)
    assert src.ids_path == rgr.DD
    assert src.ida_path == rgr.PROFILES
    assert cfg.uncertainty.ida_path == rgr.PROFILES
    assert src.time == pytest.approx(rgr.IDS_TIME)
    assert cfg.solver.mesh_path == rgr.MESH
    gc = cfg.generation
    assert gc.kinetic_source == "ida_hybrid"
    # the IDA file's own Z_eff and n_i (the default, one source per run)
    assert src.ni_source == "auto"
    assert not src.zeff_from_fuse


def test_it_is_the_production_route_at_the_recipe_identity(cfg, manifest):
    rgr = _recipe()
    gc = cfg.generation
    assert gc.reconstruction_engine == "unified"
    assert gc.jbs_self_consistent is True
    assert gc.n_equils == rgr.N_DRAWS
    assert gc.seed == rgr.SEED
    assert gc.n_inspec_target is None          # attempts, no until-N
    assert cfg.solver.nthreads == rgr.NTHREADS
    assert manifest["golden"] == "ids"
    assert manifest["provenance"]["generator_args"]["golden"] == "ids"
    assert manifest["provenance"]["generator_args"]["jphi_archival"] == \
        "input"


def test_it_was_judged_on_the_magnetics(manifest):
    """The in-spec rule of the run is the magnetics (no opt-in band), and
    every archived draw's band flag is therefore True: its ``in_spec`` is
    the coil verdict."""
    with h5py.File(_SLIM, "r") as hf:
        g = hf["scan/0"]
        rule = g.attrs.get("inspec_rule")
        rule = rule.decode() if isinstance(rule, bytes) else rule
        assert rule == "magnetics", rule
        draws = [k for k in g if k.isdigit()]
        assert draws
        for k in draws:
            assert bool(g[k].attrs["passes_draw_band"]), k
    entry = manifest["scans"]["0"]
    assert entry["n_draws"] == len(draws)
    assert 0 < entry["n_in_spec"] <= entry["n_draws"]


def test_the_rng_stream_is_drawn_from_this_golden():
    with open(_RNG) as fh:
        man = json.load(fh)
    assert man["source_fixture"] == os.path.basename(_SLIM)
