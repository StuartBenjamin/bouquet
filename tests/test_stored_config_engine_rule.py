"""A stored legacy / swb config keeps its legacy fields on reload.

Review of #78 (finding A1): ``from_dict`` judged the engine of a STORED
generation dict by ``reconstruction_engine`` alone, so a legacy or swb run
stored with that field at its default ``"unified"`` was loaded as a unified
config -- its legacy fields reset and ``bootstrap_kwargs`` dropped, with a
warning that said the stored run was unchanged.  ``solve_method`` decides."""
import warnings

import pytest

from bouquet.config import (BouquetConfig, GenerationConfig, ReconstructionSource,
                            SolverConfig, _stored_engine)


def _cfg(**gen):
    return BouquetConfig(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                         solver=SolverConfig(mesh_path="m"), output_header="h",
                         generation=GenerationConfig(**gen))


@pytest.mark.parametrize("method", ["legacy", "swb"])
def test_a_stored_legacy_or_swb_config_round_trips_with_its_legacy_fields(method):
    cfg = _cfg(solve_method=method, draw_solve_maxits=250, bootstrap_kwargs={"iterations": 3})
    d = cfg.to_dict()
    assert d["generation"]["reconstruction_engine"] == "unified"      # the default, never resolved
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        back = BouquetConfig.from_dict(d)
    g = back.generation
    assert (g.solve_method, g.draw_solve_maxits, g.bootstrap_kwargs) == (method, 250, {"iterations": 3})
    assert not [x for x in w if "stored unified config" in str(x.message)]


def test_the_stored_engine_rule():
    assert _stored_engine({"solve_method": "engine", "reconstruction_engine": "legacy"}) == "unified"
    assert _stored_engine({"solve_method": "swb", "reconstruction_engine": "unified"}) == "legacy"
    assert _stored_engine({"solve_method": "legacy"}) == "legacy"
    assert _stored_engine({"reconstruction_engine": "unified"}) == "unified"
    assert _stored_engine({}) == "legacy"
