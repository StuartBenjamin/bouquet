"""Configs stored earlier load as they were produced (finding 5 of the
2026-10-04 review).

``to_dict`` writes every field, so a stored config carries the defaults of
the day.  Fixtures: ``tests/data/stored_configs/config_<sha>.json``, the
``to_dict`` output of the SAME synthetic config (legacy and unified) at five
commits of the engine stack, produced by that commit's own code:

* a LEGACY config carrying an engine field's historical default
  (``engine_mse_jacobian="fd_broyden"``, ``engine_ids_inductive="auto"``)
  loads -- it was refused before (the field has no effect there and is
  loaded as today's default, with a warning);
* a UNIFIED config loads with the values it ran with: present fields as
  stored; a field it predates with the value it was produced with where
  that is knowable (``engine_ids_inductive`` -> "auto";
  ``engine_draw_solve_maxits`` -> the ``draw_solve_maxits`` the engine read
  then), with a warning; where it is not knowable, today's default with a
  loud warning naming the field.

Solver-free.
"""
import json
import os
import warnings

import pytest

from bouquet.config import BouquetConfig, GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, "data", "stored_configs")
SHAS = ["554edfb", "5f720ef", "ab6b95f", "de9ff62", "d874822"]


def _stored(sha, eng):
    with open(os.path.join(_DIR, f"config_{sha}.json")) as fh:
        return json.load(fh)[eng]


def _load(d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = BouquetConfig.from_dict(d)
    return c.generation, [str(x.message) for x in w]


@pytest.mark.parametrize("sha", SHAS)
def test_every_stored_legacy_config_loads(sha):
    d = _stored(sha, "legacy")
    g, msgs = _load(d)
    assert g.reconstruction_engine == "legacy"
    stored = d["generation"]
    for name, val in (("engine_mse_jacobian", "fd_broyden"),
                      ("engine_ids_inductive", "auto")):
        if stored.get(name) == val:
            # the historical default: loaded as today's, said so
            assert getattr(g, name) == getattr(GenerationConfig(), name)
            assert any(name in m and "no effect" in m for m in msgs)
    # every legacy-path setting it ran with is as stored
    for k, v in stored.items():
        if k.startswith("engine_") or k in ("separatrix_pressure",):
            continue
        if isinstance(v, list):
            continue
        assert getattr(g, k) == v, k


@pytest.mark.parametrize("sha", SHAS)
def test_every_stored_unified_config_loads_with_what_it_ran_with(sha):
    d = _stored(sha, "unified")
    stored = d["generation"]
    g, msgs = _load(d)
    assert g.reconstruction_engine == "unified"
    # present engine fields: exactly as stored (incl. the old MSE default)
    for k, v in stored.items():
        if k.startswith("engine_") and not isinstance(v, list):
            assert getattr(g, k) == v, k
    if "engine_ids_inductive" not in stored:
        assert g.engine_ids_inductive == "auto"
        assert any("engine_ids_inductive" in m and "predates" in m
                   for m in msgs)
    if "engine_draw_solve_maxits" not in stored:
        assert g.engine_draw_solve_maxits == stored["draw_solve_maxits"]
        assert g.draw_solve_maxits is None
        assert any("engine_draw_solve_maxits" in m for m in msgs)
    if sha == "d874822":
        assert msgs == []                     # a current config: silent


def test_a_unified_config_that_capped_its_draws_the_old_way():
    """Written before engine_draw_solve_maxits (the engine read
    draw_solve_maxits then): the cap moves over -- it was REFUSED before."""
    d = _stored("554edfb", "unified")
    d["generation"]["draw_solve_maxits"] = 40
    g, msgs = _load(d)
    assert g.engine_draw_solve_maxits == 40 and g.draw_solve_maxits is None


def test_an_unknowable_missing_field_warns_loudly_naming_it():
    d = _stored("d874822", "unified")
    del d["generation"]["engine_mse_jacobian"]
    g, msgs = _load(d)
    assert g.engine_mse_jacobian == GenerationConfig().engine_mse_jacobian
    assert any(m.startswith("STORED UNIFIED CONFIG LACKS "
                            "generation.engine_mse_jacobian") for m in msgs)


def test_a_loop_config_before_the_post_homotopy_field():
    d = _stored("d874822", "legacy")
    del d["generation"]["jbs_max_passes_post_homotopy"]
    g, msgs = _load(d)
    assert g.jbs_max_passes_post_homotopy == 2
    assert any("jbs_max_passes_post_homotopy" in m for m in msgs)


def test_a_live_config_is_still_strict():
    """The tolerance is for STORED configs only: a config built today with a
    non-default engine field under "legacy" is still refused."""
    with pytest.raises(ValueError, match="no effect"):
        BouquetConfig.from_dict(dict(_stored("d874822", "legacy"),
                                     generation=dict(
                                         _stored("d874822", "legacy")[
                                             "generation"],
                                         engine_mse_jacobian="fd_chord",
                                         engine_rows=["Ip"])))
    from bouquet.engine import validate_engine_settings
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            engine_mse_jacobian="fd_broyden"))
