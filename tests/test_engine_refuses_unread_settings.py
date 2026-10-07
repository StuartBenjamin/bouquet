"""Under ``reconstruction_engine="unified"`` every legacy-path setting the
engine never reads is REFUSED when set (finding 4 of the 2026-10-04 review):
before, ``structured_li_target=0.90``, ``closure_channel="structured"``,
``anchor_pressure_to_equilibrium=True``, ``jbs_loop_q0_corrector=True`` ...
were accepted and silently ignored.  Each refusal names the engine setting
that replaces it (or says nothing does).  Defaults, and the factories'
configs built with ``reconstruction_engine="unified"``, are accepted;
``workflow='custom'`` downgrades to a printed WARN, as for the MSE knobs.
``isolate_edge_jBS`` / ``perturb_jind_in_anchor`` joined the refused set on
2026-10-05 (owner-approved); the factories no longer set them for a unified
configuration.

Solver-free.
"""
import contextlib
import io
import os
import warnings

import pytest

from bouquet.config import GenerationConfig
from bouquet.engine import (ENGINE_UNREAD_LEGACY_FIELDS, engine_settings,
                            validate_engine_settings)

#: one non-default value per refused field
_SET = dict(
    closure_channel="structured", jBS_baseline_mode="ohmic",
    structured_preset="li_soft_onesided", structured_basis="gaussian",
    structured_weights="flat", structured_sigma_ind_up=0.5,
    structured_li_target=0.90, structured_li_sigma=0.05,
    structured_li_kind="li_3", structured_ip_sigma=1e4,
    structured_ip_sigma_frac=0.005, structured_soft=True,
    structured_li_max_corrector_steps=3,
    anchor_pressure_to_equilibrium=True, imas_corrective_jphi=True,
    jbs_loop_q0_corrector=True, floor_j_BS=True, swb_iterations=2,
    accept_anchor_inband=True, diagnostic_plots=True,
    # owner-approved 2026-10-05: refused too (the factories no longer set
    # them for a unified configuration)
    isolate_edge_jBS=False, perturb_jind_in_anchor=True)


def test_every_unread_field_has_a_case():
    assert set(_SET) == set(ENGINE_UNREAD_LEGACY_FIELDS)


def _unified(**kw):
    g = GenerationConfig(reconstruction_engine="unified")
    for k, v in kw.items():           # set after construction: no preset
        setattr(g, k, v)              # side effects of __post_init__
    return g


def test_the_defaults_are_accepted():
    validate_engine_settings(_unified())
    engine_settings(_unified())


@pytest.mark.parametrize("name", sorted(_SET))
def test_a_set_unread_field_is_refused_naming_its_replacement(name):
    g = _unified(**{name: _SET[name]})
    with pytest.raises(ValueError) as ei:
        validate_engine_settings(g)
    msg = str(ei.value)
    assert f"{name}=" in msg and "never reads" in msg
    assert ENGINE_UNREAD_LEGACY_FIELDS[name] in msg
    # ... and says how to get the legacy paths back (the engine is the
    # default since 2026-10-06)
    assert 'reconstruction_engine="legacy"' in msg
    # the same value under the legacy path is fine (it is read there)
    gl = GenerationConfig(reconstruction_engine="legacy")
    setattr(gl, name, _SET[name])
    validate_engine_settings(gl)


def test_homotopy_passes_without_a_homotopy_is_refused():
    g = _unified(engine_draw_homotopy=False, homotopy_passes=[(0.05, 0.1)])
    with pytest.raises(ValueError, match="homotopy_passes"):
        validate_engine_settings(g)
    validate_engine_settings(_unified(engine_draw_homotopy=False))
    validate_engine_settings(_unified(homotopy_passes=[(0.05, 0.1)]))


def test_workflow_custom_downgrades_to_a_printed_warning():
    g = _unified(structured_li_target=0.9, workflow="custom")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        validate_engine_settings(g)
    assert "WARN" in buf.getvalue() and "structured_li_target" \
        in buf.getvalue()


def _factory(factory, **kw):
    import bouquet as bq
    ex = os.path.join(os.path.dirname(__file__), os.pardir, "examples",
                      "D3D-like")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if factory == "imas":
            return bq.Bouquet.from_imas(
                os.path.join(ex, "D3Dlike_baseline_omas.json"),
                mesh=os.path.join(ex, "DIIID_mesh.h5"), time=2.3043,
                n_draws=1, **kw)
        return bq.Bouquet.from_geqdsk(
            os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
            profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
            mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1, **kw)


@pytest.mark.parametrize("factory", ["imas", "gfile"])
def test_the_factories_build_a_unified_config_the_engine_accepts(factory):
    """``reconstruction_engine="unified"`` at construction: the factory
    leaves the legacy-path workflow settings at their defaults, and the
    engine accepts the config."""
    b = _factory(factory, reconstruction_engine="unified")
    g = b.config.generation
    assert g.reconstruction_engine == "unified"
    assert g.isolate_edge_jBS is True
    assert g.perturb_jind_in_anchor is False
    validate_engine_settings(g)


@pytest.mark.parametrize("factory", ["imas", "gfile"])
def test_the_factories_default_to_the_engine(factory):
    """Without the keyword the factories build the DEFAULT engine (unified
    since 2026-10-06), exactly as with reconstruction_engine="unified"."""
    g = _factory(factory).config.generation
    assert g.reconstruction_engine == "unified"
    assert g.isolate_edge_jBS is True
    assert g.perturb_jind_in_anchor is False
    validate_engine_settings(g)


@pytest.mark.parametrize("factory", ["imas", "gfile"])
def test_the_factories_legacy_configs_are_unchanged(factory):
    """With reconstruction_engine="legacy" the factories set exactly what
    they did before: the legacy-path workflow settings."""
    for kw in (dict(reconstruction_engine="legacy"),):
        g = _factory(factory, **kw).config.generation
        assert g.reconstruction_engine == "legacy"
        assert g.isolate_edge_jBS is False
        assert g.perturb_jind_in_anchor is (factory == "imas")
        validate_engine_settings(g)


@pytest.mark.parametrize("factory", ["imas", "gfile"])
def test_switching_a_legacy_factory_config_to_the_engine_is_refused(factory):
    """A legacy factory config switched to "unified" afterwards carries the
    legacy-path values the engine never reads: refused, naming each field
    and how to build a unified config instead."""
    b = _factory(factory, reconstruction_engine="legacy")
    b.config.generation.reconstruction_engine = "unified"
    with pytest.raises(ValueError) as ei:
        validate_engine_settings(b.config.generation)
    msg = str(ei.value)
    assert "isolate_edge_jBS=False" in msg and "never reads" in msg
    assert "reconstruction_engine='unified'" in msg
    assert ("perturb_jind_in_anchor=True" in msg) is (factory == "imas")
