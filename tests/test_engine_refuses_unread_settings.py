"""Under ``reconstruction_engine="unified"`` every legacy-path setting the
engine never reads is REFUSED when set (finding 4 of the 2026-10-04 review):
before, ``structured_li_target=0.90``, ``closure_channel="structured"``,
``anchor_pressure_to_equilibrium=True``, ``jbs_loop_q0_corrector=True`` ...
were accepted and silently ignored.  Each refusal names the engine setting
that replaces it (or says nothing does).  Defaults, and the factories'
configs, are accepted; ``workflow='custom'`` downgrades to a printed WARN,
as for the MSE knobs.

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
    accept_anchor_inband=True, diagnostic_plots=True)


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
    # the same value under the legacy path is fine (it is read there)
    gl = GenerationConfig()
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


@pytest.mark.parametrize("factory", ["imas", "gfile"])
def test_the_factories_configs_are_accepted_under_the_engine(factory):
    import bouquet as bq
    ex = os.path.join(os.path.dirname(__file__), os.pardir, "examples",
                      "D3D-like")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if factory == "imas":
            b = bq.Bouquet.from_imas(
                os.path.join(ex, "D3Dlike_baseline_omas.json"),
                mesh=os.path.join(ex, "DIIID_mesh.h5"), time=2.3043,
                n_draws=1)
        else:
            b = bq.Bouquet.from_geqdsk(
                os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
                profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
                mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1)
    b.config.generation.reconstruction_engine = "unified"
    validate_engine_settings(b.config.generation)
