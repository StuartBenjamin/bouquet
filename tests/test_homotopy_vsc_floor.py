"""The homotopy's #VSC bound floor (owner-approved 2026-10-16), the VSC
regularisation weight's recorded scans and the vertical-runaway label of a
capped homotopy stage -- OFT-free.

The floor changes a homotopy coil bound so that it agrees with the in-spec
rule's VSC metric (one constant, ``_COIL_DRIFT_DENOM_FLOOR_A``); it changes
no acceptance gate.  The label is a label, never a gate.
"""
import inspect
import types

import pytest

from bouquet import boundary_balance as BB


def test_the_vsc_weight_is_one_and_its_scans_are_recorded():
    from bouquet.TokaMaker_interface import generate_bouquet
    p = inspect.signature(generate_bouquet).parameters
    assert p["vsc_soft_reg_weight"].default == 1.0
    assert BB.draw_coil_reg_weights() == (1.0e4, 1.0)
    doc = generate_bouquet.__doc__
    i = doc.index("vsc_soft_reg_weight : float")
    entry = doc[i:i + 2000]
    for text in ("2026-05", "2026-10", "1, 10 and 100 give identical",
                 "1e3 starves", "1e4 fails the sigma=0"):
        assert text in entry, text


# ---------------------------------------------------------------------------
#  the homotopy's #VSC bound: max(min|I_pair|, 10 kA)
# ---------------------------------------------------------------------------
def test_the_vsc_bound_uses_the_floor_for_a_near_zero_coil():
    import bouquet.TokaMaker_interface as T
    assert T._COIL_DRIFT_DENOM_FLOOR_A == 10_000.0     # one constant
    # a recorded coil set of the reversed-I_p kind (2 s.f.): one coil of
    # the pair carries 2.8 kA, below the floor
    base = {"F1A": 1.2e5, "F2A": -4.5e4, "F9A": -6.6e4, "F9B": 2.8e3}
    vsc = ("F9A", "F9B")
    for dF, dV in ((0.05, 0.10), (0.02, 0.05), (0.01, 0.01)):
        b = T._homotopy_coil_bounds(base, vsc, dF, dV, 50.0)
        h = max(dV * 1e4, 50.0)
        assert b["#VSC"] == pytest.approx([-h, h])
        # the bare per-coil bounds are unchanged by the floor
        assert b["F9B"] == [pytest.approx(2.8e3 - max(dF * 2.8e3, 50.0)),
                            pytest.approx(2.8e3 + max(dF * 2.8e3, 50.0))]
        assert b["F1A"] == [pytest.approx(1.2e5 * (1 - dF)),
                            pytest.approx(1.2e5 * (1 + dF))]
    assert T._vsc_bound_scale(base, vsc) == 1e4


def test_the_vsc_bound_uses_the_coil_when_both_carry_more_than_the_floor():
    import bouquet.TokaMaker_interface as T
    base = {"F1A": 1.2e5, "F9A": -6.6e4, "F9B": 4.8e4}
    b = T._homotopy_coil_bounds(base, ("F9A", "F9B"), 0.05, 0.10, 50.0)
    assert b["#VSC"] == pytest.approx([-4.8e3, 4.8e3])
    assert T._vsc_bound_scale(base, ("F9A", "F9B")) == 4.8e4
    # no pair: no #VSC bound
    assert "#VSC" not in T._homotopy_coil_bounds({"F1A": 1e5}, (), 0.05,
                                                 0.10, 50.0)


def test_vsc_saturation_is_measured_on_the_bound_denominator():
    import bouquet.TokaMaker_interface as T
    base = {"F1A": 1.2e5, "F9A": -6.6e4, "F9B": 2.8e3}
    vsc = ("F9A", "F9B")
    dV = 0.10
    # the smaller coil exactly at the channel bound (dV x 10 kA)
    cur = dict(base, F9B=2.8e3 + dV * 1e4)
    assert T._coil_bound_max_vsc(cur, base, vsc) == pytest.approx(100 * dV)
    sat_F, sat_V = T._coil_saturation(0.0, T._coil_bound_max_vsc(
        cur, base, vsc), 0.05, dV)
    assert sat_V and not sat_F
    # the old per-coil percentage (denominator floor 1 A) is unchanged for
    # the hard-factor bounds' guard
    old = T._coil_max_drifts(T._coil_drift_pct(cur, base), vsc)[1]
    assert T._coil_bound_max_vsc(cur, base, vsc, 1.0) == pytest.approx(old)
    assert old == pytest.approx(100 * dV * 1e4 / 2.8e3)
    # the fraction of the stage's bound now (1 = at it); None when unread
    gs = types.SimpleNamespace(get_coil_currents=lambda: (cur, None))
    assert T._vsc_bound_fraction_now(gs, base, vsc, dV) == pytest.approx(1.0)

    def _boom():
        raise RuntimeError("no equilibrium")
    assert T._vsc_bound_fraction_now(types.SimpleNamespace(
        get_coil_currents=_boom), base, vsc, dV) is None


def test_the_saturation_guard_measures_vsc_on_the_denominator_it_is_given():
    import bouquet.TokaMaker_interface as T
    base = {"F1A": 1.2e5, "F9A": -6.6e4, "F9B": 2.8e3}
    cur = dict(base, F9B=2.8e3 + 600.0)        # 21 % of the coil, 6 % of 10 kA
    gs = types.SimpleNamespace(get_coil_currents=lambda: (cur, None))
    floored = T._make_coil_saturation_guard(
        gs, base, ("F9A", "F9B"), 0.05, 0.10, "test", floor_A=50.0,
        vsc_denom_floor_A=T._COIL_DRIFT_DENOM_FLOOR_A)
    floored("pass")                            # 6 % < 9.9 %: not saturated
    assert floored.log[-1]["max_VSC_drift_pct"] == pytest.approx(6.0)
    plain = T._make_coil_saturation_guard(
        gs, base, ("F9A", "F9B"), 0.05, 0.10, "test", floor_A=50.0)
    with pytest.raises(T.CoilSaturated):
        plain("pass")                          # 21 % >= 9.9 %


# ---------------------------------------------------------------------------
#  the vertical-runaway label (a label, never a gate)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frac, disp, label", [
    (1.0, 1.4, "vertical_runaway"),
    (0.995, 0.11, "vertical_runaway"),
    (1.0, 0.05, None),            # at the bound, axis within 0.1 a
    (0.5, 1.4, None),             # displaced, the pair off its bound
    (None, 1.4, None),            # unknown
    (1.0, None, None),
])
def test_the_vertical_runaway_label(frac, disp, label):
    from bouquet.engine import (VERTICAL_RUNAWAY_AXIS_FRAC,
                                vertical_runaway_record)
    assert VERTICAL_RUNAWAY_AXIS_FRAC == 0.1
    r = vertical_runaway_record(frac, dict(displacement_frac_a=disp,
                                           dZ_mm=None))
    assert r["label"] == label
    assert r["threshold_frac_a"] == 0.1


def test_a_capped_homotopy_stage_carries_the_label_in_its_cap_event():
    from bouquet.engine_draws import GenerateEngineDraws
    m = GenerateEngineDraws.__new__(GenerateEngineDraws)
    m.cap_events, m._cur, m.maxits = [], {"count": 3}, 100
    m.delivered_axis = dict(R=1.7, Z=0.0, minor_radius_m=0.6)
    gs = types.SimpleNamespace(o_point=(1.7, 0.75))      # 1.25 a away
    ev = m.announce_cap("homotopy pass 1/3", RuntimeError("maxits"),
                        stage="homotopy", outcome="rejected", mygs=gs,
                        vsc_bound_frac=1.0)
    assert ev["label"] == "vertical_runaway"
    assert ev["vertical_runaway"]["axis_displacement_frac_a"] == \
        pytest.approx(0.75 / 0.6)
    ev2 = m.announce_cap("loop", RuntimeError("maxits"), stage="loop")
    assert "label" not in ev2                  # only homotopy stages
