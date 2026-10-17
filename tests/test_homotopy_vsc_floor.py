"""The homotopy's #VSC bound floor (owner-approved 2026-10-16), the VSC
regularisation weight's recorded scans and the vertical-runaway label of a
capped homotopy stage -- OFT-free.

The floor changes a homotopy coil bound, reusing the legacy in-spec VSC
metric's constant ``_COIL_DRIFT_DENOM_FLOOR_A`` (that metric ADDS it to its
scale; the default chi2 rule does not use it); it changes no acceptance
gate.  The label is a label, never a gate.
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
    # the plain per-coil percentage (denominator floor 1 A) is still the
    # measure's 1 A form (the guard's default)
    old = T._coil_max_drifts(T._coil_drift_pct(cur, base), vsc)[1]
    assert T._coil_bound_max_vsc(cur, base, vsc, 1.0) == pytest.approx(old)
    assert old == pytest.approx(100 * dV * 1e4 / 2.8e3)
    # the CHANNEL's excursion now as a fraction of the stage's #VSC bound
    # (1 = at it): the pair's anti-series drift (review of #94, P6) -- the
    # channel at its bound reads 1; a bare drift of one coil by the same
    # amount reads 0.5 and a same-sign (shaping) drift 0
    vd = dV * 1e4
    chan = dict(base, F9A=-6.6e4 + vd, F9B=2.8e3 - vd)
    gs = types.SimpleNamespace(get_coil_currents=lambda: (chan, None))
    assert T._vsc_bound_fraction_now(gs, base, vsc, dV) == pytest.approx(1.0)
    gs = types.SimpleNamespace(get_coil_currents=lambda: (cur, None))
    assert T._vsc_bound_fraction_now(gs, base, vsc, dV) == pytest.approx(0.5)
    same = dict(base, F9A=-6.6e4 + vd, F9B=2.8e3 + vd)
    gs = types.SimpleNamespace(get_coil_currents=lambda: (same, None))
    assert T._vsc_bound_fraction_now(gs, base, vsc, dV) == pytest.approx(0.0)
    # the installed half-width never goes below the per-coil floor
    gs = types.SimpleNamespace(get_coil_currents=lambda: (chan, None))
    assert T._vsc_bound_fraction_now(gs, base, vsc, dV, 4 * vd) == \
        pytest.approx(0.25)

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
    # a purely vertical displacement of disp x a (a = 0.6 m)
    r = vertical_runaway_record(frac, dict(
        displacement_frac_a=disp, minor_radius_m=0.6,
        dZ_mm=(None if disp is None else disp * 600.0)))
    assert r["label"] == label
    assert r["threshold_frac_a"] == 0.1


def test_a_radial_excursion_is_not_a_vertical_runaway():
    """Review of #94, P6: the label needs |dZ| beyond the threshold; a
    radial excursion of the same size is reported, not labelled."""
    from bouquet.engine import vertical_runaway_record
    r = vertical_runaway_record(1.0, dict(displacement_frac_a=1.4,
                                          minor_radius_m=0.6, dZ_mm=6.0))
    assert r["label"] is None and r["vsc_at_bound"] is True
    assert r["axis_displacement_frac_a"] == 1.4
    assert r["axis_dZ_frac_a"] == pytest.approx(0.01)
    assert vertical_runaway_record(1.0, dict(displacement_frac_a=1.4,
                                             dZ_mm=840.0))["label"] is None


def test_the_vsc_saturation_measure_honours_the_per_coil_floor():
    """Review of #94, P4: with drift_VSC x scale below floor_A the installed
    #VSC bound is floor_A, and the saturation measure is taken against it,
    as _coil_bound_max_f does for the F coils."""
    import bouquet.TokaMaker_interface as T
    base = {"F1A": 1.2e5, "F9A": -6.6e4, "F9B": 2.8e3}
    vsc, dV, floor = ("F9A", "F9B"), 0.001, 50.0
    b = T._homotopy_coil_bounds(base, vsc, 0.05, dV, floor)
    assert b["#VSC"] == [-50.0, 50.0]                 # floor_A, not 10 A
    cur = dict(base, F9B=2.8e3 + 20.0)                # 20 A under a 50 A bound
    m = T._coil_bound_max_vsc(cur, base, vsc, drift_VSC=dV, floor_A=floor)
    assert not T._coil_saturation(0.0, m, 0.05, dV)[1]
    assert m == pytest.approx(100 * dV * 20.0 / 50.0)
    at = dict(base, F9B=2.8e3 + 50.0)                 # at the 50 A bound
    m = T._coil_bound_max_vsc(at, base, vsc, drift_VSC=dV, floor_A=floor)
    assert T._coil_saturation(0.0, m, 0.05, dV)[1]
    # the default stages (drift_VSC >= 0.01, floor 50 A) are unaffected
    assert T._coil_bound_max_vsc(cur, base, vsc, drift_VSC=0.01,
                                 floor_A=floor) == \
        pytest.approx(T._coil_bound_max_vsc(cur, base, vsc))


def test_the_hard_factor_vsc_bound_uses_the_floored_scale():
    """Review of #94, P3: the hard-factor #VSC bound (an outer ceiling) is
    built on _vsc_bound_scale and its guard measures on the 10 kA floor, so
    it is never tighter than the homotopy's bound for the same factor."""
    import bouquet.TokaMaker_interface as T
    src = inspect.getsource(T.generate_bouquet)
    blk = src[src.index("if coil_drift_hard_factor is not None:"):
              src.index("_hard_msg = (")]
    assert "_hard * _vsc_bound_scale(_baseline_coils, _vsc_in_set)" in blk
    assert "vsc_denom_floor_A=_COIL_DRIFT_DENOM_FLOOR_A" in blk
    assert "_vsc_min_base" not in blk


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
