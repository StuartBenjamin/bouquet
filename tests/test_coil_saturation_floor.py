"""The homotopy's coil-saturation test measures each F coil's drift against
the bound actually installed for it, max(drift_F |I_base|, floor_A), so a
low-current coil whose bound is the floor is not called saturated at a tiny
fraction of that floor (owner decision 2026-10-15; before, a 10 A coil with a
50 A bound tripped at 0.5 A and every draw of the synthetic 0.7 s slice was
refused)."""
import pytest

from bouquet.TokaMaker_interface import _coil_bound_max_f, _coil_saturation


def _pct(cur, base):
    return {n: abs(cur[n] - b) / max(abs(b), 1.0) * 100.0 for n, b in base.items()}


def test_an_unfloored_coil_is_measured_exactly_as_before():
    base = {"F1A": 1000.0, "F2B": 2000.0, "VSC": 500.0}
    cur = {"F1A": 1030.0, "F2B": 2000.0, "VSC": 600.0}
    # 5 % bound = 50 A on F1A (= the 50 A floor: not floored); drift 30 A = 3.0 % either way
    assert _coil_bound_max_f(cur, base, {"VSC"}, 0.05, 50.0) == pytest.approx(3.0)
    assert max(v for n, v in _pct(cur, base).items() if n != "VSC") == pytest.approx(3.0)


def test_a_floored_coil_is_measured_against_the_floor():
    base = {"F1A": 1000.0, "F2B": 10.0, "VSC": 500.0}
    cur = {"F1A": 1000.0, "F2B": 36.0, "VSC": 500.0}      # F2B moved 26 A of its 50 A floor bound
    bare = _pct(cur, base)["F2B"]                           # 260 % of its own 10 A
    assert bare > 100.0
    m = _coil_bound_max_f(cur, base, {"VSC"}, 0.05, 50.0)
    assert m == pytest.approx(26.0 / 1000.0 * 100.0)        # 2.6 % of the 1000 A the floor stands for
    sat_F, _ = _coil_saturation(m, 0.0, 0.05, 0.10)
    assert not sat_F                                        # half its real bound: not saturated
    cur["F2B"] = 10.0 + 49.6
    sat_F, _ = _coil_saturation(_coil_bound_max_f(cur, base, {"VSC"}, 0.05, 50.0), 0.0, 0.05, 0.10)
    assert sat_F                                            # on its real bound: saturated


def test_vsc_coils_are_left_to_their_own_test():
    base = {"F1A": 1000.0, "VSC": 10.0}
    cur = {"F1A": 1000.0, "VSC": 100.0}
    assert _coil_bound_max_f(cur, base, {"VSC"}, 0.05, 50.0) == 0.0
