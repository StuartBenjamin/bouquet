"""DrawSolveGuard: the draw-loop maxits cap and the failed-solve record (solver-free)."""
import inspect
from types import SimpleNamespace

import pytest

from bouquet.TokaMaker_interface import (DRAW_SOLVE_MAXITS, DrawSolveGuard,
                                         generate_bouquet)


class FakeGS:
    """Minimal TokaMaker stand-in: settings.maxits, update_settings, solve."""

    def __init__(self, fail_on=()):
        self.settings = SimpleNamespace(maxits=800)
        self.pushed = []
        self.fail_on = set(fail_on)
        self.calls = 0

    def update_settings(self):
        self.pushed.append(self.settings.maxits)

    def solve(self, return_its=False):
        self.calls += 1
        if self.calls in self.fail_on:
            raise ValueError('Error in solve: Exceeded "maxits"      ')
        return (0, 12) if return_its else None


def _call_site(gs):
    return gs.solve()


def test_cap_applied_inside_and_restored_after():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100):
        assert gs.settings.maxits == 100
        gs.solve()
    assert gs.settings.maxits == 800
    assert gs.pushed == [100, 800]
    assert "solve" not in vars(gs), "the instance wrapper must be removed"


def test_none_keeps_the_solver_cap_but_still_records():
    gs = FakeGS(fail_on={1})
    with DrawSolveGuard(gs, None) as g:
        with pytest.raises(ValueError):
            gs.solve()
    assert gs.pushed == [] and gs.settings.maxits == 800
    assert len(g.records) == 1


def test_failures_reraise_and_are_recorded_per_draw():
    gs = FakeGS(fail_on={2, 3})
    with DrawSolveGuard(gs, 100) as g:
        g.begin_draw(0)
        _call_site(gs)
        with pytest.raises(ValueError):
            _call_site(gs)
        g.begin_draw(1)
        with pytest.raises(ValueError):
            _call_site(gs)
        _call_site(gs)
    assert g.n_solves == 4
    assert [r["draw"] for r in g.records] == [0, 1]
    r = g.failures(0)[0]
    assert r["site"] == "?"   # no bouquet frame on a test-only stack
    assert r["error"] == 'ValueError: Error in solve: Exceeded "maxits"'
    assert r["seconds"] >= 0.0
    assert g.failures(2) == []
    s = g.summary()
    assert s.startswith("[draw-solves] 2/4 solves failed (2 exceeded maxits 100)")
    assert "draws [0, 1]" in s


def test_restored_when_the_block_raises():
    gs = FakeGS()
    with pytest.raises(RuntimeError):
        with DrawSolveGuard(gs, 50):
            raise RuntimeError("boom")
    assert gs.settings.maxits == 800 and "solve" not in vars(gs)


def test_clean_summary_and_bad_cap():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100) as g:
        gs.solve()
    assert g.summary() == "[draw-solves] 1 solves, none failed (maxits 100)"
    with pytest.raises(ValueError):
        DrawSolveGuard(gs, 0)


def test_default_cap_is_even_and_matches_the_config():
    from bouquet.config import GenerationConfig
    assert DRAW_SOLVE_MAXITS % 2 == 0   # same phase of the period-2 cycle as 800
    assert GenerationConfig().draw_solve_maxits == DRAW_SOLVE_MAXITS


def test_generate_bouquet_threads_the_guard_through_the_draw_loop():
    sig = inspect.signature(generate_bouquet)
    assert sig.parameters["solve_guard"].default is None
    src = inspect.getsource(generate_bouquet)
    assert "solve_guard.begin_draw(count)" in src
    assert "diagnostics['solve_failures']" in src


def test_bouquet_generate_enters_the_guard():
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet.generate)
    assert "DrawSolveGuard(self.mygs, gc.draw_solve_maxits)" in src
    assert "solve_guard=_solve_guard" in src
    assert "print(_solve_guard.summary())" in src
