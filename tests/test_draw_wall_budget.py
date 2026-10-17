"""The per-draw wall-clock budget (GenerationConfig.draw_wall_budget_s):
checked between solves by DrawSolveGuard (never inside a running solve) and
once before a draw is archived; a draw over it is rejected as
"draw_wall_budget", recorded like every other rejection."""
import time

import pytest

class _SlowSolver:
    """A solver double whose every solve takes *dt* seconds."""

    class settings:
        maxits = 100

    def __init__(self, dt):
        self.dt = dt
        self.n = 0

    def solve(self, *a, **k):
        time.sleep(self.dt)
        self.n += 1

    def update_settings(self):
        pass


def test_the_budget_refuses_the_next_solve_never_interrupts_one():
    from bouquet.TokaMaker_interface import (DrawSolveGuard,
                                             DrawWallBudgetExceeded)
    s = _SlowSolver(0.05)
    with DrawSolveGuard(s, maxits=None, wall_budget_s=0.12) as g:
        g.begin_draw(0)
        s.solve()
        s.solve()
        s.solve()                 # started at ~0.10 s: runs to the end
        with pytest.raises(DrawWallBudgetExceeded):
            s.solve()             # ~0.15 s elapsed: refused, never started
        assert s.n == 3 and 0 in g.over_budget
        g.begin_draw(1)           # the next draw has its own clock
        s.solve()
        assert s.n == 4 and 1 not in g.over_budget


def test_no_budget_and_outside_a_draw_change_nothing():
    from bouquet.TokaMaker_interface import DrawSolveGuard
    s = _SlowSolver(0.02)
    with DrawSolveGuard(s, maxits=None, wall_budget_s=None) as g:
        g.begin_draw(0)
        for _ in range(5):
            s.solve()
        assert not g.check_wall_budget()
    with DrawSolveGuard(s, maxits=None, wall_budget_s=1e-6) as g:
        s.solve()                 # before any draw (the baseline re-solve)
        assert not g.over_budget


def test_the_rejection_code_follows_the_exception_and_its_chain():
    from bouquet.TokaMaker_interface import (DRAW_REJECTION_REASONS,
                                             DrawWallBudgetExceeded,
                                             _draw_rejection_reason)
    from bouquet.engine_draws import engine_rejection_reason
    assert "draw_wall_budget" in DRAW_REJECTION_REASONS
    e = DrawWallBudgetExceeded("over")
    assert _draw_rejection_reason(e, "perturb") == "draw_wall_budget"
    assert _draw_rejection_reason(e, "post_homotopy") == "draw_wall_budget"
    try:
        try:
            raise e
        except DrawWallBudgetExceeded as inner:
            raise RuntimeError("the loop failed") from inner
    except RuntimeError as outer:
        assert engine_rejection_reason(outer, "perturb") == "draw_wall_budget"


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), True, "600"])
def test_the_budget_setting_is_validated(bad):
    from bouquet.config import GenerationConfig
    assert GenerationConfig().draw_wall_budget_s == 600.0
    assert GenerationConfig(draw_wall_budget_s=None).draw_wall_budget_s is None
    with pytest.raises(ValueError, match="draw_wall_budget_s"):
        GenerationConfig(draw_wall_budget_s=bad)


def test_a_slow_draw_is_rejected_by_generate_and_counted(tmp_path, monkeypatch):
    """generate_bouquet end to end on the engine toy, every solve slowed: with
    a budget the draws are rejected as draw_wall_budget (recorded like every
    rejection, in the summary line too); without one they are archived."""
    import test_engine_draws as TED
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.TokaMaker_interface import (DrawSolveGuard,
                                             _rejection_summary)

    class SlowFake(FakeTokaMaker):
        def solve(self, *a, **k):
            if getattr(self, "_slow", False):
                time.sleep(0.03)
            return super().solve(*a, **k)

    holder = {}

    def fake_cls(b):
        f = SlowFake(b)
        holder["fake"] = f
        return f

    def run(budget, sub):
        import bouquet.TokaMaker_interface as TI
        real = TI.generate_bouquet

        def wrapped(mygs, *a, **k):
            mygs._slow = True
            with DrawSolveGuard(mygs, maxits=None,
                                wall_budget_s=budget) as g:
                return real(mygs, *a, solve_guard=g, **k)
        monkeypatch.setattr(TI, "generate_bouquet", wrapped)
        try:
            return TED._generate(tmp_path / sub, monkeypatch, n=2,
                                 fake_cls=fake_cls)
        finally:
            monkeypatch.setattr(TI, "generate_bouquet", real)

    diags, rej, h, G = run(None, "off")
    assert len(diags) == 2 and not rej
    diags, rej, h, G = run(0.05, "on")
    assert len(diags) == 0 and len(rej) == 2
    assert {r["reason"] for r in rej} == {"draw_wall_budget"}
    assert all(r["elapsed_s"] > 0.05 and r["wall_budget_s"] == 0.05
               for r in rej)
    assert "draw_wall_budget=2" in _rejection_summary(rej, 2, 0)


def test_the_budget_is_checked_before_each_rescue_retry():
    """Review of #94, C6: a capped draw solve that ends over the budget
    starts no rescue retry (each is a full solve); the refusal is recorded
    on the failed solve and raised as DrawWallBudgetExceeded, chained to
    the cap error."""
    from test_draw_solve_guard import FakeGS
    from bouquet.TokaMaker_interface import (DrawWallBudgetExceeded,
                                             DrawSolveGuard,
                                             _draw_rejection_reason)

    class SlowStuck(FakeGS):
        def solve(self, return_its=False):
            time.sleep(0.06)
            return super().solve(return_its=return_its)

    gs = SlowStuck(stuck={1}, escape="urf")
    with DrawSolveGuard(gs, maxits=50, retry_urf=(0.3, 0.1),
                        loose_tol=2e-5, wall_budget_s=0.03) as g:
        g.begin_draw(0)
        with pytest.raises(DrawWallBudgetExceeded) as ei:
            gs.solve()            # starts in budget, ends over it, capped
    assert len(gs.attempts) == 1                  # no retry ran
    rec = g.records[-1]
    assert rec["recovered_by"] is None
    assert rec["attempts"] == [dict(step="urf=0.3", its=None,
                                    error="not started: draw_wall_budget")]
    assert "maxits" in str(ei.value.__cause__)
    assert _draw_rejection_reason(ei.value, "perturb") == "draw_wall_budget"
    # the same draw inside its budget is rescued as before
    gs2 = FakeGS(stuck={1}, escape="urf")
    with DrawSolveGuard(gs2, maxits=50, retry_urf=(0.3, 0.1),
                        loose_tol=2e-5, wall_budget_s=60.0) as g2:
        g2.begin_draw(0)
        gs2.solve()
    assert g2.records[-1]["recovered_by"] == "urf=0.3"
