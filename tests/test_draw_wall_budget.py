"""The per-draw wall-clock budget (GenerationConfig.draw_wall_budget_s):
checked between solves by DrawSolveGuard (never inside a running solve) and
once before a draw is archived; a draw over it is rejected as
"draw_wall_budget", recorded like every other rejection."""
import time

import pytest


class _FakeClock:
    """A clock the solver double advances: the budget test must not depend
    on real wall time (a loaded machine turned 3 x 50 ms sleeps against a
    120 ms budget into a spurious failure).  Everything but ``perf_counter``
    is the real ``time`` module."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _SlowSolver:
    """A solver double whose every solve takes *dt* seconds of the clock."""

    class settings:
        maxits = 100

    def __init__(self, dt, clock=None):
        self.dt = dt
        self.n = 0
        self.clock = clock

    def solve(self, *a, **k):
        if self.clock is not None:
            self.clock.now += self.dt
        else:
            time.sleep(self.dt)
        self.n += 1

    def update_settings(self):
        pass


def test_the_budget_refuses_the_next_solve_never_interrupts_one(monkeypatch):
    import bouquet.TokaMaker_interface as TI
    from bouquet.TokaMaker_interface import (DrawSolveGuard,
                                             DrawWallBudgetExceeded)
    clock = _FakeClock()
    monkeypatch.setattr(TI, "time", clock)
    s = _SlowSolver(0.05, clock)
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


def test_the_pre_archive_budget_check_is_outside_the_coil_handling_block():
    """Review of #94, P5: the "once before archiving" check is on every
    draw, not inside ``if coil_drift is not None and _recon_Ip is not
    None:`` (where a coil_drift=None draw skipped it)."""
    import inspect
    from bouquet.TokaMaker_interface import generate_bouquet
    src = inspect.getsource(generate_bouquet)
    blk = src.index("        if coil_drift is not None and _recon_Ip is not None:")
    chk = src.index("(checked before \"\n                             \"archiving)")
    # the next statement at the block's indentation after the block opens
    nxt = src.index("\n        # The per-draw wall budget, once before archiving", blk)
    assert blk < nxt < chk
    assert src.count("check_wall_budget()") == 1


def test_the_pre_archive_budget_check_rejects_a_draw_over_its_budget(
        tmp_path, monkeypatch):
    """A draw whose solves all STARTED within the budget but whose clock is
    over it at archive time is rejected as draw_wall_budget, not archived."""
    coil_drift = 0.01
    import sys
    import test_engine_draws as TED
    import bouquet.TokaMaker_interface as TI
    from bouquet.TokaMaker_interface import DrawSolveGuard

    class ArchiveTimeOver(DrawSolveGuard):
        # in budget for every solve; over it when generate_bouquet asks
        # directly (the pre-archive check)
        def check_wall_budget(self):
            if (self.draw is not None
                    and sys._getframe(1).f_code.co_name == "generate_bouquet"):
                self.over_budget.setdefault(self.draw, 1.0)
                return True
            return False

    real = TI.generate_bouquet

    def wrapped(mygs, *a, **k):
        k["coil_drift"] = coil_drift
        with ArchiveTimeOver(mygs, maxits=None, wall_budget_s=0.5) as g:
            return real(mygs, *a, solve_guard=g, **k)
    monkeypatch.setattr(TI, "generate_bouquet", wrapped)
    try:
        diags, rej, h, G = TED._generate(tmp_path / "p5", monkeypatch, n=2)
    finally:
        monkeypatch.setattr(TI, "generate_bouquet", real)
    assert len(diags) == 0 and len(rej) == 2
    assert {r["reason"] for r in rej} == {"draw_wall_budget"}
    assert {r["stage"] for r in rej} == {"wall_budget"}
