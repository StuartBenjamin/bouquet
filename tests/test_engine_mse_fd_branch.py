"""The MSE stage's finite-difference Jacobian stays on the base branch.

On a slice whose free-boundary problem has two solutions for nearly the
same current (two flux extrema of almost equal depth, magnetic axes ~1-2 cm
apart), a perturbed FD solve can land on the OTHER solution: its column
then differences across the jump, the refreshed Jacobian's Gauss-Newton
step sends the MSE stage to the other branch and the next refresh into a
cycle between the two (measured on a reversed-Ip slice: JBSNotConverged at
the pass ceiling, a period-3 cycle).  The engine re-takes such a column with
the step halved (``engine.MSE_FD_BRANCH_AXIS_FRAC``,
``engine.MSE_FD_MAX_HALVINGS``): a finer difference step, no criterion
touched.

* :func:`bouquet.utils.structured_mse_jacobian` with an ``on_branch`` test:
  a column whose step leaves the branch is re-taken at h/2, h/4, ...; one
  that never stays keeps its original step; ``on_branch=None`` is the
  function before the option, call for call.
* On the engine's toy backend with a branch jump: the checked Jacobian
  matches the jump-free one; the mutant without halvings carries the jump;
  a state without a jump is differenced bit for bit as before.

Synthetic inputs only; no solver, no device data.
"""
import numpy as np
import pytest

import _engine_toy as T
import bouquet.engine as be
from bouquet.utils import structured_mse_jacobian

from test_engine_mse_refresh import _mse_data, _run


# ---------------------------------------------------------------------------
#  A. the FD helper
# ---------------------------------------------------------------------------
_A = np.array([[1.0, 0.5, -0.2], [0.3, -1.0, 0.8]])
_X0 = np.array([0.1, -0.2, 0.05])
_JUMP = np.array([0.4, -0.7])


def _tg_with_jump(thr, calls):
    """tan(gamma) linear in x, plus a jump once coefficient 1 moved more
    than *thr* from the base (the other branch)."""
    def f(x):
        calls.append(np.asarray(x, float).copy())
        x = np.asarray(x, float)
        out = _A @ x
        if abs(x[1] - _X0[1]) > thr:
            out = out + _JUMP
        return out
    return f


def _branch_test(thr, calls):
    def ok(i, h):
        return abs(calls[-1][1] - _X0[1]) <= thr
    return ok


def test_a_step_that_leaves_the_branch_is_halved_until_it_stays():
    calls = []
    f = _tg_with_jump(0.006, calls)
    tg0 = _A @ _X0
    J = structured_mse_jacobian(f, _X0, tg0, np.ones(3, bool), step=0.02,
                                on_branch=_branch_test(0.006, calls),
                                max_halvings=4)
    np.testing.assert_allclose(J, _A, rtol=1e-12, atol=1e-12)
    # column 1: h = 0.02 and 0.01 jumped, 0.005 stayed -> 3 solves
    steps = [c[1] - _X0[1] for c in calls if c[1] != _X0[1]]
    np.testing.assert_allclose(steps, [0.02, 0.01, 0.005])
    assert len(calls) == 2 + 3
    # without the check the column carries the jump / h
    calls2 = []
    J0 = structured_mse_jacobian(_tg_with_jump(0.006, calls2), _X0, tg0,
                                 np.ones(3, bool), step=0.02)
    np.testing.assert_allclose(J0[:, 1], _A[:, 1] + _JUMP / 0.02)
    assert len(calls2) == 3


def test_no_check_or_a_passing_check_is_the_plain_forward_difference():
    tg0 = _A @ _X0
    for kw in (dict(), dict(on_branch=lambda i, h: True, max_halvings=4),
               dict(on_branch=None, max_halvings=4)):
        calls = []
        J = structured_mse_jacobian(_tg_with_jump(1.0, calls), _X0, tg0,
                                    np.array([True, True, False]),
                                    step=0.02, **kw)
        ref = np.zeros_like(J)
        for i in (0, 1):
            xp = _X0.copy()
            xp[i] += 0.02
            ref[:, i] = (_A @ xp - tg0) / 0.02
        assert np.array_equal(J, ref)
        assert len(calls) == 2


def test_a_column_that_never_stays_keeps_its_original_step():
    calls = []
    f = _tg_with_jump(1e-9, calls)
    tg0 = _A @ _X0
    J = structured_mse_jacobian(f, _X0, tg0, np.ones(3, bool), step=0.02,
                                on_branch=_branch_test(1e-9, calls),
                                max_halvings=2)
    # h, h/2, h/4 all jumped: the original-step column, unchanged
    np.testing.assert_allclose(J[:, 1], _A[:, 1] + _JUMP / 0.02)
    assert len(calls) == 2 + 3
    with pytest.raises(ValueError, match="max_halvings"):
        structured_mse_jacobian(f, _X0, tg0, np.ones(3, bool),
                                on_branch=lambda i, h: True, max_halvings=-1)


# ---------------------------------------------------------------------------
#  B. the engine on a toy with two branches
# ---------------------------------------------------------------------------
class BranchToy(T.ToyGS):
    """The toy with a second solution: once the solved request has moved
    along the direction ``d`` by more than ``thr`` (in units of ``d``) from
    ``R_ref``, the state is on the other branch -- its magnetic axis sits
    ``jump_axis`` lower and the pitch angles carry a jump.  Reports its
    minor radius (the engine's branch-check length scale)."""

    flip = None                 # (R_ref, d, thr) or None: one branch
    jump_axis = 0.017           # [m] (the measured axis-to-axis distance)
    jump_bz = 0.05              # relative B_Z jump on the other branch

    def _other(self):
        if self.flip is None or self.state is None:
            return False
        R_ref, d, thr = self.flip
        a = float((np.asarray(self.state["R"]) - R_ref) @ d / (d @ d))
        return abs(a) > thr

    def field_at_chords(self, st):
        B, found = super().field_at_chords(st)
        if self._other():
            B = B.copy()
            B[:, 2] = B[:, 2] * (1.0 + self.jump_bz)
        return B, found

    def measure(self, want_chords=False, final=False):
        out = super().measure(want_chords=want_chords, final=final)
        if want_chords and self._other():
            out["axis"] = (T.R0, -self.jump_axis)
        return out

    def minor_radius(self):
        return T.A_MIN


def _refresh_base(toy):
    """A converged MSE reconstruction on *toy* and the refresh's base:
    ``(eng, g, lam, x_b, snap_b, tg_b, axis_b)``."""
    md, ch = _mse_data()
    toy.chords = ch
    eng, res, rec = _run(toy, md=md, ch=ch)
    assert res["converged"]
    p = eng._pending
    g, lam = p["cl_geom"], np.asarray(eng.state.lambda_bs, float)
    x_b = np.asarray(eng.state.x, float)
    m_b, snap_b = eng._mse_base_solve(g, lam, x_b)
    return eng, g, lam, x_b, snap_b, eng._tg(m_b), m_b["axis"], rec


def _arm_flip(eng, g, lam, x_b, snap_b, coef=1, thr=0.006):
    """Put the toy's branch boundary at a step of *thr* along coefficient
    *coef* from the base request."""
    eng.b.restore(snap_b)
    R_ref = np.asarray(eng.b.state["R"], float).copy()
    xp = x_b.copy()
    xp[coef] += 1.0
    d = np.asarray(eng.close(g, lam, x=xp)["jc"], float) - R_ref
    eng.b.flip = (R_ref, d, float(thr))


def test_the_checked_jacobian_matches_the_jump_free_one():
    eng, g, lam, x_b, snap_b, tg_b, ax, rec = _refresh_base(BranchToy())
    # the reconstruction on the one-branch toy: the check ran, nothing moved
    bc = [p for p in rec["phases"] if p["name"] == "mse"][-1][
        "jacobian"]["branch_check"]
    assert bc["active"] and bc["n_columns_refined"] == 0
    assert bc["threshold_m"] == pytest.approx(
        be.MSE_FD_BRANCH_AXIS_FRAC * T.A_MIN)
    J_smooth = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=ax)
    assert eng._mse_fd_last["n_columns_refined"] == 0
    n0 = eng.b.n_solves
    _arm_flip(eng, g, lam, x_b, snap_b)
    J = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=ax)
    chk = eng._mse_fd_last
    assert chk["n_columns_refined"] >= 1
    assert chk["n_columns_unresolved"] == 0
    assert 1 in [c["coefficient"] for c in chk["columns"]]
    for c in chk["columns"]:
        assert c["resolved"] and c["step_used"] < c["steps"][0]
        assert c["axis_shift_m"][0] > chk["threshold_m"]
        assert c["axis_shift_m"][-1] <= chk["threshold_m"]
    n_free = int(np.count_nonzero(eng._mse_free()))
    assert eng.b.n_solves - n0 == n_free + chk["n_extra_solves"]
    scale = np.linalg.norm(J_smooth, axis=0)
    err = np.linalg.norm(J - J_smooth, axis=0)
    assert np.all(err <= 0.05 * scale + 1e-12)


def test_mutant_without_halvings_differences_across_the_jump(monkeypatch):
    eng, g, lam, x_b, snap_b, tg_b, ax, _rec = _refresh_base(BranchToy())
    J_smooth = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=ax)
    _arm_flip(eng, g, lam, x_b, snap_b)
    monkeypatch.setattr(be, "MSE_FD_MAX_HALVINGS", 0)
    J_bad = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=ax)
    chk = eng._mse_fd_last
    assert chk["n_columns_unresolved"] >= 1
    i = 1
    assert np.linalg.norm(J_bad[:, i] - J_smooth[:, i]) \
        > 1.0 * np.linalg.norm(J_smooth[:, i])


def test_without_a_jump_the_jacobian_is_bit_for_bit_unchanged():
    """The plain toy (no minor radius: the check is inactive) and the
    one-branch toy (check active, nothing refined) give the same Jacobian,
    and both equal the plain forward difference."""
    eng, g, lam, x_b, snap_b, tg_b, ax, rec = _refresh_base(BranchToy())
    J_on = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=ax)
    J_off = eng._mse_fd(g, lam, x_b, tg_b, snap_b, axis0=None)
    assert not eng._mse_fd_last["active"]
    assert np.array_equal(J_on, J_off)
    md, ch = _mse_data()
    eng2, res2, rec2 = _run(T.ToyGS(chords=ch), md=md, ch=ch)
    fd2 = [p for p in rec2["phases"] if p["name"] == "mse"][-1]["jacobian"]
    assert fd2["branch_check"]["active"] is False
    assert fd2["branch_check"]["n_columns_refined"] == 0
    # the reconstruction itself is the same on both toys (no jump anywhere)
    assert rec2["state"]["x"] == rec["state"]["x"]
