"""The g-file reconstruction's l_i secant survives a failed SECOND solve.

``reconstruct_equilibrium`` matches l_i by a secant on the inductive scale:
it evaluates ``ind_factor = 1`` (the solved state) and ``1.05``, then
iterates.  ``_solve_and_get_li`` returns ``None`` when a solve fails and the
loop has a branch for that -- but the report of the second evaluation
formatted ``None`` with ``:.6f`` and the first loop pass subtracted from it,
so a failed second solve raised ``TypeError: unsupported format string
passed to NoneType.__format__`` instead of reaching that branch (and a
failed later solve raised ``TypeError`` one line further on).

The block is taken from the CURRENT source and run on a stand-in solver
(the pattern of ``tests/test_recon_fast_pressure.py``):

* with the second solve failing, it no longer raises and still matches l_i
  through its failed-solve branch;
* where the old block ran (no failure, or a failure it did not trip on),
  the current one takes exactly the path of the pre-change block (taken
  from the frozen copy ``tests/data/edge_pressure_prechange.py.txt`` with
  this fix's own two hunks put back, see ``_frozen``): same
  solves, same scales, same result, bit for bit.

Solver-free; no data.
"""
import inspect
import os
import textwrap

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(_HERE, "data", "edge_pressure_prechange.py.txt")

_START = "_last_good_psi = mygs.get_psi(False).copy()"
_END = "# Ensure the final state is from a converged solve"


def _block(src):
    i = src.index(_START)
    j = src.index(_END, i)
    k = src.index("_restore_psi()", j) + len("_restore_psi()")
    # back to the start of the first line, keep its indentation
    i = src.rindex("\n", 0, i) + 1
    body = textwrap.dedent(src[i:k])
    code = ("def _secant(mygs, eqdsk, j_inductive_fit, j_BS_isolated, "
            "pp_prof, psi_pad, li_target, li_tol, max_li_iters, "
            "max_step_frac):\n"
            + textwrap.indent(body, "    ")
            + "\n    return ind_1, li_1_sec\n")
    ns = {"np": np}
    exec(compile(code, "<li-secant>", "exec"), ns)
    return ns["_secant"]


def _current():
    from bouquet.TokaMaker_interface import reconstruct_equilibrium
    return _block(inspect.getsource(reconstruct_equilibrium))


_FIX_NEW = (
    ('print(f"[li match] iter 1: ind_factor={ind_1:.6f}  "\n'
     '              + (f"li={li_1_sec:.6f}  err={li_1_sec - li_target:.6f}"\n'
     '                 if li_1_sec is not None else "li=FAILED  err=N/A"))',
     'print(f"[li match] iter 1: ind_factor={ind_1:.6f}  '
     'li={li_1_sec:.6f}  err={li_1_sec - li_target:.6f}")'),
    ("err_1 = (li_1_sec - li_target) if li_1_sec is not None else None",
     "err_1 = li_1_sec - li_target"),
)


# The later last-good-state fix (the failed last secant solve hands on the
# restored state's factor).  The frozen copy may or may not have been
# re-based by it yet (the owner applies snapshot re-bases), so these are put
# back only where present -- and the pre-fix statement must be there after.
_FIX_LAST_GOOD = (
    ("        _good = (ind_0, li_0)\n", ""),
    ("        if li_1_sec is not None:\n"
     "            _update_bracket(ind_1, li_1_sec)\n"
     "            _good = (ind_1, li_1_sec)\n",
     "        if li_1_sec is not None:\n"
     "            _update_bracket(ind_1, li_1_sec)\n"),
    ("            ind_0, li_0 = _good\n",
     "            ind_0, li_0 = ind_1, li_1_sec if li_1_sec is not None "
     "else li_0\n"),
    ("                _update_bracket(ind_1, li_1_sec)\n"
     "                _good = (ind_1, li_1_sec)\n",
     "                _update_bracket(ind_1, li_1_sec)\n"),
    ("        if li_1_sec is None:\n"
     "            ind_1, li_1_sec = _good\n"
     "            print(f\"[li match] last secant solve FAILED: restoring the "
     "last \"\n"
     "                  f\"good state ind_factor={ind_1:.6f}  "
     "li={li_1_sec:.6f}\")\n"
     "            mygs.set_profiles(ffp_prof={\n"
     "                \"type\": \"jphi-linterp\",\n"
     "                \"y\": ind_1 * j_inductive_fit + j_BS_isolated,\n"
     "                \"x\": eqdsk.psi_N}, pp_prof=pp_prof)\n"
     "            _restore_psi()\n",
     "        if li_1_sec is None:\n"
     "            _restore_psi()\n"),
)


def _frozen():
    """The PRE-FIX block.  The frozen copy was re-based by this fix's own two
    hunks (it serves the edge-pressure AST test, which compares against the
    current code), so the pre-fix block is the frozen text with exactly those
    two statements put back; each put-back must actually happen.  The later
    last-good-state hunks are put back where the copy carries them."""
    with open(_FROZEN) as fh:
        txt = fh.read()
    i = txt.index("def reconstruct_equilibrium(")
    txt = txt[i:]
    for new, old in _FIX_NEW:
        assert txt.count(new) == 1, new
        txt = txt.replace(new, old)
    for new, old in _FIX_LAST_GOOD:
        if new in txt:
            assert txt.count(new) == 1, new
            txt = txt.replace(new, old)
    assert ("ind_0, li_0 = ind_1, li_1_sec if li_1_sec is not None else li_0"
            in txt)
    assert "_good = (" not in txt and "= _good" not in txt
    return _block(txt)


class _GS:
    """l_i linear in the inductive scale; the solves listed in *fail* (by
    call number, 1 = the second evaluation) raise as the solver does."""

    def __init__(self, fail=()):
        self.fail, self.n, self.scale, self.solves = set(fail), 0, 1.0, []

    def get_psi(self, _normalised):
        return np.zeros(4)

    def set_psi(self, psi, update_bounds=True):
        pass

    def set_profiles(self, ffp_prof=None, pp_prof=None):
        y = np.asarray(ffp_prof["y"], float)
        self.scale = float(y[0])           # j_inductive_fit = 1, j_BS = 0

    def solve(self):
        self.n += 1
        self.solves.append(self.scale)
        if self.n in self.fail:
            raise ValueError('Error in solve: Exceeded "maxits"')

    def get_stats(self, li_normalization="iter", lcfs_pad=None):
        return {"l_i": 0.80 + 0.50 * (self.scale - 1.0)}


class _Eq:
    psi_N = np.linspace(0.0, 1.0, 5)


def _run(fn, gs):
    return fn(gs, _Eq(), np.ones(5), np.zeros(5), None, 1e-3,
              li_target=0.81, li_tol=1e-3, max_li_iters=20,
              max_step_frac=0.10)


def test_a_failed_second_solve_reaches_the_failed_solve_branch(capsys):
    gs = _GS(fail={1})
    ind, li = _run(_current(), gs)
    out = capsys.readouterr().out
    assert "iter 1: ind_factor=1.050000  li=FAILED  err=N/A" in out
    assert "midpoint fallback" in out
    assert li is not None and abs(li - 0.81) < 1e-3
    assert gs.solves[0] == pytest.approx(1.05) and gs.n > 2


def test_a_failed_later_solve_reaches_the_failed_solve_branch(capsys):
    """The same defect one line later: the first pass after ANY failed
    secant solve subtracted from ``None``."""
    gs = _GS(fail={2, 4})
    ind, li = _run(_current(), gs)
    out = capsys.readouterr().out
    assert "li=FAILED  err=N/A" in out
    assert li is not None and abs(li - 0.81) < 1e-3


@pytest.mark.parametrize("fail", [(1,), (2, 4)])
def test_the_pre_change_block_did_raise_there(fail):
    """The defect is real (negative control on the frozen code)."""
    with pytest.raises(TypeError, match="NoneType"):
        _run(_frozen(), _GS(fail=fail))


@pytest.mark.parametrize("fail", [(), (3,)])
def test_where_the_old_code_ran_the_path_is_unchanged(fail, capsys):
    a, b = _GS(fail=fail), _GS(fail=fail)
    ra = _run(_current(), a)
    out_a = capsys.readouterr().out
    rb = _run(_frozen(), b)
    out_b = capsys.readouterr().out
    assert ra == rb and a.solves == b.solves
    assert out_a == out_b


# -- the delivered inductive factor is the one of the state actually held ---
#
# When the LAST secant solve fails, psi is restored to the last good state.
# The inductive factor handed on (``ind_1`` -> ``j_ind_li`` and the
# ``_fm["ind_1"]`` stamp the corrective iteration starts from) must be the
# one THAT state was solved with, and the solver must hold that state's
# profile too -- not the failed factor, whose l_i was never achieved.


class _StateGS:
    """A stand-in whose l_i is a property of the SOLVED state (held in psi),
    not of the last profile set: ``set_psi`` restores a state, a failed solve
    leaves a garbage state behind, and l_i is non-linear in the scale so the
    secant does not land in one step."""

    def __init__(self, fail=()):
        self.fail, self.n, self.solves = set(fail), 0, []
        self.profile_scale = 1.0
        self.state = 1.0                    # the solved state at ind = 1

    @staticmethod
    def li(s):
        return 0.80 + 0.50 * (s - 1.0) + 4.0 * (s - 1.0) ** 2

    def get_psi(self, _normalised):
        return np.array([self.state])

    def set_psi(self, psi, update_bounds=True):
        self.state = float(np.asarray(psi)[0])

    def set_profiles(self, ffp_prof=None, pp_prof=None):
        self.profile_scale = float(np.asarray(ffp_prof["y"], float)[0])

    def solve(self):
        self.n += 1
        self.solves.append(self.profile_scale)
        if self.n in self.fail:
            self.state = float("nan")       # what a failed solve leaves
            raise ValueError('Error in solve: Exceeded "maxits"')
        self.state = self.profile_scale

    def get_stats(self, li_normalization="iter", lcfs_pad=None):
        return {"l_i": self.li(self.state)}


def _run_state(fn, gs, max_li_iters):
    return fn(gs, _Eq(), np.ones(5), np.zeros(5), None, 1e-3,
              li_target=0.86, li_tol=1e-4, max_li_iters=max_li_iters,
              max_step_frac=0.10)


@pytest.mark.parametrize("max_li_iters", [3, 4, 5])
def test_a_failed_last_solve_hands_on_the_restored_states_factor(
        max_li_iters, capsys):
    n_last = max_li_iters - 1                # solve calls: 1 + (iters - 2)
    gs = _StateGS(fail={n_last})
    ind, li = _run_state(_current(), gs, max_li_iters)
    out = capsys.readouterr().out
    assert gs.n == n_last                    # the last solve was the failure
    failed = gs.solves[-1]
    assert ind != failed
    # the solver holds the state of the factor handed on, with its profile
    assert gs.state == ind and gs.profile_scale == ind
    assert li == gs.li(gs.state)
    assert ind in gs.solves[:-1]             # a GOOD evaluation's factor
    assert "last secant solve FAILED" in out


def test_the_pre_fix_block_handed_on_the_failed_factor():
    """Negative control: the pre-fix block (frozen copy) returned the failed
    scale while the solver held the restored state."""
    gs = _StateGS(fail={3})
    ind, li = _run_state(_frozen(), gs, 4)
    assert ind == gs.solves[-1]              # the FAILED factor
    assert li is None
    assert gs.state != ind


@pytest.mark.parametrize("fail", [(2,), (2, 3), (3,), (2, 4), (1, 3)])
def test_after_a_failure_the_secant_pairs_good_points_only(fail):
    """Every evaluation the secant uses as its previous point is a good one:
    the result is a state whose own l_i is the one returned, and l_i is
    matched within the iterations."""
    gs = _StateGS(fail=fail)
    ind, li = _run_state(_current(), gs, 20)
    assert gs.state == ind and li == gs.li(ind)
    assert abs(li - 0.86) < 1e-4


class _LinearStateGS(_StateGS):
    """l_i exactly LINEAR in the solved state: a secant through any two
    good (factor, l_i) pairs lands on the target in one step."""

    @staticmethod
    def li(s):
        return 0.80 + 0.50 * (s - 1.0)


def test_after_a_failure_the_next_secant_step_uses_two_good_points():
    """Behavioural pin of "the secant pairs good points" (the 2026-10-06
    review's mutant S1, which put back the old pairing and survived every
    behavioural test).  With l_i linear in the factor: 1.05 is good; the
    secant proposes 1.12, whose solve FAILS; the retreat goes halfway toward
    the last good point (1.085, good); the next secant runs through the two
    GOOD evaluations (1.05 and 1.085) and lands on the target exactly, at
    1.12.  The old pairing used the failed factor 1.12 with 1.05's l_i as
    its previous point, a wrong-signed slope that steps back to 1.05."""
    gs = _LinearStateGS(fail={2})
    ind, li = _run_state(_current(), gs, 20)
    np.testing.assert_allclose(gs.solves, [1.05, 1.12, 1.085, 1.12],
                               rtol=0.0, atol=1e-12)
    assert ind == pytest.approx(1.12, abs=1e-12)
    assert abs(li - 0.86) < 1e-12 and gs.state == ind
