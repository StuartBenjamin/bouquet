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


def _frozen():
    """The PRE-FIX block.  The frozen copy was re-based by this fix's own two
    hunks (it serves the edge-pressure AST test, which compares against the
    current code), so the pre-fix block is the frozen text with exactly those
    two statements put back; each put-back must actually happen."""
    with open(_FROZEN) as fh:
        txt = fh.read()
    i = txt.index("def reconstruct_equilibrium(")
    txt = txt[i:]
    for new, old in _FIX_NEW:
        assert txt.count(new) == 1, new
        txt = txt.replace(new, old)
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
