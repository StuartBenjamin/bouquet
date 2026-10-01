"""The solve paths are the frozen pre-change code with the edge-pressure
helper written back inline (frozen-copy pattern, at the level of the code).

``bouquet.edge_pressure`` replaced every inline ``P'`` / axis-target
statement (``pp["y"][-1] = 0.0``, ``pax = p[0]``) by one helper.  For each
function that solves, undoing that substitution on the CURRENT code
(``tests/_edge_pressure_ast.py``: each helper call becomes the inline
statements it stands for at the default settings; what the change added --
the ``edge_pressure`` keyword, the ``_edge`` local, the record writes -- is
removed) must give the AST of the frozen pre-change function
(``tests/data/edge_pressure_prechange.py.txt``).  With the helper's defaults
bit-identical to those inline statements (``tests/test_edge_pressure.py``),
every such path hands the solver the arrays it did before, at the defaults.

``generate_bouquet``, ``_post_homotopy_jbs``, ``Bouquet.generate`` and
``Bouquet.verify_sigma0_consistency`` are compared the same way against
their own (older) frozen copy in ``tests/test_engine_draws_legacy_ast.py``;
the engine backend and the engine draws are compared numerically in
``tests/test_edge_pressure.py``.

Solver-free; no data.
"""
import ast
import inspect
import os
import textwrap

import pytest

import _edge_pressure_ast as EA

_HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(_HERE, "data", "edge_pressure_prechange.py.txt")

NAMES = ["_std_candidate_solve", "perturb_kinetic_equilibrium",
         "reconstruct_equilibrium", "_forward_solve_imas_baseline",
         "_verify_sigma0_jbs_loop", "_sigma0_draw_route"]
HELPERS = {"solver_pp_profile", "solver_pprime", "solver_pax",
           "solver_pressure"}


def _frozen():
    with open(_FROZEN) as fh:
        tree = ast.parse(fh.read())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _targets():
    from bouquet import TokaMaker_interface as TI
    from bouquet.run import Bouquet
    return {"_std_candidate_solve": TI._std_candidate_solve,
            "perturb_kinetic_equilibrium": TI.perturb_kinetic_equilibrium,
            "reconstruct_equilibrium": TI.reconstruct_equilibrium,
            "_forward_solve_imas_baseline":
                Bouquet._forward_solve_imas_baseline,
            "_verify_sigma0_jbs_loop": Bouquet._verify_sigma0_jbs_loop,
            "_sigma0_draw_route": Bouquet._sigma0_draw_route}


def _current(obj):
    return ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]


def _module(node):
    return ast.Module(body=[node], type_ignores=[])


def _helper_calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id in HELPERS]


@pytest.mark.parametrize("name", NAMES)
def test_the_solve_path_is_the_frozen_code_with_the_helper_inlined(name):
    frozen = EA.common(EA.strip_docstrings(_module(_frozen()[name])))
    cur = EA.common(EA.expand(EA.strip_docstrings(
        _module(_current(_targets()[name])))))
    assert EA.dump(frozen) == EA.dump(cur), (
        f"{name}: the code differs from the frozen pre-change code beyond "
        "the edge-pressure helper substitution")


@pytest.mark.parametrize("name", NAMES)
def test_the_substitution_is_actually_there(name):
    """Not vacuous: each function calls the helper (so the expansion above
    rewrote something), the frozen copy does not, and without the expansion
    the two differ."""
    cur = _current(_targets()[name])
    assert len(_helper_calls(cur)) >= 1, name
    assert not _helper_calls(_frozen()[name]), name
    frozen = EA.common(EA.strip_docstrings(_module(_frozen()[name])))
    raw = EA.common(EA.strip_docstrings(_module(_current(_targets()[name]))))
    assert EA.dump(frozen) != EA.dump(raw)


def test_no_inline_site_is_left_in_the_package():
    """Every ``P'`` edge pin and axis target of the package goes through the
    helper: no ``...[-1] = 0.0`` on a ``P'`` array and no ``pax=<p>[0]``
    outside ``bouquet/edge_pressure.py``."""
    import re
    root = os.path.join(os.path.dirname(_HERE), "bouquet")
    pin = re.compile(r"\[\s*-1\s*\]\s*=\s*0\.0?\b")
    pax = re.compile(r"pax\s*=\s*(?:float\()?[\w.\[\]\"']+\[\s*0\s*\]")
    bad = []
    for name in ("TokaMaker_interface.py", "run.py", "engine.py",
                 "engine_draws.py", "baseline.py", "jbs_loop.py",
                 "adapters.py", "parallel.py"):
        with open(os.path.join(root, name)) as fh:
            for i, ln in enumerate(fh, 1):
                if ln.lstrip().startswith("#"):
                    continue
                if pin.search(ln) or pax.search(ln):
                    bad.append(f"{name}:{i}: {ln.strip()}")
    assert not bad, bad


def test_the_expansion_writes_the_pin():
    """The normaliser itself: a helper call expands to the derivative over
    the flux range AND the last-node zeroing (dropping either changes the
    AST)."""
    src = "y = solver_pprime(psi_N, p, psi_range, _edge)\n"
    want = "y = pchip_derivative(psi_N, p) / psi_range\ny[-1] = 0.0\n"
    assert EA.dump(EA.expand(ast.parse(src))) == EA.dump(ast.parse(want))
    nopin = "y = pchip_derivative(psi_N, p) / psi_range\n"
    assert EA.dump(EA.expand(ast.parse(src))) != EA.dump(ast.parse(nopin))
    src = "pp = solver_pp_profile(psi_N, p, (b[1] - b[0]), _edge)\n"
    want = ('pp = {"type": "linterp", "y": pchip_derivative(psi_N, p) / '
            '(b[1] - b[0]), "x": psi_N}\npp["y"][-1] = 0.0\n')
    assert EA.dump(EA.expand(ast.parse(src))) == EA.dump(ast.parse(want))
    src = "g.set_targets(Ip=Ip, pax=solver_pax(p, _edge))\n"
    want = "g.set_targets(Ip=Ip, pax=float(p[0]))\n"
    assert EA.dump(EA.common(EA.expand(ast.parse(src)))) == EA.dump(
        EA.common(ast.parse(want)))
