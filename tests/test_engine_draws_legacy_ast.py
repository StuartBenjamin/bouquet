"""The legacy draw path is bit-identical with the engine-draw hooks in place
(frozen-copy pattern, at the level of the code).

Every engine branch Stage 3 added to the legacy draw-path functions is a
STATEMENT-LEVEL block gated on one local, ``_eng`` (``if _eng is not None:``
/ ``if _eng is None: <legacy> else: <engine>``), set from
``generate_bouquet(engine_draw=...)`` / ``run._engine_gate`` (``None`` under
``reconstruction_engine="legacy"``), plus the ``engine_draw`` keyword.
Removing exactly those constructs from the CURRENT functions must give the
AST of the frozen pre-change functions (``tests/data/
engine_draws_legacy_prechange.py.txt``, the Stage 3 base) -- so with
``_eng is None`` the legacy path executes the very same statements it did
before (docstrings and comments are not compared; they carry no behaviour).

The same functions later had their inline ``P'`` / axis-target statements
replaced by the one helper of ``bouquet.edge_pressure``.  That substitution
is undone here too (``tests/_edge_pressure_ast.py``: every helper call is
written back as the inline statements it stands for at the pre-change
settings -- pin on, ``"legacy"``, no longer the defaults --
and what the change added is removed) BEFORE the comparison, so the frozen
file is still the untouched Stage 3 base; that the helper at the pre-change
settings is those inline statements bit for bit is
``tests/test_edge_pressure.py``.

Solver-free; no data.
"""
import ast
import inspect
import os
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(_HERE, "data", "engine_draws_legacy_prechange.py.txt")

#: the gate local and the keyword the hooks use
GATES = {"_eng"}
DROP_KW = {"engine_draw"}


def _gate(test):
    """``"is not"`` / ``"is"`` for ``<gate> is [not] None``, else None."""
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id in GATES and len(test.ops) == 1
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None):
        return "is not" if isinstance(test.ops[0], ast.IsNot) else "is"
    return None


class _Prune(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        g = _gate(node.test)
        if g is None:
            return node
        keep = node.orelse if g == "is not" else node.body
        return list(keep) or None

    def visit_Assign(self, node):
        if all(isinstance(t, ast.Name) and t.id in GATES
               for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg not in DROP_KW]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        a = node.args
        names = [x.arg for x in a.args]
        for kw in DROP_KW:
            if kw in names:
                i = names.index(kw)
                nd = len(a.defaults)
                first_def = len(a.args) - nd
                del a.args[i]
                if i >= first_def:
                    del a.defaults[i - first_def]
                names = [x.arg for x in a.args]
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node


def _strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b = n.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                n.body = b[1:] or [ast.Pass()]
    return tree


def _frozen():
    with open(_FROZEN) as fh:
        tree = ast.parse(fh.read())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _current(obj):
    src = textwrap.dedent(inspect.getsource(obj))
    return ast.parse(src).body[0]


def _targets():
    from bouquet import TokaMaker_interface as TI
    from bouquet.run import Bouquet
    return {"generate_bouquet": TI.generate_bouquet,
            "_post_homotopy_jbs": TI._post_homotopy_jbs,
            "generate": Bouquet.generate,
            "verify_sigma0_consistency": Bouquet.verify_sigma0_consistency,
            "_validate_workflow": Bouquet._validate_workflow}


@pytest.mark.parametrize("name", ["generate_bouquet", "_post_homotopy_jbs",
                                  "generate", "verify_sigma0_consistency",
                                  "_validate_workflow"])
def test_the_legacy_path_is_the_frozen_code_without_the_engine_branches(name):
    frozen = _strip_docstrings(ast.Module(body=[_frozen()[name]],
                                          type_ignores=[]))
    cur = _Prune().visit(ast.Module(body=[_current(_targets()[name])],
                                    type_ignores=[]))
    cur = _strip_docstrings(cur)
    import _edge_pressure_ast as EA
    frozen = EA.common(frozen)
    cur = EA.common(EA.expand(cur))
    a = ast.dump(frozen, include_attributes=False)
    b = ast.dump(cur, include_attributes=False)
    assert a == b, (f"{name}: the legacy path differs from the frozen "
                    "pre-change code beyond the gated engine branches")


def test_the_engine_branches_are_actually_there():
    """The pruning is not vacuous: each hooked function carries gated
    engine branches (so the comparison above did remove something)."""
    for name, obj in _targets().items():
        tree = _current(obj)
        n = sum(1 for x in ast.walk(tree)
                if isinstance(x, ast.If) and _gate(x.test) is not None)
        assert n >= 1, name
    tree = _current(_targets()["generate_bouquet"])
    n = sum(1 for x in ast.walk(tree)
            if isinstance(x, ast.If) and _gate(x.test) is not None)
    assert n >= 8
