"""AST normalisation that undoes the ``bouquet.edge_pressure`` substitution.

Every scattered ``pp["y"][-1] = 0.0`` / ``pax = p[0]`` site was replaced by a
call of the one helper (``solver_pp_profile`` / ``solver_pprime`` /
``solver_pax`` / ``solver_pressure``) carrying the settings object ``_edge``.
:func:`expand` rewrites the CURRENT code back to the inline statements those
calls stand for at the PRE-CHANGE settings (``edge_pprime_pin=True``,
``separatrix_pressure="legacy"``; :data:`bouquet.edge_pressure.
PRE_CHANGE_EDGE_PRESSURE` -- no longer the defaults since 2026-10-02) and
removes exactly what the change
ADDED (the ``edge_pressure`` keyword / parameter, the ``_edge`` and
``_p_lcfs`` locals, the record writes, the ``lcfs_kwargs`` splat), so the
result can be compared, as an AST, with the frozen pre-change code.  That
the helper at the pre-change settings IS those inline statements, bit for
bit, is ``tests/test_edge_pressure.py``.

:func:`common` is applied to BOTH sides: ``float(x[0])`` and ``x[0]`` are
the same axis target (the helper returns the float), and a walrus is its
value.

WHAT THIS NORMALISER CANNOT SEE (it is deliberately not extended; each
blind spot is covered by a BEHAVIOURAL test instead):

* every ``edge_pressure=`` keyword and its VALUE, and every ``_edge`` /
  ``_p_lcfs`` binding -- a site passing ``None`` (which resolves to the
  defaults, "offset" since 2026-10-02) or another settings object compares
  equal.  Covered by tests/test_edge_pressure_settings_reach.py (the
  bindings and keywords on the source, and -- executed -- the value the
  draw loop's redo and the legacy sigma=0 route actually receive);
* the ``lcfs_kwargs(...)`` splat on ``save_eqdsk`` -- which ``lcfs_pressure``
  a written g-file asks for.  Covered by
  tests/test_edge_pressure_baseline_gfile.py (the save keyword of the
  baseline and of every draw: its own ``p_sep``, or absent under "legacy");
* the record writes (``store_edge_pressure_record``, ``bl.edge_pressure =``)
  -- covered by tests/test_edge_pressure.py's archive-record tests;
* ``eq_stats_iter = None`` (the archive stage's failed ``get_stats``) --
  covered by tests/test_engine_draws_behaviour.py (the draw is archived with
  l_i NaN and an edge record without frames);
* whole ``if _eng ...`` blocks (removed by the engine-draw AST test's own
  normaliser) -- the engine branches are covered numerically by the engine
  tests;
* callees, module constants and config DEFAULTS: a changed default (e.g. the
  post-homotopy ceiling 4 -> 6) or a changed helper passes every AST test.
  The helper at the pre-change settings is pinned bit for bit by
  tests/test_edge_pressure.py; defaults by the config tests.

Solver-free; no data.
"""
import ast

#: locals the change added
DROP_LOCALS = {"_edge", "_p_lcfs"}
#: keywords / parameters / dict keys the change added
DROP_KW = {"edge_pressure"}
#: statement-level calls the change added
DROP_CALLS = {"store_edge_pressure_record"}
#: attribute assignments the change added (``bl.edge_pressure = ...``)
DROP_ATTRS = {"edge_pressure"}


def _name(node):
    return node.id if isinstance(node, ast.Name) else None


def _is_helper(node, name):
    return (isinstance(node, ast.Call) and _name(node.func) == name)


def _sub(value, index):
    return ast.Subscript(value=value, slice=ast.Constant(value=index),
                         ctx=ast.Load())


def _neg1(value):
    return ast.Subscript(
        value=value,
        slice=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
        ctx=ast.Store())


def _deriv(psi, p, r):
    return ast.BinOp(
        left=ast.Call(func=ast.Name(id="pchip_derivative", ctx=ast.Load()),
                      args=[psi, p], keywords=[]),
        op=ast.Div(), right=r)


class _Expand(ast.NodeTransformer):
    """The current code with the helper calls written out inline."""

    def _stmts(self, body):
        out = []
        for st in body:
            r = self.visit(st)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            v = getattr(node, field, None)
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                new = self._stmts(v)
                if field == "body" and not new:
                    new = [ast.Pass()]
                setattr(node, field, new)
        for field, old in ast.iter_fields(node):
            if field in ("body", "orelse", "finalbody") and isinstance(
                    old, list) and (not old or isinstance(old[0], ast.stmt)):
                continue
            if isinstance(old, list):
                new = []
                for v in old:
                    if isinstance(v, ast.AST):
                        v = self.visit(v)
                        if v is None:
                            continue
                        if isinstance(v, list):
                            new.extend(v)
                            continue
                    new.append(v)
                old[:] = new
            elif isinstance(old, ast.AST):
                new = self.visit(old)
                if new is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new)
        return node

    # ---- statements ----------------------------------------------------
    def visit_Assign(self, node):
        t = node.targets
        if len(t) == 1 and _name(t[0]) in DROP_LOCALS:
            return None
        if (len(t) == 1 and isinstance(t[0], ast.Attribute)
                and t[0].attr in DROP_ATTRS):
            return None
        if (len(t) == 1 and _name(t[0]) == "eq_stats_iter"
                and isinstance(node.value, ast.Constant)
                and node.value.value is None):
            return None
        v = node.value
        if _is_helper(v, "solver_pp_profile") and len(t) == 1:
            psi, p, r = (self.visit(a) for a in v.args[:3])
            tgt = t[0]
            d = ast.Dict(
                keys=[ast.Constant(value="type"), ast.Constant(value="y"),
                      ast.Constant(value="x")],
                values=[ast.Constant(value="linterp"), _deriv(psi, p, r),
                        psi])
            load = ast.Name(id=tgt.id, ctx=ast.Load())
            return [ast.Assign(targets=[tgt], value=d),
                    ast.Assign(targets=[_neg1(_sub(load, "y"))],
                               value=ast.Constant(value=0.0))]
        if _is_helper(v, "solver_pprime") and len(t) == 1:
            psi, p, r = (self.visit(a) for a in v.args[:3])
            tgt = t[0]
            load = ast.Name(id=tgt.id, ctx=ast.Load())
            return [ast.Assign(targets=[tgt], value=_deriv(psi, p, r)),
                    ast.Assign(targets=[_neg1(load)],
                               value=ast.Constant(value=0.0))]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
                and _name(node.value.func) in DROP_CALLS):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "edge_pressure":
            return None
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if node.body == [] or (len(node.body) == 1 and isinstance(
                node.body[0], ast.Pass) and not node.orelse):
            return None
        return node

    # ---- expressions ---------------------------------------------------
    def visit_Call(self, node):
        if _is_helper(node, "solver_pax"):
            return _sub(self.visit(node.args[0]), 0)
        if _is_helper(node, "solver_pressure"):
            return self.visit(node.args[0])
        self.generic_visit(node)
        node.keywords = [
            k for k in node.keywords
            if k.arg not in DROP_KW
            and not (k.arg is None and _is_helper(k.value, "lcfs_kwargs"))]
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        keep = [(k, v) for k, v in zip(node.keys, node.values)
                if not (isinstance(k, ast.Constant) and k.value in DROP_KW)]
        node.keys = [k for k, _ in keep]
        node.values = [v for _, v in keep]
        return node

    def visit_FunctionDef(self, node):
        a = node.args
        names = [x.arg for x in a.args]
        for kw in DROP_KW:
            if kw in names:
                i = names.index(kw)
                first_def = len(a.args) - len(a.defaults)
                del a.args[i]
                if i >= first_def:
                    del a.defaults[i - first_def]
                names = [x.arg for x in a.args]
        return self.generic_visit(node)


class _Common(ast.NodeTransformer):
    """Applied to both sides: ``float(x[0])`` is ``x[0]``; a walrus is its
    value."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if (_name(node.func) == "float" and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Subscript)
                and isinstance(node.args[0].slice, ast.Constant)
                and node.args[0].slice.value == 0):
            return node.args[0]
        return node

    def visit_NamedExpr(self, node):
        self.generic_visit(node)
        return node.value


def strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b = n.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                n.body = b[1:] or [ast.Pass()]
    return tree


def expand(tree):
    """The current code with the edge-pressure substitution undone."""
    return _Expand().visit(tree)


def common(tree):
    return _Common().visit(tree)


def dump(tree):
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)
