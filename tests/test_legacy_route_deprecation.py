"""1.4.0 deprecates the legacy route (removal in 1.5, #90): one
DeprecationWarning per run on reconstruction_engine="legacy" or
solve_method="legacy", none on the unified engine. Fast: the warning fires
where the run is resolved (prepare_baseline), before any solve, so the
baseline resolution is stubbed out."""

import os
import warnings

import pytest

import bouquet as bq
import bouquet.baseline as bqb
from bouquet.config import LEGACY_ROUTE_DEPRECATION

_HERE = os.path.dirname(os.path.abspath(__file__))
_G = os.path.join(_HERE, "data", "g_synthetic.geqdsk")
_IDA = os.path.join(_HERE, "data", "diiid_profs_synthetic.cdf")
_MESH = os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5")


class _Stop(Exception):
    pass


def _run(route):
    kw = {"reconstruction_engine": "legacy"} if route == "engine_field" else {}
    run = bq.Bouquet.from_geqdsk(_G, profiles=_IDA, mesh=_MESH, **kw)
    if route == "solve_method":
        run.generation.solve_method = "legacy"
    return run


def _deprecations(run, monkeypatch, calls=3):
    def stop(*a, **k):
        raise _Stop()
    monkeypatch.setattr(bqb, "resolve_baseline", stop)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(calls):
            try:
                run.prepare_baseline()
            except _Stop:
                pass
            except ValueError:
                pass        # the unified engine refuses without a solver; no warning there
    return [w for w in rec if issubclass(w.category, DeprecationWarning)
            and "1.5" in str(w.message)]


@pytest.mark.parametrize("route", ["engine_field", "solve_method"])
def test_legacy_route_warns_once_per_run_naming_removal(route, monkeypatch):
    if not os.path.isfile(_MESH):
        pytest.skip("example mesh absent")
    deps = _deprecations(_run(route), monkeypatch)
    assert len(deps) == 1
    msg = str(deps[0].message)
    assert msg == LEGACY_ROUTE_DEPRECATION
    assert "removed in bouquet 1.5" in msg and "#90" in msg
    # attributed to the caller, not to bouquet's own frames: the default
    # filters then show it in a script or a notebook
    assert os.path.abspath(deps[0].filename) == os.path.abspath(__file__)


def test_a_new_run_warns_again(monkeypatch):
    if not os.path.isfile(_MESH):
        pytest.skip("example mesh absent")
    assert len(_deprecations(_run("engine_field"), monkeypatch)) == 1
    assert len(_deprecations(_run("engine_field"), monkeypatch)) == 1


def test_unified_engine_does_not_warn(monkeypatch):
    if not os.path.isfile(_MESH):
        pytest.skip("example mesh absent")
    assert _deprecations(_run("default"), monkeypatch) == []
