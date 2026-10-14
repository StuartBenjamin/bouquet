"""The opt-in bistability check (``GenerationConfig.check_bistability``) --
fast, on the toy stand-in solver of ``tests/_engine_fake_gs.py``.

After the reconstruction, ONE extra solve of the same inputs from a COLD
start is compared with the delivered state (l_i(3) and q0 at the q-row radius
at the sigma=0 gates; beta_pol and the axis reported).  A miss stamps the
baseline ``bistable=True`` with the deltas and warns; the delivered
equilibrium is restored.  Off by default: no extra solve.

Synthetic inputs only; no device data.
"""
import os
import sys
import warnings

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_engine_draws as TD  # noqa: E402

toy_bouquet_solver = TD.toy_bouquet_solver
_quiet = TD._quiet


def _prepared(tmp_path, monkeypatch, *, check, fake_cls=None):
    """A toy Bouquet on the unified engine, prepared with
    ``check_bistability=check``; ``fake_cls`` the stand-in solver class."""
    import bouquet.run as br
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.solver_state import enter_bounded_coil_mode
    cls = fake_cls or FakeTokaMaker

    def _setup(self):
        gs = cls(None)
        enter_bounded_coil_mode(gs)
        return self._install_solver(gs)
    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    b = TD._bq(tmp_path)
    b.config.generation.check_bistability = check
    b.setup_solver()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b.prepare_baseline()
    return b, rec


def _n(calls, kind):
    return sum(1 for c in calls if c[0] == kind)


def test_off_by_default_no_extra_solve(tmp_path, toy_bouquet_solver,
                                       monkeypatch):
    from bouquet.config import GenerationConfig
    assert GenerationConfig().check_bistability is False
    b, rec = _prepared(tmp_path, monkeypatch, check=False)
    assert "bistability" not in b.baseline.engine
    # the reconstruction's own start only; no cold re-solve on the solver
    assert _n(b.mygs.calls, "init_psi") == 1
    assert _n(b.mygs.calls, "solve") == 0
    assert not [w for w in rec if "BISTABLE" in str(w.message)]


def test_a_single_branch_baseline_is_stamped_not_bistable(
        tmp_path, toy_bouquet_solver, monkeypatch):
    b, rec = _prepared(tmp_path, monkeypatch, check=True)
    r = b.baseline.engine["bistability"]
    assert r["checked"] is True and r["bistable"] is False
    assert r["outcome"] == "single" and r["extra_solves"] == 1
    # exactly ONE extra, cold solve: a second init_psi, then one solve
    assert _n(b.mygs.calls, "init_psi") == 2 and _n(b.mygs.calls, "solve") == 1
    assert r["deltas"]["l_i"] == 0.0 and r["deltas"]["q0"] == 0.0
    g = r["gates"]
    s = b._engine_run["engine"].delivered_meas
    assert g["dl_i"]["setting"] == "jbs_tol_li"
    assert g["dq0"]["setting"] == "jbs_tol_q0"
    assert g["dl_i"]["bound"] == b.config.generation.jbs_tol_li
    assert g["dq0"]["bound"] == b.config.generation.jbs_tol_q0
    assert r["delivered"]["l_i"] == s["li"]
    assert not [w for w in rec if "BISTABLE" in str(w.message)]


def test_a_bistable_baseline_is_stamped_and_warned(tmp_path,
                                                   toy_bouquet_solver,
                                                   monkeypatch):
    """The stand-in whose solve right after init_psi lands 0.5 % off in
    l_i(3) -- a different equilibrium of the same inputs: stamped
    bistable=True with the deltas, warned by name, and the delivered state
    is what the baseline and the solver keep."""
    b, rec = _prepared(tmp_path, monkeypatch, check=True,
                       fake_cls=TD._cold_branch_fake())
    r = b.baseline.engine["bistability"]
    li_del = b._engine_run["engine"].delivered_meas["li"]
    assert r["bistable"] is True and r["outcome"] == "bistable"
    assert r["deltas"]["l_i"] == pytest.approx(0.005 * li_del, rel=1e-9)
    assert r["gates"]["dl_i"]["passed"] is False
    assert r["cold"]["l_i"] == pytest.approx(1.005 * li_del, rel=1e-12)
    msgs = [str(w.message) for w in rec if "BISTABLE" in str(w.message)]
    assert len(msgs) == 1 and "Baseline.engine['bistability']" in msgs[0]
    # the delivered equilibrium is restored: the solver's state is the
    # reconstruction's again, and the baseline is unchanged
    assert b.mygs.toy.state["li"] == li_del
    assert b.baseline.l_i_target == li_del


def test_a_failed_cold_solve_is_recorded_not_hidden(tmp_path,
                                                    toy_bouquet_solver,
                                                    monkeypatch):
    from _engine_fake_gs import FakeTokaMaker

    class Failing(FakeTokaMaker):
        def solve(self, *a, **k):
            raise RuntimeError("synthetic cold-solve failure")
    b, rec = _prepared(tmp_path, monkeypatch, check=True, fake_cls=Failing)
    r = b.baseline.engine["bistability"]
    assert r["bistable"] is None and r["outcome"] == "cold_solve_failed"
    assert "synthetic cold-solve failure" in r["error"]
    assert [w for w in rec if "bistability unknown" in str(w.message)]
    assert b.mygs.toy.state["li"] == b.baseline.l_i_target


def test_the_check_is_engine_only(tmp_path):
    from bouquet.config import GenerationConfig
    from bouquet.engine import validate_engine_settings
    gc = GenerationConfig(reconstruction_engine="legacy",
                          check_bistability=True)
    with pytest.raises(ValueError, match="check_bistability"):
        validate_engine_settings(gc)
