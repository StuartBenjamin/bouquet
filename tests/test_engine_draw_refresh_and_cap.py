"""Engine draws: the bootstrap refresh after the first solve
(``engine_draw_bootstrap_refresh``) and ``draw_solve_maxits`` on the
homotopy / post-homotopy solves -- fast half (no GS solver).

The refresh (default OFF): after a draw's first loop solve the anchor's
kinetic Redl increment is re-evaluated on that solved geometry and the loop
RESTARTS from it instead of blending toward the start computed on the
reconstruction's geometry.  Checked on the toy Grad-Shafranov stand-in:

* the kernel hook (``run_jbs_loop(start_refresh=...)``): pass 2 solves with
  exactly the refreshed iterate, later passes blend as usual, the record
  carries the residuals before and after, a non-finite refresh is refused at
  once; ``start_refresh=None`` is covered by the frozen-kernel bit-identity
  tests (``test_engine_kernel_hook.py``);
* at ZERO perturbation the refreshed bootstrap is the stored ``lambda_BS*``
  to rounding, the first request is still bit-identical and the draw
  reproduces the reconstruction (``verify_zero_perturbation`` passes, same
  pass count);
* on perturbed draws the delivered state is the one without the refresh to
  within the loop tolerances (``jbs_tol_li``, ``jbs_rtol_j``,
  ``jbs_rtol_Ip``), in fewer or equal passes, with zero extra solves.

The cap (default ``None``): with ``draw_solve_maxits`` set, every homotopy
solve of an engine draw runs under it, and a homotopy solve (a pass or a
rollback re-solve) that stops at the cap REJECTS the draw with
``homotopy_maxits`` -- never rolled back to a looser pass and archived; a
post-homotopy pass that stops at it is ``post_homotopy_maxits``.  With the
default ``None`` nothing is re-classified.

Synthetic inputs only; no solver, no device data.
"""
import copy

import numpy as np
import pytest

import _engine_toy as T
import test_engine_draws as TD
from bouquet import engine_draws as ED
from bouquet.config import GenerationConfig
from bouquet.jbs_loop import JBSNonFinite, profile_residuals

PSI = T.PSI
_CAP_ERROR = 'Error in solve: Exceeded "maxits"'


@pytest.fixture(scope="module")
def recon():
    return TD._recon()


def _draw(ctx, b, inp, on):
    return TD._quiet(ED.run_draw, ctx, copy.deepcopy(b), inp,
                     bootstrap_refresh=on)


def _inputs(ctx, b, seed, fj=None, scale=1.01):
    from bouquet.sampling import make_rng
    return ED.sample_draw_inputs(ctx, make_rng(seed),
                                 TD._unc(ctx, f=0.03, fj=fj),
                                 b.flux_integral, scale=scale)


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def test_the_refresh_is_off_by_default_and_validated(recon):
    from bouquet.engine import engine_settings, validate_engine_settings
    assert GenerationConfig().engine_draw_bootstrap_refresh is False
    g = GenerationConfig(reconstruction_engine="unified")
    assert engine_settings(g)["draw_bootstrap_refresh"] is False
    with pytest.raises(ValueError, match="bool"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified",
            engine_draw_bootstrap_refresh=1))
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            engine_draw_bootstrap_refresh=True))
    eng, res, rec, b = recon
    assert TD._ctx(eng, res).bootstrap_refresh is False
    # off: the loop record is the one before the setting existed
    r = TD._quiet(ED.run_draw, TD._ctx(eng, res), copy.deepcopy(b),
                  TD._ctx(eng, res).zero_inputs())["record"]
    assert "bootstrap_refresh" not in r["loop"]
    # the setting reaches the context
    gc = GenerationConfig(reconstruction_engine="unified",
                          engine_draw_bootstrap_refresh=True)
    k = eng.c.kinetics
    ctx = ED.context_from_run(
        dict(engine=eng, result=res), gc,
        type("B", (), dict(psi_N_kinetic=PSI, ne=k["ne"], te=k["te"],
                           ni=k["ni"], ti=k["ti"]))())
    assert ctx.bootstrap_refresh is True


# ---------------------------------------------------------------------------
#  the kernel hook
# ---------------------------------------------------------------------------
def _affine(n=41, gain=0.1):
    """A contracting affine fixed-point map on a fixed geometry."""
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    J_fix = 1.0 + 0.5 * (1.0 - x)
    used = []

    def jmap(jbs):
        return J_fix + gain * (jbs - J_fix) + 0.3 * (1.0 - x) ** 2

    def step(jbs, k):
        used.append(np.asarray(jbs, dtype=float).copy())
        return dict(w=w, x=x, li=0.8, J=jmap(jbs))

    step.jmap = jmap
    return x, step, (lambda m: m["J"]), used


def _kernel_settings(**kw):
    from bouquet.jbs_loop import jbs_settings
    s = dict(jbs_settings(GenerationConfig()))
    s.update(max_passes=60, rtol_j=1e-6, rtol_Ip=1e-6, relax=0.8,
             on_fail="raise")
    s.update(kw)
    return s


def test_the_kernel_restarts_pass_two_from_the_refresh():
    from bouquet.jbs_loop import run_jbs_loop
    x, step, ev, used = _affine()
    calls = []

    def refresh(J, meas):
        calls.append(J.copy())
        return 0.25 * J + 0.75

    out = TD._quiet(run_jbs_loop, np.zeros_like(x), step, ev,
                    _kernel_settings(), Ip=1.0, start_refresh=refresh)
    rec = out["record"]
    assert len(calls) == 1
    np.testing.assert_array_equal(used[1], 0.25 * calls[0] + 0.75)
    # later passes blend as usual
    np.testing.assert_allclose(
        used[2], 0.2 * used[1] + 0.8 * step.jmap(used[1]), rtol=0,
        atol=1e-15)
    br = rec["bootstrap_refresh"]
    assert br["applied"] is True and br["after_pass"] == 1
    assert br["extra_solves"] == 0
    assert br["r_j_before"] == rec["r_j"][0]
    assert br["r_I_before"] == rec["r_I"][0]
    assert br["r_j_after"] == rec["r_j"][1]
    assert br["r_I_after"] == rec["r_I"][1]
    assert rec["omega"][0] is None and rec["omega"][1] is None
    assert rec["omega"][2] == 0.8
    assert rec["converged"]


def test_the_kernel_refuses_a_non_finite_refresh():
    from bouquet.jbs_loop import run_jbs_loop
    x, step, ev, used = _affine()

    def refresh(J, meas):
        out = J.copy()
        out[3] = np.nan
        return out

    with pytest.raises(JBSNonFinite, match="refreshed bootstrap"):
        TD._quiet(run_jbs_loop, np.zeros_like(x), step, ev,
                  _kernel_settings(), Ip=1.0, start_refresh=refresh)
    assert len(used) == 1                       # no solve with it


def test_the_on_pass_hook_sees_a_further_pass_before_the_restart():
    from bouquet.jbs_loop import run_jbs_loop
    x, step, ev, used = _affine()
    seen = []
    TD._quiet(run_jbs_loop, np.zeros_like(x), step, ev, _kernel_settings(),
              Ip=1.0, start_refresh=lambda J, m: J,
              on_pass=lambda k, m, J, e: seen.append(e["omega_next"]))
    assert seen[0] == 1.0 and seen[1] == 0.8 and seen[-1] is None


# ---------------------------------------------------------------------------
#  zero perturbation: the refresh is lambda_BS* to rounding
# ---------------------------------------------------------------------------
def test_the_zero_perturbation_refresh_is_the_stored_bootstrap(
        recon, monkeypatch):
    import bouquet.jbs_loop as JL
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    got = {}
    real = JL.run_jbs_loop

    def spy(*a, **k):
        f = k.get("start_refresh")
        if f is not None:
            def wrapped(J, meas):
                got["refreshed"] = np.asarray(f(J, meas), dtype=float)
                return got["refreshed"]
            k["start_refresh"] = wrapped
        return real(*a, **k)

    monkeypatch.setattr(JL, "run_jbs_loop", spy)
    off = _draw(ctx, b, ctx.zero_inputs(), False)["record"]
    out = _draw(ctx, b, ctx.zero_inputs(), True)
    r = out["record"]
    lam = ctx.lam
    d = np.max(np.abs(got["refreshed"] - lam)) / np.max(np.abs(lam))
    assert d <= 1e-12, d                      # rounding of the re-solve
    assert r["identity"]["pass1_request_bit_identical"] is True
    assert r["identity"]["pass1_request_max_abs_diff"] == 0.0
    br = r["loop"]["bootstrap_refresh"]
    assert br["applied"] and br["refresh_vs_start"]["r_j"] <= 1e-12
    assert br["source"] == ED.REFRESH_SOURCE
    assert r["loop"]["n_passes"] == off["loop"]["n_passes"]
    assert r["loop"]["pass_ok"] == off["loop"]["pass_ok"]
    assert r["solves"] == off["solves"]


def test_the_zero_perturbation_draw_still_reproduces_the_reconstruction(
        recon):
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    ctx.bootstrap_refresh = True
    s = ctx.loop
    v = TD._quiet(ED.verify_zero_perturbation, ctx, copy.deepcopy(b))
    assert v["passed"] and v["request_bit_identical"]
    assert v["r_j"] <= s["rtol_j"] and v["r_I"] <= s["rtol_Ip"]
    assert abs(v["dl_i"]) <= s["tol_li"]
    assert v["record"]["loop"]["bootstrap_refresh"]["applied"] is True


# ---------------------------------------------------------------------------
#  perturbed draws: the same fixed point, fewer or equal passes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fj", [None, 0.1])
def test_the_refresh_changes_the_path_not_the_delivered_state(recon, fj):
    from bouquet.engine import conversion_factor
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    s = ctx.loop
    fewer = 0
    for seed in (1, 2, 3, 5, 12345):
        inp = _inputs(ctx, b, seed, fj=fj)
        o0 = _draw(ctx, b, inp, False)
        o1 = _draw(ctx, b, inp, True)
        r0, r1 = o0["record"], o1["record"]
        assert r0["loop"]["converged"] and r1["loop"]["converged"]
        n0, n1 = r0["loop"]["n_passes"], r1["loop"]["n_passes"]
        assert n1 <= n0, (seed, n0, n1)
        fewer += int(n1 < n0)
        # zero extra solves: one solve per pass
        assert r1["solves"]["loop"] - r0["solves"]["loop"] == n1 - n0
        br = r1["loop"]["bootstrap_refresh"]
        assert br["applied"] and br["extra_solves"] == 0
        assert br["r_I_after"] < br["r_I_before"]
        # the delivered states agree within the loop tolerances
        assert abs(r1["delivered"]["l_i_3"] - r0["delivered"]["l_i_3"]) \
            <= s["tol_li"]
        g = o1["passes"].last["geom"]
        w = g["w_lin"] * conversion_factor(g)
        pr = profile_residuals(o1["jbs_used"], o0["jbs_used"], w, ctx.psi,
                               float(ctx.c.Ip))
        assert pr["r_j"] <= s["rtol_j"] and pr["r_I"] <= s["rtol_Ip"], pr
        j0, j1 = o0["split"]["j_phi"], o1["split"]["j_phi"]
        assert np.max(np.abs(j1 - j0)) <= s["rtol_j"] * np.max(np.abs(j0))
    assert fewer >= 1


# ---------------------------------------------------------------------------
#  draw_solve_maxits on the homotopy / post-homotopy solves
# ---------------------------------------------------------------------------
def test_a_solve_that_hit_its_cap_is_recognised():
    from bouquet.engine import EngineSolveError
    assert ED.solve_hit_iteration_cap(ValueError(_CAP_ERROR))
    try:
        try:
            raise ValueError(_CAP_ERROR)
        except ValueError as e:
            raise EngineSolveError(
                "engine GS solve failed (pass 1/1, maxits 40): x") from e
    except EngineSolveError as e:
        chained = e
    assert ED.solve_hit_iteration_cap(chained)
    # a message that only NAMES the cap is not a capped solve
    assert not ED.solve_hit_iteration_cap(EngineSolveError(
        "engine GS solve failed (pass 1/1, maxits 40): Error in solve: "
        "Non-finite value (NaN/Inf) in solution"))
    assert not ED.solve_hit_iteration_cap(RuntimeError("anything"))


def test_the_codes_are_registered_and_default_is_unchanged(recon):
    from bouquet.engine import EngineSolveError
    from bouquet.TokaMaker_interface import DRAW_REJECTION_REASONS
    assert "homotopy_maxits" in DRAW_REJECTION_REASONS
    assert "post_homotopy_maxits" in DRAW_REJECTION_REASONS
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    try:
        try:
            raise ValueError(_CAP_ERROR)
        except ValueError as e:
            raise EngineSolveError("engine GS solve failed") from e
    except EngineSolveError as e:
        exc = e
    G0 = ED.GenerateEngineDraws(ctx, unc={}, psi_pad=T.PAD)
    assert G0.maxits is None
    assert G0.rejection_reason(exc, "post_homotopy") \
        == "jbs_post_homotopy_error"
    G = ED.GenerateEngineDraws(ctx, unc={}, psi_pad=T.PAD, maxits=40)
    assert TD._quiet(G.rejection_reason, exc, "post_homotopy") \
        == "post_homotopy_maxits"
    assert G.cap_events and G.cap_events[0]["maxits"] == 40
    # the loop stage keeps its codes
    assert G.rejection_reason(exc, "perturb") == "perturb_failed"


def test_cap_solver_is_a_noop_under_a_guard_and_restores():
    from _engine_fake_gs import FakeTokaMaker
    G = ED.GenerateEngineDraws.__new__(ED.GenerateEngineDraws)
    G.maxits, G._cap_saved, G.cap_events, G._cur = 40, None, [], None
    f = FakeTokaMaker(T.ToyGS())
    f.settings.maxits = 40                     # a DrawSolveGuard's cap
    G.cap_solver(f)
    assert f.calls == [] and G._cap_saved is None
    f.settings.maxits = 80
    G.cap_solver(f)
    assert f.settings.maxits == 40 and f.calls[-1] == ("update_settings", 40)
    G.uncap_solver(f)
    G.uncap_solver(f)                          # idempotent
    assert f.settings.maxits == 80
    G.maxits = None
    G.cap_solver(f)
    assert f.settings.maxits == 80


def _generate_capped(tmp_path, monkeypatch, *, maxits, fail, n=1):
    """``generate_bouquet`` on the stand-in with homotopy solves failing
    per *fail*: ``{homotopy solve number (1-based): error text}``.  Returns
    ``(diags, rejections, G, fake, maxits seen by each homotopy solve)``."""
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    import os
    os.makedirs(str(tmp_path), exist_ok=True)
    eng, res, rec, b = TD._recon()
    ctx = TD._ctx(eng, res)
    unc = TD._unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD, maxits=maxits)
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = FakeTokaMaker(b)
    seen = []
    n_h = [0]
    real_solve = fake.solve

    def solve(*a, **k):
        cur = G._cur
        if cur is not None and cur["clock"].cur == "homotopy":
            n_h[0] += 1
            seen.append(fake.settings.maxits)
            if n_h[0] in fail:
                raise ValueError(fail[n_h[0]])
        return real_solve(*a, **k)

    fake.solve = solve
    h = str(tmp_path / "e2e")
    initialize_equilibrium_database(h)
    rej = []
    k = ctx.native
    diags = TD._quiet(
        generate_bouquet, fake, PSI, n, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=0.05,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=12345,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=rej, engine_draw=G,
        coil_filter="legacy")
    return diags, rej, G, fake, seen


def test_every_homotopy_solve_runs_under_the_cap(tmp_path, monkeypatch):
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path, monkeypatch, maxits=40, fail={})
    assert rej == [] and len(diags) == 1
    assert seen == [40, 40]                     # both homotopy passes
    assert fake.settings.maxits == 80           # put back afterwards
    assert diags[0]["engine"]["homotopy"]["solve_maxits"] == 40


@pytest.mark.parametrize("at", [1, 2])
def test_a_capped_homotopy_solve_rejects_the_draw(tmp_path, monkeypatch, at):
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path / str(at), monkeypatch, maxits=40, fail={at: _CAP_ERROR})
    # rejected with its own code -- at pass 2 too: NOT rolled back to pass
    # 1 and archived
    assert diags == []
    assert [r["reason"] for r in rej] == ["homotopy_maxits"]
    assert "maxits" in rej[0]["message"]
    assert len(seen) == at                      # no rollback re-solve
    assert G.cap_events and G.cap_events[0]["maxits"] == 40
    assert fake.settings.maxits == 80


def test_a_capped_rollback_resolve_rejects_the_draw(tmp_path, monkeypatch):
    # pass 2 fails for another reason; the rollback re-solve hits the cap
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path, monkeypatch, maxits=40,
        fail={2: "Error in solve: Non-finite value (NaN/Inf) in solution",
              3: _CAP_ERROR})
    assert diags == []
    assert [r["reason"] for r in rej] == ["homotopy_maxits"]
    assert len(seen) == 3


def test_without_a_cap_the_homotopy_is_unchanged(tmp_path, monkeypatch):
    # the same solver failure with draw_solve_maxits=None: pass 2 rolls
    # back to pass 1 and the draw is archived there, as before
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path / "a", monkeypatch, maxits=None, fail={2: _CAP_ERROR})
    assert rej == [] and len(diags) == 1
    assert diags[0]["homotopy_pass"] == 0        # (0-based) pass 1
    assert seen == [80, 80, 80]                 # pass 1, pass 2, rollback
    assert G.cap_events == []
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path / "b", monkeypatch, maxits=None, fail={1: _CAP_ERROR})
    assert diags == [] and [r["reason"] for r in rej] \
        == ["homotopy_infeasible"]


def test_a_capped_post_homotopy_solve_has_its_own_code(tmp_path,
                                                        monkeypatch):
    from bouquet.engine import EngineSolveError

    def _ph(*a, **k):
        try:
            raise ValueError(_CAP_ERROR)
        except ValueError as e:
            raise EngineSolveError("engine GS solve failed (pass 1/1, "
                                   "maxits 40): " + str(e)) from e

    monkeypatch.setattr(ED, "post_homotopy", _ph)
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path / "a", monkeypatch, maxits=40, fail={})
    assert diags == [] and [r["reason"] for r in rej] \
        == ["post_homotopy_maxits"]
    diags, rej, G, fake, seen = _generate_capped(
        tmp_path / "b", monkeypatch, maxits=None, fail={})
    assert diags == [] and [r["reason"] for r in rej] \
        == ["jbs_post_homotopy_error"]


def test_the_refresh_with_the_q0_row():
    """The q0 row kept in draws (AxisRowPin): the refresh leaves the pin's
    per-pass row update alone; same delivered state within the loop
    tolerances, q0 on target within q0_tol, fewer or equal passes."""
    b0 = T.ToyGS()
    b0.solve(T.ToyAdapter().read().anchor_request)
    q_anchor = b0.q_at(b0.state, T.PAD)
    eng, res, rec, b = TD._recon(q0=q_anchor * 0.97,
                                 engine_rows=["Ip", "l_i", "q0"])
    ctx = TD._ctx(eng, res, q0_row=True, engine_rows=("Ip", "l_i", "q0"),
                  engine_draw_q0_row=True)
    s = ctx.loop
    z = _draw(ctx, b, ctx.zero_inputs(), True)["record"]
    assert z["identity"]["pass1_request_bit_identical"] is True
    for seed in (21, 5, 3):
        inp = _inputs(ctx, b, seed, scale=1.0)
        r0 = _draw(ctx, b, inp, False)["record"]
        r1 = _draw(ctx, b, inp, True)["record"]
        assert r0["loop"]["converged"] and r1["loop"]["converged"]
        assert r1["loop"]["n_passes"] <= r0["loop"]["n_passes"]
        assert abs(r1["delivered"]["l_i_3"] - r0["delivered"]["l_i_3"]) \
            <= s["tol_li"]
        assert abs(r1["delivered"]["q0"] - r0["delivered"]["q0"]) \
            <= s["tol_q0"]
        assert abs(r1["delivered"]["q0"] - ctx.q0_target) \
            <= eng.s["q0_tol"]
