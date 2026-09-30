"""Draws on the unified engine -- fast half (no GS solver).

The engine draw (:mod:`bouquet.engine_draws`) is driven by the toy
Grad-Shafranov stand-in of the Stage 2 tests (``tests/_engine_toy.py``),
and ``generate_bouquet`` / ``Bouquet.generate`` / the parallel worker by a
TokaMaker stand-in over that toy (``tests/_engine_fake_gs.py``).  Checked:

* the zero-perturbation identity BY CONSTRUCTION: the first request is the
  stored request bit for bit (delivery correction off/on, q0 row off/on),
  the closure's Ip increment is exactly zero on pass 1, the loop's pass-1
  residuals are the reconstruction's own and the loop exits after the
  required consecutive passes (the current gate is measured one pass late),
  and the delivered draw reproduces the reconstruction within the loop
  tolerances;
* the sampler draws the legacy stream (same Generator state after the
  kinetic and auxiliary channels as perturb_kinetic_equilibrium, same
  pressure) and today's toroidal inductive sigma, and a zero sigma is
  exactly the base;
* a perturbed draw moves l_i and beta_N and records the change of l_i and
  of the poloidal flux range against the reconstruction; the Ip amplitude
  is 1.0 at zero perturbation and the exact measure holds every pass; the
  q0 row acts when enabled;
* the post-hoc filters are applied to the archived draw and recorded,
  out-of-band draws are archived with in_spec=False and not counted by
  until-N; rejections carry their DRAW_REJECTION_REASONS code;
* end to end through generate_bouquet, Bouquet.generate (+ .filter(),
  verify_sigma0_consistency) and the parallel worker entry point and merge,
  on toy archives.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import copy
import io
import os
import warnings

import numpy as np
import pytest

import _engine_toy as T
from bouquet import engine_draws as ED
from bouquet.config import GenerationConfig
from bouquet.engine import reconstruct
from bouquet.jbs_loop import (JBS_REQUIRED_CONSECUTIVE, JBSNonFinite,
                              JBSNotConverged)
from bouquet.physics import ELEMENTARY_CHARGE as EC

PSI = T.PSI
_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _kin():
    ne = 5e19 * (1 - 0.8 * PSI ** 2) + 2e18
    te = 2e3 * (1 - 0.9 * PSI ** 2) + 50.0
    return dict(ne=ne, te=te, ni=0.9 * ne, ti=1.1 * te,
                zeff=np.full(PSI.size, 1.8))


def _li0():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"]


LI0 = _li0()


def _recon(*, li=1.01, q0=None, gain=True, defect=0.02, **gc):
    """A toy reconstruction whose contract's pressure IS its kinetics'
    thermal pressure (so the sampler's pressure match is the legacy one)."""
    ad = T.ToyAdapter(li_target=LI0 * li,
                      q0_target=(None if q0 is None else q0))
    c = ad.read()
    k = _kin()
    c.kinetics = k
    c.kinetics_native = dict(psi_N=PSI, **{kk: k[kk] for kk in
                                          ("ne", "te", "ni", "ti")},
                             Zeff=k["zeff"])
    th = EC * (k["ne"] * k["te"] + k["ni"] * k["ti"])
    c.pressure = th.copy()
    c.pressure_parts = dict(thermal=th, impurity=0 * PSI, fast=0 * PSI,
                            Z_imp=None)
    b = T.ToyGS(gain=gain, defect=defect)
    b.set_inputs(pressure=c.pressure, kinetics=c.kinetics)
    eng, res, rec = _quiet(reconstruct, ad, b, T.settings(**gc), label="toy")
    assert res["converged"]
    return eng, res, rec, b


def _ctx(eng, res, *, q0_row=False, **gc):
    g = GenerationConfig(reconstruction_engine="unified", **gc)
    k = eng.c.kinetics
    native = dict(psi_N=PSI, ne=k["ne"], te=k["te"], ni=k["ni"],
                  ti=k["ti"])
    return ED.EngineDrawContext(eng, res, loop=ED.draw_loop_settings(g),
                                native=native, q0_row=q0_row)


def _unc(ctx, f=0.03, fj=None):
    k = ctx.native
    return dict(sigma_ne=f * k["ne"], sigma_te=f * k["te"],
                sigma_ni=f * k["ni"], sigma_ti=f * k["ti"],
                sigma_jphi=(f if fj is None else fj) * np.abs(ctx.request),
                n_ls=0.3, t_ls=0.3, j_ls=0.25)


@pytest.fixture(scope="module")
def recon():
    return _recon()


# ---------------------------------------------------------------------------
#  zero-perturbation identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dc", [False, True])
def test_the_first_request_is_the_stored_request_bit_for_bit(dc):
    eng, res, rec, b = _recon(engine_delivery_correction=dc)
    ctx = _ctx(eng, res)
    out = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())
    r = out["record"]
    assert r["identity"]["pass1_request_bit_identical"] is True
    assert r["identity"]["pass1_request_max_abs_diff"] == 0.0
    np.testing.assert_array_equal(out["passes"].first_request,
                                  res["state"].request)
    if dc:
        assert np.any(res["state"].delivery_correction != 0.0)


@pytest.mark.parametrize("case", ["soft_rows", "bootstrap_scalar"])
def test_the_identity_holds_for_soft_rows_and_the_scalar_preset(case):
    """The IDS-like soft rows (the Ip-row target is then the closure's
    posterior Ip, not the measured one) and a scalar preset."""
    import test_engine as TE
    if case == "soft_rows":
        ad = T.ToyAdapter(soft=True, li_target=LI0 * 1.02)
        gc = {}
    else:
        ad = T.ToyAdapter(li_target=None, jB_ind=TE._consistent_inductive())
        gc = dict(engine_preset="bootstrap_scalar", engine_rows=["Ip"])
    ad.read()
    b = T.ToyGS()
    eng, res, rec = _quiet(reconstruct, ad, b, T.settings(**gc), label="t")
    ctx = _ctx(eng, res)
    r = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    assert r["identity"]["pass1_request_bit_identical"]
    assert r["amplitude"]["per_pass"][0]["a_ind"] == 1.0
    assert r["loop"]["converged"]
    if case == "soft_rows":
        assert ctx.Ip_star != eng.c.Ip        # the posterior, recorded


def test_pass_one_closes_with_exactly_zero_increment(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    r = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    p1 = r["amplitude"]["per_pass"][0]
    assert p1["dIp_A"] == 0.0 and p1["a_ind"] == 1.0
    lp = r["loop"]
    # the pass-1 residuals ARE the reconstruction's delivered ones (Redl of
    # the same equilibrium against the same bootstrap)
    chk = rec["delivered"]["checks"]["loop"]
    # (to the re-solve's rounding: the toy's flux-range fixed point moves
    # by an ulp when the same request is solved again)
    assert lp["r_j"][0] == pytest.approx(chk["r_j"], rel=1e-6, abs=0)
    assert lp["r_I"][0] == pytest.approx(chk["r_I"], rel=1e-6, abs=0)
    assert lp["dl_i"][0] <= 1e-12
    # the loop exits after the required consecutive passes; the current
    # gate is measured one pass late, so pass 1 cannot count
    assert lp["converged"] and lp["criteria"]["current_residual"]
    assert lp["n_passes"] == JBS_REQUIRED_CONSECUTIVE + 1
    assert lp["pass_ok"] == [False, True, True]


def test_the_zero_perturbation_draw_delivers_the_reconstruction(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    s = ctx.loop
    v = _quiet(ED.verify_zero_perturbation, ctx, b)
    assert v["passed"] and v["request_bit_identical"]
    assert v["r_j"] <= s["rtol_j"] and v["r_I"] <= s["rtol_Ip"]
    assert abs(v["dl_i"]) <= s["tol_li"]
    assert v["amplitude"] == pytest.approx(1.0, abs=1e-5)
    assert v["dq0_psi_N"] == pytest.approx(T.PAD)


def test_a_zero_sigma_sample_is_exactly_the_base(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, f=0.0)
    from bouquet.sampling import make_rng
    inp = ED.sample_draw_inputs(ctx, make_rng(3), unc, b.flux_integral)
    z = ctx.zero_inputs()
    for k in ("ne", "te", "ni", "ti", "zeff"):
        np.testing.assert_array_equal(inp.kinetics[k], z.kinetics[k])
    np.testing.assert_array_equal(inp.pressure, eng.c.pressure)
    np.testing.assert_array_equal(inp.jB_ind, eng.c.jB_ind)
    inp.sampler["zero_perturbation"] = True
    r = _quiet(ED.run_draw, ctx, b, inp)["record"]
    assert r["identity"]["pass1_request_bit_identical"]


def test_a_state_that_does_not_compose_its_request_is_refused(recon):
    eng, res, rec, b = recon
    bad = dict(res)
    st = copy.copy(res["state"])
    st.request = st.request * (1.0 + 1e-12)
    bad["state"] = st
    with pytest.raises(ED.EngineDrawRefused, match="does not reproduce"):
        _ctx(eng, bad)


# ---------------------------------------------------------------------------
#  the sampler: the legacy stream
# ---------------------------------------------------------------------------
class _Stop(BaseException):
    pass


class _LegacyMock:
    """Enough of a solver for perturb_kinetic_equilibrium to sample and
    stop at its first solve."""
    psi_bounds = [-0.3, 0.0]

    def __init__(self, fi):
        self.fi = fi
        self.pp = None

    def flux_integral(self, x, p):
        return self.fi(x, p)

    def set_targets(self, **k):
        pass

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        self.pp = pp_prof

    def solve(self):
        raise _Stop()


@pytest.mark.parametrize("zeff_primary", [False, True])
def test_the_sampler_draws_the_legacy_kinetic_stream(recon, zeff_primary):
    from bouquet.jbs_loop import jbs_settings
    from bouquet.sampling import make_rng
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    from bouquet.utils import pchip_derivative
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, f=0.05)
    k = ctx.native
    if zeff_primary:
        unc.update(aux_sigmas={"zeff": 0.1 * np.ones_like(PSI),
                               "omega_tor": 1e3 * np.ones_like(PSI)},
                   aux_baselines={"zeff": np.full(PSI.size, 1.8),
                                  "omega_tor": 1e4 * (1 - PSI)},
                   aux_length_scales={"zeff": 0.4, "omega_tor": 0.3})
    else:
        unc.update(aux_sigmas={"omega_tor": 1e3 * np.ones_like(PSI)},
                   aux_baselines={"omega_tor": 1e4 * (1 - PSI)},
                   aux_length_scales={"omega_tor": 0.3})
    m = _LegacyMock(b.flux_integral)
    r1, r2 = make_rng(7), make_rng(7)
    g = GenerationConfig(reconstruction_engine="unified")
    with pytest.raises(_Stop):
        _quiet(perturb_kinetic_equilibrium, m, PSI,
               ctx.pressure_thermal_base, k["ne"], k["te"], k["ni"],
               k["ti"], ctx.request, unc["sigma_ne"], unc["sigma_te"],
               unc["sigma_ni"], unc["sigma_ti"], unc["sigma_jphi"], 0.3,
               0.3, 0.25, 1.2e6, LI0, eng.c.kinetics["zeff"], PSI.size,
               input_jinductive=0.6 * ctx.request, p_thresh=0.05, rng=r1,
               jbs_loop=jbs_settings(g, draw=True),
               aux_sigmas=unc["aux_sigmas"],
               aux_baselines=unc["aux_baselines"],
               aux_length_scales=unc["aux_length_scales"])
    inp = ED.sample_kinetics(ctx, r2, unc, b.flux_integral, p_thresh=0.05)
    # the SAME Generator state after the kinetic and auxiliary channels
    assert r1.bit_generator.state == r2.bit_generator.state
    # ... and the same drawn pressure (increment form: rounding only)
    leg = m.pp["y"] * 0.3
    mine = pchip_derivative(PSI, inp.pressure)
    mine[-1] = 0.0
    assert np.max(np.abs(leg - mine)) <= 1e-12 * np.max(np.abs(leg))
    assert set(inp.aux) == set(unc["aux_sigmas"])
    assert inp.sampler["zeff_primary"] is zeff_primary


@pytest.mark.parametrize("j_ls, bar", [(0.05, 1e-8), (0.25, 1e-3)])
def test_the_inductive_candidate_is_the_legacy_toroidal_draw(recon, j_ls,
                                                             bar):
    """The candidate's toroidal perturbation on G* IS the legacy draw of the
    same stream (sigma_jphi, j_ls) -- whatever the mean it perturbs: the
    factorised covariance does not depend on it (to rounding, which the
    near-singular kernel amplifies at the longer length scale)."""
    from bouquet.sampling import generate_perturbed_GPR, make_rng
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, fj=0.05)
    unc["j_ls"] = j_ls
    fac = ctx.s_ind * ctx.parts_star["kappa"]
    other = 0.6 * ctx.request                  # a legacy-like inductive
    r1, r2 = make_rng(11), make_rng(11)
    cand, tries, _nf = ED.sample_inductive(ctx, r1, unc)
    ref = generate_perturbed_GPR(PSI, other / other[0],
                                 sigma_profile=unc["sigma_jphi"] / other[0],
                                 length_scale=j_ls, n_samples=1, rng=r2,
                                 diag_plot=False) * other[0]
    assert tries == 1
    mine = fac * (cand - np.asarray(eng.c.jB_ind))
    sig = np.max(unc["sigma_jphi"])
    assert np.max(np.abs(mine - (ref - other))) <= bar * sig
    assert r1.bit_generator.state == r2.bit_generator.state


# ---------------------------------------------------------------------------
#  a perturbed draw
# ---------------------------------------------------------------------------
def _perturbed(ctx, b, seed=12345, f=0.03, scale=1.0):
    from bouquet.sampling import make_rng
    inp = ED.sample_draw_inputs(ctx, make_rng(seed), _unc(ctx, f=f),
                                b.flux_integral, scale=scale)
    return _quiet(ED.run_draw, ctx, b, inp)


def test_a_perturbed_draw_moves_li_and_beta_and_records_the_flux_range(
        recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    out = _perturbed(ctx, b, scale=1.01)
    r = out["record"]
    assert r["loop"]["converged"]
    d = r["deltas"]
    assert abs(d["l_i_3"]) > 1e-4 and abs(d["beta_n"]) > 0.0
    assert "attribution" not in r
    # the flux range psi_b - psi_a against the reconstruction's (the toy's
    # flux range responds to the current shape)
    fr, fr0 = r["delivered"]["flux_range"], r["reference"]["flux_range"]
    assert fr0 == pytest.approx(ctx.geom["dpsi_dpsiN"], rel=0, abs=0)
    assert d["flux_range"] == fr - fr0 and d["flux_range"] != 0.0
    assert d["flux_range_rel"] == pytest.approx((fr - fr0) / fr0,
                                                rel=1e-14)
    assert d["l_i_1"] is not None
    for k in ("l_i_3", "l_i_1", "beta_n", "q0", "q0_psi_N", "q95", "Ip",
              "delivery_check", "flux_range"):
        assert k in r["delivered"], k
    assert r["delivered"]["delivery_check"]["ok"]


def test_the_ip_amplitude_holds_the_exact_measure_every_pass(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    r = _perturbed(ctx, b, seed=5)["record"]
    for p in r["amplitude"]["per_pass"]:
        assert p["Ip_exact_measure_A"] == pytest.approx(
            ctx.Ip_star, rel=1e-12, abs=0)
    assert r["amplitude"]["final"]["a_ind"] != 1.0
    assert ctx.Ip_star == pytest.approx(eng.c.Ip, rel=1e-9)


def test_the_q0_row_acts_when_enabled():
    from bouquet.jbs_loop import AxisRowPin  # noqa: F401
    b0 = T.ToyGS()
    b0.solve(T.ToyAdapter().read().anchor_request)
    q_anchor = b0.q_at(b0.state, T.PAD)
    eng, res, rec, b = _recon(q0=q_anchor * 0.97,
                              engine_rows=["Ip", "l_i", "q0"])
    ctx = _ctx(eng, res, q0_row=True, engine_rows=("Ip", "l_i", "q0"),
               engine_draw_q0_row=True)
    # identity with the row kept
    z = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    assert z["identity"]["pass1_request_bit_identical"]
    assert z["amplitude"]["per_pass"][0]["a_bs"] == 1.0
    r = _perturbed(ctx, b, seed=21)["record"]
    assert r["loop"]["converged"] and r["loop"]["criteria"]["q0_residual"]
    qr = r["q0_row"]["record"]
    assert qr["n_row_updates"] >= 1
    tol = eng.s["q0_tol"]
    assert abs(r["delivered"]["q0"] - ctx.q0_target) <= tol


def test_the_q0_row_needs_the_reconstructions_q0_row(recon):
    eng, res, rec, b = recon
    with pytest.raises(ED.EngineDrawRefused, match="no active q0 row"):
        _ctx(eng, res, q0_row=True)
    from bouquet.engine import validate_engine_settings
    with pytest.raises(ValueError, match="engine_draw_q0_row"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_draw_q0_row=True))
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(engine_draw_q0_row=True))
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            engine_draw_homotopy=False))
    with pytest.raises(ValueError, match="bool"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_draw_homotopy=1))


# ---------------------------------------------------------------------------
#  post-hoc filters and rejection codes
# ---------------------------------------------------------------------------
def test_the_post_hoc_filters(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    li = ctx.ref["l_i"]
    fin = dict(li=li * 1.04, q_row=0.95)
    v = ED.post_hoc_verdicts(ctx, fin, l_i_tolerance=0.05,
                             constrain_sawteeth=False)
    assert v["l_i_in_band"] and v["q0_ok"] and v["in_band"]
    v = ED.post_hoc_verdicts(ctx, fin, l_i_tolerance=0.05,
                             constrain_sawteeth=True)
    assert not v["q0_ok"] and not v["in_band"] and v["reasons"]
    v = ED.post_hoc_verdicts(ctx, dict(li=li * 1.06, q_row=1.2),
                             l_i_tolerance=0.05, constrain_sawteeth=True)
    assert not v["l_i_in_band"] and v["q0_ok"] and not v["in_band"]
    assert v["q0_psi_N"] == pytest.approx(T.PAD)


def test_rejection_codes(recon):
    from bouquet.engine import EngineClosureRefused
    from bouquet.TokaMaker_interface import (DRAW_REJECTION_REASONS,
                                             CoilSaturated,
                                             DrawAnchorSolveFailed)
    eng, res, rec, b = recon
    # loop failure: a ceiling the draw cannot meet
    ctx = _ctx(eng, res, jbs_max_passes_draw=2)
    with pytest.raises(JBSNotConverged) as ei:
        _perturbed(ctx, b)
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "jbs_not_converged"
    # non-finite bootstrap: JBSNonFinite at once
    ctx = _ctx(eng, res)
    b2 = copy.deepcopy(b)
    b2.nan_redl_at = b2.n_solves + 1
    with pytest.raises(JBSNonFinite) as ei:
        _quiet(ED.run_draw, ctx, b2, ctx.zero_inputs())
    assert ei.value.record["n_passes"] == 1
    assert ED.engine_rejection_reason(ei.value, "perturb") == "jbs_non_finite"
    # saturation under the hard coil bounds
    def _sat(stage):
        raise CoilSaturated(f"{stage}: F9A on its bound", dict(stage=stage))
    with pytest.raises(CoilSaturated):
        _quiet(ED.run_draw, ctx, copy.deepcopy(b), ctx.zero_inputs(),
               coil_guard=_sat)
    assert ED.engine_rejection_reason(CoilSaturated("x"), "perturb") \
        == "coil_saturation_jbs_loop"
    assert ED.engine_rejection_reason(CoilSaturated("x"), "post_homotopy") \
        == "coil_saturation_post_homotopy"
    # a degenerate Ip amplitude
    inp = ctx.zero_inputs()
    inp.jB_ind = np.zeros_like(inp.jB_ind)
    with pytest.raises(EngineClosureRefused) as ei:
        _quiet(ED.run_draw, ctx, copy.deepcopy(b), inp)
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "engine_closure_refused"
    # the first solve failing: the anchor analog
    b3 = copy.deepcopy(b)

    def _fail(req, n_passes=1):
        raise RuntimeError("synthetic GS failure")
    b3.solve = _fail
    with pytest.raises(DrawAnchorSolveFailed) as ei:
        _quiet(ED.run_draw, ctx, b3, ctx.zero_inputs())
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "anchor_solve_failed"
    for code in ("jbs_not_converged", "jbs_non_finite",
                 "engine_closure_refused", "anchor_solve_failed",
                 "coil_saturation_jbs_loop"):
        assert code in DRAW_REJECTION_REASONS


def test_the_until_n_verdict_ands_the_band():
    G = ED.GenerateEngineDraws.__new__(ED.GenerateEngineDraws)
    G._cur = None
    ok, why = G.until_n(True, [], {ED.DRAW_BAND_FLAG: False})
    assert not ok and why == ["engine post-hoc band"]
    ok, why = G.until_n(True, [], {ED.DRAW_BAND_FLAG: True})
    assert ok and why == []
    ok, why = G.until_n(False, ["coil"], {ED.DRAW_BAND_FLAG: True})
    assert not ok and why == ["coil"]


def test_selected_ands_the_band_flag_only_where_present(tmp_path):
    import h5py
    from bouquet.filtering import _recompute_selected
    with h5py.File(tmp_path / "f.h5", "w") as hf:
        leg = hf.create_group("a")
        leg.attrs["passes_coil_filter"] = True
        leg.attrs["passes_boundary_filter"] = True
        _recompute_selected(leg)
        assert bool(leg.attrs["selected"]) is True
        eng = hf.create_group("b")
        eng.attrs["passes_coil_filter"] = True
        eng.attrs["passes_boundary_filter"] = True
        eng.attrs[ED.DRAW_BAND_FLAG] = False
        _recompute_selected(eng)
        assert bool(eng.attrs["selected"]) is False


# ---------------------------------------------------------------------------
#  end to end: generate_bouquet on a TokaMaker stand-in
# ---------------------------------------------------------------------------
def _generate(tmp_path, monkeypatch, *, n=3, l_i_tolerance=0.05,
              n_inspec_target=None, homotopy=True, seed=12345):
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    os.makedirs(str(tmp_path), exist_ok=True)
    eng, res, rec, b = _recon()
    ctx = _ctx(eng, res)
    unc = _unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD,
                               homotopy=homotopy,
                               l_i_tolerance=l_i_tolerance)
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = FakeTokaMaker(b)
    h = str(tmp_path / "e2e")
    initialize_equilibrium_database(h)
    rej = []
    k = ctx.native
    diags = _quiet(
        generate_bouquet, fake, PSI, n, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=l_i_tolerance,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=seed,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=rej, engine_draw=G,
        coil_filter="legacy", n_inspec_target=n_inspec_target)
    return diags, rej, h, G


def test_generate_bouquet_runs_engine_draws_end_to_end(tmp_path,
                                                        monkeypatch):
    import h5py
    diags, rej, h, G = _generate(tmp_path, monkeypatch)
    assert len(diags) == 3 and rej == []
    for i, d in enumerate(diags):
        e = d["engine"]
        assert e["version"] == ED.ENGINE_DRAW_VERSION
        assert e["loop"]["converged"] and d["jbs_loop"]["kind"] == "engine"
        assert e["post_hoc"]["in_band"] and d["in_spec"] is True
        assert e["homotopy"]["enabled"] and e["homotopy"]["homotopy_pass"] \
            == 1
        c = e["cost"]
        assert set(ED._Clock.STAGES) <= set(c)
        assert c["loop"]["solves"] == e["loop"]["n_passes"] \
            == c["loop"]["passes"]
        assert c["anchor"]["solves"] == 0
        assert c["homotopy"]["solves"] == 2        # two homotopy passes
        assert c["total"]["wall_s"] > 0.0
        back = ED.read_draw_engine(h, i)
        assert back["deltas"] == e["deltas"]
        assert "cost" in back
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is True
        assert bool(g0.attrs["in_spec"]) is True
        assert bool(g0.attrs["jbs_converged"]) is True


def test_an_out_of_band_draw_is_archived_and_not_counted(tmp_path,
                                                          monkeypatch):
    import h5py
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=2,
                                 l_i_tolerance=1e-9, n_inspec_target=1)
    # every draw is outside a 1e-9 band: archived, in_spec False, never
    # counted -- the loop runs to its attempt cap (max(n, 5 * target))
    assert len(diags) == 5 and rej == []
    for d in diags:
        assert d["engine"]["post_hoc"]["in_band"] is False
        assert d["in_spec"] is False
        assert d["until_n_inspec"] is False
        assert "engine post-hoc band" in d["until_n_reasons"]
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is False
        assert bool(g0.attrs["in_spec"]) is False


def test_until_n_counts_in_band_engine_draws(tmp_path, monkeypatch):
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=4,
                                 n_inspec_target=2)
    from bouquet.filtering import until_n_delivered
    assert until_n_delivered(diags) == 2 and len(diags) == 2


def test_no_homotopy_stage_when_disabled(tmp_path, monkeypatch):
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1,
                                 homotopy=False)
    e = diags[0]["engine"]
    assert e["homotopy"]["enabled"] is False
    assert e["cost"]["homotopy"]["solves"] == 0
    assert diags[0]["homotopy_pass"] == -1 and diags[0]["in_spec"]


def test_the_same_seed_draws_the_same_engine_draws(tmp_path, monkeypatch):
    a, _r, _h, _G = _generate(tmp_path / "a", monkeypatch, n=2)
    b, _r, _h, _G = _generate(tmp_path / "b", monkeypatch, n=2)
    for x, y in zip(a, b):
        assert x["engine"]["deltas"] == y["engine"]["deltas"]
        assert x["engine"]["inputs"] == y["engine"]["inputs"]


# ---------------------------------------------------------------------------
#  Bouquet.generate / verify_sigma0_consistency / the parallel worker
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_bouquet_solver(monkeypatch):
    """The engine's backend -> a toy whose pressure and kinetics are the
    contract's (so the reconstruction and its draws agree), created once
    per stand-in solver; the draws' backend -> the same toy."""
    import bouquet.engine as be
    from _engine_fake_gs import FakeTokaMaker

    def _backend(mygs, contract, **kw):
        if getattr(mygs, "toy", None) is None:
            t = T.ToyGS(psi=contract.psi_N, Ip=contract.Ip)
            t.set_inputs(pressure=contract.pressure,
                         kinetics=contract.kinetics)
            mygs.toy = t
        return mygs.toy

    monkeypatch.setattr(be, "TokaMakerBackend", _backend)
    monkeypatch.setattr(be, "_lcfs_deviation_mm", lambda m, p: (2.5, 7.0))
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)

    def _setup(self):
        self.mygs = FakeTokaMaker(None)
        return self

    import bouquet.run as br
    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    return _backend


def _bq(tmp_path, n=2):
    import bouquet as bq
    b = bq.Bouquet.from_geqdsk(
        os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(_EX, "DIIID_mesh.h5"), n_draws=n,
        header=str(tmp_path / "bq"))
    g = b.config.generation
    g.reconstruction_engine = "unified"
    g.engine_rows = ["Ip"]
    g.seed = 12345
    return b


def test_bouquet_generate_runs_on_the_engine(tmp_path, toy_bouquet_solver):
    from bouquet.engine import load_baseline_engine
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"] and v["request_bit_identical"]
    assert v["invariant"] == "engine-draw"
    diags = _quiet(b.generate)
    assert len(diags) == 2 and b.draw_rejections == []
    assert all(d["engine"]["loop"]["converged"] for d in diags)
    rec = load_baseline_engine(b.config.output_header,
                               scan_key=b.config.generation.scan_key)
    assert rec is not None and rec["draws"]["version"] \
        == ED.ENGINE_DRAW_VERSION
    sel = _quiet(b.filter)
    assert sel["boundary"]["n_total"] == 2
    # the archived engine draws export like any draw: g-file bundle and a
    # perturbed IDS (the example OMAS file as the template)
    import json
    from bouquet.io.imas import write_imas_draw
    bun = _quiet(b.export_bundle, str(tmp_path / "bundle"),
                 formats=("geqdsk",), selection="all")
    assert len(bun) == 2 and all(os.path.isfile(v["geqdsk"])
                                 for v in bun.values())
    out = _quiet(write_imas_draw, b.config.output_header, 0,
                 os.path.join(_EX, "D3Dlike_baseline_omas.json"),
                 str(tmp_path / "d0.json"), time=2.3043)
    with open(out) as fh:
        ids = json.load(fh)
    assert ids["core_profiles"]["profiles_1d"]


def test_a_legacy_baseline_is_not_drawn_on_the_engine(tmp_path,
                                                       toy_bouquet_solver):
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    b.config.generation.reconstruction_engine = "legacy"
    b.config.generation.engine_rows = ("Ip", "l_i")
    with pytest.raises(NotImplementedError, match="now 'legacy'"):
        b.generate()


def test_the_parallel_worker_entry_point_and_merge(tmp_path,
                                                    toy_bouquet_solver):
    import h5py
    from bouquet.engine import load_baseline_engine
    from bouquet.parallel import merge_archives, run_shard
    b = _bq(tmp_path, n=3)
    out = str(tmp_path / "par")
    metas = [_quiet(run_shard, b.config, w, 2, n_equils_total=3,
                    seed_base=7, out_header=out, scan_key=0,
                    threads_per_worker=1, verbose=True) for w in (0, 1)]
    assert [m["n"] for m in metas] == [2, 1]
    assert metas[0]["li_target"] == metas[1]["li_target"]
    path, n = merge_archives([m["path"] for m in metas], out, scan_key=0)
    assert n == 3
    with h5py.File(path, "r") as hf:
        for i in range(3):
            g = hf[f"scan/0/{i}"]
            assert "engine_json" in g.attrs
            assert ED.DRAW_BAND_FLAG in g.attrs
    for i in range(3):
        assert ED.read_draw_engine(out, i, scan_key=0)["version"] \
            == ED.ENGINE_DRAW_VERSION
    assert load_baseline_engine(out, scan_key=0)["draws"] is not None


def test_the_probe_draw_mode_runs_on_the_toy(tmp_path, toy_bouquet_solver):
    """The solver probe's sigma0 stage and draw mode, on the stand-in (so
    the cluster run cannot fail on the probe's own bookkeeping)."""
    import json
    import sys
    sys.path.insert(0, os.path.join(_HERE, "probes"))
    import measure_engine as ME
    from bouquet.jbs_loop import jsonable
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    z = _quiet(ME._sigma0, b)
    assert z["passed"] and z["request_bit_identical"]
    assert z["cost"]["anchor"]["solves"] == 0
    d = _quiet(ME._draws, b, 2, ME.DEFAULT_SEED)
    assert d["attempts"] == 2 == d["archived"] + d["rejected"]
    assert set(d["stage_totals"]) == set(ED._Clock.STAGES)
    json.dumps(jsonable(d))
