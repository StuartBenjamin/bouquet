"""The unified reconstruction engine -- fast half (no GS solver).

The engine (:mod:`bouquet.engine`) is driven here by a TOY Grad-Shafranov
stand-in (``tests/_engine_toy.py``): a deterministic map from the requested
current to a delivered state with the couplings the engine has to handle (Ip
imposed with a uniform factor, a localised delivery defect, a flux range that
responds to l_i with the measured log-gain of 2, a Redl bootstrap that follows
the geometry, q ~ 1/j(axis), pitch angles from the enclosed current).  On it:

* the engine converges for Ip + l_i, Ip + l_i + q0 and Ip + l_i + MSE (both
  Jacobian schemes, hard and soft rows), with ONLY the existing tolerances;
* every row's residual at convergence is evaluated on the DELIVERED state;
* the soft row's delivered residual is the one the fit weighed;
* the scalar presets reduce exactly to the scalar closures;
* the delivery correction, when ON, drives intended - achieved to the toy's
  floor;
* unreachable targets and non-finite bootstraps fail loudly; the flag policy
  flags;
* the stored state reproduces the delivered request bit for bit;
* the config field defaults to "legacy", old configs load as "legacy", and
  the engine options are validated by name;
* with "legacy" prepare_baseline() never enters the engine, and the draws
  refuse an engine baseline (Stage 3).

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import io
import json

import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import (EngineClosureRefused, ENGINE_ATTR,
                            UnifiedEngine, complete_geometry, compose,
                            conversion_factor, convergence_table,
                            engine_settings, gfile_li_row_tol,
                            load_baseline_engine, pressure_term, reconstruct,
                            store_baseline_engine, validate_engine_settings)
from bouquet.jbs_loop import JBSNonFinite, JBSNotConverged


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _anchor_li_q():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"], b.q_at(b.state, T.PAD)


LI0, Q0 = _anchor_li_q()


def _run(adapter, backend=None, **gc):
    b = backend if backend is not None else T.ToyGS()
    adapter.read()
    return _quiet(reconstruct, adapter, b, T.settings(**gc), label="toy") \
        + (b,)


def _mse_data(amp=0.04, sig=0.01):
    """Pitch angles of a toy equilibrium whose current differs from the
    anchor by a mid-radius bump (E_r-corrected, as the engine requires)."""
    from bouquet.mse import mse_chords
    ch0 = T.toy_chords()
    bt = T.ToyGS(chords=ch0)
    req = T.ToyAdapter().read().anchor_request
    bt.solve(req * (1.0 + amp * np.exp(-0.5 * ((T.PSI - 0.45) / 0.15) ** 2)))
    B = bt.field_at_chords(bt.state)
    tg = B[:, 2] / B[:, 1]
    md = dict(R=ch0["R"], Z=ch0["Z"], tgamma=tg, sigma=np.abs(tg) * sig,
              weight=np.ones(tg.size), A1=ch0["A1"], A2=ch0["A2"],
              A3=ch0["A3"], A4=ch0["A4"], er_corrected=True)
    ch = mse_chords(md)
    ch["psi"] = ch0["psi"]
    return md, ch


# ---------------------------------------------------------------------------
#  composition
# ---------------------------------------------------------------------------
def test_composition_is_the_field_aligned_conversion_plus_the_pressure_term():
    g = T.ToyGS().geometry(0.3)
    a, b, c = T.base_inductive(), 0.3 * T.base_inductive(), \
        0.1 * T.base_inductive()
    J, parts = compose(g, a, b, c, s_ind=1.1, s_bs=0.9)
    kap = g["F"] * g["inv_R"] / g["B2"]
    P = g["pprime"] * (g["R_avg"] - g["F"] ** 2 * g["inv_R"] / g["B2"])
    np.testing.assert_array_equal(parts["kappa"], conversion_factor(g))
    np.testing.assert_array_equal(parts["pressure"], pressure_term(g))
    np.testing.assert_allclose(J, kap * (1.1 * a + 0.9 * b + c) + P,
                               rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(P, pressure_term(g), rtol=0, atol=0)
    g0 = dict(g, pprime=0.0 * g["pprime"])
    assert np.all(pressure_term(g0) == 0.0)


# ---------------------------------------------------------------------------
#  convergence on the toy
# ---------------------------------------------------------------------------
def test_ip_and_li_converge_on_the_delivered_state():
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    eng, res, rec, b = _run(ad)
    assert res["converged"] and res["loop_converged"]
    d = rec["delivered"]
    assert d["ok"]
    # the hard g-file-like row: delivered l_i on the target to the g-file tol
    assert abs(d["checks"]["l_i"]["delivered"] - LI0 * 1.01) \
        <= gfile_li_row_tol()
    # the loop's own residuals re-checked on the delivered equilibrium
    assert d["checks"]["loop"]["ok"]
    assert d["checks"]["current_residual"]["ok"]
    # solve counts: anchor (two passes), one per loop pass, delivery (two)
    n = rec["phases"][0]["record"]["n_passes"]
    assert rec["solves"] == dict(anchor=2, loop=n, delivery=2, total=n + 4)
    assert b.n_solves == n + 4
    # the gate (decision 9) and the l_i row are loop criteria here
    crit = rec["phases"][0]["record"]["criteria"]
    assert crit["current_residual"] and crit["li_row"] and crit["dl_i"]
    json.dumps(rec)                       # JSON-safe end to end


def test_ip_li_and_q0_converge_and_q0_is_met_on_the_delivered_state():
    ad = T.ToyAdapter(li_target=LI0 * 1.01, q0_target=Q0 * 0.97)
    eng, res, rec, b = _run(ad, engine_rows=["Ip", "l_i", "q0"])
    assert res["converged"]
    q = rec["delivered"]["checks"]["q0"]
    assert q["ok"] and abs(q["residual"]) <= T.settings()["q0_tol"]
    assert q["dq0_ok"]
    pin = rec["phases"][0]["record"]["q0_pin"]
    assert pin["n_row_updates"] >= 1
    assert rec["phases"][0]["record"]["criteria"]["q0_residual"]


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("scheme", ["fd_broyden", "fd_chord"])
def test_ip_li_and_mse_converge(soft, scheme):
    md, ch = _mse_data()
    ad = T.ToyAdapter(soft=soft, li_target=LI0 * 1.01,
                      mse=dict(chords=ch, er_terms="toy"))
    eng, res, rec, b = _run(ad, T.ToyGS(chords=ch),
                            engine_rows=["Ip", "l_i", "mse"], mse_data=md,
                            engine_mse_jacobian=scheme)
    assert res["converged"]
    fd = rec["phases"][1]["jacobian"]
    # the Jacobian: one base solve + one per free coefficient (2K = 8)
    assert fd["n_free"] == 8 and fd["n_solves"] == 1 + 8
    assert rec["solves"]["mse_fd"] == 9
    if scheme == "fd_broyden":
        assert fd["n_broyden_updates"] >= 1
    else:
        assert fd["n_broyden_updates"] == 0
    m = rec["delivered"]["checks"]["mse"]
    assert m["ok"] and m["dtg_max_sigma"] <= T.settings()["mse_tol_sigma"]
    assert rec["phases"][1]["record"]["criteria"]["mse_chords"]
    # the chords moved the reconstruction toward the data: chi2 on the
    # delivered state below that of the same reconstruction without them
    from bouquet.mse import mse_chi2
    ad0 = T.ToyAdapter(soft=soft, li_target=LI0 * 1.01)
    _e0, _r0, _rec0, b0 = _run(ad0, T.ToyGS(chords=ch))
    B0 = b0.field_at_chords(b0.state)
    assert m["chi2"] < mse_chi2(B0[:, 2] / B0[:, 1], ch)[0]


def test_every_row_is_evaluated_on_the_delivered_state():
    md, ch = _mse_data()
    ad = T.ToyAdapter(li_target=LI0 * 1.01, q0_target=Q0 * 0.98,
                      mse=dict(chords=ch, er_terms="toy"))
    eng, res, rec, b = _run(ad, T.ToyGS(chords=ch),
                            engine_rows=["Ip", "l_i", "q0", "mse"],
                            mse_data=md)
    assert res["converged"]
    st = b.state                    # the backend is left on the delivery
    np.testing.assert_array_equal(st["R"], res["state"].request)
    c = rec["delivered"]["checks"]
    assert c["l_i"]["delivered"] == st["li"]
    assert c["q0"]["delivered"] == b.q_at(st, T.PAD)
    B = b.field_at_chords(st)
    np.testing.assert_allclose(c["mse"]["tgamma"], B[:, 2] / B[:, 1],
                               rtol=1e-14)
    # and the Redl bootstrap of that state against the one it carries
    assert c["loop"]["ok"]


def test_the_soft_rows_delivered_residual_is_the_one_the_fit_weighed():
    ad = T.ToyAdapter(soft=True, li_target=LI0 * 1.02)
    eng, res, rec, b = _run(ad)
    assert res["converged"]
    li = rec["delivered"]["checks"]["l_i"]
    s = T.settings()
    # delivered residual in sigma vs the fit's own (model + discrepancy)
    fit_res = (li["predicted_plus_d"] - li["target"]) / 0.04
    assert abs(li["residual_sigma"] - fit_res) <= s["structured_li_tol"] / 0.04
    assert abs(li["error"]) <= s["structured_li_tol"]
    assert not li["hard"]


def test_the_stored_state_reproduces_the_delivered_request_bit_for_bit():
    """The zero-perturbation identity's foundation: composing on the stored
    geometry snapshot with x*, lambda_BS* (and Delta) IS the request."""
    from bouquet.utils import structured_basis_eval
    for dc in (False, True):
        ad = T.ToyAdapter(li_target=LI0 * 1.01)
        eng, res, rec, b = _run(ad, engine_delivery_correction=dc)
        st = res["state"]
        Phi = structured_basis_eval(eng.basis, eng.psi)
        K = Phi.shape[0]
        _, parts = compose(st.geom, eng.c.jB_ind, st.lambda_bs, eng.c.jB_fix)
        s_ind = 1.0 + st.x[:K] @ Phi
        s_bs = 1.0 + st.x[K:] @ Phi
        R = (s_ind * parts["ind"] + s_bs * parts["bs"]
             + (parts["fix"] + parts["pressure"]))
        if dc:
            R = R + st.delivery_correction
        np.testing.assert_array_equal(R, st.request)
        # ... and the recorded state carries what a draw needs
        r = st.record()
        for k in ("x", "lambda_bs", "li_discrepancy", "geometry_snapshot",
                  "li_geom_snapshot", "request"):
            assert r[k] is not None


# ---------------------------------------------------------------------------
#  presets = the scalar closures, exactly
# ---------------------------------------------------------------------------
def _consistent_inductive():
    """A toy inductive whose raw components carry Ip on the anchor geometry
    (a self-consistent source: the scalar channels then need a scale ~1)."""
    c = T.ToyAdapter().read()
    b = T.ToyGS()
    b.solve(c.anchor_request, n_passes=2)
    a = b.measure()
    g = complete_geometry(a["geom"])
    _, p = compose(g, c.jB_ind, a["redl"], c.jB_fix)
    lin = lambda j: float(np.trapezoid(g["w_lin"] * j, g["psi_N"]))  # noqa
    f = (c.Ip - g["c_affine"] - lin(p["bs"]) - lin(p["fix"] + p["pressure"])
         ) / lin(p["ind"])
    return f * c.jB_ind


def _engine_for(preset, rows, **kw):
    ad = T.ToyAdapter(li_target=None, jB_ind=_consistent_inductive(), **kw)
    c = ad.read()
    b = T.ToyGS()
    b.solve(c.anchor_request, n_passes=2)
    anchor = b.measure()
    return UnifiedEngine(c, b, T.settings(engine_preset=preset,
                                          engine_rows=rows), anchor=anchor)


def test_the_bootstrap_scalar_preset_is_the_bootstrap_channel():
    from bouquet.utils import close_ip, closure_sign_convention
    eng = _engine_for("bootstrap_scalar", ["Ip"])
    g = eng.state.geom
    cl = eng.close(g, eng.state.lambda_bs)
    out = cl["out"]
    assert np.ptp(out["s_bs"]) == 0.0 and np.all(out["s_ind"] == 1.0)
    p = cl["parts"]
    jf = p["fix"] + p["pressure"]
    lin = [np.trapezoid(g["w_lin"] * j, g["psi_N"]) for j in
           (p["ind"], p["bs"], jf)]
    sgn, ipt, cs = closure_sign_convention(*lin, g["c_affine"], eng.c.Ip)
    ohm, bs = close_ip("bootstrap", ipt, cs, *lin)
    assert ohm == 1.0
    assert out["s_bs"][0] == pytest.approx(bs, rel=1e-12, abs=0)


def test_the_sawtooth_two_scalar_preset_is_the_q0_closure():
    from bouquet.utils import close_ip_q0, closure_sign_convention
    eng = _engine_for("sawtooth_two_scalar", ["Ip", "q0"],
                      q0_target=Q0 * 0.97)
    g = eng.state.geom
    cl = eng.close(g, eng.state.lambda_bs)
    out = cl["out"]
    p = cl["parts"]
    jf = p["fix"] + p["pressure"]
    lin = [np.trapezoid(g["w_lin"] * j, g["psi_N"]) for j in
           (p["ind"], p["bs"], jf)]
    sgn, ipt, cs = closure_sign_convention(*lin, g["c_affine"], eng.c.Ip)
    ax = cl["axis"]
    ohm, bs = close_ip_q0(ipt, cs, *lin, ax["j_ind0"], ax["j_bs0"],
                          ax["j_fix0"], ax["j_ref0"])
    assert out["s_ind"][0] == pytest.approx(ohm, rel=1e-12, abs=0)
    assert out["s_bs"][0] == pytest.approx(bs, rel=1e-12, abs=0)


def test_the_two_scalar_preset_falls_back_when_the_gate_rejects_q0():
    ad = T.ToyAdapter(li_target=None, q0_target=Q0)
    c = ad.read()
    c.rows["q0"]["admitted"] = False
    c.rows["q0"]["gate_basis"] = "|q0_dd|"
    b = T.ToyGS()
    b.solve(c.anchor_request, n_passes=2)
    eng = _quiet(UnifiedEngine, c, b, T.settings(
        engine_preset="sawtooth_two_scalar", engine_rows=["Ip", "q0"]),
        anchor=b.measure())
    assert eng.preset == "bootstrap_scalar" and "q0" not in eng.rows
    assert eng.notices and "falls back" in eng.notices[0]


def test_the_scalar_bootstrap_preset_converges_end_to_end():
    ad = T.ToyAdapter(li_target=None, jB_ind=_consistent_inductive())
    eng, res, rec, b = _run(ad, engine_preset="bootstrap_scalar",
                            engine_rows=["Ip"])
    assert res["converged"]
    assert rec["settings"]["preset_in_force"] == "bootstrap_scalar"


# ---------------------------------------------------------------------------
#  the delivery correction
# ---------------------------------------------------------------------------
def test_the_delivery_correction_drives_intended_minus_achieved_to_the_floor():
    out = {}
    for dc in (False, True):
        ad = T.ToyAdapter(li_target=LI0 * 1.01)
        eng, res, rec, b = _run(ad, T.ToyGS(defect=0.05),
                                engine_delivery_correction=dc)
        assert res["converged"]
        st = b.state
        I = eng.delivered_closure["jc"]
        A = st["A"] / st["c"]
        out[dc] = float(np.max(np.abs(I - A)) / np.max(np.abs(A)))
        if dc:
            assert rec["delivered"]["checks"]["l_i"]["ok"]
            assert np.any(res["state"].delivery_correction != 0.0)
    # OFF: the toy's defect (5 % bump) is delivered as is (~1 % of peak);
    # ON: one Newton step per pass removes it to the loop's own floor
    assert out[False] > 5e-3
    assert out[True] < 0.1 * out[False]


# ---------------------------------------------------------------------------
#  failures are loud
# ---------------------------------------------------------------------------
def test_an_unreachable_l_i_target_fails_loudly():
    ad = T.ToyAdapter(li_target=LI0 * 1.6)
    with pytest.raises((EngineClosureRefused, JBSNotConverged)) as ei:
        _run(ad)
    assert str(ei.value)


def test_a_non_finite_bootstrap_raises_even_under_the_flag_policy():
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    with pytest.raises(JBSNonFinite):
        _run(ad, T.ToyGS(nan_redl_at=4), jbs_loop_on_fail="flag")


def test_the_flag_policy_delivers_a_flagged_result():
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    eng, res, rec, b = _run(ad, jbs_loop_on_fail="flag", jbs_max_passes=3)
    assert not res["converged"] and not res["loop_converged"]
    assert "fail_message" in rec["phases"][0]["record"]
    with pytest.raises(JBSNotConverged):
        _run(T.ToyAdapter(li_target=LI0 * 1.01), jbs_max_passes=3)


# ---------------------------------------------------------------------------
#  settings / config
# ---------------------------------------------------------------------------
def _cfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    return BouquetConfig(source=ImasSource(ids_path="x.json"),
                         solver=SolverConfig(mesh_path="m.h5"),
                         output_header="t",
                         generation=GenerationConfig(**gen))


def test_the_engine_defaults_to_legacy_and_old_configs_load_as_legacy():
    from bouquet.config import BouquetConfig
    c = _cfg()
    assert c.generation.reconstruction_engine == "legacy"
    d = c.to_dict()
    for k in ("reconstruction_engine", "engine_preset", "engine_rows",
              "engine_delivery_correction", "engine_mse_jacobian"):
        del d["generation"][k]
    assert BouquetConfig.from_dict(d).generation.reconstruction_engine \
        == "legacy"


def test_unified_settings_round_trip():
    from bouquet.config import BouquetConfig
    c = _cfg(reconstruction_engine="unified",
             engine_rows=["Ip", "l_i", "q0"],
             engine_delivery_correction=True, engine_mse_jacobian="fd_chord")
    c2 = BouquetConfig.from_dict(json.loads(c.to_json()))
    s = engine_settings(c2.generation)
    assert s["rows"] == ("Ip", "l_i", "q0")
    assert s["delivery_correction"] is True
    assert s["mse_jacobian"] == "fd_chord"
    assert s["loop"]["gate_current_residual"] is True


@pytest.mark.parametrize("gen, match", [
    (dict(reconstruction_engine="new"), "reconstruction_engine"),
    (dict(engine_preset="bootstrap_scalar"), "no effect"),
    (dict(engine_delivery_correction=True), "no effect"),
    (dict(reconstruction_engine="unified", engine_preset="x"),
     "engine_preset"),
    (dict(reconstruction_engine="unified", engine_rows=["l_i"]), "'Ip'"),
    (dict(reconstruction_engine="unified", engine_rows="Ip"), "list"),
    (dict(reconstruction_engine="unified", engine_rows=["Ip", "x"]),
     "unknown"),
    (dict(reconstruction_engine="unified", engine_preset="bootstrap_scalar",
          engine_rows=["Ip", "l_i"]), "admits"),
    (dict(reconstruction_engine="unified",
          engine_preset="sawtooth_two_scalar", engine_rows=["Ip"]), "q0"),
    (dict(reconstruction_engine="unified", engine_delivery_correction=1),
     "bool"),
    (dict(reconstruction_engine="unified", engine_mse_jacobian="exact"),
     "engine_mse_jacobian"),
    (dict(reconstruction_engine="unified", engine_rows=["Ip", "mse"]),
     "mse_data"),
    (dict(reconstruction_engine="unified", jbs_self_consistent=False),
     "jbs_self_consistent"),
    (dict(reconstruction_engine="unified", jbs_init="swb"), "anchor"),
])
def test_bad_engine_settings_are_refused_by_name(gen, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**gen)


def test_the_convergence_table_quotes_every_constant_from_its_home():
    from bouquet.jbs_loop import (JBS_RELAX_FLOOR, JBS_REQUIRED_CONSECUTIVE,
                                  MSE_CHORD_OFFSET_TOL_SIGMA)
    from bouquet.utils import IP_ROUNDTRIP_TOL_PCT
    t = {r["criterion"]: r for r in convergence_table(T.settings())}
    assert t["r_j"]["value"] == 1e-3 and t["r_I"]["value"] == 1e-4
    assert t["|dl_i|"]["value"] == 1e-3
    assert t["|dq0| (q0 row active)"]["value"] == 2e-3
    assert t["|q0 - q0_target| (q0 row active)"]["value"] == 0.01
    assert t["l_i, g-file hard row"]["value"] == 1e-3 == gfile_li_row_tol()
    assert t["l_i, IDS hard row / soft-row discrepancy"]["value"] == 0.005
    assert t["MSE tan(gamma) change [sigma_eff]"]["value"] \
        == MSE_CHORD_OFFSET_TOL_SIGMA
    assert t["consecutive passes"]["value"] == JBS_REQUIRED_CONSECUTIVE
    assert t["pass ceiling"]["value"] == 8
    assert t["omega floor"]["value"] == JBS_RELAX_FLOOR
    assert t["closure scale bounds"]["value"] == [0.2, 5.0]
    assert t["Ip round trip [%]"]["value"] == IP_ROUNDTRIP_TOL_PCT


# ---------------------------------------------------------------------------
#  wiring
# ---------------------------------------------------------------------------
class _Sentinel(Exception):
    pass


def test_legacy_prepare_baseline_never_enters_the_engine(monkeypatch):
    import bouquet.baseline as bb
    import bouquet.engine as be
    from bouquet.run import Bouquet

    def _no_engine(bq):
        raise AssertionError("the engine was entered with 'legacy'")

    def _legacy(cfg, mygs):
        raise _Sentinel("legacy resolver reached")

    monkeypatch.setattr(be, "prepare_engine_baseline", _no_engine)
    monkeypatch.setattr(bb, "resolve_baseline", _legacy)
    bq = Bouquet(_cfg())
    with pytest.raises(_Sentinel):
        bq.prepare_baseline()


def test_unified_prepare_baseline_dispatches_to_the_engine(monkeypatch):
    import bouquet.engine as be
    from bouquet.run import Bouquet
    seen = {}

    def _eng(bq):
        seen["bq"] = bq
        return "engine-baseline"

    monkeypatch.setattr(be, "prepare_engine_baseline", _eng)
    bq = Bouquet(_cfg(reconstruction_engine="unified"))
    assert bq.prepare_baseline() == "engine-baseline"
    assert seen["bq"] is bq


def test_the_draws_refuse_an_engine_baseline():
    from bouquet.run import Bouquet
    bq = Bouquet(_cfg(reconstruction_engine="unified"))
    bq.baseline = object()
    bq.mygs = object()
    with pytest.raises(NotImplementedError, match="Stage 3"):
        bq.generate()
    with pytest.raises(NotImplementedError, match="Stage 3"):
        bq._refuse_unified_engine_draws("verify_sigma0_consistency()")
    # a legacy config with a legacy baseline is not refused
    bq2 = Bouquet(_cfg())
    bq2._refuse_unified_engine_draws("generate()")


# ---------------------------------------------------------------------------
#  archive: an ADDED _baseline attribute
# ---------------------------------------------------------------------------
def test_the_engine_record_is_stored_beside_the_baseline(tmp_path):
    import h5py
    from bouquet.utils import initialize_equilibrium_database
    h = str(tmp_path / "arch")
    initialize_equilibrium_database(h)
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    eng, res, rec, b = _run(ad)
    with h5py.File(h + ".h5", "a") as hf:
        hf.create_group("_baseline")
    assert load_baseline_engine(h) is None
    store_baseline_engine(h, rec)
    back = load_baseline_engine(h)
    assert back["version"] == rec["version"]
    assert back["converged"] is True
    with h5py.File(h + ".h5", "r") as hf:
        assert ENGINE_ATTR in hf["_baseline"].attrs
        assert int(hf.attrs["schema_version"]) == 3
