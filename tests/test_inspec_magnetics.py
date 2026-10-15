"""The in-spec rule is the magnetics (owner decision 2026-10-14).

A draw is IN SPEC when it passes the configured coil test and the boundary
cut (and was not rejected before archiving).  The l_i band is no longer part
of the default; it and a beta_p band are opt-in, for a source that states a
measurement uncertainty.  Lambda = beta_p + l_i(3)/2 is stored and reported,
never filtered on.  Checked here, fast (no GS solver; the toy engine and its
TokaMaker stand-in, hand-built archives):

* the rule and its opt-in bands (``bouquet.filtering.InspecBands``), their
  validation, and that a band never passes an unmeasured quantity;
* Lambda's definition and its value in the engine's post-hoc record;
* the stored-config compatibility: a config stored before the rule at the
  old default ``l_i_tolerance=0.05`` loads with the band OFF and one note;
  an explicit stored value keeps its band; a current config round-trips;
* the archive flags (per band, ``passes_draw_band``, ``inspec_rule``) and
  the re-filter of an archive judged under the old unconditional band;
* end to end on the toy: engine draws far outside the old 5 % band are in
  spec and counted by until-N; the explicit band still rejects them; the
  rule and Lambda definition are stamped; ``Bouquet.filter()`` re-judges an
  old archive; ``merge_archives`` carries the rule and refuses a mix.

Synthetic inputs only; no solver, no device data.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

import test_engine_draws as TED
from test_engine_draws import toy_bouquet_solver  # noqa: F401  (fixture)

from bouquet import engine_draws as ED
from bouquet import filtering as F
from bouquet.config import BouquetConfig, GenerationConfig


# --------------------------------------------------------------------------
#  the rule and its opt-in bands
# --------------------------------------------------------------------------
def test_the_default_rule_is_the_magnetics():
    b = F.resolve_inspec_bands(GenerationConfig())
    assert GenerationConfig().l_i_tolerance is None
    assert b.rule() == "magnetics" and not b.any_on
    # an l_i 30 % off the reconstruction is in spec on the magnetics alone
    v = b.verdicts(l_i=1.3, l_i_reference=1.0, beta_p=0.4)
    assert v["in_band"] is True and v["l_i_in_band"] is None
    assert v["beta_p_in_band"] is None and v["q0_ok"] is None
    assert b.flags(v) == {"passes_draw_band": True}


def test_the_fractional_l_i_band_is_the_old_band_when_set():
    b = F.resolve_inspec_bands(GenerationConfig(l_i_tolerance=0.05))
    assert b.rule() == "magnetics+li"
    assert b.verdicts(l_i=1.049, l_i_reference=1.0)["in_band"]
    v = b.verdicts(l_i=1.051, l_i_reference=1.0)
    assert not v["in_band"] and v["l_i_in_band"] is False and v["reasons"]
    assert b.flags(v) == {"passes_l_i_band": False, "passes_draw_band": False}


def test_the_sigma_l_i_band_takes_its_width_from_the_stated_sigma():
    b = F.resolve_inspec_bands(GenerationConfig(l_i_sigma=0.02))
    assert b.rule() == "magnetics+li" and b.nsigma == 2.0
    assert b.verdicts(l_i=0.939, l_i_reference=0.9)["in_band"]   # 1.95 sigma
    assert not b.verdicts(l_i=0.941, l_i_reference=0.9)["in_band"]
    b3 = F.resolve_inspec_bands(GenerationConfig(l_i_sigma=0.02,
                                                 inspec_band_nsigma=3.0))
    v = b3.verdicts(l_i=0.959, l_i_reference=0.9)
    assert v["in_band"] and v["l_i_half_width"] == pytest.approx(0.06)


def test_the_beta_p_band_is_a_diamagnetic_measurement():
    b = F.resolve_inspec_bands(GenerationConfig(beta_p_target=0.60,
                                                beta_p_sigma=0.05))
    assert b.rule() == "magnetics+beta_p"
    v = b.verdicts(l_i=0.9, l_i_reference=0.9, beta_p=0.69)
    assert v["in_band"] and v["beta_p_z"] == pytest.approx(1.8)
    v = b.verdicts(l_i=0.9, l_i_reference=0.9, beta_p=0.71)
    assert not v["in_band"] and b.flags(v)["passes_beta_p_band"] is False
    both = F.resolve_inspec_bands(GenerationConfig(
        l_i_sigma=0.02, beta_p_target=0.6, beta_p_sigma=0.05,
        constrain_sawteeth=True))
    assert both.rule() == "magnetics+li+beta_p+q0"


def test_an_unmeasured_quantity_fails_a_band_that_is_on():
    b = F.InspecBands(l_i_sigma=0.02, beta_p_target=0.6, beta_p_sigma=0.05,
                      constrain_sawteeth=True)
    v = b.verdicts(l_i=float("nan"), l_i_reference=0.9, beta_p=None, q0=None)
    assert v["l_i_in_band"] is False and v["beta_p_in_band"] is False
    assert v["q0_ok"] is False and not v["in_band"]
    # ... and an unmeasured quantity does not matter while its band is off
    assert F.InspecBands().verdicts(l_i=None, beta_p=None)["in_band"]


@pytest.mark.parametrize("kw, match", [
    (dict(l_i_tolerance=0.05, l_i_sigma=0.02), "set one, not both"),
    (dict(beta_p_target=0.6), "both beta_p_target and beta_p_sigma"),
    (dict(beta_p_sigma=0.05), "both beta_p_target and beta_p_sigma"),
    (dict(l_i_sigma=0.0), "must be a finite number > 0"),
    (dict(l_i_tolerance=-0.05), "must be a finite number > 0"),
    (dict(beta_p_target=0.6, beta_p_sigma=float("nan")), "finite"),
    (dict(inspec_band_nsigma=0), "must be a finite number > 0"),
    (dict(l_i_sigma=True), "must be a finite number > 0"),
])
def test_a_malformed_band_is_refused_at_construction(kw, match):
    with pytest.raises(ValueError, match=match):
        GenerationConfig(**kw)


def test_the_band_record_round_trips():
    b = F.InspecBands(l_i_sigma=0.02, beta_p_target=0.6, beta_p_sigma=0.05,
                      nsigma=2.5, constrain_sawteeth=True)
    back = F.InspecBands.from_record(json.loads(json.dumps(b.record())))
    assert back == b and back.rule() == b.rule()
    assert F.InspecBands() != b


def test_the_legacy_construction_tolerance_is_unchanged():
    assert F.l_i_construction_tolerance(None) == 0.05
    assert F.l_i_construction_tolerance(GenerationConfig()) == 0.05
    assert F.l_i_construction_tolerance(
        GenerationConfig(l_i_tolerance=0.01)) == 0.01


# --------------------------------------------------------------------------
#  Lambda
# --------------------------------------------------------------------------
def test_lambda_is_beta_p_plus_half_l_i3():
    assert F.shafranov_lambda(0.5, 0.9) == pytest.approx(0.95)
    assert np.isnan(F.shafranov_lambda(float("nan"), 0.9))
    assert np.isnan(F.shafranov_lambda(None, 0.9))
    d = F.LAMBDA_DEFINITION
    for phrase in ("beta_p + l_i(3)/2", "mu0 I_p / L_p", "FULL pressure",
                   "li_normalization='iter'", "magnetic-axis",
                   "shape-weighted", "never filtered"):
        assert phrase in d, phrase


class _Ctx:
    ref = {"l_i": 0.90, "q_row_psi_N": 1e-3}


def test_the_engine_post_hoc_record_reports_beta_p_and_lambda():
    fin = dict(li=0.99, q_row=1.1,
               pressure_frames={"full": {"beta_pol": 62.0, "beta_n": 1.6}})
    v = ED.post_hoc_verdicts(_Ctx(), fin)
    # 10 % off in l_i, still in band: no band on by default
    assert v["in_band"] is True and v["rule"] == "magnetics"
    assert v["beta_p"] == pytest.approx(0.62)
    assert v["Lambda"] == pytest.approx(0.62 + 0.99 / 2)
    assert v["q0_ok"] is True          # the gate is off: historical spelling
    # the explicit tolerance still bands (back-compatible keyword)
    v = ED.post_hoc_verdicts(_Ctx(), fin, l_i_tolerance=0.05)
    assert v["in_band"] is False and v["l_i_in_band"] is False
    # an opt-in beta_p band reads the full-pressure frame
    v = ED.post_hoc_verdicts(_Ctx(), fin, bands=F.InspecBands(
        beta_p_target=0.60, beta_p_sigma=0.005))
    assert v["beta_p_in_band"] is False and v["beta_p_z"] == pytest.approx(4.0)
    # no frames (a backend that reports none): beta_p and Lambda are NaN,
    # which matters only to a beta_p band
    v = ED.post_hoc_verdicts(_Ctx(), dict(li=0.99, q_row=1.1))
    assert np.isnan(v["beta_p"]) and np.isnan(v["Lambda"]) and v["in_band"]


# --------------------------------------------------------------------------
#  stored-config compatibility
# --------------------------------------------------------------------------
def _stored_dict(**gen):
    from bouquet.config import ReconstructionSource, SolverConfig
    c = BouquetConfig(source=ReconstructionSource(geqdsk_path="g.geqdsk",
                                                  profiles_path="p.peqdsk"),
                      solver=SolverConfig(mesh_path="m.h5"),
                      output_header="h")
    d = c.to_dict()
    d["generation"].update(gen)
    return d


def _load(d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = BouquetConfig.from_dict(d)
    return c.generation, [str(x.message) for x in w]


def test_a_stored_config_at_the_old_default_loads_with_the_band_off():
    d = _stored_dict(l_i_tolerance=0.05)
    del d["generation"]["inspec_band_nsigma"]     # written before the rule
    g, msgs = _load(d)
    assert g.l_i_tolerance is None
    assert F.resolve_inspec_bands(g).rule() == "magnetics"
    notes = [m for m in msgs if "2026-10-14 in-spec rule" in m]
    assert len(notes) == 1 and "\n" not in notes[0]
    assert "l_i_tolerance=0.05" in notes[0] and "OFF" in notes[0]


@pytest.mark.parametrize("tol", [0.01, 0.03, 0.1])
def test_a_stored_explicit_tolerance_keeps_its_band(tol):
    d = _stored_dict(l_i_tolerance=tol)
    del d["generation"]["inspec_band_nsigma"]
    g, msgs = _load(d)
    assert g.l_i_tolerance == tol
    assert F.resolve_inspec_bands(g).rule() == "magnetics+li"
    assert not [m for m in msgs if "2026-10-14 in-spec rule" in m]


@pytest.mark.parametrize("tol", [None, 0.05])
def test_a_current_config_round_trips_silently(tol):
    d = BouquetConfig.from_dict(_stored_dict(l_i_tolerance=tol)).to_dict()
    assert "inspec_band_nsigma" in d["generation"]
    g, msgs = _load(d)
    # an explicit 0.05 written AFTER the rule is an explicit band, kept
    assert g.l_i_tolerance == tol
    assert not [m for m in msgs if "in-spec rule" in m]


def test_a_stored_config_without_the_field_loads_loud_with_the_band_off():
    d = _stored_dict()
    del d["generation"]["l_i_tolerance"]
    del d["generation"]["inspec_band_nsigma"]
    g, msgs = _load(d)
    assert g.l_i_tolerance is None
    assert any("LACKS generation.l_i_tolerance" in m for m in msgs)


# --------------------------------------------------------------------------
#  archive flags and the re-filter of an old archive
# --------------------------------------------------------------------------
_LI_REF = 0.90


def _old_engine_archive(path, lis=(0.90, 0.93, 0.96, 0.80), legacy=(),
                        beta_p=0.60):
    """An archive as the pre-rule code left it: engine draws whose
    ``passes_draw_band`` is the unconditional 5 % l_i band, coil and
    boundary flags all passing, no ``inspec_rule`` stamp."""
    from bouquet.engine import write_engine_json
    with h5py.File(path, "w") as hf:
        sg = hf.create_group("scan/0")
        bl = sg.create_group("_baseline")
        bl.attrs["l_i_target"] = _LI_REF
        for i, li in enumerate(lis):
            g = sg.create_group(str(i))
            g.attrs["l_i(3)"] = li
            g.attrs["passes_coil_filter"] = True
            g.attrs["passes_boundary_filter"] = True
            g.attrs["edge_pressure_json"] = json.dumps(
                {"frames": {"full": {"beta_pol": 100.0 * beta_p}}})
            if i in legacy:
                F._recompute_selected(g)
                continue
            old = abs(li - _LI_REF) / _LI_REF <= 0.05
            write_engine_json(g, {"post_hoc": dict(
                l_i_3=li, l_i_reference=_LI_REF, l_i_tolerance=0.05,
                l_i_in_band=old, q0=0.95 if i == 1 else 1.2,
                q0_psi_N=1e-3, in_band=old)})
            g.attrs[ED.DRAW_BAND_FLAG] = old
            F._recompute_selected(g)
    return str(path)


def test_the_old_archive_selects_on_the_old_band(tmp_path):
    p = _old_engine_archive(tmp_path / "a.h5")
    assert F.select_indices(p, scan_key="0") == [0, 1]
    assert F.read_inspec_rule(p, scan_key="0") == (None, None)


def test_refilter_resets_an_old_archive_to_the_magnetics(tmp_path):
    p = _old_engine_archive(tmp_path / "a.h5")
    out = F.refilter_inspec(p, scan_key="0", verbose=False)
    assert out["rule"] == "magnetics" and out["n_selected"] == 4
    assert F.select_indices(p, scan_key="0") == [0, 1, 2, 3]
    rule, bands = F.read_inspec_rule(p, scan_key="0")
    assert rule == "magnetics" and bands == F.InspecBands()
    flags = F.read_filter_flags(p, scan_key="0")
    for i, rec in flags.items():
        assert rec["passes_draw_band"] is True
        assert "passes_l_i_band" not in rec
        assert rec["beta_p"] == pytest.approx(0.60)
        assert rec["Lambda"] == pytest.approx(0.60 + rec_li(p, i) / 2)
    with h5py.File(p, "r") as hf:
        assert hf["scan/0"].attrs[F.LAMBDA_DEFINITION_ATTR] \
            == F.LAMBDA_DEFINITION


def rec_li(p, i):
    with h5py.File(p, "r") as hf:
        return float(hf[f"scan/0/{i}"].attrs["l_i(3)"])


def test_refilter_applies_and_removes_an_opt_in_band(tmp_path):
    p = _old_engine_archive(tmp_path / "a.h5", legacy=(3,))
    # a stated 1-sigma of 0.02 -> +/-0.04 around 0.90: draws 0 and 1
    F.refilter_inspec(p, F.InspecBands(l_i_sigma=0.02), scan_key="0",
                      verbose=False)
    fl = F.read_filter_flags(p, scan_key="0")
    assert [fl[i]["passes_l_i_band"] for i in range(4)] == \
        [True, True, False, False]
    assert F.select_indices(p, scan_key="0") == [0, 1]
    assert F.read_inspec_rule(p, scan_key="0")[0] == "magnetics+li"
    # the q0 gate is judged on engine draws only (draw 1 has q0 0.95)
    F.refilter_inspec(p, F.InspecBands(constrain_sawteeth=True),
                      scan_key="0", verbose=False)
    fl = F.read_filter_flags(p, scan_key="0")
    assert "passes_l_i_band" not in fl[0]
    assert fl[1]["passes_q0_gate"] is False and fl[0]["passes_q0_gate"]
    assert "passes_q0_gate" not in fl[3]          # the legacy draw
    assert F.select_indices(p, scan_key="0") == [0, 2, 3]
    # back to the default: every opt-in flag gone, all selected
    F.refilter_inspec(p, None, scan_key="0", verbose=False)
    fl = F.read_filter_flags(p, scan_key="0")
    assert not any(k in r for r in fl.values() for k in F._BAND_FLAGS)
    assert "passes_draw_band" not in fl[3]        # never invented on legacy
    assert F.select_indices(p, scan_key="0") == [0, 1, 2, 3]


def test_a_beta_p_band_reads_the_archived_full_frame(tmp_path):
    p = _old_engine_archive(tmp_path / "a.h5", beta_p=0.70)
    F.refilter_inspec(p, F.InspecBands(beta_p_target=0.60, beta_p_sigma=0.02),
                      scan_key="0", verbose=False)
    assert F.select_indices(p, scan_key="0") == []
    F.refilter_inspec(p, F.InspecBands(beta_p_target=0.66, beta_p_sigma=0.02),
                      scan_key="0", verbose=False)
    assert F.select_indices(p, scan_key="0") == [0, 1, 2, 3]


def test_the_band_tables_report_beta_p_and_lambda(tmp_path):
    import bouquet as bq
    p = _old_engine_archive(tmp_path / "a.h5")
    F.refilter_inspec(p, scan_key="0", verbose=False)
    with h5py.File(p, "a") as hf:
        hf["scan/0"].attrs["coil_filter"] = "chi2"
    recs = bq.draw_band(p, "0", lambda v: F_shafranov(v), baseline=False)
    assert recs["Lambda"].values == pytest.approx(
        {i: 0.60 + li / 2 for i, li in enumerate((0.90, 0.93, 0.96, 0.80))})
    assert recs["Lambda"].provenance["inspec_rule"] == "magnetics"
    assert not any("in-spec rule is not recorded" in s
                   for s in recs["Lambda"].provenance["limitations"])


def F_shafranov(view):
    from bouquet.stats import _shafranov_entries
    return _shafranov_entries(view)


def test_an_unstamped_archive_is_flagged_in_the_band_limitations(tmp_path):
    import bouquet as bq
    p = _old_engine_archive(tmp_path / "a.h5")
    with h5py.File(p, "a") as hf:
        hf["scan/0"].attrs["coil_filter"] = "chi2"
    recs = bq.draw_band(p, "0", F_shafranov, baseline=False)
    r = recs["beta_p"]
    assert r.provenance["inspec_rule"] == "unrecorded" and r.n_used == 2
    assert any("in-spec rule is not recorded" in s
               for s in r.provenance["limitations"])


# --------------------------------------------------------------------------
#  end to end on the toy engine
# --------------------------------------------------------------------------
def _spread_li(diags):
    return max(abs(d["engine"]["post_hoc"]["l_i_rel"]) for d in diags)


def test_engine_draws_outside_the_old_band_are_in_spec(tmp_path, monkeypatch):
    """Draws whose l_i lies outside a tight band are archived in spec under
    the default rule and counted by until-N; the same draws with the band
    set explicitly are out of spec and not counted."""
    diags, rej, h, G = TED._generate(tmp_path / "m", monkeypatch, n=2,
                                     l_i_tolerance=None, n_inspec_target=2)
    assert rej == [] and len(diags) == 2
    assert G.bands.rule() == "magnetics"
    for d in diags:
        ph = d["engine"]["post_hoc"]
        assert ph["in_band"] is True and ph["l_i_in_band"] is None
        assert d["in_spec"] is True and d["until_n_inspec"] is True
        assert "Lambda" in d["engine"]["archived"]
    assert _spread_li(diags) > 1e-9          # a real l_i spread, not judged
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is True
        assert F.L_I_BAND_FLAG not in g0.attrs
    # the explicit tolerance below that spread: archived, out, not counted
    diags, rej, h, G = TED._generate(tmp_path / "t", monkeypatch, n=2,
                                     l_i_tolerance=1e-9, n_inspec_target=1)
    assert G.bands.rule() == "magnetics+li"
    assert all(d["in_spec"] is False and d["until_n_inspec"] is False
               for d in diags)
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[F.L_I_BAND_FLAG]) is False
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is False


def _toy_run(tmp_path, n=2, **gen):
    b = TED._bq(tmp_path, n=n)
    for k, v in gen.items():
        setattr(b.config.generation, k, v)
    b.setup_solver()
    TED._quiet(b.prepare_baseline)
    TED._quiet(b.generate)
    return b


def test_generate_stamps_the_rule_and_filter_keeps_it(tmp_path,
                                                      toy_bouquet_solver):
    b = _toy_run(tmp_path)
    h, sk = b.config.output_header, b.config.generation.scan_key
    rule, bands = F.read_inspec_rule(h, scan_key=sk)
    assert rule == "magnetics" and bands == F.InspecBands()
    with h5py.File(h + ".h5", "r") as hf:
        gp = [k for k in hf["scan"]][0]
        rec = json.loads(hf[f"scan/{gp}"].attrs[F.INSPEC_BANDS_ATTR])
        assert rec["route"] == "engine"
        assert hf[f"scan/{gp}"].attrs[F.LAMBDA_DEFINITION_ATTR] \
            == F.LAMBDA_DEFINITION
    sel = TED._quiet(b.filter)
    assert sel["inspec"] == {"rule": "magnetics", "refiltered": False,
                             "previous_rule": "magnetics"}


def test_filter_rejudges_an_archive_made_under_the_old_rule(
        tmp_path, toy_bouquet_solver, capsys):
    b = _toy_run(tmp_path)
    h, sk = b.config.output_header, b.config.generation.scan_key
    # make it look like a pre-rule archive: no stamp, band flags False
    with h5py.File(h + ".h5", "a") as hf:
        gs = hf[f"scan/{[k for k in hf['scan']][0]}"]
        for k in (F.INSPEC_RULE_ATTR, F.INSPEC_BANDS_ATTR):
            del gs.attrs[k]
        for k in gs:
            if k.isdigit():
                gs[k].attrs[ED.DRAW_BAND_FLAG] = False
    sel = b.filter()
    assert sel["inspec"]["refiltered"] is True
    assert sel["inspec"]["previous_rule"] is None
    assert "predates the in-spec rule record" in capsys.readouterr().out
    assert F.read_inspec_rule(h, scan_key=sk)[0] == "magnetics"
    fl = F.read_filter_flags(h, scan_key=sk)
    assert all(r["passes_draw_band"] for r in fl.values())


def test_a_measurement_band_is_refused_on_the_legacy_route(tmp_path):
    b = TED._bq(tmp_path)
    b.config.generation.reconstruction_engine = "legacy"
    b.config.generation.l_i_sigma = 0.02
    with pytest.raises(ValueError, match="refilter_inspec"):
        b._inspec_bands()
    b.config.generation.l_i_sigma = None
    b.config.generation.l_i_tolerance = 0.02
    bands, extra = b._inspec_bands()
    # on the legacy route the tolerance is the draw CONSTRUCTION's, not a
    # post-hoc band
    assert bands.rule() == "magnetics"
    assert extra["l_i_construction_tolerance"] == 0.02


def test_merge_carries_the_rule_and_refuses_a_mix(tmp_path,
                                                  toy_bouquet_solver):
    from bouquet.parallel import merge_archives, run_shard
    b = TED._bq(tmp_path, n=2)
    out = str(tmp_path / "par")
    metas = [TED._quiet(run_shard, b.config, w, 2, n_equils_total=2,
                        seed_base=7, out_header=out, scan_key=0,
                        threads_per_worker=1, verbose=True) for w in (0, 1)]
    paths = [m["path"] for m in metas]
    path, n = TED._quiet(merge_archives, paths, out + "_m", scan_key=0)
    with h5py.File(path, "r") as hf:
        a = hf["scan/0"].attrs
        assert a[F.INSPEC_RULE_ATTR] == "magnetics"
        man = json.loads(a["parallel_manifest_json"])
        assert man["inspec_rule"] == "magnetics"
        assert man["inspec_bands"]["rule"] == "magnetics"
    # one shard re-judged under another rule: the merge refuses
    F.refilter_inspec(paths[1], F.InspecBands(l_i_sigma=0.5), scan_key=0,
                      verbose=False)
    with pytest.raises(RuntimeError, match="different in-spec rules"):
        TED._quiet(merge_archives, paths, out + "_x", scan_key=0)


# --------------------------------------------------------------------------
#  review F1: the stamp is the rule AS APPLIED (the q0 gate switched off)
# --------------------------------------------------------------------------
def _sawtoothing_run(tmp_path):
    """constrain_sawteeth=True on a reconstruction whose q(0) < 1: the toy's
    q scale is lowered after the reconstruction, so generate_bouquet's entry
    check sees q0 < 1 and switches the gate off for the scan."""
    b = TED._bq(tmp_path, n=3)
    b.config.generation.constrain_sawteeth = True
    b.setup_solver()
    TED._quiet(b.prepare_baseline)
    b.mygs.toy.q_scale *= 0.3
    TED._quiet(b.generate)
    return b


def test_a_switched_off_q0_gate_is_not_stamped(tmp_path, toy_bouquet_solver):
    b = _sawtoothing_run(tmp_path)
    h, sk = b.config.output_header, b.config.generation.scan_key
    rule, bands = F.read_inspec_rule(h, scan_key=sk)
    assert rule == "magnetics" and bands.constrain_sawteeth is False
    rec = F.read_inspec_record(h, scan_key=sk)
    q0g = rec[F.Q0_GATE_RECORD_KEY]
    assert q0g["configured"] is True and q0g["applied"] is False
    assert q0g["q0"] < 1.0 and "q(0)" in q0g["reason"]
    # the baseline's engine record says the same
    from bouquet.engine import load_baseline_engine
    eng = load_baseline_engine(h, scan_key=sk)
    assert eng["draws"]["inspec_bands"]["constrain_sawteeth"] is False
    assert eng["draws"]["q0_gate"]["applied"] is False
    # the draws were judged without the gate: q0 < 1, still in band
    ph = [ED.read_draw_engine(h, i, scan_key=sk)["post_hoc"]
          for i in F.list_equilibrium_indices(h, scan_key=sk)]
    assert ph and all(p["q0"] < 1.0 and p["constrain_sawteeth"] is False
                      and p["in_band"] for p in ph)


def test_a_refilter_under_the_stamped_rule_reproduces_selected(
        tmp_path, toy_bouquet_solver):
    b = _sawtoothing_run(tmp_path)
    h, sk = b.config.output_header, b.config.generation.scan_key
    sel = TED._quiet(b.filter)
    # filter() sees the stamp as the rule this config applied: no re-judge
    assert sel["inspec"]["refiltered"] is False
    assert sel["inspec"]["rule"] == "magnetics"
    before = F.select_indices(h, scan_key=sk, selection="selected")
    assert before                                  # something to reproduce
    F.refilter_inspec(h, F.read_inspec_rule(h, scan_key=sk)[1], scan_key=sk,
                      verbose=False)
    assert F.select_indices(h, scan_key=sk, selection="selected") == before
    # ... and the switched-off record survives that re-stamp, so a later
    # filter() still does not apply the gate generation switched off
    assert F.q0_gate_switched_off(
        F.read_inspec_record(h, scan_key=sk)[F.Q0_GATE_RECORD_KEY])
    sel = TED._quiet(b.filter)
    assert sel["inspec"]["refiltered"] is False
    assert F.select_indices(h, scan_key=sk, selection="selected") == before


# --------------------------------------------------------------------------
#  review F2: the archived in_spec is the in-spec rule's verdict
# --------------------------------------------------------------------------
def _rule_run(tmp_path, monkeypatch, *, coil_ok, drift_ok, **extra):
    """An engine run whose configured coil test says *coil_ok* (a stand-in
    for the chi2 predicate) and whose legacy +/-2 % drift test says
    *drift_ok*, independently."""
    def _pred(*a, **k):
        def p(diagnostics, draw_currents=None):
            return bool(coil_ok), {"chi2_nu": 0.5 if coil_ok else 9.0,
                                   "max_abs_z": 0.7 if coil_ok else 5.0,
                                   "coil_nu": 4, "worst_coil": "F1A"}
        model = {"kind": "stand-in", "acceptance": {
            "chi2_max": 2.0, "z_max": None, "source": "test",
            "calibrated_nu": None, "nu_sigma": 4}}
        return p, "chi2", model
    monkeypatch.setattr(F, "make_coil_predicate", _pred)
    monkeypatch.setattr(F, "passes_coil_spec", lambda *a, **k: bool(drift_ok))
    return TED._generate(tmp_path, monkeypatch, n=2, l_i_tolerance=None,
                         n_inspec_target=2, **extra)


def _archived(h, i):
    with h5py.File(h + ".h5", "r") as hf:
        g = hf[f"scan/0/{i}"] if "scan" in hf else hf[str(i)]
        return dict(g.attrs)


def test_a_chi2_in_spec_draw_beyond_the_drift_band_is_in_spec(
        tmp_path, monkeypatch):
    diags, rej, h, G = _rule_run(tmp_path, monkeypatch, coil_ok=True,
                                 drift_ok=False)
    assert rej == [] and len(diags) == 2
    flags = next(iter(F.read_filter_flags(h).values()))
    for i, d in enumerate(diags):
        assert d["in_spec"] is True and d["until_n_inspec"] is True
        assert d["coil_drift_in_spec"] is False
        ph = ED.read_draw_engine(h, i)["post_hoc"]
        assert ph["in_spec"] is True and ph["coil_in_spec"] is True
        assert ph["coil_drift_in_spec"] is False
        assert ph["in_spec_source"] == F.IN_SPEC_RULE_SOURCE
        a = _archived(h, i)
        assert bool(a["in_spec"]) is True
        assert bool(a[F.COIL_DRIFT_IN_SPEC_ATTR]) is False
        assert a[F.IN_SPEC_SOURCE_ATTR] == F.IN_SPEC_RULE_SOURCE
        assert flags[i]["in_spec"] is True
        assert flags[i]["coil_drift_in_spec"] is False
        assert flags[i]["in_spec_source"] == F.IN_SPEC_RULE_SOURCE


@pytest.mark.parametrize("why", ["coil", "boundary"])
def test_a_drift_in_band_draw_outside_the_rule_is_out_of_spec(
        tmp_path, monkeypatch, why):
    """The reverse: inside the +/-2 % drift band, but failing the
    configured coil test -- or the boundary cut the drift test never
    had."""
    extra = ({} if why == "coil" else dict(inspec_rms_max_mm=1e-9))
    diags, rej, h, G = _rule_run(tmp_path, monkeypatch,
                                 coil_ok=(why != "coil"), drift_ok=True,
                                 **extra)
    assert rej == [] and diags
    for i, d in enumerate(diags):
        assert d["in_spec"] is False and d["until_n_inspec"] is False
        assert d["coil_drift_in_spec"] is True
        ph = ED.read_draw_engine(h, i)["post_hoc"]
        assert ph["in_spec"] is False and ph["coil_drift_in_spec"] is True
        assert ph["in_spec"] == (ph["coil_in_spec"] and ph["in_band"])
        a = _archived(h, i)
        assert bool(a["in_spec"]) is False
        assert bool(a[F.COIL_DRIFT_IN_SPEC_ATTR]) is True


def test_an_archive_before_the_rule_verdict_reads_its_drift_in_spec():
    """No in_spec_source: in_spec IS the drift verdict -- reported under
    both names, and the reader says which it used."""
    old = F.in_spec_fields({"in_spec": False})
    assert old == {"in_spec": False, "coil_drift_in_spec": False,
                   "in_spec_source": F.IN_SPEC_DRIFT_SOURCE}
    new = F.in_spec_fields({"in_spec": True, "coil_drift_in_spec": False,
                            "in_spec_source": b"inspec_rule"})
    assert new == {"in_spec": True, "coil_drift_in_spec": False,
                   "in_spec_source": F.IN_SPEC_RULE_SOURCE}
    assert F.in_spec_fields({}) == {}


def test_an_old_archive_reads_through_read_filter_flags_and_drawview(
        tmp_path, monkeypatch):
    from bouquet.archive import BouquetArchive
    diags, rej, h, G = _rule_run(tmp_path, monkeypatch, coil_ok=True,
                                 drift_ok=False)
    # strip the new fields: the shape of an archive written before them,
    # whose in_spec was the drift verdict
    with h5py.File(h + ".h5", "a") as hf:
        for i in range(len(diags)):
            g = hf[f"scan/0/{i}"] if "scan" in hf else hf[str(i)]
            del g.attrs[F.COIL_DRIFT_IN_SPEC_ATTR]
            del g.attrs[F.IN_SPEC_SOURCE_ATTR]
            g.attrs["in_spec"] = False
    fl = next(iter(F.read_filter_flags(h).values()))
    assert all(r["in_spec"] is False and r["coil_drift_in_spec"] is False
               and r["in_spec_source"] == F.IN_SPEC_DRIFT_SOURCE
               for r in fl.values())
    ar = BouquetArchive(h + ".h5")
    d0 = ar.scan(ar.scan_keys[0])[0].flags
    assert d0["in_spec_source"] == F.IN_SPEC_DRIFT_SOURCE


# --------------------------------------------------------------------------
#  _apply_inspec_rule: a config the rule cannot be resolved from is warned
#  about; anything else propagates
# --------------------------------------------------------------------------
class _NoRule:
    def __init__(self, exc):
        self._exc = exc

    def _inspec_bands(self):
        raise self._exc


@pytest.mark.parametrize("exc", [KeyError("l_i_sigma"),
                                 TypeError("bad field")])
def test_an_unresolvable_rule_is_warned_about(exc):
    from bouquet.run import Bouquet
    with pytest.warns(UserWarning, match=type(exc).__name__):
        out = Bouquet._apply_inspec_rule(_NoRule(exc), "unused", None)
    assert out == {"rule": None, "refiltered": False, "previous_rule": None}


def test_another_error_resolving_the_rule_propagates():
    from bouquet.run import Bouquet
    with pytest.raises(RuntimeError, match="boom"):
        Bouquet._apply_inspec_rule(_NoRule(RuntimeError("boom")), "unused",
                                   None)
