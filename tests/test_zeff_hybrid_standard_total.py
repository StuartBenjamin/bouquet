"""The ida_hybrid ``ni_source="standard"`` sub-route under the route-independent
Z_eff rule (owner decision 2026-10-18).

1. The Z_eff handed to the bootstrap coefficients is the TOTAL Z_eff (all
   ions, fast included, over n_e) on every route: on this sub-route the dd's
   own total (the validated ``zeff`` dataset, else the all-ion sum), with
   ``zeff_includes_fast`` following the beam.
2. The species derivations (thermal n_i, Z_imp, p_imp) keep the
   thermal-numerator Z_eff.
3. The Z_eff draw perturbs the measured total Z_eff: absolute sigma, centred
   on the base of 1.  On this sub-route the sigma is the IDA file's Z_eff
   envelope (carbon-propagated > VB-measured), eligible although the
   baseline is dd-derived; the assumed template stays the loud fallback.
4. The beam fraction is held fixed (the dd's z_fast / z2_fast).

The measured IDA routes ("Zeff", "CER", "all") are unchanged: their
eligibility carries no pairing, so their envelopes, bases and draws are those
of the legacy call (asserted bit-identical below).
"""
import warnings

import h5py
import numpy as np
import pytest

import test_ni_fast_subtraction as NF
from bouquet.baseline import (ZEFF_PAIRING_HYBRID_DD_TOTAL,
                              ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL,
                              hybrid_dd_total_zeff_subroute,
                              hybrid_zeff_from_fuse_total,
                              resolve_uncertainty, resolve_zeff_envelope,
                              zeff_sigma_eligibility)
from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                            SolverConfig, UncertaintyConfig)
from bouquet.io.ida import read_ida
from bouquet.io.imas import read_imas_baseline
from bouquet.kinetic_sampler import KineticBase, sample_kinetics
from bouquet.physics import main_ion_density_from_zeff
from bouquet.sampling import make_rng

Z_IMP = NF.Z_IMP
_EC = 1.602176634e-19


def _src(ddp, cdf, **kw):
    return ImasSource(ids_path=ddp, time=1.0, ida_path=cdf,
                      impurity_Z=Z_IMP, **kw)


def _read(ddp, cdf, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_imas_baseline(_src(ddp, cdf, **kw),
                                  kinetic_source="ida_hybrid")


def _cfg(src, sigma_file, **unc):
    return BouquetConfig(
        source=src, solver=SolverConfig(mesh_path="unused"),
        output_header="unused",
        generation=GenerationConfig(kinetic_source="ida_hybrid"),
        uncertainty=UncertaintyConfig(ida_path=sigma_file, **unc))


def _strip(cdf, *names):
    with h5py.File(cdf, "a") as f:
        for n in names:
            del f[n]


def _thermal_dd(ddp):
    return NF.TestDefaultsArePre56._dd_thermal(None, ddp)


def _kbase(bl, env):
    return KineticBase(
        psi_kin=np.asarray(bl.psi_N_kinetic, dtype=float), ne=bl.ne,
        te=bl.te, ni=bl.ni, ti=bl.ti, sigma_ne=env["sigma_ne"],
        sigma_te=env["sigma_te"], sigma_ni=env["sigma_ni"],
        sigma_ti=env["sigma_ti"], n_ls=env["n_ls"], t_ls=env["t_ls"],
        aux_sigmas=env["aux_sigmas"], aux_baselines=env["aux_baselines"],
        aux_length_scales=env["aux_length_scales"], Z_imp=bl.Z_imp,
        z_fast=bl.z_fast, z2_fast=bl.z2_fast,
        zeff_includes_fast=bl.zeff_includes_fast,
        ni_from_zeff=env["ni_from_zeff"], zeff_dne=env.get("zeff_dne"),
        clips=env["kinetic_clips"])


def _draws(bl, env, n=4, seed=20261018):
    b = _kbase(bl, env)
    x = b.psi_kin

    def _thermal(d):
        return np.trapezoid(_EC * (d.ne * d.te + d.ni * d.ti), x)
    return [sample_kinetics(b, make_rng(seed + k), _thermal, _thermal(b))
            for k in range(n)]


# ---------------------------------------------------------------------------
#  rules 1 and 2: the base is the dd's total, the species the thermal numerator
# ---------------------------------------------------------------------------
def _store(kind):
    """dd_mutate: no zeff, the total as stored, or total + 0.6 (matches
    neither definition: C2 replaces it by the all-ion sum)."""
    if kind != "neither":
        return NF._store_zeff(kind)

    def mutate(dd):
        NF._store_zeff("all")(dd)
        cp = dd["core_profiles"]["profiles_1d"][0]
        cp["zeff"] = (np.asarray(cp["zeff"]) + 0.6).tolist()
    return mutate


@pytest.mark.parametrize("stored", [None, "all", "neither"])
def test_the_base_is_the_dds_total_and_the_species_the_thermal_numerator(
        tmp_path, stored):
    """No stored zeff (the all-ion sum), a stored total (used as stored), a
    stored array matching neither definition (C2: replaced by the all-ion
    sum).  The base handed to the bootstrap is the total in every case; n_i
    and Z_imp are identical whatever is stored (thermal numerator)."""
    import json
    ddp, cdf, *_ = NF._build(tmp_path, dd_mutate=_store(stored))
    bl = _read(ddp, cdf, ni_source="standard")
    ne_dd, zth = _thermal_dd(ddp)
    total = zth + bl.z2_fast / ne_dd
    d = bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]
    assert d["definition"] == "total"
    if stored == "all":
        cp = json.load(open(ddp))["core_profiles"]["profiles_1d"][0]
        np.testing.assert_array_equal(bl.Zeff, np.asarray(cp["zeff"], float))
        assert d["source"] == "dataset"
        np.testing.assert_allclose(bl.Zeff, total, rtol=1e-12)
    else:
        np.testing.assert_array_equal(bl.Zeff, total)
        assert d["source"] == "from_densities"
    assert bl.Zeff[0] > zth[0] + 0.1                          # really the total
    assert bl.zeff_includes_fast is True
    np.testing.assert_array_equal(bl.aux["zeff"], bl.Zeff)  # draw centre
    assert d["draw_centred_on_base"] is True
    assert bl.li_metrics["kinetics_route"]["ni_source"] == "standard"
    # rule 2: n_i from the THERMAL numerator on the IDA n_e with the beam's
    # charge in quasineutrality (owner-approved correction 2026-10-18; it
    # was the no-beam form), Z_imp likewise
    ida = bl.aux["ida_profiles"][1]
    np.testing.assert_array_equal(
        bl.ni, main_ion_density_from_zeff(
            np.interp(bl.psi_N_kinetic, ida.psi_N, ida.ne),
            zth, Z_IMP, z_fast=bl.z_fast, z2_fast=bl.z2_fast,
            zeff_includes_fast=False))
    (tmp_path / "ref").mkdir()
    ref = _read(*NF._build(tmp_path / "ref")[:2], ni_source="standard")
    np.testing.assert_array_equal(bl.ni, ref.ni)
    assert bl.Z_imp == ref.Z_imp


def test_the_floor_at_one_is_the_right_floor(tmp_path):
    """With the total and zeff_includes_fast=True, the physical lower edge
    of the Z_eff window is 1 for the hydrogenic beam of the fixture (the
    thermal-numerator window's edge n_e,th/n_e < 1 is not reached)."""
    ddp, cdf, *_ = NF._build(tmp_path)
    bl = _read(ddp, cdf, ni_source="standard")
    env = resolve_uncertainty(_cfg(_src(ddp, cdf, ni_source="standard"),
                                   cdf), bl)
    lo, _ = _kbase(bl, env).zeff_bounds_raw(bl.ne, bl.Z_imp)
    np.testing.assert_allclose(lo, 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
#  rule 3: the envelope is the IDA file's measured one, stamped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested, tier, field", [
    ("auto", "carbon-propagated", "sigma_Zeff_carbon"),
    ("measured", "VB-measured", "sigma_Zeff")])
def test_the_envelope_is_the_ida_tier_on_the_total_base(
        tmp_path, requested, tier, field):
    ddp, cdf, *_ = NF._build(tmp_path)
    bl = _read(ddp, cdf, ni_source="standard")
    src = _src(ddp, cdf, ni_source="standard")
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # an available tier: no fallback
        env = resolve_uncertainty(_cfg(src, cdf,
                                       zeff_sigma_source=requested), bl)
    ida = read_ida(cdf, time=1.0, ni_source="standard", impurity_Z=Z_IMP)
    expect = np.interp(bl.psi_N_kinetic, ida.psi_N, getattr(ida, field))
    np.testing.assert_array_equal(env["aux_sigmas"]["zeff"], expect)
    np.testing.assert_array_equal(env["aux_baselines"]["zeff"], bl.Zeff)
    m = env["zeff_sigma_tier"]
    assert m["tier"] == tier and m["eligible"] is True
    assert m["ineligible_reason"] == "" and m["skipped"] == []
    assert m["label"].endswith(ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL)
    assert env["sigma_source"]["zeff"]["kind"] == "ida"
    p = m["pairing"]
    assert p["reason"] == ZEFF_PAIRING_HYBRID_DD_TOTAL
    assert p["case"] == "ida_hybrid ni_source='standard'"
    assert p["base_definition"]["definition"] == "total"
    assert p["base_definition"]["source"] == "from_densities"
    assert p["draw_centre"].startswith("the base")
    assert "conditional at fixed beam fraction" in p["conditional"]
    assert p["width_source"] == m["label"] and p["measured"] is True
    assert p["tier"] == tier


def test_no_zeff_envelope_in_the_file_falls_back_loudly(tmp_path):
    """The assumed template stays the fallback, warned and recorded."""
    ddp, cdf, *_ = NF._build(tmp_path)
    _strip(cdf, "Zeff_err", "n_12C6_err")
    bl = _read(ddp, cdf, ni_source="standard")
    with pytest.warns(UserWarning, match="SCALAR tier"):
        env = resolve_uncertainty(
            _cfg(_src(ddp, cdf, ni_source="standard"), cdf), bl)
    m = env["zeff_sigma_tier"]
    assert m["tier"] == "scalar" and m["warned"] is True
    assert "FALLBACK" in m["label"]
    assert not m["label"].endswith(ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL)
    assert m["pairing"]["measured"] is False
    assert {s["tier"] for s in m["skipped"]} == {"carbon-propagated",
                                                 "VB-measured"}
    assert all("missing dataset" in s["reason"] for s in m["skipped"])


def test_the_draws_are_centred_on_the_total_with_the_ida_width(tmp_path):
    """Fixed-seed draws on the sub-route: the Z_eff channel moves around the
    total base; at zero width the drawn Z_eff is the base exactly.  (n_i:
    tests/test_hybrid_standard_ni_beam.py.)"""
    ddp, cdf, *_ = NF._build(tmp_path)
    bl = _read(ddp, cdf, ni_source="standard")
    env = resolve_uncertainty(_cfg(_src(ddp, cdf, ni_source="standard"),
                                   cdf), bl)
    ds = _draws(bl, env, n=3)
    assert all(not np.array_equal(d.aux["zeff"], bl.Zeff) for d in ds)
    zero = dict(env, aux_sigmas=dict(env["aux_sigmas"],
                                     zeff=0.0 * env["aux_sigmas"]["zeff"]),
                sigma_ne=0.0 * env["sigma_ne"], sigma_te=0.0 * env["sigma_te"],
                sigma_ni=0.0 * env["sigma_ni"], sigma_ti=0.0 * env["sigma_ti"])
    d0 = _draws(bl, zero, n=1)[0]
    np.testing.assert_array_equal(d0.aux["zeff"], bl.Zeff)


# ---------------------------------------------------------------------------
#  eligibility: the sub-route's own case; unrelated files still refused
# ---------------------------------------------------------------------------
def test_the_subroute_is_an_explicit_allowed_case(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    src = _src(ddp, cdf, ni_source="standard")
    assert hybrid_dd_total_zeff_subroute(src, "ida_hybrid") is True
    ok, why, pair = zeff_sigma_eligibility(src, cdf, "ida_hybrid",
                                           with_pairing=True)
    assert ok and why == ""
    assert pair["reason"] == ZEFF_PAIRING_HYBRID_DD_TOTAL
    # the legacy two-tuple is unchanged for every caller
    assert zeff_sigma_eligibility(src, cdf, "ida_hybrid") == (True, "")


def test_an_unrelated_file_is_still_refused_on_the_subroute(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    (tmp_path / "other").mkdir()
    other = str(tmp_path / "other" / "other_vintage.cdf")
    NF._ida_cdf(other)
    src = _src(ddp, cdf, ni_source="standard")
    ok, why, pair = zeff_sigma_eligibility(src, other, "ida_hybrid",
                                           with_pairing=True)
    assert not ok and "genuinely different file" in why and pair is None
    bl = _read(ddp, cdf, ni_source="standard")
    with warnings.catch_warnings():
        # the separate sigma file's own slice-match notice is not the point
        warnings.simplefilter("ignore", UserWarning)
        env = resolve_uncertainty(_cfg(src, other), bl)
    m = env["zeff_sigma_tier"]
    assert m["tier"] == "scalar" and m["eligible"] is False
    assert "genuinely different file" in m["ineligible_reason"]
    assert "pairing" not in m


@pytest.mark.parametrize("case", ["fuse kinetics", "classification"])
def test_the_dd_routes_without_ida_kinetics_are_still_refused(tmp_path, case):
    """The plain dd route (FUSE kinetics) with an IDA sigma file, and the
    EXPERIMENTAL classification on ida_hybrid: refused as before."""
    ddp, cdf, *_ = NF._build(tmp_path)
    if case == "fuse kinetics":
        src = ImasSource(ids_path=ddp, time=1.0, impurity_Z=Z_IMP)
        ks = "fuse"
    else:
        src = _src(ddp, cdf, ni_source="standard", zeff_fast_ions=True)
        ks = "ida_hybrid"
    assert hybrid_dd_total_zeff_subroute(src, ks) is False
    ok, why = zeff_sigma_eligibility(src, cdf, ks)
    assert not ok and "IMAS/FUSE source" in why


def test_a_baseline_with_a_thermal_base_is_not_the_subroute(tmp_path):
    """A baseline whose recorded Z_eff definition is not the total (one read
    before the decision) does not take the measured tiers."""
    ddp, cdf, *_ = NF._build(tmp_path)
    src = _src(ddp, cdf, ni_source="standard")
    bl = _read(ddp, cdf, ni_source="standard")
    assert hybrid_dd_total_zeff_subroute(src, "ida_hybrid", baseline=bl)
    lm = dict(bl.li_metrics)
    zp = dict(lm["zeff_dd_provenance"])
    zp["zeff_definition"] = dict(zp["zeff_definition"],
                                 definition="thermal-numerator")
    lm["zeff_dd_provenance"] = zp
    bl.li_metrics = lm
    assert not hybrid_dd_total_zeff_subroute(src, "ida_hybrid", baseline=bl)
    assert not zeff_sigma_eligibility(src, cdf, "ida_hybrid",
                                      baseline=bl)[0]


# ---------------------------------------------------------------------------
#  the measured routes: bases, envelopes and draws bit-identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("route", ["Zeff", "CER", "all"])
def test_the_measured_routes_are_bit_identical(tmp_path, route):
    """Base: the IDA Z_eff, stamped (total, ida).  Eligibility: same-file,
    no pairing.  Envelope, label and meta: those of the legacy
    resolve_zeff_envelope call (no pairing argument), bit for bit, with no
    "pairing" key.  Draws: fixed-seed draws on that legacy envelope are the
    run's draws, bit for bit.  zeff_from_fuse=False set explicitly (the
    default) is the same read bit for bit, with the measured route's own
    record and no zeff_from_fuse pairing (Fix 2 of 2026-10-18 moves only
    zeff_from_fuse=True)."""
    ddp, cdf, *_ = NF._build(tmp_path)
    bl = _read(ddp, cdf, ni_source=route)
    src = _src(ddp, cdf, ni_source=route)
    ida = bl.aux["ida_profiles"][1]
    (tmp_path / "explicit").mkdir()
    ddx, cdx, *_ = NF._build(tmp_path / "explicit")
    blx = _read(ddx, cdx, ni_source=route, zeff_from_fuse=False)
    for k in ("Zeff", "ne", "te", "ni", "ti", "p_fast", "z_fast", "z2_fast"):
        np.testing.assert_array_equal(getattr(blx, k), getattr(bl, k))
    np.testing.assert_array_equal(blx.aux["zeff"], bl.aux["zeff"])
    assert (blx.Z_imp, blx.zeff_includes_fast) == (bl.Z_imp,
                                                    bl.zeff_includes_fast)
    for k in ("sigma_ne_ida", "sigma_te_ida", "sigma_ni_ida", "sigma_ti_ida"):
        np.testing.assert_array_equal(blx.aux[k], bl.aux[k])
    rec = bl.li_metrics["zeff_dd_provenance"]
    assert blx.li_metrics["zeff_dd_provenance"] == rec
    assert (rec["convention"], rec["source"]) == ("thermal-only", "default")
    assert rec["baseline_zeff_from"].startswith("IDA")
    assert not hybrid_zeff_from_fuse_total(src, "ida_hybrid", baseline=bl)
    np.testing.assert_array_equal(
        bl.Zeff, np.interp(bl.psi_N_kinetic, ida.psi_N, ida.Zeff))
    np.testing.assert_array_equal(bl.aux["zeff"], bl.Zeff)
    assert bl.zeff_includes_fast is True
    d = bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]
    assert (d["definition"], d["source"]) == ("total", "ida")
    assert not hybrid_dd_total_zeff_subroute(src, "ida_hybrid", baseline=bl)
    assert zeff_sigma_eligibility(src, cdf, "ida_hybrid", baseline=bl,
                                  with_pairing=True) == (True, "", None)
    cfg = _cfg(src, cdf)
    env = resolve_uncertainty(cfg, bl)
    m = env["zeff_sigma_tier"]
    assert "pairing" not in m
    assert ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL not in m["label"]
    # the legacy call, with the inputs resolve_uncertainty reads
    unc = cfg.uncertainty
    _x = np.asarray(ida.psi_N, dtype=float)

    def _k(a):
        return np.interp(bl.psi_N_kinetic, _x, np.asarray(a, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        legacy_env, legacy_label, legacy_meta = resolve_zeff_envelope(
            unc.zeff_sigma_source, unc.zeff_scalar_sigma, bl.Zeff,
            zeff_is_ida=True,
            measured_sigma=(None if ida.sigma_Zeff is None
                            else _k(ida.sigma_Zeff)),
            measured_source=str(getattr(ida, "sigma_Zeff_source", "?")),
            carbon_sigma=(None if ida.sigma_Zeff_carbon is None
                          else _k(ida.sigma_Zeff_carbon)),
            carbon_source=str(getattr(ida, "sigma_Zeff_carbon_source", "?")),
            ineligible_reason="", ida_in_play=True, ladder="resolved",
            psi_N=np.asarray(bl.psi_N_kinetic, dtype=float))
    np.testing.assert_array_equal(env["aux_sigmas"]["zeff"], legacy_env)
    assert m["label"] == legacy_label
    assert {k: v for k, v in m.items() if k != "ladder"} == legacy_meta
    legacy = dict(env, aux_sigmas=dict(env["aux_sigmas"], zeff=legacy_env))
    for a, b in zip(_draws(bl, env), _draws(bl, legacy)):
        for k in ("ne", "te", "ni", "ti"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        np.testing.assert_array_equal(a.aux["zeff"], b.aux["zeff"])
