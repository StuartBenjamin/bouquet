"""zeff_from_fuse=True on a measured ida_hybrid route: the bootstrap gets the
dd's TOTAL Z_eff (owner decision 2026-10-18, the Z_eff definition rule).

The base handed to the Sauter / Redl coefficients is the dd's total Z_eff
(the validated ``zeff`` dataset, else the all-ion sum over n_e), with
``zeff_includes_fast`` following the beam; the species (n_i from the IDA
Z_eff with the beam subtracted, Z_imp = impurity_Z) are those of the same
route without the option, bit for bit; the draw is centred on the base and
the IDA envelope is carried absolute, recorded as its own pairing.
"""

import numpy as np
import pytest

import test_ni_fast_subtraction as NF
import test_zeff_hybrid_standard_total as HS
from bouquet.baseline import (ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL,
                              ZEFF_PAIRING_HYBRID_FROM_FUSE_TOTAL,
                              hybrid_zeff_from_fuse_total, resolve_uncertainty,
                              zeff_sigma_eligibility)

SPECIES = ("ne", "te", "ti", "ni", "p_fast", "z_fast", "z2_fast")


@pytest.mark.parametrize("stored", [None, "all"])
@pytest.mark.parametrize("route", ["Zeff", "CER", "all"])
def test_the_base_is_the_total_and_the_species_are_untouched(
        tmp_path, route, stored):
    ddp, cdf, *_ = NF._build(tmp_path, dd_mutate=NF._store_zeff(stored))
    bl = HS._read(ddp, cdf, ni_source=route, zeff_from_fuse=True)
    ref = HS._read(ddp, cdf, ni_source=route)
    ne_dd, zth = HS._thermal_dd(ddp)
    total = zth + bl.z2_fast / ne_dd
    np.testing.assert_allclose(bl.Zeff, total, rtol=1e-12)
    assert bl.Zeff[0] > zth[0] + 0.1                       # not the thermal
    assert bl.zeff_includes_fast is True
    np.testing.assert_array_equal(bl.aux["zeff"], bl.Zeff)  # draw centre
    d = bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]
    assert d["definition"] == "total"
    assert d["source"] == ("dataset" if stored == "all" else "from_densities")
    assert d["draw_centred_on_base"] is True
    rec = bl.li_metrics["zeff_dd_provenance"]
    assert rec["convention"] == "thermal+fast"
    assert "zeff_from_fuse=True" in rec["basis"]
    assert rec["baseline_zeff_from"] == "dd"
    assert bl.li_metrics["zeff_source"] == "dd"
    # the species: exactly those of the route without the option
    for k in SPECIES:
        np.testing.assert_array_equal(getattr(bl, k), getattr(ref, k))
    assert bl.Z_imp == ref.Z_imp == NF.Z_IMP
    for k in ("sigma_ne_ida", "sigma_te_ida", "sigma_ni_ida", "sigma_ti_ida"):
        np.testing.assert_array_equal(bl.aux[k], ref.aux[k])
    assert bl.aux["ni_fast_meta"] == ref.aux["ni_fast_meta"]


def test_without_a_beam_the_numerators_coincide(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path, fast_frac=0.0)
    bl = HS._read(ddp, cdf, ni_source="all", zeff_from_fuse=True)
    _, zth = HS._thermal_dd(ddp)
    assert bl.zeff_includes_fast is False and bl.z2_fast is None
    np.testing.assert_array_equal(bl.Zeff, zth)


def test_the_envelope_is_carried_absolute_and_the_pairing_recorded(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    src = HS._src(ddp, cdf, ni_source="all", zeff_from_fuse=True)
    bl = HS._read(ddp, cdf, ni_source="all", zeff_from_fuse=True)
    ref = HS._read(ddp, cdf, ni_source="all")
    assert hybrid_zeff_from_fuse_total(src, "ida_hybrid", baseline=bl)
    ok, why, pair = zeff_sigma_eligibility(src, cdf, "ida_hybrid",
                                           baseline=bl, with_pairing=True)
    assert ok and why == ""
    assert pair["reason"] == ZEFF_PAIRING_HYBRID_FROM_FUSE_TOTAL
    env = resolve_uncertainty(HS._cfg(src, cdf), bl)
    env_ref = resolve_uncertainty(
        HS._cfg(HS._src(ddp, cdf, ni_source="all"), cdf), ref)
    np.testing.assert_array_equal(env["aux_sigmas"]["zeff"],
                                  env_ref["aux_sigmas"]["zeff"])
    m = env["zeff_sigma_tier"]
    assert m["tier"] == env_ref["zeff_sigma_tier"]["tier"]
    assert m["label"] == (env_ref["zeff_sigma_tier"]["label"] + " "
                          + ZEFF_PAIRING_HYBRID_DD_TOTAL_LABEL)
    p = m["pairing"]
    assert p["reason"] == ZEFF_PAIRING_HYBRID_FROM_FUSE_TOTAL
    assert p["measured"] is True
    assert p["base_definition"]["definition"] == "total"
    # zeff_from_fuse never pairs n_i with the Z_eff draw: its own sigma_ni
    assert env["ni_from_zeff"] is False
    # at zero width the drawn Z_eff is the base exactly
    zero = dict(env, aux_sigmas=dict(env["aux_sigmas"],
                                     zeff=0.0 * env["aux_sigmas"]["zeff"]),
                sigma_ne=0.0 * env["sigma_ne"], sigma_te=0.0 * env["sigma_te"],
                sigma_ni=0.0 * env["sigma_ni"], sigma_ti=0.0 * env["sigma_ti"])
    d0 = HS._draws(bl, zero, n=1)[0]
    np.testing.assert_array_equal(d0.aux["zeff"], bl.Zeff)


def test_the_classification_keeps_its_own_rule(tmp_path):
    """zeff_fast_ions=True (EXPERIMENTAL) with zeff_from_fuse=True: the
    classified dd Z_eff, no zeff_from_fuse pairing."""
    ddp, cdf, *_ = NF._build(tmp_path, dd_mutate=NF._store_zeff("thermal"))
    src = HS._src(ddp, cdf, ni_source="all", zeff_from_fuse=True,
                  zeff_fast_ions=True)
    bl = HS._read(ddp, cdf, ni_source="all", zeff_from_fuse=True,
                  zeff_fast_ions=True)
    _, zth = HS._thermal_dd(ddp)
    assert bl.zeff_includes_fast is False
    np.testing.assert_allclose(bl.Zeff, zth, rtol=1e-12)
    assert not hybrid_zeff_from_fuse_total(src, "ida_hybrid", baseline=bl)
    assert zeff_sigma_eligibility(src, cdf, "ida_hybrid", baseline=bl,
                                  with_pairing=True)[2] is None


def test_a_thermal_base_baseline_is_not_the_case(tmp_path):
    """A baseline recording a thermal-numerator base (read before the
    decision) is not paired as a total base."""
    ddp, cdf, *_ = NF._build(tmp_path)
    src = HS._src(ddp, cdf, ni_source="all", zeff_from_fuse=True)
    bl = HS._read(ddp, cdf, ni_source="all", zeff_from_fuse=True)
    bl.li_metrics["zeff_dd_provenance"]["zeff_definition"] = dict(
        bl.li_metrics["zeff_dd_provenance"]["zeff_definition"],
        definition="thermal-numerator")
    assert not hybrid_zeff_from_fuse_total(src, "ida_hybrid", baseline=bl)
    assert zeff_sigma_eligibility(src, cdf, "ida_hybrid", baseline=bl,
                                  with_pairing=True) == (True, "", None)
