"""The stored-Z_eff consistency check (C2) after the last pre-tag review.

* F2: each ion weighs with its CHARGE STATE (``ion[].z_ion``) when the dd
  carries it, else its ``z_n``, in the check and in the all-ion fallback.
* F3: a rejected dataset's ``zeff_error_upper`` is not the width either.
* F4: a stored zeff closer to the thermal numerator says so in every draw.
* F6: the check covers every node with n_e > 0 and finite inputs.

Fast: the plain dd reader on the synthetic beam dd of
``test_ni_fast_subtraction``; no solve.
"""

import json
import warnings

import numpy as np
import pytest

import test_ni_fast_subtraction as NF
from bouquet.config import ImasSource
from bouquet.io.imas import ZEFF_DATASET_RTOL, read_imas_baseline

Z = NF.Z_IMP
Z_N_AR, Z_ION_AR = 18.0, 16.0          # argon, not fully stripped


def _total(cp, charge):
    """sum_s q_s^2 (n_s^th + n_s^fast) / n_e with q_s = charge(ion)."""
    ne = np.asarray(cp["electrons"]["density_thermal"], dtype=float)
    num = np.zeros_like(ne)
    for ion in cp["ion"]:
        q = charge(ion)
        n = np.asarray(ion["density_thermal"], dtype=float)
        n = n + np.asarray(ion.get("density_fast", 0.0 * n), dtype=float)
        num += q * q * n
    return num / ne


def _z_n(ion):
    return float(ion["element"][0]["z_n"])


def _z_ion(ion):
    return float(ion.get("z_ion", ion["element"][0]["z_n"]))


def _with_argon(store=None, scale=1.0):
    """dd_mutate: add a 0.2 % argon seed at charge state 16 (z_ion) and
    store zeff = scale x the IMAS total (charge-state weighted)."""
    def mutate(dd):
        cp = dd["core_profiles"]["profiles_1d"][0]
        ne = np.asarray(cp["electrons"]["density_thermal"], dtype=float)
        cp["ion"].append({
            "density_thermal": (2e-3 * ne).tolist(),
            "temperature": cp["ion"][0]["temperature"], "label": "Ar",
            "z_ion": Z_ION_AR, "element": [{"z_n": Z_N_AR, "a": 40.0}]})
        if store == "imas":
            cp["zeff"] = (scale * _total(cp, _z_ion)).tolist()
    return mutate


def _read(ddp, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(
            ImasSource(ids_path=ddp, time=1.0, impurity_Z=Z, **kw),
            kinetic_source="fuse")
    return bl, [str(x.message) for x in w]


def _cp(ddp):
    return json.load(open(ddp))["core_profiles"]["profiles_1d"][0]


# --------------------------------------------------------------------- F2
def test_a_correct_total_with_a_partially_ionised_impurity_is_accepted(
        tmp_path):
    ddp, *_ = NF._build(tmp_path, dd_mutate=_with_argon("imas"))
    cp = _cp(ddp)
    stored = np.asarray(cp["zeff"], dtype=float)
    # the z_n-weighted sum is beyond the 5 % tolerance (what C2 compared)
    z_n_sum = _total(cp, _z_n)
    assert np.max(np.abs(stored - z_n_sum) / stored) > ZEFF_DATASET_RTOL
    bl, msgs = _read(ddp)
    assert not [m for m in msgs if "matches neither" in m]
    d = bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]
    assert d["source"] == "dataset" and d["matches"] == "total"
    assert d["d_all"] < 1e-12
    np.testing.assert_array_equal(bl.Zeff, stored)


def test_the_fallback_total_is_charge_state_weighted(tmp_path):
    ddp, *_ = NF._build(tmp_path, dd_mutate=_with_argon("imas", scale=2.0))
    bl, msgs = _read(ddp)
    assert [m for m in msgs if "matches neither" in m]
    d = bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]
    assert d["reason"] == "dataset inconsistent"
    np.testing.assert_allclose(bl.Zeff, _total(_cp(ddp), _z_ion),
                               rtol=1e-12)


def test_the_species_derivations_keep_z_n(tmp_path):
    """n_i and Z_imp are untouched by z_ion (they assume z_n, as before)."""
    a, *_ = NF._build(tmp_path, dd_mutate=_with_argon())
    (tmp_path / "b").mkdir()

    def _no_z_ion(dd):
        _with_argon()(dd)
        del dd["core_profiles"]["profiles_1d"][0]["ion"][-1]["z_ion"]
    b, *_ = NF._build(tmp_path / "b", dd_mutate=_no_z_ion)
    bla, _ = _read(a)
    blb, _ = _read(b)
    np.testing.assert_array_equal(bla.ni, blb.ni)
    assert bla.Z_imp == blb.Z_imp
    # the bootstrap's total differs exactly by the charge-state weighting
    ne = np.asarray(bla.ne)
    nar = 2e-3 * ne
    np.testing.assert_allclose(
        blb.Zeff - bla.Zeff, (Z_N_AR ** 2 - Z_ION_AR ** 2) * nar / ne,
        rtol=1e-9)


@pytest.mark.parametrize("store", [None, "total", "stale"])
def test_the_carbon_fixtures_are_bit_identical(tmp_path, store):
    """No z_ion (every shipped dd), or a fully stripped z_ion = z_n: the
    check and the fallback are the z_n sums, bit for bit."""
    def mutate(dd, z_ion):
        cp = dd["core_profiles"]["profiles_1d"][0]
        if z_ion:
            for ion in cp["ion"]:
                ion["z_ion"] = ion["element"][0]["z_n"]
        if store is not None:
            z = _total(cp, _z_n)
            cp["zeff"] = (z if store == "total" else 2.0 * z).tolist()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, *_ = NF._build(tmp_path / "a", dd_mutate=lambda dd: mutate(dd, False))
    b, *_ = NF._build(tmp_path / "b", dd_mutate=lambda dd: mutate(dd, True))
    bla, _ = _read(a)
    blb, _ = _read(b)
    cp = _cp(a)
    ne = np.asarray(cp["electrons"]["density_thermal"], dtype=float)
    num = np.zeros_like(ne)
    for ion in cp["ion"]:                 # the pre-fix z_n accumulation
        Zs = _z_n(ion)
        num += np.asarray(ion["density_thermal"], dtype=float) * Zs * Zs
    z2 = np.zeros_like(ne)
    for ion in cp["ion"]:
        if "density_fast" in ion:
            Zs = _z_n(ion)
            z2 += Zs * Zs * np.asarray(ion["density_fast"], dtype=float)
    expect = (np.asarray(cp["zeff"], dtype=float) if store == "total"
              else num / ne + z2 / np.clip(ne, 1e-30, None))
    for bl in (bla, blb):
        np.testing.assert_array_equal(bl.Zeff, expect)
        np.testing.assert_array_equal(bl.ni, bla.ni)
        assert bl.Z_imp == bla.Z_imp
    assert (bla.li_metrics["zeff_dd_provenance"]["zeff_definition"]
            == blb.li_metrics["zeff_dd_provenance"]["zeff_definition"])


# --------------------------------------------------------------------- F3
def _store_with_sigma(scale):
    """dd_mutate: store zeff = scale x the dd's total and a 5 % absolute
    zeff_error_upper beside it."""
    def mutate(dd):
        cp = dd["core_profiles"]["profiles_1d"][0]
        z = scale * _total(cp, _z_n)
        cp["zeff"] = z.tolist()
        cp["zeff_error_upper"] = (0.05 * z).tolist()
    return mutate


def _plain_env(bl, ddp):
    from bouquet.baseline import resolve_uncertainty
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                SolverConfig, UncertaintyConfig)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=ddp, time=1.0, impurity_Z=Z),
        solver=SolverConfig(mesh_path="unused"), output_header="unused",
        generation=GenerationConfig(kinetic_source="fuse"),
        uncertainty=UncertaintyConfig())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = resolve_uncertainty(cfg, bl)
    return env, [str(x.message) for x in w]


def test_an_accepted_dataset_keeps_its_own_width(tmp_path):
    ddp, *_ = NF._build(tmp_path, dd_mutate=_store_with_sigma(1.0))
    bl, _ = _read(ddp)
    env, msgs = _plain_env(bl, ddp)
    meta = env["zeff_sigma_tier"]
    assert meta["tier"] == "IMAS-measured"
    assert "IMAS-measured" not in [s["tier"] for s in meta["skipped"]]
    assert not [m for m in msgs if "dataset rejected" in m]


def test_a_rejected_dataset_lends_no_width(tmp_path):
    """C2 rejected the stored zeff (x2): its zeff_error_upper is skipped
    with reason 'dataset rejected', the ladder falls to the assumed
    template, and the fallback warning fires (no IDA file in play)."""
    ddp, *_ = NF._build(tmp_path, dd_mutate=_store_with_sigma(2.0))
    bl, _ = _read(ddp)
    assert (bl.li_metrics["zeff_dd_provenance"]["zeff_definition"]["reason"]
            == "dataset inconsistent")
    env, msgs = _plain_env(bl, ddp)
    meta = env["zeff_sigma_tier"]
    assert meta["tier"] == "scalar" and meta["warned"] is True
    imas = [s for s in meta["skipped"] if s["tier"] == "IMAS-measured"]
    assert len(imas) == 1 and imas[0]["reason"].startswith("dataset rejected")
    assert "FALLBACK" in meta["label"]
    assert env["sigma_source"]["zeff"]["kind"] == "assumed"
    fallback = [m for m in msgs if "dataset rejected" in m]
    assert len(fallback) == 1 and "SCALAR tier" in fallback[0]
