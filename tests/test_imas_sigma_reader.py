"""The IMAS reader takes core_profiles *_error_upper as the kinetic sigma envelopes.

Precedence per channel: sigma_profiles > an IDA .cdf > the IMAS source's own
error fields (sigma_mode "auto" / "direct") > the assumed default; the winner
is stamped in resolve_uncertainty()'s "sigma_source".  On ida_hybrid the IDA
file's envelopes win, as before.  Uses the synthetic dd fixture (its slices
carry the radial templates, tests/test_fixture_sigma.py).
"""
import gzip
import json
import os

import numpy as np
import pytest

from bouquet.baseline import resolve_uncertainty
from bouquet.config import BouquetConfig, ImasSource, SolverConfig, UncertaintyConfig
from bouquet.io.imas import clear_dd_cache, read_imas_baseline, read_imas_sigmas
from bouquet.uncertainties import sigma_template

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DD_GZ = os.path.join(DATA, "dd_synthetic.json.gz")
IDA = os.path.join(DATA, "diiid_profs_synthetic.cdf")
T = 1.2          # a ramp-template slice


def _dd():
    with gzip.open(DD_GZ, "rt") as fh:
        return json.load(fh)


def _slice(dd=None, t=T):
    dd = dd or _dd()
    cp = dd["core_profiles"]
    return cp["profiles_1d"][int(np.argmin(np.abs(np.asarray(cp["time"]) - t)))]


def _psi_N(cp1):
    psi = np.asarray(cp1["grid"]["psi"], float)
    return (psi - psi[0]) / (psi[-1] - psi[0])


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if "_error_" not in k}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _write(tmp_path, dd, name="dd.json"):
    p = str(tmp_path / name)
    with open(p, "w") as fh:
        json.dump(dd, fh)
    clear_dd_cache()
    return p


def _cfg(ids_path, unc=None, **src):
    return BouquetConfig(source=ImasSource(ids_path=ids_path, time=T, **src),
                         solver=SolverConfig(mesh_path="unused"), output_header="unused",
                         uncertainty=unc or UncertaintyConfig())


@pytest.fixture(scope="module")
def DD(tmp_path_factory):
    """The dd fixture, unzipped (the reader takes plain JSON, as the solver tests do)."""
    p = str(tmp_path_factory.mktemp("dd") / "dd_synthetic.json")
    with gzip.open(DD_GZ, "rb") as fi, open(p, "wb") as fo:
        fo.write(fi.read())
    clear_dd_cache()
    return p


# ---------------------------------------------------------------------------
#  read_imas_sigmas
# ---------------------------------------------------------------------------
def test_reads_the_five_channels_of_a_slice():
    cp1 = _slice()
    sig, rec = read_imas_sigmas(cp1)
    assert set(sig) == {"ne", "te", "ni", "ti", "zeff"}
    x = _psi_N(cp1)
    main = cp1["ion"][0]
    assert float(main["element"][0]["z_n"]) == 1.0
    for ch, prof, q in (("ne", cp1["electrons"]["density_thermal"], "ne"),
                        ("te", cp1["electrons"]["temperature"], "te"),
                        ("ni", main["density_thermal"], "ni"),
                        ("ti", main["temperature"], "ti"), ("zeff", cp1["zeff"], "zeff")):
        np.testing.assert_allclose(sig[ch], sigma_template(x, q, "ramp") * np.abs(prof),
                                   rtol=1e-5)
        assert rec[ch]["used"] and rec[ch]["how"] == "upper"
    assert rec["ne"]["field"] == "electrons.density_thermal_error_upper"
    assert rec["ti"]["field"] == "ion[main].temperature_error_upper"
    assert rec["zeff"]["field"] == "zeff_error_upper"


def test_symmetrisation_fallbacks_and_refusals():
    cp1 = _slice()
    el = cp1["electrons"]
    up = np.asarray(el["temperature_error_upper"])
    el["temperature_error_lower"] = (0.5 * up).tolist()         # both -> the mean
    main = cp1["ion"][0]
    main["temperature_error_lower"] = main.pop("temperature_error_upper")   # lower only
    main["density_thermal_error_upper"] = [-1.0] * len(main["density_thermal"])
    cp1["zeff_error_upper"] = cp1["zeff_error_upper"][:-1]     # wrong length
    el["density"] = el["density_thermal"]                      # 'density' only as fallback
    el["density_error_upper"] = [1.0] * len(el["density"])
    sig, rec = read_imas_sigmas(cp1)
    np.testing.assert_allclose(sig["te"], 0.75 * up, rtol=1e-12)
    assert rec["te"]["how"] == "mean of upper and lower"
    assert rec["ti"]["how"] == "lower" and "ti" in sig
    assert "ni" not in sig and "negative" in rec["ni"]["reason"]
    assert "zeff" not in sig and "wrong length" in rec["zeff"]["reason"]
    assert rec["ne"]["field"] == "electrons.density_thermal_error_upper"
    del el["density_thermal_error_upper"]
    sig, rec = read_imas_sigmas(cp1)
    assert rec["ne"]["field"] == "electrons.density_error_upper"
    np.testing.assert_array_equal(sig["ne"], 1.0)
    sig, rec = read_imas_sigmas(_strip(_slice()))
    assert sig == {} and all(not r["used"] and r["reason"].startswith("absent")
                             for r in rec.values())


# ---------------------------------------------------------------------------
#  through read_imas_baseline + resolve_uncertainty
# ---------------------------------------------------------------------------
def test_an_imas_source_with_error_fields_uses_them(DD):
    clear_dd_cache()
    bl = read_imas_baseline(ImasSource(ids_path=DD, time=T))
    cp1 = _slice()
    for ch in ("ne", "te", "ni", "ti", "zeff"):
        assert np.shape(bl.aux[f"sigma_{ch}_imas"]) == bl.psi_N_kinetic.shape
    env = resolve_uncertainty(_cfg(DD), bl)
    np.testing.assert_array_equal(env["sigma_ne"], cp1["electrons"]["density_thermal_error_upper"])
    np.testing.assert_array_equal(env["sigma_ti"], cp1["ion"][0]["temperature_error_upper"])
    src = env["sigma_source"]
    for ch in ("ne", "te", "ni", "ti"):
        assert src[ch]["kind"] == "imas", src
        assert "IMAS core_profiles" in src[ch]["source"]
    assert src["zeff"]["kind"] == "imas" and src["zeff"]["tier"] == "IMAS-measured"
    assert env["zeff_sigma_tier"]["tier"] == "IMAS-measured"
    np.testing.assert_array_equal(env["aux_sigmas"]["zeff"], cp1["zeff_error_upper"])


def test_without_error_fields_the_assumed_defaults_apply(tmp_path):
    p = _write(tmp_path, _strip(_dd()))
    bl = read_imas_baseline(ImasSource(ids_path=p, time=T))
    assert not any(k.endswith("_imas") for k in bl.aux)
    env = resolve_uncertainty(_cfg(p), bl)
    for ch in ("ne", "te", "ni", "ti", "zeff"):
        assert env["sigma_source"][ch]["kind"] == "assumed", ch
    assert env["zeff_sigma_tier"]["tier"] == "scalar"


def test_ensemble_sigma_mode_does_not_read_the_error_fields(DD):
    clear_dd_cache()
    bl = read_imas_baseline(ImasSource(ids_path=DD, time=T))
    env = resolve_uncertainty(_cfg(DD, UncertaintyConfig(sigma_mode="ensemble")), bl)
    assert {env["sigma_source"][c]["kind"] for c in ("ne", "te", "ni", "ti", "zeff")} == {"assumed"}


def test_an_ida_sigma_file_still_wins_the_kinetic_channels(DD):
    clear_dd_cache()
    bl = read_imas_baseline(ImasSource(ids_path=DD, time=T))
    with pytest.warns(UserWarning, match="IMAS-MEASURED tier"):
        env = resolve_uncertainty(_cfg(DD, UncertaintyConfig(ida_path=IDA), impurity_Z=6.0), bl)
    for ch in ("ne", "te", "ni", "ti"):
        assert env["sigma_source"][ch]["kind"] == "ida", ch
    # the dd route's Z_eff baseline is the dd's: the IDA tiers are ineligible,
    # and the dd's own Z_eff envelope is the one that pairs with it
    assert env["zeff_sigma_tier"]["tier"] == "IMAS-measured"


def test_ida_hybrid_is_unchanged_the_ida_envelopes_win(DD):
    clear_dd_cache()
    src = dict(ida_path=IDA, impurity_Z=6.0)
    bl = read_imas_baseline(ImasSource(ids_path=DD, time=T, **src), kinetic_source="ida_hybrid")
    assert not any(k.endswith("_imas") for k in bl.aux) and "imas_sigma_fields" not in bl.aux
    cfg = _cfg(DD, UncertaintyConfig(ida_path=IDA), **src)
    cfg.generation.kinetic_source = "ida_hybrid"
    env = resolve_uncertainty(cfg, bl)
    for ch in ("ne", "te", "ni", "ti"):
        assert env["sigma_source"][ch]["kind"] == "ida", ch
    assert env["zeff_sigma_tier"]["tier"] != "IMAS-measured"
    # bit-identical to the same read of the dd without any error field
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dd.json")
        with open(p, "w") as fh:
            json.dump(_strip(_dd()), fh)
        clear_dd_cache()
        bl0 = read_imas_baseline(ImasSource(ids_path=p, time=T, **src),
                                 kinetic_source="ida_hybrid")
        cfg0 = _cfg(p, UncertaintyConfig(ida_path=IDA), **src)
        cfg0.generation.kinetic_source = "ida_hybrid"
        env0 = resolve_uncertainty(cfg0, bl0)
    for k in ("sigma_ne", "sigma_te", "sigma_ni", "sigma_ti", "sigma_jphi"):
        np.testing.assert_array_equal(env[k], env0[k])
    np.testing.assert_array_equal(env["aux_sigmas"]["zeff"], env0["aux_sigmas"]["zeff"])


def test_a_set_scalar_loses_to_the_imas_envelope_loudly(DD):
    clear_dd_cache()
    bl = read_imas_baseline(ImasSource(ids_path=DD, time=T))
    with pytest.warns(UserWarning, match="te_scalar_sigma=0.2 was set but IGNORED"):
        env = resolve_uncertainty(_cfg(DD, UncertaintyConfig(te_scalar_sigma=0.2)), bl)
    assert env["sigma_source"]["te"]["kind"] == "imas"
