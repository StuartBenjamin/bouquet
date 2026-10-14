"""The IDA-hybrid default: ONE thermal-kinetics source per run (owner
decision, 2026-10).

When an IDA file supplies the kinetics (``kinetic_source="ida_hybrid"``),
the IDA file's own Z_eff is the default: ``ImasSource.ni_source="auto"``
resolves to the measured route (``"all"``: the VB+CER mean where both routes
have an envelope, else whichever does), with n_i from it.  The dd's Z_eff is
used only when asked for (``ni_source="standard"`` or
``zeff_from_fuse=True``).  An IDA file with no Z_eff information at all (no
``Zeff``, no ``n_12C6``) is REFUSED on that default.  The dd's fast-ion
pressure is the one declared cross-source exception (``include_pfast=True``),
and with it the beam density is subtracted from the IDA (total) n_i
(``ni_subtract_fast=None`` follows ``include_pfast`` on this route), so the
fast ions are counted once.  Every baseline stamps which Z_eff it used
(``li_metrics["zeff_source"]``: ``"ida"`` / ``"dd"``) and the resolved route
(``li_metrics["kinetics_route"]``), warning-free.

Synthetic inputs only; no device data.
"""
import warnings

import h5py
import numpy as np
import pytest

import test_ni_fast_subtraction as NF
from bouquet import experimental as X
from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                            SolverConfig, UncertaintyConfig)
from bouquet.io.imas import read_imas_baseline


def _src(ddp, cdf, **kw):
    return ImasSource(ids_path=ddp, time=1.0, ida_path=cdf,
                      impurity_Z=NF.Z_IMP, **kw)


def _read(ddp, cdf, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        bl = read_imas_baseline(_src(ddp, cdf, **kw),
                                kinetic_source="ida_hybrid")
    return bl, rec


def _strip(cdf, *names):
    with h5py.File(cdf, "a") as f:
        for n in names:
            del f[n]


# ---------------------------------------------------------------------------
#  the setting and its resolution
# ---------------------------------------------------------------------------
def test_the_default_is_auto_and_resolves_per_run():
    s = ImasSource(ids_path="dd.json", ida_path="i.cdf")
    assert s.ni_source == X.AUTO_NI_SOURCE == "auto"
    assert s.ni_subtract_fast is None and s.include_pfast is True
    assert X.resolved_ni_source(s, "ida_hybrid") == X.MEASURED_NI_SOURCE \
        == "all"
    # FUSE kinetics: the dd's own Z_eff, as before
    assert X.resolved_ni_source(s, "fuse") == "standard"
    assert X.resolved_ni_source(ImasSource(ids_path="dd.json"),
                                "ida_hybrid") == "standard"
    # an explicit value is honoured as given
    for v in ("standard", "Zeff", "CER", "all"):
        assert X.resolved_ni_source(
            ImasSource(ids_path="dd.json", ida_path="i.cdf", ni_source=v),
            "ida_hybrid") == v
    # the fast-ion options follow the #79 rule on the measured route
    assert X.resolved_ni_subtract_fast(s, "ida_hybrid") is True
    assert X.resolved_ni_subtract_fast(
        ImasSource(ids_path="dd.json", ida_path="i.cdf", include_pfast=False),
        "ida_hybrid") is False
    assert X.resolved_ni_subtract_fast(s, "fuse") is False
    assert X.resolved_ni_subtract_fast(
        ImasSource(ids_path="dd.json", ida_path="i.cdf",
                   ni_source="standard"), "ida_hybrid") is False


def test_the_measured_route_is_no_longer_experimental(tmp_path):
    for k in ("ida_ion_route", "ida_ni_beam_subtraction"):
        assert k not in X.REGISTRY
        assert X.GRADUATED[k]["known_limitations"]
    cfg = BouquetConfig(
        source=_src(str(tmp_path / "dd.json"), str(tmp_path / "i.cdf")),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out"),
        generation=GenerationConfig(kinetic_source="ida_hybrid"),
        uncertainty=UncertaintyConfig())
    # the default IDA-hybrid run enables nothing experimental (the sampler
    # clips stay opt-in: no clipping beyond the route's own counted clamps)
    assert X.experimental_features_enabled(cfg) == []
    assert X.kinetic_clips_on(cfg) is False


# ---------------------------------------------------------------------------
#  read_imas_baseline end to end
# ---------------------------------------------------------------------------
def test_the_default_takes_the_ida_zeff_and_counts_the_fast_ions_once(
        tmp_path):
    ddp, cdf, ni_total, ni_fast, ni_th = NF._build(tmp_path)
    bl, rec = _read(ddp, cdf)
    ida = bl.aux["ida_profiles"][1]
    assert ida.zeff_provenance["ni_source"] == "all"
    assert ida.zeff_provenance["source"] == "VB+CER mean"
    np.testing.assert_allclose(bl.Zeff, ida.Zeff, rtol=1e-12)
    assert bl.zeff_includes_fast is True             # a measured Z_eff
    # the beam density is subtracted (fast ions in p_fast, not twice)
    assert bl.aux["ni_fast_meta"]["applied"] is True
    np.testing.assert_allclose(bl.ni, ni_th, rtol=2e-6)
    assert np.any(bl.p_fast)
    assert not [w for w in rec if "counted twice" in str(w.message)
                or "include_pfast" in str(w.message)]
    # the warning-free stamps
    assert bl.li_metrics["zeff_source"] == "ida"
    kr = bl.li_metrics["kinetics_route"]
    assert kr["ni_source"] == "all" and kr["ni_source_setting"] == "auto"
    assert kr["zeff_source"] == "ida"
    assert kr["ni_subtract_fast"] is True and kr["include_pfast"] is True
    assert "p_fast" in kr["cross_source"]
    assert bl.li_metrics["zeff_dd_provenance"]["baseline_zeff_from"] \
        .startswith("IDA")


def test_the_dd_zeff_is_an_explicit_choice(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    bl, _ = _read(ddp, cdf, ni_source="standard")
    assert bl.li_metrics["zeff_source"] == "dd"
    assert bl.li_metrics["kinetics_route"]["ni_source"] == "standard"
    assert bl.aux["ni_fast_meta"]["applied"] is False
    assert bl.zeff_includes_fast is False
    bl2, _ = _read(ddp, cdf, zeff_from_fuse=True)
    assert bl2.li_metrics["zeff_source"] == "dd"
    np.testing.assert_allclose(bl2.Zeff, bl.Zeff, rtol=1e-12)


def test_the_two_defaults_differ_on_a_slice_whose_zeffs_differ(tmp_path):
    """Numbers move: where the dd's Z_eff is not the IDA one, the default
    baseline Z_eff is the IDA one."""
    def _dd_zeff_off(dd):
        ion = dd["core_profiles"]["profiles_1d"][0]["ion"][1]
        ion["density_thermal"] = (2.0 * np.asarray(
            ion["density_thermal"])).tolist()
    ddp, cdf, *_ = NF._build(tmp_path, dd_mutate=_dd_zeff_off)
    new, _ = _read(ddp, cdf)
    old, _ = _read(ddp, cdf, ni_source="standard")
    assert not np.allclose(new.Zeff, old.Zeff, rtol=1e-3)
    np.testing.assert_allclose(new.Zeff, new.aux["ida_profiles"][1].Zeff,
                               rtol=1e-12)


def test_carbon_only_file_takes_the_cer_route_alone(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    _strip(cdf, "Zeff", "Zeff_err")
    bl, _ = _read(ddp, cdf)
    prov = bl.aux["ida_profiles"][1].zeff_provenance
    assert prov["source"] == "CER" and prov["weights"] == {"VB": 0.0,
                                                           "CER": 1.0}
    assert bl.li_metrics["zeff_source"] == "ida"


def test_a_file_with_no_zeff_information_is_refused(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    _strip(cdf, "Zeff", "Zeff_err", "n_12C6", "n_12C6_err")
    with pytest.raises(ValueError) as ei:
        _read(ddp, cdf)
    msg = str(ei.value)
    assert "no Z_eff information" in msg
    assert "ni_source='standard'" in msg and "zeff_from_fuse=True" in msg
    # the explicit choices take the dd's Z_eff and read the file
    for kw in (dict(ni_source="standard"), dict(zeff_from_fuse=True)):
        bl, _ = _read(ddp, cdf, **kw)
        assert bl.li_metrics["zeff_source"] == "dd", kw
        assert bl.li_metrics["kinetics_route"]["ni_source"] == "standard"
        assert bl.aux["ida_profiles"][1].zeff_provenance["source"] \
            .startswith("dd")
