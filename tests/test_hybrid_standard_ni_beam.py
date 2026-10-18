"""ida_hybrid ni_source="standard": the base n_i carries the beam's charge
(owner decision 2026-10-18).

Quasineutrality with a beam is n_e = n_i + Z n_z + z_fast and the thermal
numerator is Z_eff,th n_e = n_i + Z^2 n_z, so
n_i = [n_e (Z - Z_eff,th) - Z z_fast]/(Z - 1); without a beam it is the old
n_e (Z - Z_eff,th)/(Z - 1).  The fixture's dd and IDA file describe the same
carbon plasma, so the corrected n_i is the dd's own thermal deuterons and the
implied impurity charge is carbon's.
"""

import warnings

import numpy as np
import pytest

import test_ni_fast_subtraction as NF
import test_zeff_hybrid_standard_total as HS
from bouquet.baseline import resolve_uncertainty
from bouquet.io.imas import _merge_ida_kinetics
from bouquet.kinetic_sampler import CLIP_COUNTERS
from bouquet.physics import main_ion_density_from_zeff

Z = NF.Z_IMP


def test_with_a_beam_n_i_is_the_thermal_deuterons_and_z_imp_is_carbon(
        tmp_path):
    ddp, cdf, _, _, ni_th = NF._build(tmp_path)
    bl = HS._read(ddp, cdf, ni_source="standard")
    _, zth = HS._thermal_dd(ddp)
    assert np.any(bl.z_fast)
    np.testing.assert_allclose(bl.ni, ni_th, rtol=1e-12)
    # the old no-beam form overstated it by Z z_fast/(Z - 1)
    old = main_ion_density_from_zeff(bl.ne, np.clip(zth, 1.0, Z), Z)
    assert np.all(old[:-1] > bl.ni[:-1])
    np.testing.assert_allclose(
        (old - bl.ni)[zth >= 1.0], (Z * bl.z_fast / (Z - 1.0))[zth >= 1.0],
        rtol=1e-12, atol=1e-12 * float(np.max(bl.ne)))   # rounding at z_fast=0
    assert bl.Z_imp == pytest.approx(Z, rel=1e-9)
    meta = bl.aux["ni_fast_meta"]
    assert meta["applied"] is False
    assert meta["n_floored"] == 0 and meta["n_capped"] == 0
    assert "z_fast" in meta["evidence"]


def test_without_a_beam_n_i_is_the_old_form_bit_for_bit(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path, fast_frac=0.0)
    bl = HS._read(ddp, cdf, ni_source="standard")
    _, zth = HS._thermal_dd(ddp)
    assert bl.z_fast is None
    np.testing.assert_array_equal(
        bl.ni, main_ion_density_from_zeff(bl.ne, np.clip(zth, 1.0, Z), Z))
    assert bl.Z_imp == pytest.approx(Z, rel=1e-9)
    meta = bl.aux["ni_fast_meta"]
    assert "n_floored" not in meta and "n_capped" not in meta
    assert meta["evidence"].endswith("no subtraction")


def test_the_thermal_window_is_applied_and_counted(tmp_path):
    """n_i < 0 (beam charge beyond what Z_eff,th leaves) is floored at 0,
    n_i > n_e - z_fast (n_z < 0) capped there; each counted and warned."""
    cdf = str(tmp_path / "ida.cdf")
    ne, *_ = NF._ida_cdf(cdf)
    psi = np.linspace(0.0, 1.0, NF.N)
    zf = 0.2 * ne
    zth = np.full(NF.N, 1.3)
    zf[:5], zth[:5] = 0.1 * ne[:5], 0.5           # below 0.9: n_z < 0
    zf[-5:], zth[-5:] = 0.9 * ne[-5:], 1.5        # above 6 * 0.1: n_i < 0
    with pytest.warns(UserWarning, match="floored at 0 on 5 node.*on 5"):
        out = _merge_ida_kinetics(
            psi, ne, ne, zth + 0.1, cdf, 1.0, Z, ni_source="standard",
            z_fast=zf, z2_fast=zf, zeff_for_ni=zth)
    ni, meta = out[3], out[11]
    assert meta["n_floored"] == 5 and meta["n_capped"] == 5
    np.testing.assert_array_equal(ni[-5:], 0.0)
    np.testing.assert_array_equal(ni[:5], ne[:5] - zf[:5])
    mid = slice(5, -5)
    np.testing.assert_allclose(
        ni[mid], (ne[mid] * (Z - zth[mid]) - Z * zf[mid]) / (Z - 1.0),
        rtol=1e-12)
    np.testing.assert_array_equal(out[4], zth + 0.1)   # the run's Z_eff


def test_the_zero_width_draw_needs_no_n_i_clip(tmp_path):
    """The sampler's n_i ceiling n_e - z_fast no longer binds on the base:
    at zero width the draw is the base and no clip fires."""
    ddp, cdf, *_ = NF._build(tmp_path)
    bl = HS._read(ddp, cdf, ni_source="standard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env = resolve_uncertainty(
            HS._cfg(HS._src(ddp, cdf, ni_source="standard"), cdf), bl)
    zero = dict(env, aux_sigmas=dict(env["aux_sigmas"],
                                     zeff=0.0 * env["aux_sigmas"]["zeff"]),
                sigma_ne=0.0 * env["sigma_ne"], sigma_te=0.0 * env["sigma_te"],
                sigma_ni=0.0 * env["sigma_ni"], sigma_ti=0.0 * env["sigma_ti"])
    assert env["kinetic_clips"]
    d0 = HS._draws(bl, zero, n=1)[0]
    assert set(d0.clips) == set(CLIP_COUNTERS)
    assert not any(d0.clips.values()), d0.clips
    np.testing.assert_allclose(d0.ni, bl.ni, rtol=1e-12)
