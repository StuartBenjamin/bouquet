"""The unified engine's source adapters (:mod:`bouquet.adapters`) -- fast.

On the repository's synthetic inputs (the D3D-like example g-file + p-file
and the synthetic OMAS file; no solver):

* each adapter produces the contract (kinetics, fixed pressure, parallel
  components, boundary, rows, signs);
* the g-file's ``<j.B>`` (identity I0 on the reader's own surfaces) composes
  back to the g-file's own ``<j_phi>`` (identity I2) to rounding, and a file
  read with the wrong COCOS is refused;
* the g-file inductive is smoothed with EXACTLY the legacy
  ``fit_inductive_profile`` basis, without its amplitude search;
* the IDS inductive is ``|B0| j_ohmic`` in the positive frame, the rows are
  the source's own (Ip soft, ``li_3`` soft, q0 at the measurement radius);
* a raw-E_r MSE request is refused, E_r-corrected chords are accepted;
* identity (I2) on STORED geometry of the synthetic golden fixture (read-only
  h5py): the draw's own g-file composes back exactly, and the conversion
  factor ``F<1/R>/<B^2>`` from its surfaces agrees with TokaMaker's archived
  flux-surface averages to the level the verification report measured.

Synthetic inputs only.
"""
import os

import numpy as np
import pytest

from bouquet.adapters import (EngineInputRefused, GFileAdapter, IdsAdapter,
                              gfile_parallel_current, inductive_basis,
                              mse_rows, validate_contract)
from bouquet.engine import compose, conversion_factor

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")
_GOLD = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_TIME = 2.3043


def _gcfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=_GEQ, profiles_path=_PF),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


def _icfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    return BouquetConfig(
        source=ImasSource(ids_path=_OMAS, time=_TIME),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


@pytest.fixture(scope="module")
def gfile():
    ad = GFileAdapter(_gcfg().source, _gcfg())
    c = ad.read()
    return ad, c


@pytest.fixture(scope="module")
def ids():
    import warnings
    from bouquet.baseline import resolve_baseline
    cfg = _icfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
    ad = IdsAdapter(cfg.source, cfg, bl)
    return ad, ad.read(), bl


def _geom(parts):
    return dict(F=parts["F"], R_avg=parts["R_avg"], inv_R=parts["inv_R"],
                B2=parts["B2"], pprime=parts["pprime"])


# ---------------------------------------------------------------------------
#  g-file
# ---------------------------------------------------------------------------
def test_the_gfile_adapter_reads_the_contract(gfile):
    ad, c = gfile
    assert c.kind == "gfile" and c.jB_ind is None      # before the anchor
    n = c.psi_N.size
    for k in ("ne", "te", "ni", "ti", "zeff"):
        assert c.kinetics[k].shape == (n,) and np.all(c.kinetics[k] > 0)
    assert np.all(c.pressure >= 0) and c.pressure[0] > c.pressure[-1]
    np.testing.assert_allclose(
        c.pressure, sum(c.pressure_parts[k] for k in
                        ("thermal", "impurity", "fast")), rtol=0, atol=0)
    assert c.Ip == pytest.approx(abs(float(ad.eqdsk.Ip)))
    li = c.rows["l_i"]
    assert li["hard"] and li["kind"] == "li_3" and li["tol"] == 1e-3
    assert li["target"] == float(ad.eqdsk.li["li(2)"])
    # q0 at the measurement radius (like radii), with the sawtooth gate
    q = c.rows["q0"]
    assert q["psi"] == pytest.approx(1e-3)
    assert q["target"] == pytest.approx(float(np.interp(
        1e-3, ad.eqdsk.psi_N, np.abs(ad.eqdsk.qpsi))))
    assert q["admitted"] is (abs(q["q0_source_axis"]) <= 1.1)
    assert c.rows["mse"] is None
    assert c.signs["current_sign"] == 1.0
    assert c.boundary.shape[1] == 2 and c.boundary.shape[0] > 10


def test_the_gfile_parallel_current_composes_back_to_its_own_jphi(gfile):
    """Identity (I2) on the g-file's own surfaces, with any split."""
    ad, c = gfile
    jB, parts = gfile_parallel_current(ad.eqdsk)
    assert parts["frame_identity_max_rel"] < 1e-9
    J, p = compose(_geom(parts), 0.6 * jB, 0.3 * jB, 0.1 * jB)
    scale = float(np.max(np.abs(parts["jphi_in"])))
    assert float(np.max(np.abs(J - parts["jphi_in"]))) / scale < 1e-9
    # the anchor request is the legacy reconstruction's total
    np.testing.assert_array_equal(
        c.anchor_request, np.abs(ad.eqdsk.j_tor_averaged_direct))
    # the pressure-driven term is a measurable piece of the edge current
    frac = p["pressure"] / parts["jphi_in"]
    assert 0.0 < float(np.median(frac[c.psi_N > 0.9])) < 1.0


def test_the_gfile_adapter_finalizes_with_the_anchor_bootstrap(gfile):
    ad, c = gfile
    jB, parts = gfile_parallel_current(ad.eqdsk)
    psi = c.psi_N
    redl = 0.2 * float(np.max(jB)) * np.exp(-0.5 * ((psi - 0.95) / 0.03) ** 2)
    ad2 = GFileAdapter(ad.source, ad.config)
    ad2.read()
    c2 = ad2.finalize(redl, _geom(parts))
    validate_contract(c2)
    assert np.all(c2.jB_ind >= 0.0)
    np.testing.assert_array_equal(c2.jB_bs_anchor, redl)
    expect = inductive_basis(psi, jB - redl - c2.jB_fix,
                             k=int(ad.source.n_k),
                             psi_bridge=float(ad.source.psi_bridge))
    np.testing.assert_array_equal(c2.jB_ind, expect)
    # no fixed parts configured: the fixed <j.B> is zero
    assert not np.any(c2.jB_fix)


def test_a_gfile_read_with_the_wrong_cocos_is_refused():
    import h5py
    from bouquet.io.geqdsk import GEQDSKEquilibrium
    with h5py.File(_GOLD, "r") as f:
        raw = bytes(f["scan/0/0/eqdsk"][()])
    g = GEQDSKEquilibrium.from_bytes(raw, cocos=1)   # it is written COCOS 7
    with pytest.raises(EngineInputRefused, match="sign"):
        gfile_parallel_current(g)


def test_the_inductive_basis_is_the_legacy_basis_without_the_amplitude(
        gfile, monkeypatch):
    """fit_inductive_profile returns ind_scale * basis; the adapter's copy
    returns basis -- the same to rounding."""
    import bouquet.TokaMaker_interface as ti
    ad, c = gfile
    jB, _ = gfile_parallel_current(ad.eqdsk)
    psi = c.psi_N
    resid = jB - 0.3 * jB * np.exp(-0.5 * ((psi - 0.95) / 0.03) ** 2)
    # the amplitude search's proxy, replaced by a linear stand-in
    monkeypatch.setattr(ti, "calc_cylindrical_li_proxy",
                        lambda mygs, j, pad: float(np.sum(j)) * 1e-9)
    fit = ti.fit_inductive_profile(None, resid, np.zeros_like(psi), psi,
                                   1e-3, float(np.sum(resid)) * 0.9e-9,
                                   k=5, psi_bridge=0.99)
    assert fit["ind_scale"] != 1.0
    np.testing.assert_allclose(fit["j_inductive_fit"] / fit["ind_scale"],
                               inductive_basis(psi, resid, k=5,
                                               psi_bridge=0.99),
                               rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
#  IDS
# ---------------------------------------------------------------------------
def test_the_ids_adapter_reads_the_contract(ids):
    import json
    ad, c, bl = ids
    assert c.kind == "ids"
    validate_contract(c)
    with open(_OMAS) as fh:
        dd = json.load(fh)
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    np.testing.assert_allclose(c.jB_ind, B0 * np.asarray(cp["j_ohmic"]),
                               rtol=1e-15)
    assert "j_ohmic" in c.provenance["inductive"]
    assert c.rows["Ip"]["hard"] is False
    assert c.rows["Ip"]["sigma"] == pytest.approx(0.005 * bl.Ip_target)
    li = c.rows["l_i"]
    gq = dd["equilibrium"]["time_slice"][2]["global_quantities"]
    assert li["kind"] == "li_3" and li["target"] == float(gq["li_3"])
    assert li["hard"] is False and li["sigma"] == 0.04
    q = c.rows["q0"]
    assert q["psi"] == pytest.approx(1e-3) and q["target"] > 0
    assert c.signs["b0_sign"] == -1.0 and c.signs["current_sign"] == 1.0
    # pressure: thermal + impurity + fast, NO p_diff
    assert "no p_diff" in c.provenance["pressure"]
    np.testing.assert_array_equal(c.anchor_request, bl.j_phi)


def test_the_ids_residual_inductive(ids):
    import json
    ad, c, bl = ids
    ad2 = IdsAdapter(ad.source, ad.config, bl, inductive="residual")
    c2 = ad2.read()
    with open(_OMAS) as fh:
        cp = json.load(fh)["core_profiles"]["profiles_1d"][2]
    B0 = 1.8
    nbi = c2.jB_fix / B0
    np.testing.assert_allclose(
        c2.jB_ind, B0 * (np.asarray(cp["j_total"])
                         - np.asarray(cp["j_bootstrap"]) - nbi), rtol=1e-12)
    assert c2.provenance["inductive"].startswith("residual")


def test_the_ids_adapter_refuses_without_b0(ids, tmp_path):
    import json
    ad, c, bl = ids
    with open(_OMAS) as fh:
        dd = json.load(fh)
    del dd["equilibrium"]["vacuum_toroidal_field"]["b0"]
    p = tmp_path / "no_b0.json"
    p.write_text(json.dumps(dd))
    from bouquet.config import ImasSource
    src = ImasSource(ids_path=str(p), time=_TIME)
    with pytest.raises(EngineInputRefused, match="b0"):
        IdsAdapter(src, ad.config, bl).read()


def _mse_block(**over):
    n = 6
    md = dict(R=np.linspace(1.7, 2.2, n), Z=np.zeros(n),
              tgamma=np.linspace(0.02, 0.1, n), sigma=np.full(n, 0.002),
              weight=np.ones(n), A1=np.ones(n), A2=np.ones(n),
              A3=np.zeros(n), A4=np.zeros(n))
    md.update(over)
    return md


def test_raw_er_mse_is_refused_and_corrected_mse_is_accepted():
    from bouquet.config import GenerationConfig
    g = GenerationConfig(reconstruction_engine="unified",
                         engine_rows=["Ip", "l_i", "mse"],
                         mse_data=_mse_block(er_corrected=True))
    r = mse_rows(g)
    assert r["chords"]["n_active"] == 6 and r["chords"]["er_corrected"]
    for bad in (dict(), dict(A5=np.ones(6), Er=np.full(6, 1e3))):
        g.mse_data = _mse_block(**bad)
        with pytest.raises(EngineInputRefused, match="E_r"):
            mse_rows(g)


# ---------------------------------------------------------------------------
#  identity (I2) on the golden fixture's stored geometry (read-only)
# ---------------------------------------------------------------------------
def test_identity_I2_on_the_golden_fixtures_stored_geometry():
    import h5py
    from bouquet.io.geqdsk import GEQDSKEquilibrium
    with h5py.File(_GOLD, "r") as f:
        raw = bytes(f["scan/0/0/eqdsk"][()])
        fsa = {k: f["scan/0/0/eq_fsa/" + k][()]
               for k in ("psi_N", "F", "avg_inv_R", "avg_B2")}
    g = GEQDSKEquilibrium.from_bytes(raw, cocos=7)
    jB, parts = gfile_parallel_current(g)
    J, _ = compose(_geom(parts), 0.6 * jB, 0.3 * jB, 0.1 * jB)
    scale = float(np.max(np.abs(parts["jphi_in"])))
    assert float(np.max(np.abs(J - parts["jphi_in"]))) / scale < 1e-9
    # the conversion factor from the g-file's own surfaces against
    # TokaMaker's archived averages of the same draw: the verification
    # report measured the archived averages against independent contour
    # averages at 7e-6 to 1.3e-3 (worst at psi_N 0.97); the same level
    # here, over psi_N 0.01-0.99
    kap_g = np.interp(fsa["psi_N"], g.psi_N, conversion_factor(_geom(parts)))
    kap_t = np.abs(fsa["F"]) * fsa["avg_inv_R"] / fsa["avg_B2"]
    m = (fsa["psi_N"] >= 0.01) & (fsa["psi_N"] <= 0.99)
    assert float(np.max(np.abs(kap_g[m] / kap_t[m] - 1.0))) <= 1.3e-3
