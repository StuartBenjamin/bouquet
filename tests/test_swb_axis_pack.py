"""GenerationConfig.swb_axis_pack: SWB on the run grid plus near-axis nodes
(coords.axis_pack_grid).  Grid properties, off is bit-identical, the
run-grid <-> packed-grid round trip, config guards and the mocked SWB path."""

import types

import numpy as np
import pytest

from bouquet import coords
from bouquet.config import BouquetConfig, GenerationConfig, swb_config_problems
from bouquet.run import Bouquet

from test_swb_saw import _cfg, _run, fake_swb, saw_oft  # noqa: F401  (fixtures)

PHI_RHO = (np.arange(257) / 256.0) ** 2          # FUSE-like: uniform in rho
PHI_UNI = np.linspace(0.0, 1.0, 129)              # uniform in Phi_N: first rho 0.088
PSI_UNI = np.linspace(0.0, 1.0, 101)


# ---- the grid ---------------------------------------------------------------
class TestGrid:
    @pytest.mark.parametrize("x", [PHI_RHO, PHI_UNI, PSI_UNI])
    def test_off(self, x):
        xs, idx = coords.axis_pack_grid(x, None)
        assert idx is None and np.array_equal(xs, x)

    def test_already_fine_is_off(self):
        xs, idx = coords.axis_pack_grid(PHI_RHO, 0.005)     # run grid dr = 0.0039
        assert idx is None and np.array_equal(xs, PHI_RHO)

    @pytest.mark.parametrize("x", [PHI_RHO, PHI_UNI, PSI_UNI])
    @pytest.mark.parametrize("h0,rp", [(0.01, 0.15), (0.002, 0.15), (0.005, 0.3),
                                       (0.001, 0.1)])
    def test_properties(self, x, h0, rp):
        xs, idx = coords.axis_pack_grid(x, h0, rp)
        if idx is None:
            assert np.sqrt(x[1]) <= h0
            return
        assert np.array_equal(xs[idx], x)                    # run nodes kept exactly
        assert np.all(np.diff(xs) > 0) and xs[0] == x[0] and xs[-1] == x[-1]
        r, rr = np.sqrt(xs), np.sqrt(x)
        dr = np.diff(r)
        assert dr[0] <= h0 * (1 + 1e-12)                     # axis spacing
        g = dr[1:] / dr[:-1]
        base = np.diff(rr)
        assert g.max() <= coords.AXIS_PACK_GROWTH + 1e-9     # outward growth limit
        assert (1 / g).max() <= max(2.0, (base[:-1] / base[1:]).max()) + 1e-9
        n_in = np.diff(idx) - 1
        assert n_in.sum() == xs.size - x.size
        ri = rr[:-1]
        stop = np.nonzero((ri >= rp) & (n_in == 0))[0]       # packing ends there
        if stop.size:
            assert not n_in[stop[0]:].any()
        w = np.cos(0.5 * np.pi * np.minimum(ri / rp, 1.0)) ** 2
        cap = np.where(ri < rp, h0 / np.where(w > 0, w, 1.0), np.inf)
        assert np.all(base / (n_in + 1) <= cap * (1 + 1e-9))

    def test_coarse_phi_grid_resolved(self):
        xs, idx = coords.axis_pack_grid(PHI_UNI, 0.01, 0.15)
        assert np.sqrt(PHI_UNI[1]) > 0.08
        assert np.sqrt(xs[1]) <= 0.01 and np.sum(np.sqrt(xs) < 0.1) >= 10


class TestRoundTrip:
    def test_to_from(self):
        x = PHI_UNI
        xs, idx = coords.axis_pack_grid(x, 0.01)
        y = np.cos(3 * x) + x ** 2
        ys = coords.to_swb(x, xs, y)
        assert np.array_equal(ys[idx], y)                    # exact at the run nodes
        np.testing.assert_array_equal(ys, np.interp(xs, x, y))
        lin = 2.0 - 3.0 * x
        np.testing.assert_allclose(coords.to_swb(x, xs, lin), 2.0 - 3.0 * xs, atol=1e-14)
        assert coords.to_swb(x, xs, None) is None and coords.to_swb(x, xs, 1.5) == 1.5
        res = {"total_j_phi": ys, "j_saw": list(ys), "saw_rho_m": 0.3, "n": 3,
               "short": np.ones(4)}
        out = coords.from_swb(res, idx, xs.size)
        assert np.array_equal(out["total_j_phi"], y) and np.array_equal(out["j_saw"], y)
        assert out["saw_rho_m"] == 0.3 and out["short"].size == 4
        assert np.array_equal(out["swb_packed"]["total_j_phi"], ys)
        assert set(out["swb_packed"]) == {"total_j_phi", "j_saw"}


# ---- config -----------------------------------------------------------------
class TestConfig:
    def test_default_off(self):
        gc = GenerationConfig()
        assert gc.swb_axis_pack is None and gc.swb_axis_pack_rho == 0.15

    @pytest.mark.parametrize("kw", [dict(swb_axis_pack=0.0), dict(swb_axis_pack=1e-5),
                                    dict(swb_axis_pack=0.1), dict(swb_axis_pack=True),
                                    dict(swb_axis_pack="auto"),
                                    dict(swb_axis_pack_rho=0.0), dict(swb_axis_pack_rho=0.6)])
    def test_bad_values(self, kw):
        with pytest.raises(ValueError, match=next(iter(kw))):
            GenerationConfig(imas_baseline="swb", **kw)

    def test_roundtrip(self):
        cfg = _cfg(swb_axis_pack=0.004, swb_axis_pack_rho=0.2)
        g2 = BouquetConfig.from_json(cfg.to_json()).generation
        assert (g2.swb_axis_pack, g2.swb_axis_pack_rho) == (0.004, 0.2)

    def test_problems(self, saw_oft, monkeypatch):
        assert swb_config_problems(_cfg(swb_axis_pack=0.005)) == []
        assert any("swb_axis_pack_rho" in m for m in swb_config_problems(
            _cfg(swb_axis_pack=0.05, swb_axis_pack_rho=0.04)))
        monkeypatch.setattr(coords, "_swb_params",
                            lambda: frozenset({"psi_N", "jphi_fixed", "p_fixed"}))
        assert any("swb_axis_pack" in m and "lacks x" in m
                   for m in swb_config_problems(_cfg(swb_axis_pack=0.005)))

    def test_closure_path_refuses(self):
        ns = types.SimpleNamespace(
            config=types.SimpleNamespace(
                source=types.SimpleNamespace(),
                generation=GenerationConfig(swb_axis_pack=0.005,
                                            reconstruction_engine="legacy")),
            _resolve_engine_defaults=lambda: None,
            _check_jbs_loop_workflow=lambda gc: None,
            _check_structured_mse_reachable=lambda cfg: None)
        with pytest.raises(ValueError, match="swb_axis_pack"):
            Bouquet.prepare_baseline(ns)


# ---- mocked SWB -------------------------------------------------------------
def _calls_equal(a, b):
    assert set(a) == set(b)
    for k in a:
        va, vb = a[k], b[k]
        assert (np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb), k


class TestMockedSWB:
    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_off_bit_identical(self, fake_swb, saw_q):
        ref = _run(swb_saw_q=saw_q)
        ref._swb_imas_baseline()
        n_ref = len(fake_swb)
        ns = _run(swb_saw_q=saw_q, swb_axis_pack=None)
        ns._swb_imas_baseline()
        for a, b in zip(fake_swb[:n_ref], fake_swb[n_ref:]):
            _calls_equal(a, b)
            assert np.array_equal(a["x"], ns.baseline.psi_N)
        for k in ("j_phi", "j_inductive", "j_BS"):
            assert np.array_equal(getattr(ns.baseline, k), getattr(ref.baseline, k))
        assert "swb_axis_pack" not in ns.baseline.ip_closure
        assert "swb_packed" not in ns.baseline.swb_baseline

    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_packed_inputs_outputs(self, fake_swb, saw_q):
        ns = _run(swb_saw_q=saw_q, swb_axis_pack=0.02, swb_axis_pack_rho=0.3)
        bl = ns.baseline
        x = bl.psi_N.copy()
        ns._swb_imas_baseline()
        xs, idx = coords.axis_pack_grid(x, 0.02, 0.3)
        assert idx is not None
        for kw in fake_swb:
            assert np.array_equal(kw["x"], xs)
            assert np.array_equal(kw["jphi_fixed"][idx], bl.swb_jphi_fixed)
            if saw_q is not None:
                assert np.array_equal(kw["jphi_saw"][idx], bl.swb_jphi_saw)
            else:
                assert "jphi_saw" not in kw
        # outputs on the run grid; accounting identity holds there
        assert bl.j_phi.shape == x.shape and bl.j_BS.shape == x.shape
        np.testing.assert_allclose(bl.j_BS, 1e5 * (1.0 - x) ** 2, rtol=1e-14)
        pk = np.max(np.abs(bl.j_phi))
        other = bl.j_other if saw_q is None else bl.j_other - bl.j_sawteeth + bl.j_saw
        np.testing.assert_allclose(bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + other,
                                   bl.j_phi, rtol=0, atol=1e-12 * pk)
        ic = bl.ip_closure
        assert ic["swb_axis_pack_added"] == xs.size - x.size > 0
        assert ic["swb_axis_pack"] == 0.02 and ic["swb_axis_pack_rho"] == 0.3
        packed = bl.swb_baseline["swb_packed"]
        assert np.array_equal(packed["x"], xs) and packed["total_j_phi"].size == xs.size
        assert ("j_saw" in packed) == (saw_q is not None)

    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_sigma0_repeat(self, fake_swb, saw_q):
        ns = _run(swb_saw_q=saw_q, swb_axis_pack=0.02, swb_axis_pack_rho=0.3)
        ns._swb_imas_baseline()
        out = ns._verify_sigma0_swb()
        assert out["passed"]
        _calls_equal(fake_swb[-1], fake_swb[1])
