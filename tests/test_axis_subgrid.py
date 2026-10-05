"""Near-axis sub-grid flattening (bouquet.axis_subgrid, GenerationConfig.
swb_saw_axis_flatten): the pure function, the cut, the mesh radius, config
guards and the swb prepare-path accounting (SWB mocked)."""

import types

import numpy as np
import pytest

from bouquet import axis_subgrid as AS
from bouquet.config import GenerationConfig, BouquetConfig, swb_config_problems
from bouquet.run import Bouquet
from test_swb_saw import _cfg, _run, fake_swb, saw_oft  # noqa: F401  (fixtures)

RHO = np.arange(257) / 256.0          # FUSE's grid: uniform in rho_tor
S = RHO ** 2


def _wiggly(s=S):
    """Smooth peaked profile plus near-axis structure (extrema ~0.02 apart in rho)."""
    rho = np.sqrt(s)
    return (1.6e6 * (1.0 - s) ** 1.2
            + 1e5 * np.exp(-(rho / 0.03) ** 2)
            + 6e4 * np.sin(2 * np.pi * rho / 0.035) * np.exp(-(rho / 0.09) ** 4))


def _trapz(s, j):
    return float(np.sum(0.5 * (j[1:] + j[:-1]) * np.diff(s)))


# ---- flatten_axis_subgrid ----------------------------------------------------
class TestFlatten:
    def test_enclosed_current_and_identity_outside(self):
        j = _wiggly()
        out, s_used = AS.flatten_axis_subgrid(S, j, 0.09 ** 2)
        k = int(np.searchsorted(S, s_used))
        assert s_used == S[k] and S[k - 1] < 0.09 ** 2 <= S[k]
        assert np.array_equal(out[k:], j[k:])
        assert abs(_trapz(S, out) - _trapz(S, j)) <= 1e-13 * _trapz(S, np.abs(j))
        assert abs(_trapz(S[:k + 1], out[:k + 1]) - _trapz(S[:k + 1], j[:k + 1])) \
            <= 1e-12 * _trapz(S[:k + 1], np.abs(j[:k + 1]))

    def test_c1_and_axis_regular(self):
        j = _wiggly()
        out, s_used = AS.flatten_axis_subgrid(S, j, 0.09 ** 2)
        k = int(np.searchsorted(S, s_used))
        c2, c1, c0 = np.polyfit(S[:k], out[:k], 2)
        assert np.allclose(np.polyval([c2, c1, c0], S[:k]), out[:k], rtol=1e-10)
        assert np.isclose(np.polyval([c2, c1, c0], S[k]), j[k], rtol=1e-10)   # C0
        dj = np.gradient(j, S, edge_order=2)[k]
        assert np.isclose(2 * c2 * S[k] + c1, dj, rtol=1e-8)                 # C1
        # a polynomial in s = rho^2 is even in rho: no extremum left inside but the axis
        assert AS.axis_extrema(S, out, S[k]).size <= 1
        assert AS.axis_extrema(S, j, S[k]).size >= 3

    def test_quadratic_is_a_fixed_point(self):
        j = 1.7e6 - 2e6 * S + 5e5 * S ** 2
        out, _ = AS.flatten_axis_subgrid(S, j, 0.1 ** 2)
        np.testing.assert_allclose(out, j, rtol=1e-12)

    @pytest.mark.parametrize("s_cut", [0.0, S[1], 1.0])
    def test_bad_cut_refused(self, s_cut):
        with pytest.raises(ValueError, match="s_cut"):
            AS.flatten_axis_subgrid(S, _wiggly(), s_cut)


# ---- detection and cut ------------------------------------------------------
class TestCut:
    def test_monotone_is_left_alone(self):
        j = 1.6e6 * (1.0 - S) ** 1.2
        assert AS.axis_extrema(S, j, 0.2 ** 2).size == 0
        assert AS.axis_cut(S, j, 0.08) is None

    def test_cut_snaps_to_next_extremum(self):
        j = _wiggly()
        s_cut = AS.axis_cut(S, j, 0.076)
        ext = AS.axis_extrema(S, j, 1.0)
        assert s_cut >= 0.076 ** 2 and s_cut in S[ext]
        assert not np.any((S[ext] >= 0.076 ** 2) & (S[ext] < s_cut))

    def test_no_extremum_in_reach_uses_rho_res(self):
        j = 1.6e6 * (1.0 - S) ** 1.2
        j[:6] += 1e4 * np.cos(np.pi * RHO[:6] / RHO[5])    # structure only near axis
        assert AS.axis_cut(S, j, 0.08) == 0.08 ** 2


def _square_mesh(n=60, L=1.2):
    """n x n squares split in two triangles over [-L, L]^2: cell area (2L/n)^2 / 2."""
    g = np.linspace(-L, L, n + 1)
    R, Z = np.meshgrid(g, g, indexing="ij")
    r = np.c_[R.ravel(), Z.ravel(), 0 * R.ravel()]
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    return r, np.r_[np.c_[a, b, c], np.c_[a, c, d]], 0.5 * (2 * L / n) ** 2


def test_mesh_axis_rho():
    r, lc, a_cell = _square_mesh()
    th = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    bnd = np.c_[np.cos(th), 1.5 * np.sin(th)]          # area 1.5 pi
    a_lcfs = 0.5 * abs(np.dot(bnd[:, 0], np.roll(bnd[:, 1], 1))
                       - np.dot(bnd[:, 1], np.roll(bnd[:, 0], 1)))
    for n in (10.0, 20.0):
        assert np.isclose(AS.mesh_axis_rho(r, lc, bnd, n_cells=n),
                          np.sqrt(n * a_cell / a_lcfs), rtol=1e-12)


# ---- baseline bookkeeping ---------------------------------------------------
def _bl(coord="phi_n", saw=True):
    x = S if coord == "phi_n" else RHO ** 2 / (1.0 + 0.5 * RHO ** 2) * 1.5   # psi_N(rho)
    j_ind = 1.4e6 * (1.0 - S) ** 1.5
    j_bs = 1.5e5 * np.sqrt(RHO) * (1.0 - S)
    j_nbi = 7e4 * (1.0 - S)
    j_st = (_wiggly() - 1.6e6 * (1.0 - S) ** 1.2) if saw else np.zeros_like(S)
    j_fus = 1e3 * (1.0 - S)
    bl = types.SimpleNamespace(
        psi_N=x, coord=coord, j_inductive=j_ind, j_BS=j_bs, j_NBI=j_nbi,
        j_RF=np.zeros_like(S), j_other=j_fus + j_st, j_sawteeth=j_st,
        j_phi=j_ind + j_bs + j_nbi + j_fus + j_st)
    if coord == "psi_n":
        bl.fuse_currents = {"psi_norm": x, "rho_tor_norm": RHO}
    return bl


class TestBaseline:
    @pytest.mark.parametrize("coord", ["phi_n", "psi_n"])
    def test_change_booked_on_sawteeth(self, coord):
        bl = _bl(coord)
        old = {k: np.array(getattr(bl, k)) for k in vars(bl) if k.startswith("j_")}
        rec = AS.flatten_baseline_saw(bl, 0.09)
        d = bl.j_phi - old["j_phi"]
        assert np.any(d) and np.array_equal(d == 0, RHO >= rec["saw_axis_rho_cut"])
        for k in ("j_inductive", "j_BS", "j_NBI", "j_RF"):
            assert np.array_equal(getattr(bl, k), old[k])
        np.testing.assert_allclose(bl.j_other - old["j_other"], d, rtol=0, atol=1e-9)
        np.testing.assert_allclose(bl.j_sawteeth - old["j_sawteeth"], d, rtol=0, atol=1e-9)
        tot = bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + bl.j_other
        np.testing.assert_allclose(tot, bl.j_phi, rtol=0, atol=1e-9)
        # jphi_fixed (j_phi - j_ind - j_BS - j_sawteeth) is unchanged
        np.testing.assert_allclose(
            bl.j_phi - bl.j_inductive - bl.j_BS - bl.j_sawteeth,
            old["j_phi"] - old["j_inductive"] - old["j_BS"] - old["j_sawteeth"],
            rtol=0, atol=1e-9)
        assert abs(rec["saw_axis_enclosed_change"]) < 1e-12
        assert 0 < rec["saw_axis_moved_frac"] < 0.05 and rec["saw_axis_n_extrema"] >= 3

    def test_guards_quiet(self, capsys):
        rec = AS.flatten_baseline_saw(_bl(), 0.09, rho_res=0.08)
        assert np.isclose(rec["saw_axis_cut_over_res"], rec["saw_axis_rho_cut"] / 0.08)
        assert not (rec["saw_axis_warn_wide"] or rec["saw_axis_warn_moved"])
        assert "WARN" not in capsys.readouterr().out

    def test_guard_wide_cut(self, capsys):
        rec = AS.flatten_baseline_saw(_bl(), 0.09, rho_res=0.04)
        assert rec["saw_axis_cut_over_res"] > AS.MAX_CUT_RATIO and rec["saw_axis_warn_wide"]
        assert "mesh radius" in capsys.readouterr().out

    def test_guard_moved_current(self, capsys):
        bl = _bl()
        bl.j_sawteeth = bl.j_sawteeth + 1e6 * np.cos(np.pi * RHO / 0.1) * (RHO < 0.25)
        bl.j_phi = bl.j_phi + 1e6 * np.cos(np.pi * RHO / 0.1) * (RHO < 0.25)
        rec = AS.flatten_baseline_saw(bl, 0.2)
        assert rec["saw_axis_moved_frac"] > AS.MAX_MOVED_FRAC and rec["saw_axis_warn_moved"]
        assert rec["saw_axis_cut_over_res"] is None and not rec["saw_axis_warn_wide"]
        assert "moved" in capsys.readouterr().out

    def test_auto_noop_when_resolved(self):
        bl = _bl()
        bl.j_phi = 1.6e6 * (1.0 - S) ** 1.2
        j0 = bl.j_phi.copy()
        rec = AS.flatten_baseline_saw(bl, "auto", rho_res=0.08)
        assert rec["saw_axis_rho_cut"] is None and np.array_equal(bl.j_phi, j0)

    def test_no_sawteeth_skipped(self):
        bl = _bl(saw=False)
        j0 = bl.j_phi.copy()
        rec = AS.flatten_baseline_saw(bl, 0.09)
        assert rec["saw_axis_flatten_skipped"] and np.array_equal(bl.j_phi, j0)

    def test_psi_run_needs_a_rho_map(self):
        bl = _bl("psi_n")
        del bl.fuse_currents
        with pytest.raises(RuntimeError, match="rho_tor"):
            AS.flatten_baseline_saw(bl, 0.09)


# ---- config -----------------------------------------------------------------
class TestConfig:
    def test_default_off(self):
        gc = GenerationConfig()
        assert gc.swb_saw_axis_flatten is None and gc.swb_saw_axis_flatten_cells == 20.0

    @pytest.mark.parametrize("v", ["auto", 0.09])
    def test_accepted(self, v, saw_oft):
        cfg = _cfg(swb_saw_axis_flatten=v)
        assert swb_config_problems(cfg) == []
        g2 = BouquetConfig.from_json(cfg.to_json()).generation
        assert g2.swb_saw_axis_flatten == v

    @pytest.mark.parametrize("kw", [dict(swb_saw_axis_flatten="on"),
                                    dict(swb_saw_axis_flatten=0.0),
                                    dict(swb_saw_axis_flatten=0.6),
                                    dict(swb_saw_axis_flatten=True),
                                    dict(swb_saw_axis_flatten_cells=0.0)])
    def test_bad_values_refused(self, kw):
        with pytest.raises(ValueError, match=next(iter(kw))):
            GenerationConfig(imas_baseline="swb", **kw)

    def test_sawteeth_in_ohmic_refused(self, saw_oft):
        cfg = _cfg(swb_saw_axis_flatten="auto")
        cfg.source.sawteeth_in_ohmic = True
        assert any("sawteeth_in_ohmic" in m for m in swb_config_problems(cfg))

    def test_closure_path_refuses(self):
        ns = types.SimpleNamespace(
            config=types.SimpleNamespace(
                generation=GenerationConfig(swb_saw_axis_flatten=0.09)),
            _check_coord=lambda: None)
        with pytest.raises(ValueError, match="swb_saw_axis_flatten"):
            Bouquet.prepare_baseline(ns)


# ---- swb prepare path (mocked SWB) ------------------------------------------
def _wiggly_run(**gen):
    ns = _run(**gen)
    bl = ns.baseline
    x = bl.psi_N                                   # 33 nodes on [0, 1]: use it as Phi_N
    bl.coord = "phi_n"
    rho = np.sqrt(x)
    bump = 2e5 * np.cos(np.pi * rho / 0.25) * (rho < 0.375)    # structure inside rho 0.375
    bl.j_sawteeth = bl.j_sawteeth + bump
    bl.j_other = bl.j_other + bump
    bl.j_phi = bl.j_phi + bump
    return ns


class TestPreparePath:
    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_accounting(self, fake_swb, saw_q):
        ref = _wiggly_run(swb_saw_q=saw_q)
        ref._swb_imas_baseline()
        ns = _wiggly_run(swb_saw_q=saw_q, swb_saw_axis_flatten=0.45)
        bl = ns.baseline
        st0 = bl.j_sawteeth.copy()
        ns._swb_imas_baseline()
        pk = np.max(np.abs(bl.j_phi))
        # the non-sawteeth fixed current is untouched (saw off: jphi_fixed takes Delta)
        def rest(b):
            return (b.swb_jphi_fixed - b.j_sawteeth
                    + (0.0 if b.swb_jphi_saw is None else b.swb_jphi_saw))
        np.testing.assert_allclose(rest(bl), rest(ref.baseline), rtol=0, atol=1e-9 * pk)
        assert not np.array_equal(bl.j_sawteeth, st0)
        if saw_q is not None:
            np.testing.assert_array_equal(bl.swb_jphi_saw, bl.j_sawteeth)
            assert np.array_equal(fake_swb[-1]["jphi_saw"], bl.swb_jphi_saw)
            other = bl.j_other - bl.j_sawteeth + bl.j_saw
        else:
            other = bl.j_other
        np.testing.assert_allclose(bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + other,
                                   bl.j_phi, rtol=0, atol=1e-12 * pk)
        ic = bl.ip_closure
        assert ic["saw_axis_rho_cut"] >= 0.45 and ic["saw_axis_moved_frac"] > 0
        assert "saw_axis_rho_cut" not in ref.baseline.ip_closure

    def test_auto_reads_the_mesh(self, fake_swb):
        ns = _wiggly_run(swb_saw_q=1.025, swb_saw_axis_flatten="auto",
                         swb_saw_axis_flatten_cells=40.0)
        r, lc, a_cell = _square_mesh()
        ns._mesh_cells = (r, lc)
        th = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
        ns._boundary_RZ = np.c_[0.3 * np.cos(th), 0.45 * np.sin(th)]
        ns._swb_imas_baseline()
        ic = ns.baseline.ip_closure
        assert np.isclose(ic["saw_axis_rho_res"],
                          AS.mesh_axis_rho(r, lc, ns._boundary_RZ, n_cells=40.0))
        assert ic["saw_axis_rho_cut"] is not None
        assert ic["saw_axis_rho_cut"] >= ic["saw_axis_rho_res"]
        assert ic["saw_axis_cut_over_res"] >= 1.0 and not ic["saw_axis_warn_wide"]
