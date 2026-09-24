"""IMAS read/write in a toroidal-flux (phi_n) run.

The reader relabels the core_profiles nodes by their own rho_tor_norm² and
places equilibrium.profiles_1d by the equilibrium's rho_tor_norm²; the draw
writer lands a phi_n archive on the template's Φ_N nodes and samples the
draw's geometry at the draw's ψ_N of those nodes.
"""
import json
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.config import ImasSource
from bouquet.io.imas import read_imas_baseline, write_imas_draw
from bouquet.physics import toroidal_to_parallel
from test_ni_fast_subtraction import N, _dd
from test_imas_export import (_B0, _EQ_FSA, _GEQ, _J_BS, _J_IND, _J_PHI,
                              _PEQ, _PF, _make_archive, _make_template)

PSI = np.linspace(0.0, 1.0, N)
RHO = PSI ** 0.55                     # core_profiles' real rho_tor_norm
NEQ = 41
PSI_EQ = np.linspace(0.0, 1.0, NEQ)
RHO_EQ = PSI_EQ ** 0.45               # the equilibrium's own (different) map


def _norm(phi):
    return (phi - phi[0]) / (phi[-1] - phi[0])


def _write_dd(tmp_path, rho=RHO, rho_eq=RHO_EQ, eq_q=True):
    ne = 5.0e19 * (1.0 - 0.8 * PSI ** 2) + 1e18
    nc = 1.0e18 * (1.0 - 0.5 * PSI ** 2)
    dd, _, _ = _dd(4.0e19 * (1.0 - 0.7 * PSI ** 2), ne, nc)
    cp = dd["core_profiles"]["profiles_1d"][0]
    if rho is not None:
        cp["grid"]["rho_tor_norm"] = np.asarray(rho).tolist()
    # equilibrium on its own grid, with its own rho_tor_norm
    eqp = dd["equilibrium"]["time_slice"][0]["profiles_1d"]
    for k in ("pressure", "j_tor"):
        eqp[k] = np.interp(PSI_EQ, PSI, eqp[k]).tolist()
    eqp["psi"] = PSI_EQ.tolist()
    if rho_eq is not None:
        eqp["rho_tor_norm"] = np.asarray(rho_eq).tolist()
    eqp.pop("q", None)
    if eq_q:
        eqp["q"] = (1.0 + 3.0 * PSI_EQ ** 2).tolist()
    p = tmp_path / "dd.json"
    p.write_text(json.dumps(dd))
    return str(p), eqp


def _read(ddp, coord):
    return read_imas_baseline(ImasSource(ids_path=ddp, time=1.0, coord=coord),
                              allow_incomplete_pressure=True)


class TestReadPhi:
    def test_phi_n_is_a_relabel_of_psi_n(self, tmp_path):
        ddp, _ = _write_dd(tmp_path)
        bp, bf = _read(ddp, "psi_n"), _read(ddp, "phi_n")
        np.testing.assert_array_equal(bp.psi_N, PSI)
        np.testing.assert_allclose(bf.psi_N, _norm(RHO ** 2), rtol=0, atol=1e-15)
        for k in ("ne", "te", "ni", "ti", "j_phi", "j_BS"):
            np.testing.assert_array_equal(getattr(bf, k), getattr(bp, k))

    def test_the_equilibrium_is_placed_by_its_own_rho(self, tmp_path):
        ddp, eqp = _write_dd(tmp_path)
        bp, bf = _read(ddp, "psi_n"), _read(ddp, "phi_n")
        phi_eq = _norm(RHO_EQ ** 2)
        exp_p = np.interp(bf.psi_N, phi_eq, eqp["pressure"])
        exp_j = np.interp(bf.psi_N, phi_eq, eqp["j_tor"])
        np.testing.assert_allclose(bf.p_equilibrium, exp_p, rtol=1e-12)
        np.testing.assert_allclose(bf.jphi_diff + bf.j_phi, exp_j, rtol=1e-12)
        # psi_n keeps the psi_N placement, which differs here
        np.testing.assert_allclose(
            bp.p_equilibrium, np.interp(PSI, PSI_EQ, eqp["pressure"]), rtol=1e-12)
        assert np.max(np.abs(bf.p_equilibrium - bp.p_equilibrium)) > 1e-3 * np.max(exp_p)

    def test_rho_tor_is_phi_n(self, tmp_path):
        ddp, _ = _write_dd(tmp_path)
        bf, br = _read(ddp, "phi_n"), _read(ddp, "rho_tor")
        assert br.coord == "phi_n"
        for k in ("psi_N", "ne", "j_phi", "p_equilibrium", "jphi_diff"):
            np.testing.assert_array_equal(getattr(br, k), getattr(bf, k))

    @pytest.mark.parametrize("which", ["cp", "eq"])
    @pytest.mark.parametrize("bad", ["placeholder", "missing"])
    def test_an_unusable_rho_is_refused(self, tmp_path, which, bad):
        rho = {"placeholder": np.sqrt, "missing": lambda p: None}[bad]
        kw = {"rho": rho(PSI)} if which == "cp" else {"rho_eq": rho(PSI_EQ), "eq_q": False}
        ddp, _ = _write_dd(tmp_path, **kw)
        with pytest.raises(ValueError, match="rho_tor_norm"):
            _read(ddp, "phi_n")
        _read(ddp, "psi_n")                  # psi_n never looks at rho

    def test_an_equilibrium_without_rho_is_placed_by_its_q(self, tmp_path):
        from bouquet.coords import phi_n_from_q
        ddp, eqp = _write_dd(tmp_path, rho_eq=None)
        bf = _read(ddp, "phi_n")
        phi_eq = phi_n_from_q(PSI_EQ, eqp["q"])[1]
        np.testing.assert_allclose(
            bf.p_equilibrium, np.interp(bf.psi_N, phi_eq, eqp["pressure"]), rtol=1e-12)

    def test_a_rho_grid_off_0_to_1_is_renormalised(self, tmp_path):
        rho, rho_eq = 0.02 + 0.97 * RHO, 0.01 + 0.98 * RHO_EQ
        ddp, eqp = _write_dd(tmp_path, rho=rho, rho_eq=rho_eq)
        bf = _read(ddp, "phi_n")
        np.testing.assert_allclose(bf.psi_N, _norm(rho ** 2), rtol=0, atol=1e-15)
        assert bf.psi_N[0] == 0.0 and bf.psi_N[-1] == 1.0
        np.testing.assert_allclose(
            bf.p_equilibrium,
            np.interp(bf.psi_N, _norm(rho_eq ** 2), eqp["pressure"]), rtol=1e-12)


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestWriteDrawPhi:
    def _setup(self, tmp_path, coord):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=True)
        self.zeff = np.linspace(2.0, 1.5, 20)
        self.kin = {"n_e": 1.0, "T_e": 2.0, "n_i": 3.0, "T_i": 4.0}
        with h5py.File(arc, "a") as hf:
            g = hf["scan/0/0"]
            for k, s in self.kin.items():      # distinct per channel
                del g[k]
                g.create_dataset(k, data=s * np.linspace(1.0, 0.1, 20))
            g.create_dataset("aux_zeff", data=self.zeff)
            if coord == "phi_n":
                hf.require_group("scan/0/_baseline").attrs["profile_coord"] = "phi_n"
        tmpl = str(tmp_path / "tmpl.json"); psi = _make_template(tmpl)
        psiN_t = _norm(psi)
        rho = psiN_t ** 0.4
        t = json.load(open(tmpl))
        t["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = rho.tolist()
        json.dump(t, open(tmpl, "w"))
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        return cp, psi, psiN_t, rho

    def _geq(self):
        from bouquet.io.geqdsk import read_geqdsk
        return read_geqdsk(_GEQ)

    def _check_channels(self, cp, x):
        pk = np.linspace(0, 1, 20)
        s = self.kin
        np.testing.assert_allclose(cp["electrons"]["density_thermal"],
                                   np.interp(x, pk, s["n_e"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(cp["electrons"]["temperature"],
                                   np.interp(x, pk, s["T_e"] * np.linspace(1, .1, 20)), rtol=1e-12)
        ion = cp["ion"][0]
        np.testing.assert_allclose(ion["density_thermal"],
                                   np.interp(x, pk, s["n_i"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(ion["temperature"],
                                   np.interp(x, pk, s["T_i"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(cp["zeff"], np.interp(x, pk, self.zeff), rtol=1e-12)
        np.testing.assert_allclose(cp["j_tor"], np.interp(x, _PEQ, _J_PHI), rtol=1e-12)

    def _geom(self, psiN):
        return {"F": np.interp(psiN, _PF, _EQ_FSA["F"]),
                "avg_inv_R": np.interp(psiN, _PF, _EQ_FSA["avg_inv_R"]),
                "avg_B2": np.interp(psiN, _PF, _EQ_FSA["avg_B2"]),
                "avg_inv_R2": np.interp(psiN, _PF, _EQ_FSA["avg_inv_R2"]),
                "B0": _B0}

    def test_a_phi_archive_lands_on_rho2_and_uses_the_draw_psi(self, tmp_path):
        cp, _, psiN_t, rho = self._setup(tmp_path, "phi_n")
        x = rho ** 2
        self._check_channels(cp, x)
        geq = self._geq()
        psiN_d = np.interp(x, _norm(np.asarray(geq.rhovn) ** 2), geq.psi_N)
        assert np.max(np.abs(psiN_d - psiN_t)) > 1e-3   # the maps differ
        geom = self._geom(psiN_d)
        for k, j in (("j_total", _J_PHI), ("j_ohmic", _J_IND), ("j_bootstrap", _J_BS)):
            np.testing.assert_allclose(
                cp[k], toroidal_to_parallel(np.interp(x, _PEQ, j), geom=geom), rtol=1e-10)
        np.testing.assert_allclose(
            cp["grid"]["psi"],
            geq.psi_axis + psiN_d * (geq.psi_boundary - geq.psi_axis), rtol=1e-12)
        np.testing.assert_allclose(cp["grid"]["rho_tor_norm"], rho, rtol=0, atol=0)

    def test_a_psi_archive_is_unchanged(self, tmp_path):
        cp, psi, psiN_t, _ = self._setup(tmp_path, "psi_n")
        self._check_channels(cp, psiN_t)
        np.testing.assert_array_equal(cp["grid"]["psi"], psi)
        np.testing.assert_allclose(
            cp["j_total"],
            toroidal_to_parallel(np.interp(psiN_t, _PEQ, _J_PHI),
                                 geom=self._geom(psiN_t)), rtol=1e-10)
