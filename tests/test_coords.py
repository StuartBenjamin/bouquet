"""bouquet.coords: the run coordinate at the OpenFUSIONToolkit boundary."""
import numpy as np
import pytest

from bouquet import coords


class _Eq:
    """Stand-in solver: psi_bounds plus a toroidal-flux map ψ_N = Φ_N**0.8."""
    psi_bounds = (-0.4, 0.6)

    def __init__(self):
        self.calls = []

    def get_torflux_map(self, x, inverse=False):
        self.calls.append(inverse)
        x = np.asarray(x, float)
        return (x ** 0.8 if inverse else x ** 1.25), np.ones_like(x)


@pytest.fixture
def swb_params(monkeypatch):
    def _set(names):
        monkeypatch.setattr(coords, "_SWB_PARAMS", frozenset(names), raising=False)
    return _set


X = np.array([0.0, 0.05, 0.2, 0.5, 0.9, 1.0])


def test_psi_profile_dicts_carry_no_coord():
    d = coords.oft_prof("jphi-linterp", X, X ** 2)
    assert set(d) == {"type", "x", "y"}
    pp = coords.pp_prof(_Eq(), X, 1.0 - X ** 2)
    assert "coord" not in pp and pp["type"] == "linterp"


def test_phi_profile_tags():
    assert coords.oft_prof("jphi-linterp", X, X, coords.PHI)["coord"] == "phi_n_relabel"
    assert coords.oft_prof("linterp", X, X, coords.PHI)["coord"] == "phi_n"


def test_pp_scale_is_the_same_in_both_coordinates():
    p = 1.0 - X ** 2
    a = coords.pp_prof(_Eq(), X, p)["y"]
    b = coords.pp_prof(_Eq(), X, p, coords.PHI)["y"]
    np.testing.assert_array_equal(a, b)
    from bouquet.utils import pchip_derivative
    np.testing.assert_array_equal(a, pchip_derivative(X, p) / 1.0)


def test_psi_of():
    eq = _Eq()
    np.testing.assert_array_equal(coords.psi_of(eq, X, psi_pad=0.01),
                                  np.clip(X, 0.01, 0.99))
    assert eq.calls == []
    got = coords.psi_of(eq, X, coords.PHI, psi_pad=0.01)
    np.testing.assert_allclose(got, np.clip(X ** 0.8, 0.01, 0.99))
    assert eq.calls == [True]


def test_window_x():
    eq = _Eq()
    for wc in ("psi_n", "native"):
        np.testing.assert_array_equal(coords.window_x(eq, X, coords.PSI, wc), X)
    np.testing.assert_array_equal(coords.window_x(eq, X, coords.PHI, "native"), X)
    np.testing.assert_allclose(coords.window_x(eq, X, coords.PHI, "psi_n"), X ** 0.8)
    with pytest.raises(ValueError):
        coords.window_x(eq, X, coords.PSI, "rho")


def test_rho_tor_input_becomes_phi_n():
    c, x = coords.resolve_input_coord("rho_tor", np.array([0.0, 0.5, 1.0]))
    assert c == coords.PHI
    np.testing.assert_array_equal(x, [0.0, 0.25, 1.0])
    assert coords.resolve_input_coord("psi_n", X)[1] is X
    with pytest.raises(ValueError):
        coords.resolve_input_coord("psi", X)


@pytest.mark.parametrize("arg", ["x", "psi_N"])
def test_swb_grid_on_a_toolkit_with_a_grid_argument(swb_params, arg):
    swb_params({"mygs", "ne", arg})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), xi)
    assert list(coords.swb_grid_kwargs(xi)) == [arg]
    assert coords.swb_grid_kwargs(xi, coords.PHI)["coord"] == coords.PHI
    np.testing.assert_array_equal(coords.swb_seed(xi), (1 - xi ** 1.5) ** 1.5)


def test_swb_grid_prefers_x(swb_params):
    swb_params({"x", "psi_N"})
    assert list(coords.swb_grid_kwargs(X)) == ["x"]


def test_swb_grid_on_a_legacy_toolkit(swb_params):
    swb_params({"mygs", "ne"})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), np.linspace(0, 1, 4))
    assert coords.swb_grid_kwargs(xi) == {}


def test_seed_matches_oft_power_flux_fun_on_a_uniform_grid(swb_params):
    swb_params({"x"})
    s = np.linspace(0.0, 1.0, 129)
    ref = np.power(1.0 - np.power(np.linspace(0.0, 1.0, 129), 1.5), 1.5)
    np.testing.assert_array_equal(coords.swb_seed(s), ref)


def test_check_backend():
    coords.check_backend(coords.PSI)
    with pytest.raises(ValueError):
        coords.check_backend("rho_tor")


def test_window_x_rejects_an_unknown_window_coord():
    with pytest.raises(ValueError):
        coords.window_x(_Eq(), X, coords.PHI, "phi_n")


def test_psi_at_is_the_identity_object_in_a_psi_run():
    assert coords.psi_at(_Eq(), X) is X


class TestDdPhiN:
    def _cp(self, rho):
        return {"grid": {"rho_tor_norm": list(rho)}}

    def test_relabels_the_nodes(self):
        from bouquet.io.imas import _dd_phi_n
        pn = np.linspace(0, 1, 11)
        rho = pn ** 0.4
        np.testing.assert_allclose(_dd_phi_n(self._cp(rho), pn), rho ** 2)

    @pytest.mark.parametrize("rho", [None, "placeholder", "short", "decreasing"])
    def test_refuses_a_grid_that_does_not_place_the_nodes(self, rho):
        from bouquet.io.imas import _dd_phi_n
        pn = np.linspace(0, 1, 11)
        cp = {"grid": {}}
        if rho == "placeholder":
            cp = self._cp(np.sqrt(pn))
        elif rho == "short":
            cp = self._cp(pn[:-1] ** 0.4)
        elif rho == "decreasing":
            cp = self._cp((pn ** 0.4)[::-1])
        with pytest.raises(ValueError):
            _dd_phi_n(cp, pn)


class TestCheckCoord:
    def _run(self, coord="psi_n", **gen):
        import bouquet as bq
        from bouquet.config import (BouquetConfig, GenerationConfig,
                                    ImasSource, SolverConfig)
        cfg = BouquetConfig(source=ImasSource(ids_path="x.json", coord=coord),
                            solver=SolverConfig(mesh_path="m.h5"),
                            generation=GenerationConfig(**gen),
                            output_header="t")
        return bq.Bouquet(cfg)

    def test_psi_run_passes(self):
        assert self._run()._check_coord() == coords.PSI

    def test_rho_tor_is_a_phi_run(self, monkeypatch):
        monkeypatch.setattr(coords, "check_backend", lambda c: None)
        assert self._run("rho_tor")._check_coord() == coords.PHI

    def test_bad_window_coord(self):
        with pytest.raises(ValueError, match="window_coord"):
            self._run(window_coord="phi_n")._check_coord()

    def test_phi_run_refuses_the_python_solve(self, monkeypatch):
        monkeypatch.setattr(coords, "check_backend", lambda c: None)
        run = self._run("phi_n")
        run.config.generation.bootstrap_kwargs = {"use_python_solve": True}
        with pytest.raises(ValueError, match="use_python_solve"):
            run._check_coord()

    def test_phi_run_refuses_a_toolkit_without_support(self, monkeypatch):
        import sys
        # Also hide submodules an earlier test may have imported.
        for m in [k for k in sys.modules if k.split(".")[0] == "OpenFUSIONToolkit"] + ["OpenFUSIONToolkit"]:
            monkeypatch.setitem(sys.modules, m, None)
        monkeypatch.delattr(coords, "_SWB_PARAMS", raising=False)
        with pytest.raises(RuntimeError, match="toroidal-flux"):
            self._run("phi_n")._check_coord()


def test_profile_coord_defaults_on_an_old_archive(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bouquet.plotting import _profile_coord
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        f.create_group("_baseline")
    assert _profile_coord(str(p)) == "psi_n"
    with h5py.File(p, "a") as f:
        f["_baseline"].attrs["profile_coord"] = "phi_n"
    assert _profile_coord(str(p)) == "phi_n"


def test_seed_is_the_same_physical_profile_in_a_phi_run(swb_params):
    swb_params({"x"})
    xphi = np.array([0.0, 0.1, 0.4, 1.0])
    psi = xphi ** 0.8
    np.testing.assert_array_equal(coords.swb_seed(xphi, psi), (1 - psi ** 1.5) ** 1.5)
    swb_params(set())                       # legacy toolkit: its own uniform grid
    np.testing.assert_array_equal(coords.swb_seed(xphi, psi),
                                  (1 - np.linspace(0, 1, 4) ** 1.5) ** 1.5)


def test_seed_psi_picks_the_seed_coordinate():
    eq, xphi = _Eq(), np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(coords.seed_psi(eq, xphi, coords.PHI, "psi_n"), xphi ** 0.8)
    assert coords.seed_psi(eq, xphi, coords.PHI, "native") is None
    assert coords.seed_psi(eq, X, coords.PSI, "psi_n") is X
    with pytest.raises(ValueError):
        coords.seed_psi(eq, X, coords.PSI, "phi_n")


def test_bad_seed_coord_is_refused_in_prepare():
    run = TestCheckCoord()._run(seed_coord="rho")
    with pytest.raises(ValueError, match="seed_coord"):
        run._check_coord()


def test_phi_n_from_q():
    psi = np.linspace(0.0, 1.2, 61)
    q = 1.0 + 3.0 * psi ** 2
    inside, phi = coords.phi_n_from_q(psi, q)
    assert inside.sum() == 51 and phi[0] == 0.0 and phi[-1] == pytest.approx(1.0)
    x = psi[inside]
    np.testing.assert_allclose(phi, (x + x ** 3) / 2.0, atol=2e-4)
    # a grid that stops short of 1 is closed at 1 by interpolation
    _, phi2 = coords.phi_n_from_q(psi[:-2] * 1.0, q[:-2])
    assert phi2[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        coords.phi_n_from_q(psi[1:], q[1:])


class TestIdaHybridPhi:
    """ida_hybrid in a phi_n run places the IDA fits by the file's own q."""

    def _build(self, tmp_path, ida_q=True):
        h5py = pytest.importorskip("h5py")
        import json
        from test_ni_fast_subtraction import _build
        ddp, cdf, *_ = _build(tmp_path)
        psi = np.linspace(0.0, 1.0, 33)
        dd = json.loads(open(ddp).read())
        # the dd's own map (q = 1 + 2 psi^2) differs from the IDA's (1 + 4 psi^2)
        dd["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = \
            np.sqrt((3 * psi + 2 * psi ** 3) / 5).tolist()
        dd["equilibrium"]["time_slice"][0]["profiles_1d"]["rho_tor_norm"] = \
            np.sqrt((3 * psi + 2 * psi ** 3) / 5).tolist()
        open(ddp, "w").write(json.dumps(dd))
        if ida_q:
            with h5py.File(cdf, "a") as f:
                f["q"] = (1.0 + 4.0 * psi ** 2)[None, :]
        return ddp, cdf, psi

    def _read(self, ddp, cdf, coord):
        from bouquet.io.imas import read_imas_baseline
        from bouquet.config import ImasSource
        return read_imas_baseline(
            ImasSource(ids_path=ddp, time=1.0, ida_path=cdf, impurity_Z=6.0,
                       coord=coord), kinetic_source="ida_hybrid")

    def test_te_is_placed_by_the_ida_phi_n(self, tmp_path):
        ddp, cdf, psi = self._build(tmp_path)
        bl = self._read(ddp, cdf, "phi_n")
        ida = bl.aux["ida_profiles"][1]
        _, phi_ida = coords.phi_n_from_q(ida.psi_N, ida.q)
        np.testing.assert_allclose(bl.te, np.interp(bl.psi_N, phi_ida, ida.te), rtol=1e-12)
        np.testing.assert_array_equal(bl.psi_map[1], phi_ida)
        # not the psi_N placement the dd's own map would give
        blp = self._read(ddp, cdf, "psi_n")
        assert np.max(np.abs(bl.te - blp.te)) > 0.01 * np.max(blp.te)

    def test_the_envelope_follows_the_same_map(self, tmp_path):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        ddp, cdf, _ = self._build(tmp_path)
        bl = self._read(ddp, cdf, "phi_n")
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ddp, time=1.0, ida_path=cdf,
                              impurity_Z=6.0, coord="phi_n"),
            solver=SolverConfig(mesh_path="unused"), output_header="unused")
        env = resolve_uncertainty(cfg, bl)
        np.testing.assert_allclose(env["sigma_te"], bl.aux["sigma_te_ida"], rtol=1e-12)

    def test_an_ida_without_q_is_refused(self, tmp_path):
        ddp, cdf, _ = self._build(tmp_path, ida_q=False)
        with pytest.raises(ValueError, match="no q"):
            self._read(ddp, cdf, "phi_n")


class TestReconKineticPhi:
    """g-file path: IDA nodes by the IDA's q, p-file nodes by the g-file's map."""

    def test_ida_uses_its_own_q(self):
        from bouquet.baseline import _kinetic_phi_n
        psi = np.linspace(0.0, 1.2, 61)
        q = 1.0 + 3.0 * psi ** 2
        psi_eq = np.linspace(0.0, 1.0, 11)
        inside, phi = _kinetic_phi_n(psi, q, psi_eq, psi_eq ** 0.5)
        np.testing.assert_array_equal(phi, coords.phi_n_from_q(psi, q)[1])
        assert inside.sum() == 51

    def test_pfile_uses_the_gfile_map(self):
        from bouquet.baseline import _kinetic_phi_n
        psi = np.linspace(0.0, 1.1, 23)
        psi_eq = np.linspace(0.0, 1.0, 11)
        inside, phi = _kinetic_phi_n(psi, None, psi_eq, psi_eq ** 1.25)
        np.testing.assert_allclose(phi, np.interp(psi[inside], psi_eq, psi_eq ** 1.25))

    def test_the_ida_loader_keeps_q(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from test_ni_fast_subtraction import _ida_cdf
        from bouquet.baseline import _load_kinetic_profiles
        from bouquet.config import ReconstructionSource
        cdf = str(tmp_path / "ida.cdf")
        _ida_cdf(cdf)
        with h5py.File(cdf, "a") as f:
            f["q"] = (1.0 + np.linspace(0, 1, 33) ** 2)[None, :]
        kin = _load_kinetic_profiles(ReconstructionSource(
            geqdsk_path="g", profiles_path=cdf, time=1.0))
        np.testing.assert_allclose(kin["q"], 1.0 + np.linspace(0, 1, 33) ** 2)
