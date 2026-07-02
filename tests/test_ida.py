"""Unit tests for bouquet.io.ida.read_ida -- both field-validated layouts.

Builds tiny synthetic IDA ``.cdf`` files with h5py (the IDA file is netCDF4 =
HDF5), so no proprietary data is needed:

  * direct   : (n_time, n_radial) profiles + companion ``*_err`` datasets;
  * ensemble : (n_time, n_samples, n_radial) posterior samples, no ``*_err``.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.io.ida import read_ida


def _write_direct(path, nr=32):
    psi = np.linspace(0, 1.2, nr)
    ne = 5e19 * (1 - 0.8 * (psi / 1.2) ** 2)
    te = 3000.0 * (1 - 0.9 * (psi / 1.2) ** 2) + 50.0
    ti = 0.9 * te
    zeff = 1.0 + 0.8 * (psi / 1.2)
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0, 3500.0])           # ms, 2 slices
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", ti), ("Zeff", zeff)]:
            f[k] = np.stack([v, v])                       # (n_time, n_radial)
        for k, v in [("n_e_err", 0.05 * ne), ("T_e_err", 0.04 * te),
                     ("T_12C6_err", 0.06 * ti)]:
            f[k] = np.stack([v, v])


def _write_ensemble(path, nr=24, ns=256, seed=0):
    rng = np.random.default_rng(seed)
    psi = np.linspace(0, 1.2, nr)
    ne0 = 6e19 * (1 - 0.7 * (psi / 1.2) ** 2)
    te0 = 2500.0 * (1 - 0.85 * (psi / 1.2) ** 2) + 40.0
    ti0 = 0.9 * te0
    zf0 = 1.0 + 1.0 * (psi / 1.2)

    def samples(mu, frac):
        return mu[None, :] * (1.0 + frac * rng.standard_normal((ns, nr)))

    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])                    # ms, single slice
        f["psi_n"] = np.stack([np.tile(psi, (ns, 1))])    # (1, ns, nr)
        f["n_e"] = np.stack([samples(ne0, 0.05)])         # (1, ns, nr)
        f["T_e"] = np.stack([samples(te0, 0.04)])
        f["T_12C6"] = np.stack([samples(ti0, 0.06)])
        f["Zeff"] = np.stack([samples(zf0, 0.03)])
    return dict(ne0=ne0, te0=te0, ti0=ti0, zf0=zf0, psi=psi)


class TestDirectLayout:
    def test_reads_and_derives_ni(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        ida = read_ida(str(p), time=3.0, impurity_Z=6.0)
        assert ida.psi_N[-1] == pytest.approx(1.2)
        assert np.all(ida.ni <= ida.ne) and np.all(ida.ni > 0)
        # sigma comes straight from *_err
        assert np.allclose(ida.sigma_ne, 0.05 * ida.ne, rtol=1e-6)
        assert ida.time == pytest.approx(3.0)

    def test_time_selection(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        ida = read_ida(str(p), time=3.5, impurity_Z=6.0)
        assert ida.time == pytest.approx(3.5)

    def test_direct_mode_rejects_ensemble(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        _write_ensemble(str(p))
        with pytest.raises(ValueError, match="3-D posterior"):
            read_ida(str(p), sigma_mode="direct")


class TestEnsembleLayout:
    def test_auto_detect_mean_and_band(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        truth = _write_ensemble(str(p))
        ida = read_ida(str(p), impurity_Z=6.0)              # auto-detect
        # sample mean recovers the underlying profile
        assert np.allclose(ida.ne, truth["ne0"], rtol=0.02)
        assert np.allclose(ida.te, truth["te0"], rtol=0.03)
        # percentile band is a positive ~5% fraction on ne
        frac = np.median(ida.sigma_ne[ida.ne > 0] / ida.ne[ida.ne > 0])
        assert 0.02 < frac < 0.09
        assert np.all(ida.ni <= ida.ne) and np.all(ida.ni > 0)

    def test_std_method(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        _write_ensemble(str(p))
        ida = read_ida(str(p), sigma_method="std", impurity_Z=6.0)
        assert np.all(ida.sigma_te >= 0)

    def test_ensemble_mode_rejects_direct(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        with pytest.raises(ValueError, match="2-D direct"):
            read_ida(str(p), time=3.0, sigma_mode="ensemble")
