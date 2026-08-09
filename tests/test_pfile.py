"""Tests for the bouquet p-file reader/writer."""

import os
import tempfile

import numpy as np
import pytest

from bouquet.io.pfile import PFile, _read_pfile, _write_pfile, _NT_TO_KPA, read_pfile

_EXAMPLE_PFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples", "D3D-like",
    "D3Dlike_Hmode_baseline.peqdsk")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_test_pfile(path, npts=64):
    """Write a synthetic p-file with known profiles."""
    psi = np.linspace(0, 1, npts)

    # Parabolic-ish profiles
    ne_data = 3.0 * (1 - psi ** 2)
    te_data = 5.0 * (1 - psi ** 1.5)
    ni_data = 2.8 * (1 - psi ** 2)
    ti_data = 4.5 * (1 - psi ** 1.5)
    ptot_data = 100.0 * (1 - psi ** 2) ** 2

    # Simple derivatives (not physically exact, just for format testing)
    ne_der = np.gradient(ne_data, psi)
    te_der = np.gradient(te_data, psi)
    ni_der = np.gradient(ni_data, psi)
    ti_der = np.gradient(ti_data, psi)
    ptot_der = np.gradient(ptot_data, psi)

    profiles = [
        ("ne", "10^20/m^3", ne_data, ne_der),
        ("te", "KeV", te_data, te_der),
        ("ni", "10^20/m^3", ni_data, ni_der),
        ("ti", "KeV", ti_data, ti_der),
        ("ptot", "KPa", ptot_data, ptot_der),
    ]

    lines = []
    for key, units, data, deriv in profiles:
        lines.append(f"{npts} psinorm {key}({units}) d{key}/dpsiN\n")
        for i in range(npts):
            lines.append(f" {psi[i]:f}   {data[i]:f}   {deriv[i]:f}\n")

    # Ion species block
    lines.append("3 N Z A of ION SPECIES\n")
    lines.append(" 6.000000   6.000000   12.000000\n")  # Carbon
    lines.append(" 1.000000   1.000000   2.000000\n")   # Deuterium (main)
    lines.append(" 1.000000   1.000000   2.000000\n")   # Deuterium (beam)

    with open(path, "w") as f:
        f.writelines(lines)

    return psi, {
        "ne": ne_data,
        "te": te_data,
        "ni": ni_data,
        "ti": ti_data,
        "ptot": ptot_data,
    }


@pytest.fixture
def pfile_path(tmp_path):
    """Create a temporary synthetic p-file and return its path."""
    path = str(tmp_path / "p123456.01234")
    generate_test_pfile(path)
    return path


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_keys_present(self, pfile_path):
        raw = _read_pfile(pfile_path)
        assert "ne" in raw
        assert "te" in raw
        assert "ni" in raw
        assert "ti" in raw
        assert "ptot" in raw
        assert "N Z A" in raw

    def test_profile_shape(self, pfile_path):
        raw = _read_pfile(pfile_path)
        assert raw["ne"]["psinorm"].shape == (64,)
        assert raw["ne"]["data"].shape == (64,)
        assert raw["ne"]["derivative"].shape == (64,)

    def test_psinorm_range(self, pfile_path):
        raw = _read_pfile(pfile_path)
        psi = raw["ne"]["psinorm"]
        assert psi[0] == pytest.approx(0.0)
        assert psi[-1] == pytest.approx(1.0)

    def test_units_parsed(self, pfile_path):
        raw = _read_pfile(pfile_path)
        assert raw["ne"]["units"] == "10^20/m^3"
        assert raw["te"]["units"] == "KeV"
        assert raw["ptot"]["units"] == "KPa"

    def test_deriv_label_parsed(self, pfile_path):
        raw = _read_pfile(pfile_path)
        assert raw["ne"]["deriv_label"] == "dne/dpsiN"

    def test_ion_species(self, pfile_path):
        raw = _read_pfile(pfile_path)
        nza = raw["N Z A"]
        assert len(nza["N"]) == 3
        assert nza["Z"][0] == pytest.approx(6.0)  # Carbon
        assert nza["A"][1] == pytest.approx(2.0)  # Deuterium

    def test_data_values(self, pfile_path):
        raw = _read_pfile(pfile_path)
        # ne at psi=0 should be 3.0 (from 3*(1-0))
        assert raw["ne"]["data"][0] == pytest.approx(3.0, abs=1e-5)
        # ne at psi=1 should be 0.0
        assert raw["ne"]["data"][-1] == pytest.approx(0.0, abs=1e-5)

    def test_file_order_preserved(self, pfile_path):
        raw = _read_pfile(pfile_path)
        keys = list(raw.keys())
        assert keys == ["ne", "te", "ni", "ti", "ptot", "N Z A"]


# ---------------------------------------------------------------------------
# PFile class tests
# ---------------------------------------------------------------------------

class TestPFile:
    def test_properties(self, pfile_path):
        pf = PFile(pfile_path)
        assert pf.ne is not None
        assert pf.te is not None
        assert pf.ni is not None
        assert pf.ti is not None
        assert pf.ptot is not None
        assert len(pf.ne) == 64

    def test_missing_property_returns_none(self, pfile_path):
        pf = PFile(pfile_path)
        assert pf.omgeb is None
        assert pf.er is None

    def test_contains(self, pfile_path):
        pf = PFile(pfile_path)
        assert "ne" in pf
        assert "omgeb" not in pf
        assert "N Z A" in pf

    def test_getitem(self, pfile_path):
        pf = PFile(pfile_path)
        entry = pf["ne"]
        assert "psinorm" in entry
        assert "data" in entry
        assert "derivative" in entry

    def test_keys(self, pfile_path):
        pf = PFile(pfile_path)
        assert "ne" in pf.keys
        assert "N Z A" in pf.keys

    def test_len(self, pfile_path):
        pf = PFile(pfile_path)
        assert len(pf) == 6  # 5 profiles + N Z A

    def test_iter(self, pfile_path):
        pf = PFile(pfile_path)
        keys = list(pf)
        assert keys[0] == "ne"

    def test_psinorm_for(self, pfile_path):
        pf = PFile(pfile_path)
        psi = pf.psinorm_for("ne")
        assert psi is not None
        assert psi[0] == pytest.approx(0.0)
        assert pf.psinorm_for("N Z A") is None
        assert pf.psinorm_for("nonexistent") is None

    def test_derivative_for(self, pfile_path):
        pf = PFile(pfile_path)
        d = pf.derivative_for("ne")
        assert d is not None
        assert len(d) == 64

    def test_units_for(self, pfile_path):
        pf = PFile(pfile_path)
        assert pf.units_for("ne") == "10^20/m^3"

    def test_ion_species(self, pfile_path):
        pf = PFile(pfile_path)
        sp = pf.ion_species
        assert sp is not None
        assert len(sp["A"]) == 3

    def test_repr(self, pfile_path):
        pf = PFile(pfile_path)
        r = repr(pf)
        assert "PFile" in r
        assert "5 profiles" in r

    def test_convenience_function(self, pfile_path):
        pf = read_pfile(pfile_path)
        assert isinstance(pf, PFile)
        assert pf.ne is not None


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_write_read_roundtrip(self, pfile_path, tmp_path):
        pf = PFile(pfile_path)
        out = str(tmp_path / "roundtrip.pfile")
        pf.save(out)

        pf2 = PFile(out)
        np.testing.assert_allclose(pf2.ne, pf.ne, atol=1e-5)
        np.testing.assert_allclose(pf2.te, pf.te, atol=1e-5)
        np.testing.assert_allclose(
            pf2.psinorm_for("ne"), pf.psinorm_for("ne"), atol=1e-5
        )

    def test_ion_species_roundtrip(self, pfile_path, tmp_path):
        pf = PFile(pfile_path)
        out = str(tmp_path / "roundtrip2.pfile")
        pf.save(out)

        pf2 = PFile(out)
        np.testing.assert_allclose(
            pf2.ion_species["A"], pf.ion_species["A"], atol=1e-5
        )

    def test_from_bytes(self, pfile_path):
        with open(pfile_path, "rb") as f:
            raw = f.read()
        pf = PFile.from_bytes(raw)
        assert pf.ne is not None
        assert len(pf.ne) == 64


# ---------------------------------------------------------------------------
# Remap tests
# ---------------------------------------------------------------------------

class TestRemap:
    def test_remap_to_reference_key(self, pfile_path):
        pf = PFile(pfile_path)
        remapped = pf.remap(key="ne")
        # All profiles should now share ne's grid
        np.testing.assert_array_equal(
            remapped.psinorm_for("te"), remapped.psinorm_for("ne")
        )
        np.testing.assert_array_equal(
            remapped.psinorm_for("ptot"), remapped.psinorm_for("ne")
        )

    def test_remap_to_int(self, pfile_path):
        pf = PFile(pfile_path)
        remapped = pf.remap(psinorm=128)
        assert len(remapped.ne) == 128
        assert remapped.psinorm_for("ne")[0] == pytest.approx(0.0)
        assert remapped.psinorm_for("ne")[-1] == pytest.approx(1.0)

    def test_remap_to_array(self, pfile_path):
        pf = PFile(pfile_path)
        target = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        remapped = pf.remap(psinorm=target)
        np.testing.assert_array_equal(remapped.psinorm_for("ne"), target)
        assert len(remapped.ne) == 5

    def test_remap_preserves_ion_species(self, pfile_path):
        pf = PFile(pfile_path)
        remapped = pf.remap(psinorm=32)
        assert remapped.ion_species is not None
        assert len(remapped.ion_species["A"]) == 3

    def test_remap_interpolation_reasonable(self, pfile_path):
        pf = PFile(pfile_path)
        remapped = pf.remap(psinorm=128)
        # ne at psi=0 should still be ~3.0
        assert remapped.ne[0] == pytest.approx(3.0, abs=0.01)
        # ne at psi=1 should still be ~0.0
        assert remapped.ne[-1] == pytest.approx(0.0, abs=0.01)

    def test_remap_missing_key_raises(self, pfile_path):
        pf = PFile(pfile_path)
        with pytest.raises(KeyError):
            pf.remap(key="nonexistent")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.pfile")
        with open(path, "w") as f:
            f.write("")
        pf = PFile(path)
        assert len(pf) == 0
        assert pf.ne is None

    def test_different_grids(self, tmp_path):
        """Profiles can have different psinorm grids."""
        path = str(tmp_path / "mixed.pfile")
        psi_a = np.linspace(0, 1, 32)
        psi_b = np.linspace(0, 1, 64)
        lines = []
        lines.append(f"32 psinorm ne(10^20/m^3) dne/dpsiN\n")
        for i in range(32):
            lines.append(f" {psi_a[i]:f}   {1.0:f}   {0.0:f}\n")
        lines.append(f"64 psinorm te(KeV) dte/dpsiN\n")
        for i in range(64):
            lines.append(f" {psi_b[i]:f}   {2.0:f}   {0.0:f}\n")

        with open(path, "w") as f:
            f.writelines(lines)

        # CONTRACT CHANGE (per-quantity grid fix): reading now unifies the
        # grid by default, because consumers pair one quantity's values with
        # another's grid and silently misplace the profile.  The parser still
        # reads the native grids -- opt out to get them.
        pf = PFile(path)
        assert len(pf.ne) == 32
        assert len(pf.te) == 32            # resampled onto ne's grid
        assert len(pf.psinorm_for("te", native=True)) == 64   # native kept

        raw = PFile(path, remap=False)
        assert len(raw.ne) == 32
        assert len(raw.te) == 64           # untouched, as read

    def test_no_ion_species(self, tmp_path):
        """File without N Z A block."""
        path = str(tmp_path / "noions.pfile")
        psi = np.linspace(0, 1, 8)
        lines = [f"8 psinorm ne(10^20/m^3) dne/dpsiN\n"]
        for i in range(8):
            lines.append(f" {psi[i]:f}   {1.0:f}   {0.0:f}\n")

        with open(path, "w") as f:
            f.writelines(lines)

        pf = PFile(path)
        assert pf.ne is not None
        assert pf.ion_species is None


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_new_empty(self):
        pf = PFile.new()
        assert len(pf) == 0
        assert pf.ne is None

    def test_set_profile(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 32)
        data = 3.0 * (1 - psi ** 2)
        pf.set_profile("ne", psi, data)
        assert pf.ne is not None
        assert len(pf.ne) == 32
        assert pf.ne[0] == pytest.approx(3.0)
        assert pf.units_for("ne") == "10^20/m^3"

    def test_set_profile_auto_derivative(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 64)
        data = 2.0 * (1 - psi)
        pf.set_profile("te", psi, data)
        deriv = pf.derivative_for("te")
        # d/dpsi of 2*(1-psi) = -2
        np.testing.assert_allclose(deriv, -2.0, atol=0.1)

    def test_set_profile_explicit_derivative(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 16)
        data = np.ones(16)
        deriv = np.zeros(16)
        pf.set_profile("ni", psi, data, derivative=deriv)
        np.testing.assert_array_equal(pf.derivative_for("ni"), 0.0)

    def test_set_profile_custom_units(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 8)
        pf.set_profile("ne", psi, np.ones(8), units="m^-3")
        assert pf.units_for("ne") == "m^-3"

    def test_set_ion_species(self):
        pf = PFile.new()
        pf.set_ion_species(N=[6, 1, 1], Z=[6, 1, 1], A=[12, 2, 2])
        sp = pf.ion_species
        assert sp is not None
        assert sp["Z"][0] == pytest.approx(6.0)
        assert len(sp["A"]) == 3

    def test_compute_derivatives(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 64)
        pf.set_profile("ne", psi, 3.0 * (1 - psi))
        # Overwrite derivative with garbage
        pf["ne"]["derivative"] = np.ones(64) * 999.0
        pf.compute_derivatives()
        # Should now be close to -3.0
        np.testing.assert_allclose(
            pf.derivative_for("ne"), -3.0, atol=0.1
        )

    def test_new_roundtrip(self, tmp_path):
        pf = PFile.new()
        psi = np.linspace(0, 1, 32)
        pf.set_profile("ne", psi, 3.0 * (1 - psi ** 2))
        pf.set_profile("te", psi, 5.0 * (1 - psi))
        pf.set_ion_species(N=[6, 1], Z=[6, 1], A=[12, 2])

        out = str(tmp_path / "constructed.pfile")
        pf.save(out)
        pf2 = PFile(out)
        np.testing.assert_allclose(pf2.ne, pf.ne, atol=1e-5)
        np.testing.assert_allclose(pf2.te, pf.te, atol=1e-5)
        assert len(pf2.ion_species["A"]) == 2


# ---------------------------------------------------------------------------
# Physics computation tests
# ---------------------------------------------------------------------------

class TestPhysics:
    @pytest.fixture
    def kinetic_pfile(self):
        """Build a PFile with realistic-ish kinetic profiles."""
        pf = PFile.new()
        psi_N = np.linspace(0, 1, 128)
        # Profiles in pfile units
        ne = 3.0 * (1 - psi_N ** 2)         # 10^20/m^3
        te = 5.0 * (1 - psi_N ** 1.5)       # keV
        ni = 2.8 * (1 - psi_N ** 2)          # 10^20/m^3
        ti = 4.5 * (1 - psi_N ** 1.5)        # keV

        pf.set_profile("ne", psi_N, ne)
        pf.set_profile("te", psi_N, te)
        pf.set_profile("ni", psi_N, ni)
        pf.set_profile("ti", psi_N, ti)

        # Carbon impurity + deuterium main + deuterium beam
        pf.set_ion_species(N=[6, 1, 1], Z=[6, 1, 1], A=[12, 2, 2])
        return pf, psi_N

    def test_compute_quasineutrality(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()
        nz1 = pf._get_data("nz1")
        assert nz1 is not None
        # nz1 = (ne - ni) / Z  (nb=0)
        expected = (pf.ne - pf.ni) / 6.0
        np.testing.assert_allclose(nz1, expected)

    def test_compute_quasineutrality_with_nb(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        nb = 0.1 * np.ones(128)
        pf.set_profile("nb", psi_N, nb)
        with pytest.warns(UserWarning, match="negative nz1"):
            pf.compute_quasineutrality()
        # With nb=0.1, some grid points have ne < ni + nb so nz1 goes
        # negative.  The function warns but does NOT clamp — the caller
        # decides how to handle negative impurity densities.
        expected = (pf.ne - pf.ni - nb) / 6.0
        np.testing.assert_allclose(pf._get_data("nz1"), expected)

    def test_compute_zeff(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()
        psi, zeff = pf.compute_zeff()

        # Manual: ne=3*(1-x^2), ni=2.8*(1-x^2), nz1=(ne-ni)/6
        # At psi=0: ne=3, ni=2.8, nz1=0.2/6, nb=0
        # Z_main=1, Z_imp=6, Z_beam=1
        # Zeff = (ni*1^2 + nz1*6^2 + nb*1^2) / ne
        #      = (2.8 + 0.2/6*36) / 3 = (2.8 + 1.2) / 3
        expected_0 = (2.8 * 1**2 + (0.2 / 6.0) * 6**2) / 3.0
        assert zeff[0] == pytest.approx(expected_0, rel=1e-10)
        # Zeff should be >= 1 everywhere (boundary defaults to 1.0)
        assert np.all(zeff >= 1.0)
        # Should NOT be stored in the pfile
        assert "zeff" not in pf

    def test_compute_pressure_units(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()
        pf.compute_pressure()

        ptot = pf.ptot
        assert ptot is not None

        # Manual calculation at psi=0
        ne0, te0 = 3.0, 5.0
        ni0, ti0 = 2.8, 4.5
        nz1_0 = (ne0 - ni0) / 6.0
        expected_0 = _NT_TO_KPA * (ne0 * te0 + (ni0 + nz1_0) * ti0)
        assert ptot[0] == pytest.approx(expected_0, rel=1e-6)
        # Sanity: should be order ~100s of kPa for tokamak-like profiles
        assert 10 < ptot[0] < 1000

    def test_compute_pressure_with_pb(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pb = 10.0 * np.ones(128)  # 10 kPa fast-ion pressure
        pf.set_profile("pb", psi_N, pb)
        pf.compute_quasineutrality()
        pf.compute_pressure()

        # At psi=0, ptot should include the 10 kPa from pb
        ne0, te0 = 3.0, 5.0
        ni0, ti0 = 2.8, 4.5
        nb0 = 0.0  # nb not set, defaults to 0 in quasineutrality
        nz1_0 = (ne0 - ni0 - nb0) / 6.0
        expected_0 = _NT_TO_KPA * (ne0 * te0 + (ni0 + nz1_0) * ti0) + 10.0
        assert pf.ptot[0] == pytest.approx(expected_0, rel=1e-6)

    def test_compute_diamagnetic_rotations(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()

        # Realistic psi in Weber (DIII-D-like: psi_axis ~ -1.5, psi_bdy ~ 0)
        psi_axis = -1.5
        psi_bdy = 0.0
        psi = psi_N * (psi_bdy - psi_axis) + psi_axis

        pf.compute_diamagnetic_rotations(psi)

        omgpp = pf.omgpp
        ommpp = pf._get_data("ommpp")
        omepp = pf._get_data("omepp")

        assert omgpp is not None
        assert ommpp is not None
        assert omepp is not None

        # Sign conventions: impurity and main ion dia are negative,
        # electron dia is positive (in the interior where gradients exist)
        interior = slice(5, -5)
        assert np.all(omgpp[interior] <= 0)
        assert np.all(ommpp[interior] <= 0)
        assert np.all(omepp[interior] >= 0)

    def test_compute_rotation_decomposition(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()

        psi_axis = -1.5
        psi = psi_N * (0.0 - psi_axis) + psi_axis
        pf.compute_diamagnetic_rotations(psi)

        # Without equilibrium data: just rotation algebra
        pf.compute_rotation_decomposition()
        assert pf.omgeb is not None
        assert pf._get_data("ommvb") is not None
        assert pf._get_data("omevb") is not None

        # omgeb = omgvb + omgpp (omgvb=0 since not set)
        np.testing.assert_allclose(pf.omgeb, pf.omgpp)

        # ommvb = omgeb - ommpp
        np.testing.assert_allclose(
            pf._get_data("ommvb"),
            pf.omgeb - pf._get_data("ommpp"),
        )

    def test_compute_rotation_with_equilibrium(self, kinetic_pfile):
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()

        psi_axis = -1.5
        psi = psi_N * (0.0 - psi_axis) + psi_axis

        pf.compute_diamagnetic_rotations(psi)

        # DIII-D-like midplane equilibrium data
        R = 1.7 + 0.5 * psi_N
        Bp = 0.3 * np.ones(128)
        Bt = 2.0 * np.ones(128)

        pf.compute_rotation_decomposition(R=R, Bp=Bp, Bt=Bt, psi=psi)

        assert pf.er is not None
        assert pf._get_data("omghb") is not None

        # er = omgeb * R * Bp
        np.testing.assert_allclose(
            pf.er, pf.omgeb * R * Bp, rtol=1e-10
        )

    def test_quasineutrality_requires_ions(self):
        pf = PFile.new()
        psi = np.linspace(0, 1, 16)
        pf.set_profile("ne", psi, np.ones(16))
        pf.set_profile("ni", psi, np.ones(16))
        with pytest.raises(ValueError, match="Ion species"):
            pf.compute_quasineutrality()

    def test_full_workflow(self, kinetic_pfile, tmp_path):
        """End-to-end: build profiles, compute physics, save, reload."""
        pf, psi_N = kinetic_pfile
        pf.compute_quasineutrality()

        psi = psi_N * 1.5  # simple psi in Wb
        pf.compute_diamagnetic_rotations(psi)

        R = 1.7 * np.ones(128)
        Bp = 0.3 * np.ones(128)
        Bt = 2.0 * np.ones(128)
        pf.compute_rotation_decomposition(R=R, Bp=Bp, Bt=Bt, psi=psi)
        pf.compute_pressure()
        pf.compute_derivatives()

        out = str(tmp_path / "workflow.pfile")
        pf.save(out)
        pf2 = PFile(out)

        assert len(pf2) > 10  # should have many profiles
        np.testing.assert_allclose(pf2.ne, pf.ne, atol=1e-5)
        np.testing.assert_allclose(pf2.ptot, pf.ptot, atol=1e-3)
        np.testing.assert_allclose(pf2.omgeb, pf.omgeb, atol=1e-5)


# ---------------------------------------------------------------------------
#  per-quantity psinorm grids (issue: automatic p-file remapping)
# ---------------------------------------------------------------------------
class TestPerQuantityGrids:
    r"""An Osborne p-file stores a SEPARATE radial grid per quantity.

    On real DIII-D files ``te``'s grid sits up to ~0.2 in psi_N from ``ne``'s,
    so pairing one quantity's values with another's grid misplaces the profile
    (~13 % of peak T_e, worst at the pedestal) and makes index-wise arithmetic
    across quantities -- ``ne - ni - nb`` in quasi-neutrality -- ill-posed.
    Both shipped example p-files happen to carry ONE grid for every quantity,
    so a synthetic fixture cannot exercise this; the fixture below is built to
    diverge on purpose.
    """

    @staticmethod
    def _multigrid_pfile(tmp_path):
        """A p-file whose te grid is deliberately offset from ne's."""
        n = 64
        g_ne = np.linspace(0.0, 1.0, n)
        g_te = np.linspace(0.0, 1.0, n) ** 1.4          # same span, denser edge
        ne = 5.0 * (1.0 - 0.8 * g_ne ** 2)
        te = 3.0 * (1.0 - 0.9 * g_te ** 2)
        ni = 4.0 * (1.0 - 0.8 * g_ne ** 2)
        lines = []
        for key, g, d, unit in (("ne", g_ne, ne, "10^20/m^3"),
                                ("te", g_te, te, "KeV"),
                                ("ni", g_ne, ni, "10^20/m^3")):
            lines.append(f"{n} psinorm {key}({unit}) d{key}/dpsiN\n")
            dv = np.gradient(d, g)
            for i in range(n):
                lines.append(f" {g[i]:f}   {d[i]:f}   {dv[i]:f}\n")
        lines.append("1 N Z A of ION SPECIES\n")
        lines.append(" 6.000000   6.000000  12.000000\n")
        p = tmp_path / "p999999.01000"
        p.write_text("".join(lines))
        return p, g_ne, g_te

    def test_native_grids_actually_differ_in_the_fixture(self, tmp_path):
        """Guard the guard: a single-grid fixture would make everything below
        pass for the wrong reason."""
        path, g_ne, g_te = self._multigrid_pfile(tmp_path)
        pf = read_pfile(path, remap=False)
        assert not np.allclose(pf.psinorm_for("te"), pf.psinorm_for("ne"))
        assert np.abs(np.asarray(pf.psinorm_for("te"))
                      - np.asarray(pf.psinorm_for("ne"))).max() > 0.05

    def test_remap_is_on_by_default_and_unifies_the_grid(self, tmp_path):
        path, _, _ = self._multigrid_pfile(tmp_path)
        with pytest.warns(UserWarning, match="psinorm grid per quantity"):
            pf = read_pfile(path)
        np.testing.assert_allclose(pf.psinorm_for("te"), pf.psinorm_for("ne"))

    def test_remap_interpolates_rather_than_relabels(self, tmp_path):
        """The values must be resampled through te's OWN grid, not simply
        re-tagged with ne's -- relabelling is the defect being fixed."""
        path, g_ne, g_te = self._multigrid_pfile(tmp_path)
        raw = read_pfile(path, remap=False)
        te_native = np.asarray(raw.te, dtype=float)
        # grids as STORED -- the writer keeps 6 decimals, so the generator's
        # full-precision arrays are not what was read back
        g_ne_f = np.asarray(raw.psinorm_for("ne"), dtype=float)
        g_te_f = np.asarray(raw.psinorm_for("te"), dtype=float)
        with pytest.warns(UserWarning):
            pf = read_pfile(path)
        expected = np.interp(g_ne_f, g_te_f, te_native)
        np.testing.assert_allclose(np.asarray(pf.te, dtype=float), expected,
                                   rtol=1e-10)
        assert not np.allclose(np.asarray(pf.te, dtype=float), te_native)

    def test_single_grid_file_does_not_warn(self, tmp_path):
        """No false alarm on files that already share one grid (both shipped
        examples do)."""
        import warnings as _w
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            read_pfile(_EXAMPLE_PFILE)
        assert not [r for r in rec
                    if "psinorm grid per quantity" in str(r.message)]

    def test_native_grids_survive_the_remap(self, tmp_path):
        path, g_ne, g_te = self._multigrid_pfile(tmp_path)
        with pytest.warns(UserWarning):
            pf = read_pfile(path)
        np.testing.assert_allclose(pf.psinorm_for("te", native=True), g_te,
                                   atol=1e-6)
        assert not np.allclose(pf.psinorm_for("te", native=True),
                               pf.psinorm_for("te"))

    def test_unmodified_file_round_trips_on_native_grids(self, tmp_path):
        """Reading and re-writing a provenance copy must not silently rewrite
        the radial grids."""
        path, _, g_te = self._multigrid_pfile(tmp_path)
        with pytest.warns(UserWarning):
            pf = read_pfile(path)
        back = PFile.from_bytes(pf.to_bytes(), remap=False)
        np.testing.assert_allclose(back.psinorm_for("te"), g_te, atol=1e-6)

    def test_modified_file_writes_the_modified_state(self, tmp_path):
        """A perturbed draw legitimately lives on the common grid."""
        path, g_ne, _ = self._multigrid_pfile(tmp_path)
        with pytest.warns(UserWarning):
            pf = read_pfile(path)
        pf.set_profile("te", pf.psinorm_for("te"),
                       np.asarray(pf.te, dtype=float) * 1.5)
        back = PFile.from_bytes(pf.to_bytes(), remap=False)
        np.testing.assert_allclose(back.psinorm_for("te"), g_ne, atol=1e-6)

    def test_quasineutrality_precondition_is_met_after_remap(self, tmp_path):
        """`ne - ni - nb` is index-wise; on mixed grids it is not well posed
        and can manufacture negative impurity density."""
        path, _, _ = self._multigrid_pfile(tmp_path)
        with pytest.warns(UserWarning):
            pf = read_pfile(path)
        np.testing.assert_allclose(pf.psinorm_for("ni"), pf.psinorm_for("ne"))
        pf.compute_quasineutrality()
        assert np.all(np.asarray(pf["nz1"]["data"], dtype=float) >= 0.0)

    def test_detached_remap_copy_is_a_full_pfile(self, tmp_path):
        """remap(overwrite=False) must return a usable object, not a shell."""
        path, _, g_te = self._multigrid_pfile(tmp_path)
        src = read_pfile(path, remap=False)
        out = src.remap()
        assert out is not src
        np.testing.assert_allclose(out.psinorm_for("te", native=True), g_te,
                                   atol=1e-6)
        assert isinstance(out.to_bytes(), bytes)
        np.testing.assert_allclose(src.psinorm_for("te"), g_te,
                                   atol=1e-6)                     # untouched
