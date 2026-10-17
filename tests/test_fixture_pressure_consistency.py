"""Each synthetic g-file carries the pressure its paired profile file gives.

The g-file route rebuilds the reconstruction's pressure from the profile file:
e (n_e T_e + n_i T_i) from an IDA-layout file (n_i from Z_eff by
single-impurity quasineutrality; the file has no fast-ion channel), plus the
beam pressure ``pb`` from a p-file.  A g-file solved with a different pressure
(e.g. the total, fast ions included, against a thermal-only IDA file) makes
every reconstruction on the pair recover the wrong beta, so each pair is
checked here (tests/data/README.md, FIXTURE_SPEC.md sections 3 and 3b):

* ``g_synthetic.geqdsk`` + ``diiid_profs_synthetic.cdf``: thermal only;
* ``g_synthetic_pfast.geqdsk`` + ``p_synthetic.peqdsk``: thermal + pb;
* ``g_synthetic_lmode.geqdsk`` + ``ida_synthetic_lmode.cdf``: thermal (no beam).
"""
import os

import numpy as np
import pytest

from bouquet.io.geqdsk import read_geqdsk
from bouquet.io.pfile import read_pfile
from bouquet.physics import ELEMENTARY_CHARGE, main_ion_density_from_zeff

h5py = pytest.importorskip("h5py")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
Z_IMP = 6.0
#: the g-file's axis pressure vs the profile file's, relative
AXIS_TOL = 0.02

PAIRS = (("g_synthetic.geqdsk", "diiid_profs_synthetic.cdf"),
         ("g_synthetic_pfast.geqdsk", "p_synthetic.peqdsk"),
         ("g_synthetic_lmode.geqdsk", "ida_synthetic_lmode.cdf"))


def _ida(name):
    with h5py.File(os.path.join(DATA, name), "r") as f:
        return {k: np.asarray(f[k][()], float) for k in
                ("psi_n", "n_e", "T_e", "T_12C6", "Zeff", "q")}


def _profile_pressure(name):
    """``(psi_N, p [Pa])``: the pressure the g-file route reads from the file."""
    if name.endswith(".cdf"):
        d = _ida(name)
        ne, te, ti = d["n_e"][0], d["T_e"][0], d["T_12C6"][0]
        ni = main_ion_density_from_zeff(ne, np.clip(d["Zeff"][0], 1.0, Z_IMP), Z_IMP)
        return d["psi_n"], ELEMENTARY_CHARGE * (ne * te + ni * ti)
    pf = read_pfile(os.path.join(DATA, name))
    nt = (np.asarray(pf.ne) * np.asarray(pf.te) + np.asarray(pf.ni) * np.asarray(pf.ti))
    return (np.asarray(pf.psinorm_for("ne"), float),
            ELEMENTARY_CHARGE * 1e23 * nt + 1e3 * np.asarray(pf.pb, float))


@pytest.mark.parametrize("gfile, profiles", PAIRS)
def test_gfile_axis_pressure_is_its_profile_files(gfile, profiles):
    p_g = np.abs(np.asarray(read_geqdsk(os.path.join(DATA, gfile)).pres, float))
    x, p = _profile_pressure(profiles)
    assert x[0] == 0.0
    rel = p_g[0] / p[0] - 1.0
    assert abs(rel) < AXIS_TOL, (
        f"{gfile} axis pressure {p_g[0] / 1e3:.2f} kPa vs {profiles} "
        f"{p[0] / 1e3:.2f} kPa: {100 * rel:+.2f} % (tolerance {100 * AXIS_TOL:g} %)")


def test_the_total_pressure_gfile_needs_the_beam_pressure():
    """Without ``pb`` the p-file's thermal pressure is far below the total
    g-file's: the pair is consistent only with the fast ions."""
    p_g = abs(float(read_geqdsk(os.path.join(DATA, "g_synthetic_pfast.geqdsk")).pres[0]))
    pf = read_pfile(os.path.join(DATA, "p_synthetic.peqdsk"))
    thermal = ELEMENTARY_CHARGE * 1e23 * (pf.ne[0] * pf.te[0] + pf.ni[0] * pf.ti[0])
    assert pf.pb[0] > 0.0 and pf.pb[-1] == 0.0
    assert p_g / thermal - 1.0 > 0.5


@pytest.mark.parametrize("gfile, ida", (PAIRS[0], PAIRS[2]))
def test_ida_q_is_its_gfiles_q(gfile, ida):
    """The IDA file was fitted on its g-file: its q is that g-file's q
    (the writer's own interpolation, so exact)."""
    d = _ida(ida)
    qg = np.abs(np.asarray(read_geqdsk(os.path.join(DATA, gfile)).qpsi, float))
    q = np.interp(np.clip(d["psi_n"], 0.0, 1.0), np.linspace(0.0, 1.0, qg.size), qg)
    np.testing.assert_allclose(d["q"][0], q, rtol=1e-12, atol=0.0)


def test_pfile_and_ida_file_carry_the_same_slice():
    """p_synthetic.peqdsk and diiid_profs_synthetic.cdf are the same slice's
    kinetics: equal n_e, T_e, T_i and Z_eff at their common nodes (psi_N
    0, 0.01, ..., 0.89) to the p-file's six decimals."""
    d = _ida("diiid_profs_synthetic.cdf")
    pf = read_pfile(os.path.join(DATA, "p_synthetic.peqdsk"))
    pf.compute_quasineutrality()
    xp, zeff = pf.compute_zeff()
    xi = d["psi_n"]
    common = [(i, int(np.argmin(np.abs(xp - v)))) for i, v in enumerate(xi)
              if v < 0.895 and np.min(np.abs(xp - v)) < 1e-9]
    assert len(common) == 90
    ii, jj = map(np.asarray, zip(*common))
    np.testing.assert_allclose(np.asarray(pf.ne)[jj] * 1e20, d["n_e"][0][ii], rtol=0, atol=1e14)
    np.testing.assert_allclose(np.asarray(pf.te)[jj] * 1e3, d["T_e"][0][ii], rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(pf.ti)[jj] * 1e3, d["T_12C6"][0][ii], rtol=0, atol=1e-3)
    np.testing.assert_allclose(zeff[jj], d["Zeff"][0][ii], rtol=0, atol=1e-3)
