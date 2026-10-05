"""read_imas_baseline converts FUSE currents exactly to TokaMaker jphi.

Synthetic FUSE-shaped dd: two equilibrium slices, core_profiles.j_tor built
from j_total with the EARLIER slice's geometry (A6, as FUSE does on a
time-dependent run), a bootstrap and a beam source.  Checks the slice pairing,
the exact conversions (docs/current-conventions.md A5-A7), that the components
sum to j_phi, the jphi_diff anchor, and the loud fallback for a dd without the
equilibrium flux-surface averages.
"""
import json
import warnings

import numpy as np
import pytest

from bouquet.config import ImasSource
from bouquet.io.imas import NBI_SOURCE_INDEX, read_imas_baseline
from bouquet.physics import (jpar_to_jphi_tokamaker, jphi_tokamaker_pressure_term,
                             jphi_tokamaker_to_jtor_imas, jtor_imas_to_jphi_tokamaker)

_EC = 1.602176634e-19
_N = 41
_RHO = np.linspace(0.0, 1.0, _N)            # shared by eq and cp: knots, no interp
_PSI = np.linspace(-2.4, -0.5, _N)          # COCOS 11, increasing outward
_B0 = [-1.92, -1.95]                        # per equilibrium slice


def _eq_geometry(shift):
    """FUSE-named equilibrium profiles; ``shift`` distinguishes the slices."""
    x = _RHO
    return {
        "rho_tor_norm": x.tolist(),
        "f": (-3.31 + 0.07 * x**2 + shift).tolist(),
        "gm8": (1.74 - 0.30 * x**2 + shift).tolist(),           # <R>
        "gm9": (0.573 + 0.13 * x**2 - shift / 10).tolist(),     # <1/R>
        "gm1": ((0.573 + 0.13 * x**2) ** 2 * (1 + 0.06 * x**2)).tolist(),
        "gm5": (3.60 + 1.9 * x**3 + shift).tolist(),            # <B^2>
        "dpressure_dpsi": (-3.0e4 * x * (1 - 0.5 * x)
                           - 2.0e4 * np.exp(-((x - 0.95) / 0.03) ** 2)).tolist(),
    }


def _geom(p1, b0):
    a = {k: np.asarray(v, float) for k, v in p1.items()}
    return {"F": a["f"], "avg_R": a["gm8"], "avg_inv_R": a["gm9"],
            "avg_inv_R2": a["gm1"], "avg_B2": a["gm5"],
            "pprime": -2 * np.pi * a["dpressure_dpsi"], "B0": b0}


def _dd(with_geometry=True):
    x = _RHO
    ne = 5e19 * (1 - 0.7 * x**2) + 1e18
    te = 3e3 * (1 - 0.9 * x**2) + 50
    ni, ti = 0.9 * ne, te.copy()
    nC = (ne - ni) / 6.0
    p_eq = _EC * (ne * te + ni * ti + nC * ti)
    j_ohm = 1.4e6 * (1 - x**2) ** 1.5 + 1e4                  # <J.B>/B0
    j_bs = 6e5 * np.exp(-((x - 0.93) / 0.05) ** 2) + 1e3
    j_nbi = 8e4 * np.exp(-(x / 0.4) ** 2)
    j_total = j_ohm + j_bs + j_nbi
    eq_p1 = [_eq_geometry(0.0), _eq_geometry(0.01)]
    paired = _geom(eq_p1[0], _B0[0])                          # the EARLIER slice
    j_tor = jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(j_total, paired) + jphi_tokamaker_pressure_term(paired),
        paired)                                               # A6, as FUSE writes it
    slices = []
    for k, t in enumerate((0.98, 1.0)):
        p1 = {"psi": _PSI.tolist(), "pressure": p_eq.tolist(),
              "j_tor": (j_tor * (1.0 + 0.02 * k * x)).tolist()}
        if with_geometry:
            p1.update(eq_p1[k])
        slices.append({"time": t, "profiles_1d": p1,
                       "global_quantities": {"ip": 1.3e6, "li_3": 0.9}})
    sp = lambda n_, t_, z: {"density_thermal": n_.tolist(), "temperature": t_.tolist(),
                            "element": [{"z_n": z}]}
    grid = {"psi": _PSI.tolist()}
    if with_geometry:
        grid["rho_tor_norm"] = x.tolist()
    return {
        "equilibrium": {"time": [0.98, 1.0],
                        "vacuum_toroidal_field": {"r0": 1.69, "b0": _B0},
                        "time_slice": slices},
        "core_profiles": {"time": [1.0], "profiles_1d": [{
            "time": 1.0, "grid": grid,
            "j_total": j_total.tolist(), "j_tor": j_tor.tolist(),
            "j_ohmic": j_ohm.tolist(), "j_bootstrap": j_bs.tolist(),
            "electrons": {"density_thermal": ne.tolist(), "temperature": te.tolist()},
            "ion": [sp(ni, ti, 1.0), sp(nC, ti, 6.0)]}]},
        "core_sources": {"time": [1.0], "source": [
            {"identifier": {"index": NBI_SOURCE_INDEX, "name": "beam"},
             "profiles_1d": [{"time": 1.0, "j_parallel": j_nbi.tolist()}]}]},
    }, dict(j_total=j_total, j_tor=j_tor, j_bs=j_bs, j_nbi=j_nbi)


def _read(tmp_path, dd):
    path = tmp_path / "dd.json"
    path.write_text(json.dumps(dd))
    return read_imas_baseline(ImasSource(ids_path=str(path), time=1.0),
                              p_fast_reduction="sum")


def test_reader_converts_exactly_on_the_paired_slice(tmp_path, capsys):
    dd, raw = _dd()
    bl = _read(tmp_path, dd)
    out = capsys.readouterr().out
    assert "equilibrium t=0.9800" in out                  # the earlier slice won
    g = _geom(dd["equilibrium"]["time_slice"][0]["profiles_1d"], _B0[0])
    pt = jphi_tokamaker_pressure_term(g)
    assert np.allclose(bl.j_phi, jtor_imas_to_jphi_tokamaker(raw["j_tor"], g), rtol=1e-12)
    assert np.allclose(bl.j_BS, jpar_to_jphi_tokamaker(raw["j_bs"], g) + pt, rtol=1e-12)
    assert np.allclose(bl.j_NBI, jpar_to_jphi_tokamaker(raw["j_nbi"], g), rtol=1e-12)
    # the total's own parallel current closes on j_phi (j_tor came from it)
    assert np.allclose(jpar_to_jphi_tokamaker(raw["j_total"], g) + pt, bl.j_phi,
                       rtol=1e-12)
    # J_TM != J_IMAS: the conversion is not vacuous on this geometry
    assert np.max(np.abs(bl.j_phi / raw["j_tor"] - 1)) > 1e-2


def test_reader_components_sum_to_j_phi(tmp_path):
    dd, _ = _dd()
    bl = _read(tmp_path, dd)
    total = bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF
    assert np.allclose(total, bl.j_phi, rtol=0, atol=1e-9 * np.max(np.abs(bl.j_phi)))
    # the inductive residual is the ohmic <J.B>'s field-aligned image
    g = _geom(dd["equilibrium"]["time_slice"][0]["profiles_1d"], _B0[0])
    j_ohm = np.asarray(dd["core_profiles"]["profiles_1d"][0]["j_ohmic"])
    assert np.allclose(bl.j_inductive, jpar_to_jphi_tokamaker(j_ohm, g), rtol=1e-10)


def test_jphi_diff_uses_the_anchor_slice_own_geometry(tmp_path):
    dd, _ = _dd()
    bl = _read(tmp_path, dd)
    p1 = dd["equilibrium"]["time_slice"][1]["profiles_1d"]   # nearest to T
    eq_jphi = jtor_imas_to_jphi_tokamaker(np.asarray(p1["j_tor"]), _geom(p1, _B0[1]))
    assert np.allclose(bl.jphi_diff, eq_jphi - bl.j_phi, rtol=1e-12, atol=1e-6)


def test_dd_without_geometry_falls_back_loudly(tmp_path):
    dd, raw = _dd(with_geometry=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = _read(tmp_path, dd)
    assert any("WITHOUT exact conversion" in str(x.message) for x in w)
    c = raw["j_tor"] / raw["j_total"]                         # legacy treatment
    assert np.allclose(bl.j_phi, raw["j_tor"])
    assert np.allclose(bl.j_BS, raw["j_bs"] * c)
    assert np.allclose(bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF, bl.j_phi)


def test_pairing_prefers_the_slice_that_reproduces_j_tor(tmp_path, capsys):
    # if j_tor came from the NEAREST slice instead, that slice is chosen
    dd, raw = _dd()
    g1 = _geom(dd["equilibrium"]["time_slice"][1]["profiles_1d"], _B0[1])
    jt = jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(raw["j_total"], g1) + jphi_tokamaker_pressure_term(g1), g1)
    dd["core_profiles"]["profiles_1d"][0]["j_tor"] = jt.tolist()
    bl = _read(tmp_path, dd)
    assert "equilibrium t=1.0000" in capsys.readouterr().out
    assert np.allclose(bl.j_phi, jtor_imas_to_jphi_tokamaker(jt, g1), rtol=1e-12)


if __name__ == "__main__":                                   # pragma: no cover
    pytest.main([__file__, "-q"])
