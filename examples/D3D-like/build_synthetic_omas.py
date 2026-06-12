#!/usr/bin/env python3
"""Build a committable, non-proprietary synthetic OMAS/IMAS baseline (L->H).

Produces ``D3Dlike_baseline_omas.json`` -- a minimal FUSE-style data dictionary
for exercising bouquet's ``ImasSource`` path, with a 3-slice **L->H transition**:

  * ne/Te/Ti for each slice come from the EPED-like pedestal-evolution model
    (SPARC ``lh_pedestal_evolution``) seeded with our synthetic H-mode target,
  * each slice is reconstructed on the SAME g-file (fixed Ip + boundary), so the
    bootstrap is recomputed self-consistently -- as the pedestal grows the
    bootstrap grows and the inductive current shrinks to hold Ip fixed,
  * synthetic extra profiles are added at the verified IMAS locations so the
    switchboard perturbation path has something to read: toroidal rotation
    (ion.rotation_frequency_tor), radial E-field (e_field.radial), Zeff
    (profiles_1d.zeff), and thermal diffusivities chi_e/chi_i (core_transport).
    (Production FUSE files carry only rotation; chi/E_r are supplied manually.)

Run with ``bouquet`` (OFT) and the SPARC lh_pedestal_evolution importable:
    PYTHONPATH=$OFT:.:/Users/.../SPARC python examples/D3D-like/build_synthetic_omas.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# local dev helper: the EPED-like model lives in the SPARC tree
sys.path.append("/Users/danielburgess/Desktop/plasma/SPARC")

import bouquet as bq
from bouquet.io.geqdsk import read_geqdsk
from bouquet.io.pfile import read_pfile
from bouquet.TokaMaker_interface import reconstruct_equilibrium
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
from lh_pedestal_evolution.pedestal_evolution import evolve_lh_transition

EC = 1.602176634e-19
Z_CARBON = 6.0
HERE = os.path.dirname(os.path.abspath(__file__))
GEQDSK = os.path.join(HERE, "D3Dlike_Hmode_baseline.geqdsk")
PFILE = os.path.join(HERE, "D3Dlike_Hmode_baseline.peqdsk")
MESH = os.path.join(HERE, "DIIID_mesh.h5")
OUT = os.path.join(HERE, "D3Dlike_baseline_omas.json")

TIMES = [2.10, 2.20, 2.3043]   # L-mode -> intermediate -> H-mode


# --- generic dimensionless current shapes (smooth fits, no scaffold data) ------
def c_par_to_tor(x):       # j_tor / j_total
    return 1.02 - 0.12 * x ** 1.6


def nbi_frac(x):           # beam-driven j / total (carved from ohmic)
    return 0.038 * np.exp(-(x / 0.70) ** 2) + 0.003


def pfast_frac(x):         # isotropic fast-ion pressure / thermal pressure
    return 0.08 * np.exp(-(x / 0.60) ** 2) + 0.03


# --- synthetic EXTRA profiles (evolve L->H via frac in [0,1]) -------------------
def omega_tor(x, frac):
    """Toroidal rotation [rad/s], core-peaked, rising with the transition."""
    return (6.0e4 + 5.0e4 * frac) * (1.0 - 0.9 * x ** 2)


def e_r_radial(x, frac):
    """Radial E-field [V/m]: weak core gradient + an edge well deepening L->H."""
    return 4.0e3 * x - (1.0e4 + 4.5e4 * frac) * np.exp(-((x - 0.95) / 0.035) ** 2)


def chi(x, frac, ion=False):
    """Thermal diffusivity [m^2/s]: rises to the edge in L-mode, with an edge
    transport barrier (suppression near psi_N~0.95) forming as L->H."""
    base = (0.5 + 2.5 * x ** 2) * (0.7 if ion else 1.0)
    barrier = 1.0 - 0.85 * frac * np.exp(-((x - 0.95) / 0.05) ** 2)
    return base * barrier


def main():
    eq = read_geqdsk(GEQDSK, cocos=1)
    psi_N = np.asarray(eq.psi_N, dtype=float)

    pf = read_pfile(PFILE)
    pf.compute_quasineutrality()
    ppf = np.asarray(pf.psinorm_for("ne"), dtype=float)
    ne_H = np.interp(psi_N, ppf, pf.ne) * 1e20
    te_H = np.interp(psi_N, ppf, pf.te) * 1e3
    ti_H = np.interp(psi_N, ppf, pf.ti) * 1e3
    ni_ratio = np.interp(psi_N, ppf, np.asarray(pf.ni, float) / np.asarray(pf.ne, float))
    _, zeff_pf = pf.compute_zeff()
    Zeff = np.clip(np.interp(psi_N, ppf, zeff_pf), 1.0, None)

    # EPED-like L->H sequence (target = our synthetic H-mode)
    scan = evolve_lh_transition(psi_N, ne=ne_H, te=te_H, ti=ti_H,
                                n_steps=len(TIMES), n_ne_steps=2, width_growth=0.5)

    # one solver for all slices
    b = bq.Bouquet(bq.BouquetConfig(
        source=bq.ReconstructionSource(geqdsk_path=GEQDSK, profiles_path=PFILE,
                                       cocos=1, psi_pad=1e-3),
        solver=bq.SolverConfig(mesh_path=MESH, nthreads=4, order=3),
        output_header="/tmp/_lh_builder")).setup_solver()
    iso = np.column_stack([eq.boundary_R, eq.boundary_Z])
    isow = np.ones(len(iso)) * 200.0
    guess = create_power_flux_fun(len(psi_N), 1.5, 1.5)["y"]

    c = c_par_to_tor(psi_N)
    nbi_shape = nbi_frac(psi_N)
    pf_shape = pfast_frac(psi_N)
    li1 = float(eq.li.get("li(1)", eq.li.get("li(1)_EFIT")))
    li2, li3 = float(eq.li["li(2)"]), float(eq.li["li(3)"])
    bnd_r = np.asarray(eq.boundary_R, dtype=float).tolist()
    bnd_z = np.asarray(eq.boundary_Z, dtype=float).tolist()
    psi_list, rho = psi_N.tolist(), np.sqrt(psi_N).tolist()

    eq_slices, cp_slices, src_slices, ct_slices = [], [], [], []
    for k, t in enumerate(TIMES):
        frac = k / (len(TIMES) - 1)                # 0 (L) -> 1 (H)
        ne = np.asarray(scan.ne[k], dtype=float)
        te = np.asarray(scan.te[k], dtype=float)
        ti = np.asarray(scan.ti[k], dtype=float)
        ni = ne * ni_ratio
        nC = np.clip((ne - ni) / Z_CARBON, 0.0, None)

        # reconstruct this slice on the fixed g-file -> self-consistent currents
        res = reconstruct_equilibrium(
            b.mygs, eq, ne, te, ni, ti, Zeff, iso, isow, 1e-3,
            guess_jinductive=guess, n_k=5, psi_bridge=0.99,
            rescale_j_BS=False, shelf_psi_N=0.0, initialize_psi=True)
        j_tor = np.asarray(res["j_phi_fit"], dtype=float)
        j_ind = np.asarray(res["j_inductive_fit"], dtype=float)
        j_bs = np.asarray(res["j_BS_used"], dtype=float)

        # carve a small NBI out of the inductive; store as IMAS parallel quantities
        j_nbi_tor = nbi_shape * j_tor
        j_total = j_tor / c
        j_bootstrap = j_bs / c
        j_NBI = j_nbi_tor / c
        j_ohmic = (j_ind - j_nbi_tor) / c
        j_non_inductive = j_bootstrap + j_NBI

        p_fast = pf_shape * EC * (ne * te + ni * ti + nC * ti)
        om = omega_tor(psi_N, frac)
        er = e_r_radial(psi_N, frac)
        chie = chi(psi_N, frac, ion=False)
        chii = chi(psi_N, frac, ion=True)

        eq_slices.append({
            "time": t,
            "global_quantities": {"ip": float(eq.Ip), "li_1": li1, "li_2": li2, "li_3": li3},
            "boundary": {"outline": {"r": bnd_r, "z": bnd_z}},
        })
        cp_slices.append({
            "time": t,
            "grid": {"psi": psi_list, "rho_tor_norm": rho},
            "j_tor": j_tor.tolist(), "j_total": j_total.tolist(),
            "j_ohmic": j_ohmic.tolist(), "j_bootstrap": j_bootstrap.tolist(),
            "j_non_inductive": j_non_inductive.tolist(),
            "zeff": Zeff.tolist(),
            "e_field": {"radial": er.tolist()},
            "electrons": {"density_thermal": ne.tolist(), "temperature": te.tolist(),
                          "pressure_fast_perpendicular": np.zeros_like(psi_N).tolist(),
                          "pressure_fast_parallel": np.zeros_like(psi_N).tolist()},
            "ion": [
                {"label": "D", "element": [{"z_n": 1.0, "a": 2.0}],
                 "density_thermal": ni.tolist(), "temperature": ti.tolist(),
                 "rotation_frequency_tor": om.tolist(),
                 "pressure_fast_perpendicular": p_fast.tolist(),
                 "pressure_fast_parallel": p_fast.tolist()},
                {"label": "C12", "element": [{"z_n": 6.0, "a": 12.0}],
                 "density_thermal": nC.tolist(), "temperature": ti.tolist(),
                 "pressure_fast_perpendicular": np.zeros_like(psi_N).tolist(),
                 "pressure_fast_parallel": np.zeros_like(psi_N).tolist()},
            ],
        })
        src_slices.append({"j_parallel": j_NBI.tolist()})
        ct_slices.append({
            "time": t,
            "grid": {"rho_tor_norm": rho},
            "electrons": {"energy": {"d": chie.tolist()}},
            "total_ion_energy": {"d": chii.tolist()},
        })
        print(f"  t={t:.4f} ({'L' if k==0 else 'H' if k==len(TIMES)-1 else 'mid'}): "
              f"Ip={abs(b.mygs.get_globals()[0])/1e6:.3f}MA  jBS_max={np.max(j_bs):.2e}  "
              f"jind0={j_ind[0]:.2e}  Te0={te[0]/1e3:.2f}keV")

    dd = {
        "equilibrium": {
            "time": list(TIMES),
            "vacuum_toroidal_field": {"r0": float(eq.R_center),
                                      "b0": [float(eq.B_center)] * len(TIMES)},
            "time_slice": eq_slices,
        },
        "core_profiles": {"time": list(TIMES), "profiles_1d": cp_slices},
        "core_sources": {"time": list(TIMES),
                         "source": [{"identifier": {"name": "nbi_synthetic", "index": 2},
                                     "profiles_1d": src_slices}]},
        "core_transport": {"time": list(TIMES),
                           "model": [{"identifier": {"name": "anomalous", "index": 1},
                                      "profiles_1d": ct_slices}]},
    }
    with open(OUT, "w") as fh:
        json.dump(dd, fh)
    print(f"wrote {OUT} with {len(TIMES)} L->H slices")
    print(f"  extras: ion.rotation_frequency_tor, e_field.radial, zeff, "
          f"core_transport.electrons.energy.d (chi_e), total_ion_energy.d (chi_i)")


if __name__ == "__main__":
    main()
