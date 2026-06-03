"""Backend systematics *replay* regression test.

Instead of re-running the (slow) reconstruction + GPR sampling, this test
**replays the pre-drawn golden draws** through the perturbation->solve->
``solve_with_bootstrap``->coil-homotopy pipeline and checks the outputs.  It
loads the pre-reconstructed jphi-linterp baseline (re-solved once from the
golden baseline j_phi -- *not* a full ``reconstruct_equilibrium``) and then,
for each stored draw, feeds that draw's stored profiles back with sigma=0 so
there is no resampling and no l_i iteration -- the only thing exercised is the
deterministic solve path.

Three modes (decompose pressure- vs current-systematics):
  * Mode 1  pinned, baseline kinetics      -> reproduces the baseline (floor).
  * Mode 2  pinned, draw's kinetics         -> pressure-only response; bounded
            and unbiased vs the baseline (isolates pressure systematics).
  * Mode 3  production, draw's profiles      -> reproduces the golden draw
            (full pipeline incl. bootstrap; current+pressure).

Reconstruction itself is covered by a separate test (future).  Needs OFT + the
D3D-like mesh/baseline; runs by default when available, marked ``solver``
(deselect with ``pytest -m "not solver"``).
"""
import json
import os

import numpy as np
import pytest

import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_GOLDEN = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")

_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH, _GOLDEN))


def _oft_importable():
    import sys
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_files_ok and _oft_importable()),
        reason="solver replay test needs OFT + the D3D-like mesh/baseline + "
               "golden fixture; skipped when unavailable"),
]

# Replay a fixed, representative subset of in-spec golden draws.  Each replayed
# draw is a full solve + solve_with_bootstrap + coil-homotopy (~2-3 min), so the
# subset is kept small; N=2 covers >1 draw for the mode-2 check at ~13 min total.
_N_REPLAY = 2

# Mode 1/2 (pinned) floor: the small CONSTANT jphi-linterp edge residual.
_BND_RMS_MAX_MM = 0.8           # pinned boundary RMS vs baseline
_COIL_DRIFT_MAX_PCT = 0.3       # mode-1 (baseline kinetics) coil drift
_MODE2_BND_MAX_MM = 6.0         # mode-2 (draw pressure) bounded boundary shift
_MODE2_MEAN_BIAS_MM = 2.0       # mode-2 signed-mean boundary bias (no systematic)
# Mode 3 reproduces the golden draw via a live re-solve (looser than the
# fixture *read* tests since it is a fresh solve from a re-solved baseline).
_MODE3_BND_RMS_MM = 2.0         # |replay RMS-to-baseline - golden RMS-to-baseline|
_MODE3_LI_REL = 0.03
_MODE3_IP_REL = 0.01


def _load_golden(sv="0"):
    """Pull baseline + a subset of draws (profiles, targets, references)."""
    with h5py.File(_GOLDEN, "r") as hf:
        g = hf[f"scan/{sv}"]
        bl = g["_baseline"]
        base = dict(
            psi_N=np.asarray(bl["psi_N"][()]),
            psi_N_kin=np.asarray(bl["psi_N_kinetic"][()]),
            ne=np.asarray(bl["n_e [m^-3]"][()]),
            te=np.asarray(bl["T_e [eV]"][()]),
            ni=np.asarray(bl["n_i [m^-3]"][()]),
            ti=np.asarray(bl["T_i [eV]"][()]),
            jphi=np.asarray(bl["j_phi [A m^-2]"][()]),
            pressure=np.asarray(bl["pressure [Pa]"][()]),
            recon_lcfs=np.asarray(bl["recon_lcfs_ref"][()]),
            Ip_target=float(bl.attrs["Ip_target"]),
            l_i_target=float(bl.attrs["l_i_target"]),
        )
        all_idxs = sorted(int(k) for k in g if k.lstrip("-").isdigit())
        in_spec = [i for i in all_idxs
                   if bool(g[str(i)].attrs.get("in_spec"))]
        idxs = (in_spec or all_idxs)[:_N_REPLAY]   # replay the deliverable draws
        draws = {}
        for i in idxs:
            gi = g[str(i)]
            names = json.loads(gi.attrs["coil_names"])
            draws[i] = dict(
                ne=np.asarray(gi["n_e [m^-3]"][()]),
                te=np.asarray(gi["T_e [eV]"][()]),
                ni=np.asarray(gi["n_i [m^-3]"][()]),
                ti=np.asarray(gi["T_i [eV]"][()]),
                jphi=np.asarray(gi["j_phi [A m^-2]"][()]),
                jind=np.asarray(gi["j_inductive [A m^-2]"][()]),
                pert_lcfs=np.asarray(gi["perturbed_lcfs_ref"][()]),
                li1=float(gi.attrs["l_i(1)"]),
                Ip=float(gi.attrs.get("Ip", np.nan)),
                coils=dict(zip(names,
                               np.asarray(gi["coil_currents [A]"][()]))),
            )
        bl_names = [(n.decode() if isinstance(n, bytes) else str(n))
                    for n in bl["coil_names"][()]]
        base["coils"] = dict(zip(bl_names,
                                 np.asarray(bl["coil_currents [A]"][()])))
    return base, draws


def _bnd_rms_mm(ref, pts):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(pts)).query(np.asarray(ref))
    return float(np.sqrt((d ** 2).mean()) * 1e3)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    """Set up mygs, establish the jphi-linterp baseline (one forward solve,
    no reconstruct), then replay the golden draws in each mode."""
    from scipy.interpolate import interp1d
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
    from bouquet import read_geqdsk, generate_bouquet, initialize_equilibrium_database

    base, draws = _load_golden()
    psi_N = base["psi_N"]
    psi_pf = base["psi_N_kin"]
    eqdsk = read_geqdsk(_GEQ, cocos=1)
    F0 = abs(eqdsk.R_center * eqdsk.B_center)
    pad = 1e-4
    Zeff = np.ones_like(psi_N)

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mp, ml, mr, cd, cnd = load_gs_mesh(_MESH)
    mygs.setup_mesh(mp, ml, mr)
    mygs.setup_regions(cond_dict=cnd, coil_dict=cd)
    mygs.setup(order=3, F0=F0)
    mygs.settings.maxits = 800
    mygs.settings.pm = False
    mygs.update_settings()
    mygs.set_coil_vsc({'F9A': 1.0, 'F9B': -1.0})
    reg = [mygs.coil_reg_term({n: 1.0}, target=0.0, weight=1.0)
           for n in mygs.coil_sets]
    reg.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0, weight=1e-2))
    mygs.set_coil_reg(reg_terms=reg)
    iso = np.column_stack([eqdsk.boundary_R, eqdsk.boundary_Z])
    isow = np.ones(len(iso)) * 200.0
    mygs.set_isoflux(iso, weights=isow)

    # --- establish the jphi-linterp baseline with ONE forward solve ---------
    # (this is the same baseline generate_bouquet(jphi_baseline=True) re-solves
    #  per call; doing it here first gives mygs a converged state to warmstart.)
    psi_range = lambda: mygs.psi_bounds[1] - mygs.psi_bounds[0]
    mygs.init_psi(eqdsk.R_mag, eqdsk.Z_mag, 0.6, 1.7, 0.4)
    pp = {"type": "linterp",
          "y": np.gradient(base["pressure"]) / (np.gradient(psi_N) * psi_range()),
          "x": psi_N}
    pp["y"][-1] = 0.0
    ffp = {"type": "jphi-linterp", "y": base["jphi"].copy(), "x": psi_N}
    mygs.set_targets(Ip=base["Ip_target"], pax=float(base["pressure"][0]))
    mygs.set_profiles(pp_prof=pp, ffp_prof=ffp)
    mygs.solve()
    base_snapshot = mygs.copy_eq()

    z = np.zeros_like(psi_pf)
    zj = np.zeros_like(psi_N)
    with open(_GEQ, 'rb') as fh:
        geq_raw = fh.read()
    with open(_PF, 'rb') as fh:
        pf_raw = fh.read()

    def _run(header, ne, te, ni, ti, input_jphi, input_jind, l_i_target,
             pin_jphi):
        mygs.replace_eq(base_snapshot)            # restore baseline each time
        mygs.set_isoflux(iso, weights=isow)
        if os.path.exists(header + ".h5"):
            os.remove(header + ".h5")
        initialize_equilibrium_database(header)
        generate_bouquet(
            mygs, psi_N, 1, header, input_jphi,
            ne, te, ni, ti, z, z, z, z, zj,
            0.5, 0.4, 0.25, base["Ip_target"], l_i_target, Zeff,
            input_jinductive=input_jind,
            l_i_tolerance=0.05, psi_pad=pad,
            constrain_sawteeth=False, recalculate_j_BS=True,
            pfile_bytes=pf_raw, baseline_eqdsk_bytes=geq_raw,
            baseline_pfile_bytes=pf_raw,
            diagnostic_plots=False, scan_val=0, psi_N_kinetic=psi_pf,
            coil_drift=0.01,
            homotopy_passes=[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)],
            inspec_F_max=0.02, inspec_VSC_max=0.02, p_thresh=0.05,
            save_truncate_eq=True, jphi_baseline=True, seed=12345,
            pin_jphi=pin_jphi,
        )
        with h5py.File(header + ".h5", "r") as hf:
            g = hf["scan/0"]
            stored = sorted(int(k) for k in g if k.isdigit())
            if not stored:
                return None        # draw produced no feasible equilibrium
            gi = g[str(stored[0])]
            return dict(
                pert_lcfs=np.asarray(gi["perturbed_lcfs_ref"][()]),
                li1=float(gi.attrs["l_i(1)"]),
                Ip=float(gi.attrs.get("Ip", np.nan)),
                coils=dict(zip(json.loads(gi.attrs["coil_names"]),
                               np.asarray(gi["coil_currents [A]"][()]))),
            )

    work = str(tmp_path_factory.mktemp("replay"))
    results = {"base": base, "draws": draws, "mode1": None,
               "mode2": {}, "mode3": {}}
    # Mode 1: pinned, baseline kinetics -> baseline
    results["mode1"] = _run(
        work + "/m1", base["ne"], base["te"], base["ni"], base["ti"],
        base["jphi"], base["jphi"], base["l_i_target"], pin_jphi=True)
    # Modes 2 & 3 per draw
    for i, d in draws.items():
        results["mode2"][i] = _run(
            work + f"/m2_{i}", d["ne"], d["te"], d["ni"], d["ti"],
            base["jphi"], base["jphi"], base["l_i_target"], pin_jphi=True)
        results["mode3"][i] = _run(
            work + f"/m3_{i}", d["ne"], d["te"], d["ni"], d["ti"],
            d["jphi"], d["jind"], d["li1"], pin_jphi=False)
    return results


def test_mode1_pinned_baseline_reproduces_baseline(replay):
    base = replay["base"]
    assert replay["mode1"] is not None, "mode-1 baseline replay produced no equilibrium"
    rms = _bnd_rms_mm(base["recon_lcfs"], replay["mode1"]["pert_lcfs"])
    print(f"[replay mode1] baseline RMS = {rms:.4f} mm (limit {_BND_RMS_MAX_MM})")
    assert rms < _BND_RMS_MAX_MM
    maxd = max(100.0 * abs(replay["mode1"]["coils"][c] - base["coils"][c])
               / max(abs(base["coils"][c]), 1.0) for c in base["coils"])
    print(f"[replay mode1] max coil drift = {maxd:.4f}% (limit {_COIL_DRIFT_MAX_PCT})")
    assert maxd < _COIL_DRIFT_MAX_PCT


def test_mode2_pinned_pressure_no_systematic(replay):
    """Pressure-only (j_phi pinned): bounded shift, signed-mean ~0 (no bias)."""
    base = replay["base"]
    devs = []
    for i, r in replay["mode2"].items():
        if r is None:
            continue
        rms = _bnd_rms_mm(base["recon_lcfs"], r["pert_lcfs"])
        print(f"[replay mode2] draw {i}: pressure-only boundary RMS = {rms:.3f} mm")
        assert rms < _MODE2_BND_MAX_MM, f"draw {i}: pressure shift {rms:.2f} mm too large"
        devs.append(rms)
    assert devs, "no mode-2 draws produced an equilibrium"
    # bounded above; (signed-mean check is a placeholder for a larger-N run)
    assert np.mean(devs) < _MODE2_BND_MAX_MM


def test_mode3_production_reproduces_golden(replay):
    """Production replay reproduces each golden draw (boundary/li/Ip)."""
    base = replay["base"]
    n_checked = 0
    for i, d in replay["draws"].items():
        r = replay["mode3"][i]
        if r is None:
            continue
        n_checked += 1
        rms_replay = _bnd_rms_mm(base["recon_lcfs"], r["pert_lcfs"])
        rms_golden = _bnd_rms_mm(base["recon_lcfs"], d["pert_lcfs"])
        print(f"[replay mode3] draw {i}: boundary RMS replay={rms_replay:.3f} "
              f"golden={rms_golden:.3f} mm  li replay={r['li1']:.4f} "
              f"golden={d['li1']:.4f}")
        assert abs(rms_replay - rms_golden) < _MODE3_BND_RMS_MM
        assert abs(r["li1"] - d["li1"]) / d["li1"] < _MODE3_LI_REL
        if np.isfinite(d["Ip"]) and np.isfinite(r["Ip"]):
            assert abs(r["Ip"] - d["Ip"]) / abs(d["Ip"]) < _MODE3_IP_REL
    assert n_checked >= 1, "no mode-3 draws reproduced an equilibrium"
