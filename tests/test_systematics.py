"""Backend systematics regression test (the fast analog of the systematics
notebook's Mode 1).

Runs a small ``pin_jphi=True``, ``sigma=0`` bouquet from the shipped D3D-like
baseline and asserts every draw reproduces the baseline -- boundary RMS ~0 mm
and coil drift ~0 % -- so a future change that subtly introduces a systematic
offset/bias trips this test.

It needs the live GS solver (OpenFUSIONToolkit) + the DIII-D mesh, so it runs
**by default whenever those are available** and skips gracefully otherwise.
It is marked ``solver`` so the fast unit loop can deselect it with
``pytest -m "not solver"`` (it takes a few minutes: one reconstruction + a
couple of solves).
"""
import os

import numpy as np
import pytest

# ---- availability gate (the solver + mesh + baseline must be present) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
# Self-contained: use the in-repo example baseline/mesh (the mesh is a local,
# gitignored artifact placed alongside the shipped baseline g/p-files).
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")

_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))


def _oft_importable():
    """OpenFUSIONToolkit must be importable.  Resolution order (no
    user-specific absolute paths):
      1. already on PYTHONPATH / pip-installed,
      2. the ``OFT_PYTHONPATH`` env var, if set,
      3. a sibling ``OpenFUSIONToolkit/build_release/python`` checkout
         (the standard dev layout).
    """
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
        reason="solver systematics test needs OFT + the D3D-like mesh/baseline "
               "(mesh is a local artifact); skipped when unavailable"),
]

# Thresholds chosen from the observed pinned-σ=0 floor (very stable across
# draws): boundary RMS ~0.525 mm, max coil drift ~0.054 %.  These are the
# small, CONSTANT jphi-linterp representation residual -- not a bias.  The
# limits sit modestly above that floor so a future change that subtly
# introduces a systematic (extra ~0.3 mm / ~0.25 %) trips the test, while
# leaving headroom for normal solver/build variation.
_BND_RMS_MAX_MM = 0.8
_COIL_DRIFT_MAX_PCT = 0.3
_N_DRAWS = 2


@pytest.fixture(scope="module")
def pinned_sigma0_run(tmp_path_factory):
    """Reconstruct the baseline and run a 2-draw pinned, sigma=0 bouquet."""
    from scipy.interpolate import interp1d
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
    import bouquet
    from bouquet import (read_geqdsk, reconstruct_equilibrium,
                         generate_bouquet, initialize_equilibrium_database)
    from bouquet.io.pfile import read_pfile

    work = tmp_path_factory.mktemp("systematics")
    header = str(work / "sys_pinned_s0")

    eqdsk = read_geqdsk(_GEQ, cocos=1)
    psi_N = eqdsk.psi_N
    F0 = abs(eqdsk.R_center * eqdsk.B_center)
    pad_psi = 1e-4

    pf = read_pfile(_PF)
    if pf.ion_species is None:
        pf.set_ion_species(N=[6, 1, 1], Z=[6, 1, 1], A=[12, 2, 2])
    pf.compute_quasineutrality()
    psi_pf, Zeff = pf.compute_zeff()
    ne_SI = interp1d(psi_pf, pf.ne*1e20, fill_value='extrapolate')(psi_N)
    te_SI = interp1d(psi_pf, pf.te*1e3,  fill_value='extrapolate')(psi_N)
    ni_SI = interp1d(psi_pf, pf.ni*1e20, fill_value='extrapolate')(psi_N)
    ti_SI = interp1d(psi_pf, pf.ti*1e3,  fill_value='extrapolate')(psi_N)
    Zeff_eq = np.clip(interp1d(psi_pf, Zeff, fill_value='extrapolate')(psi_N),
                      1.0, None)

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

    guess = create_power_flux_fun(len(psi_N), 1.5, 1.5)['y']
    result = reconstruct_equilibrium(
        mygs, eqdsk, ne_SI, te_SI, ni_SI, ti_SI, Zeff_eq, iso, isow, pad_psi,
        guess_jinductive=guess, n_k=5, psi_bridge=0.99, rescale_j_BS=False,
        shelf_psi_N=0.0, initialize_psi=True)
    Ip_target = abs(eqdsk.Ip)
    l_i_target = float(mygs.get_stats(lcfs_pad=pad_psi,
                                      li_normalization='std')['l_i'])

    ne_kin = pf.ne*1e20
    z = np.zeros_like(psi_pf)
    zj = np.zeros_like(result['j_phi_fit'])

    initialize_equilibrium_database(header)
    with open(_GEQ, 'rb') as fh:
        geq_raw = fh.read()
    with open(_PF, 'rb') as fh:
        pf_raw = fh.read()
    mygs.set_isoflux(result['isoflux_pts'], weights=result['weights'])

    generate_bouquet(
        mygs, psi_N, _N_DRAWS, header, result['j_phi_fit'],
        ne_kin, pf.te*1e3, pf.ni*1e20, pf.ti*1e3,
        z, z, z, z, zj,                       # sigma = 0 everywhere
        0.5, 0.4, 0.25, Ip_target, l_i_target, Zeff_eq,
        input_jinductive=result['j_inductive_fit'],
        l_i_tolerance=0.05, psi_pad=pad_psi,
        constrain_sawteeth=False, recalculate_j_BS=True,
        pfile_bytes=pf_raw, baseline_eqdsk_bytes=geq_raw,
        baseline_pfile_bytes=pf_raw,
        diagnostic_plots=False, scan_val=0, psi_N_kinetic=psi_pf,
        coil_drift=0.01,
        homotopy_passes=[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)],
        inspec_F_max=0.02, inspec_VSC_max=0.02, p_thresh=0.05,
        save_truncate_eq=True, jphi_baseline=True, seed=12345,
        pin_jphi=True,
    )
    return header + ".h5"


def test_pinned_sigma0_boundary_is_baseline(pinned_sigma0_run):
    """σ=0 pinned: every draw's LCFS must sit on the baseline (~0 mm RMS)."""
    import h5py
    from scipy.spatial import cKDTree
    with h5py.File(pinned_sigma0_run, "r") as hf:
        g = hf["scan/0"]
        ref = np.asarray(g["_baseline"]["recon_lcfs_ref"][()])
        draws = sorted(int(k) for k in g if k.isdigit())
        assert draws, "no draws were archived"
        for i in draws:
            gi = g[str(i)]
            assert "perturbed_lcfs_ref" in gi
            p = np.asarray(gi["perturbed_lcfs_ref"][()])
            d, _ = cKDTree(p).query(ref)
            rms = np.sqrt((d**2).mean()) * 1e3
            print(f"[systematics] pinned σ=0 draw {i}: boundary RMS = "
                  f"{rms:.4f} mm (limit {_BND_RMS_MAX_MM})")
            assert rms < _BND_RMS_MAX_MM, (
                f"draw {i}: boundary RMS {rms:.3f} mm exceeds "
                f"{_BND_RMS_MAX_MM} mm -- a systematic offset crept in")


def test_pinned_sigma0_coils_are_baseline(pinned_sigma0_run):
    """σ=0 pinned: coil currents must not drift from the baseline (~0 %)."""
    import json
    import h5py
    with h5py.File(pinned_sigma0_run, "r") as hf:
        g = hf["scan/0"]
        bl = g["_baseline"]
        names = [(n.decode() if isinstance(n, bytes) else str(n))
                 for n in bl["coil_names"][()]]
        base = dict(zip(names, np.asarray(bl["coil_currents [A]"])))
        draws = sorted(int(k) for k in g if k.isdigit())
        maxd = 0.0
        for i in draws:
            gi = g[str(i)]
            nm = json.loads(gi.attrs["coil_names"])
            v = np.asarray(gi["coil_currents [A]"])
            cur = dict(zip(nm, v))
            for c in names:
                if abs(base[c]) < 1.0:
                    continue
                drift = 100.0 * abs(cur[c] - base[c]) / abs(base[c])
                maxd = max(maxd, drift)
                assert drift < _COIL_DRIFT_MAX_PCT, (
                    f"draw {i} coil {c}: drift {drift:.3f}% exceeds "
                    f"{_COIL_DRIFT_MAX_PCT}% -- a systematic bias crept in")
        print(f"[systematics] pinned σ=0 max coil drift = {maxd:.4f}% "
              f"(limit {_COIL_DRIFT_MAX_PCT})")
