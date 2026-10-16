"""ida_hybrid kinetics held on Φ_N vs ψ_N: where they land in real space.

IDA fits its profiles on one equilibrium's flux surfaces (``g_synthetic_pfast``);
bouquet re-solves with the dd's current profile, which differs from that
equilibrium's, so the surfaces move.  Profiles held on ψ_N move with them;
profiles held on Φ_N (the IDA file's own q map) stay tied to the geometry.

Fixtures (tests/data, see tests/data/README.md): ``dd_synthetic.json.gz``
(currents), ``diiid_profs_synthetic_pfast.cdf`` (kinetics, on g_synthetic_pfast's ψ_N
and q) and ``g_synthetic_pfast.geqdsk`` (the equilibrium IDA was fitted on; also the
LCFS).  The pair is the TOTAL-pressure g-file, the equilibrium the dd's slice
produces: the thermal-only ``g_synthetic.geqdsk`` is not, so on it the premise
(the dd re-solve reproduces the IDA geometry up to the current-profile change)
does not hold.

Each IDA node's surface on the run is compared with its surface on g_synthetic_pfast all
the way round: the area between the two over the perimeter (the area-averaged
normal displacement), so an outward axis shift that a reshaped current cancels on
the outboard side is still seen inboard.  Every solve runs in a subprocess
(tests/_harness.py).  Marked ``solver``; a demonstration, run only with
``BOUQUET_RUN_DEMOS=1``.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()



import numpy as np
import pytest

from test_phi_solver_draws import _oft_phi_ready

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic_pfast.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic_pfast.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))
_TIME = 1.200
_METHODS = ("legacy", "swb", "engine")
_NODES = (0.1, 0.98)                      # IDA ψ_N range compared

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(os.environ.get("BOUQUET_RUN_DEMOS") != "1",
                       reason="demonstration: BOUQUET_RUN_DEMOS=1 pytest -m solver "
                              "tests/test_phi_imas_solver.py"),
    pytest.mark.skipif(not (all(os.path.isfile(p) for p in (_DD, _IDA, _GEQ, _MESH))
                            and _oft_phi_ready()),
                       reason="needs OFT with toroidal-flux support"),
]


# ---------------------------------------------------------------------------
#  probes (subprocess)
# ---------------------------------------------------------------------------
_THETA = np.linspace(-np.pi, np.pi, 721)[:-1]   # poloidal angle about g_synthetic_pfast's axis


def _surfaces(geq, psi_N, axis):
    """r(_THETA) [m] about ``axis`` of the closed surfaces ``psi_N`` of a parsed g-file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.contour(geq.R_grid, geq.Z_grid, geq.psi_N_RZ, levels=np.unique(psi_N))
    plt.close(fig)
    r_of = {}
    for lev, segs in zip(cs.levels, cs.allsegs):
        for s in segs:
            th = np.arctan2(s[:, 1] - axis[1], s[:, 0] - axis[0])
            if len(s) > 20 and abs(np.sum(np.diff(np.unwrap(th)))) > 6.0:   # winds round the axis
                k = np.argsort(th)
                r_of[lev] = np.interp(_THETA, th[k], np.hypot(s[k, 0] - axis[0], s[k, 1] - axis[1]),
                                      period=2 * np.pi)
                break
    return np.array([r_of[v] for v in psi_N])


def _probe(work, coord, method):
    import warnings
    import bouquet as bq
    from bouquet import coords
    from bouquet.io.geqdsk import read_geqdsk
    from scipy.interpolate import RectBivariateSpline
    ddp = os.path.join(work, "dd.json")
    with gzip.open(_DD, "rb") as fi, open(ddp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    kw = ({"reconstruction_engine": "legacy"} if method == "legacy"
          else {"solve_method": method})
    run = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, n_draws=1, ida_path=_IDA,
                               LCFS_geqdsk=_GEQ, impurity_Z=6.0,
                               header=os.path.join(work, "run"), **kw)
    run.source.coord = coord
    run.config.generation.seed = 42
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    mygs = run.mygs
    gsolved = os.path.join(work, "solved.geqdsk")
    mygs.save_eqdsk(gsolved, nr=257, nz=257, lcfs_pad=1e-3)
    g_ref, g_run = read_geqdsk(_GEQ), read_geqdsk(gsolved)

    ida = bq.read_ida(_IDA, time=_TIME)
    psi_ida = np.asarray(ida.psi_N, float)
    inside, phi_ida = coords.phi_n_from_q(psi_ida, ida.q, bracket=True)
    psi_in = psi_ida[inside]
    sel = (psi_in >= _NODES[0]) & (psi_in <= _NODES[1])
    psi_nodes = psi_in[sel]
    # where the run holds each IDA node: at its ψ_N, or at the solved ψ_N of its Φ_N
    psi_run = (psi_nodes if coord == "psi_n"
               else coords.psi_at(mygs, phi_ida[sel], coords.PHI))
    # each node's surface all the way round, on g_synthetic_pfast and on the run, about
    # g_synthetic_pfast's axis: the area between them over the perimeter is the
    # area-averaged normal displacement
    axis = (g_ref.R_mag, g_ref.Z_mag)
    r_ref, r_run = _surfaces(g_ref, psi_nodes, axis), _surfaces(g_run, psi_run, axis)
    dth = _THETA[1] - _THETA[0]
    dl = np.hypot(r_ref, np.gradient(r_ref, dth, axis=1))      # arc length per radian
    dr = 1e3 * 0.5 * np.sum(np.abs(r_run ** 2 - r_ref ** 2), axis=1) / np.sum(dl, axis=1)
    i_out, i_in = np.argmin(np.abs(_THETA)), np.argmin(np.pi - np.abs(_THETA))
    # the run's own n_e at the points of each reference surface, against IDA's
    # value on it, averaged round the surface the same way
    R = axis[0] + r_ref * np.cos(_THETA)
    Z = axis[1] + r_ref * np.sin(_THETA)
    psi_pts = RectBivariateSpline(g_run.Z_grid, g_run.R_grid, g_run.psi_N_RZ).ev(Z, R)
    bl = run.baseline
    ne_run = np.interp(psi_pts, coords.psi_at(mygs, np.asarray(bl.psi_N, float), coord),
                       np.asarray(bl.ne, float))
    ne_ida = np.asarray(ida.ne, float)[inside][sel]
    ne_dev = np.sum(np.abs(ne_run - ne_ida[:, None]) * dl, axis=1) / np.sum(dl, axis=1)
    i = int(np.argmax(dr))
    out = {"max_dr_mm": float(dr[i]), "at_psi_N": float(psi_nodes[i]),
           "dr_mm": dr.tolist(), "psi_N": psi_nodes.tolist(),
           "dr_out_mm": (1e3 * (r_run - r_ref)[:, i_out]).tolist(),
           "dr_in_mm": (1e3 * (r_run - r_ref)[:, i_in]).tolist(),
           "ne_misfit": float(np.max(ne_dev) / np.max(ne_ida))}
    with open(os.path.join(work, "probe.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    env = _harness.subprocess_env(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                                  MPLBACKEND="Agg")
    keys = [(m, c) for m in _METHODS for c in ("psi_n", "phi_n")]
    works = {k: str(tmp_path_factory.mktemp(f"ida_{k[0]}_{k[1]}")) for k in keys}
    procs = {k: subprocess.Popen([sys.executable, os.path.abspath(__file__), works[k], k[1], k[0]],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in keys}
    out = {}
    for k, pr in procs.items():
        _, err = pr.communicate()
        path = os.path.join(works[k], "probe.json")
        out[k] = (json.load(open(path)) if pr.returncode == 0 and os.path.isfile(path)
                  else {"_error": f"rc={pr.returncode}\n{err[-4000:]}"})
    return out


def _pair(probes, method):
    p = {c: probes[(method, c)] for c in ("psi_n", "phi_n")}
    for c, v in p.items():
        if "_error" in v:
            pytest.fail(f"{method}/{c} probe failed: {v['_error']}")
    return p["psi_n"], p["phi_n"]


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", _METHODS)
def test_phi_held_kinetics_stay_nearer_their_ida_placement(probes, method):
    # area-averaged normal displacement of each IDA node's surface (inboard and
    # outboard together); measured on the 1.2 s pair (g_synthetic_pfast, 2026-10-16):
    # ψ_N-held up to 4.8-5.7 mm, Φ_N-held up to 0.8-1.1 mm
    psi, phi = _pair(probes, method)
    worse = [(x, a, b) for x, a, b in zip(psi["psi_N"], psi["dr_mm"], phi["dr_mm"]) if b >= a]
    assert not worse, f"Φ_N-held not nearer at (ψ_N, ψ-held mm, Φ-held mm): {worse[:5]}"
    i = int(np.argmax(phi["dr_mm"]))
    assert phi["max_dr_mm"] < 10.0, (
        f"Φ_N-held {phi['max_dr_mm']:.2f} mm at ψ_N {phi['at_psi_N']:.3f} (outboard "
        f"{phi['dr_out_mm'][i]:+.2f}, inboard {phi['dr_in_mm'][i]:+.2f} mm)")


@pytest.mark.parametrize("method", _METHODS)
def test_phi_held_ne_matches_ida_in_real_space(probes, method):
    psi, phi = _pair(probes, method)
    # n_e at the points of each IDA surface, averaged round it; measured on the
    # 1.2 s pair (g_synthetic_pfast, 2026-10-16): 0.58-0.69 % of peak (ψ_N-held) vs
    # 0.16-0.17 % (Φ_N-held)
    assert phi["ne_misfit"] < psi["ne_misfit"], (psi["ne_misfit"], phi["ne_misfit"])


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    _probe(*sys.argv[1:4])
