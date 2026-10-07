#!/usr/bin/env python
"""Measure the unified reconstruction engine with the live solver -- a
MEASUREMENT (pytest does not collect this file); ``tests/test_engine_solver.py``
runs it and asserts on its JSON.

For each synthetic example of the repository (``examples/D3D-like``: the
g-file + p-file, and the OMAS modelling source at t = 2.3043 s) the baseline
is built with ``reconstruction_engine="unified"``, each part in its own
interpreter (``OFT_env`` is a per-process singleton).  Per part it records:

* converged or not, passes per phase, the delivered-state checks (loop
  residuals, every row), the full engine record (per pass: coefficients,
  l_i, q, request - achieved, the uniform Ip factor);
* the GS solve count per stage and the wall time;
* the DISTANCE-TO-INPUT table, in the units of the three-state comparison
  report (g-file: l_i(3) matched and l_i(1) free, q on axis and at
  psi_N = 0.02, q95, the q-profile max/rms over psi_N 0.05-0.95, the
  achieved <j_phi> against the input core (psi_N < 0.8) / edge (>= 0.8)
  max/rms in % of the input's peak, beta_N, W_MHD, beta_p, the LCFS
  distance rms/max, requested - achieved core/edge; modelling source: the
  same against the IDS's own li_3, q profile, j_tor and boundary);
* the delivered state RE-SOLVED once from itself with its stored request:
  the change in l_i, q0, q95 and the achieved current.

Parts: ``recon``, ``imas`` (rows Ip + l_i), ``imas_q0`` (+ the q0 row, which
the source's sawtooth gate admits), ``recon_dc`` (the delivery correction
ON).  Usage::

    python tests/probes/measure_engine.py OUTDIR [--parts recon,imas,...]

Writes ``OUTDIR/engine_<part>.json`` per part (always, with the error in
place of the numbers when a stage raises) and ``OUTDIR/engine_measurement
.json`` (all parts).  Single-threaded (``nthreads=1``, every BLAS/OpenMP
count 1); no network.  OpenFUSIONToolkit is found as the solver tests find
it (``OFT_PYTHONPATH``, else the sibling checkout's
``build_release/python``).  ``BQ_ENGINE_PROBE_OUT=<dir>`` also copies every
part's JSON there.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import _harness  # noqa: E402

_harness.ensure_repo_on_syspath()

_EXAMPLE = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like")
_OMAS = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.3043

#: part -> (source, GenerationConfig fields set on top of the engine switch)
PARTS = {
    "recon": ("recon", dict()),
    "imas": ("imas", dict()),
    "imas_q0": ("imas", dict(engine_rows=["Ip", "l_i", "q0"])),
    "recon_dc": ("recon", dict(engine_delivery_correction=True)),
}


def oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
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


# ---------------------------------------------------------------------------
#  the child: one part in one interpreter
# ---------------------------------------------------------------------------
def _pct_stats(d, ref_peak, mask):
    import numpy as np
    v = 100.0 * np.asarray(d, float)[mask] / ref_peak
    return dict(max=float(np.max(np.abs(v))),
                rms=float(np.sqrt(np.mean(v ** 2))))


def _q_profile(mygs, psi):
    import numpy as np
    pq = np.ascontiguousarray(np.clip(np.asarray(psi, float), 1e-3,
                                      1.0 - 1e-3))
    return np.asarray(mygs.get_q(psi=pq)[1], dtype=float)


def _distance_gfile(b, bl, psi_pad):
    """The three-state comparison report's table for the engine state."""
    import numpy as np
    from bouquet.engine import _lcfs_deviation_mm
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.TokaMaker_interface import _corrective_output_jphi
    mygs = b.mygs
    eq = read_geqdsk(b.config.source.geqdsk_path, cocos=b.config.source.cocos)
    psi = np.asarray(eq.psi_N, float)
    jin = np.abs(np.asarray(eq.j_tor_averaged_direct, float))
    peak = float(np.max(jin))
    A = np.asarray(_corrective_output_jphi(mygs, psi, psi_pad), float)
    R = np.asarray(bl.j_phi, float)
    st_i = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    st_s = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="std")
    qin = np.abs(np.asarray(eq.qpsi, float))
    qp = np.abs(_q_profile(mygs, psi))
    m = (psi >= 0.05) & (psi <= 0.95)
    rel = qp[m] / qin[m] - 1.0
    core, edge = psi < 0.8, psi >= 0.8
    bnd = np.column_stack([eq.boundary_R, eq.boundary_Z])
    rms, mx = _lcfs_deviation_mm(mygs, bnd)
    betas = eq.betas
    W_in = 1.5 * float(eq.volume_integral(eq.pres)[-1]) / 1e6
    from bouquet.physics import SOLVER_Q0_PSI_N
    q002 = float(np.abs(_q_profile(mygs, [SOLVER_Q0_PSI_N, 0.5]))[0])
    q_in_002 = float(np.interp(SOLVER_Q0_PSI_N, psi, qin))
    return dict(
        li3_matched=dict(input=float(eq.li["li(2)"]),
                         engine=float(st_i["l_i"]),
                         delta=float(st_i["l_i"]) - float(eq.li["li(2)"])),
        li1_free=dict(input=float(eq.li["li(1)_EFIT"]),
                      engine=float(st_s["l_i"]),
                      delta=float(st_s["l_i"]) - float(eq.li["li(1)_EFIT"])),
        q_axis=dict(input=float(qin[0]), engine=float(qp[0]),
                    delta=float(qp[0] - qin[0]), input_psi_N=0.0,
                    engine_psi_N=1e-3,
                    note=("NOT like radii: the input's qpsi[0] is on axis, "
                          "the engine's sample is the psi_N = 1e-3 clip "
                          "(kept for the comparison report's row)")),
        q_002=dict(input=q_in_002, engine=q002, delta=q002 - q_in_002,
                   psi_N=float(SOLVER_Q0_PSI_N),
                   code_q0=float(st_i.get("q_0", float("nan"))),
                   note=("like radii (physics.SOLVER_Q0_PSI_N, get_stats' "
                         "q_0 radius), as reconstruction_metrics' q0")),
        q95=dict(input=float(np.interp(0.95, psi, qin)),
                 engine=float(st_i["q_95"]),
                 delta=float(st_i["q_95"]) - float(np.interp(0.95, psi,
                                                             qin))),
        q_profile_rel_pct=dict(max=100.0 * float(np.max(np.abs(rel))),
                               rms=100.0 * float(np.sqrt(np.mean(rel ** 2)))),
        jphi_vs_input_pct_of_peak=dict(core=_pct_stats(A - jin, peak, core),
                                       edge=_pct_stats(A - jin, peak, edge),
                                       edge_argmax_psiN=float(psi[edge][
                                           int(np.argmax(np.abs(
                                               (A - jin)[edge])))])),
        requested_minus_achieved_pct_of_peak=dict(
            core=_pct_stats(R - A, peak, core),
            edge=_pct_stats(R - A, peak, edge)),
        beta_n=dict(input=float(betas.get("beta_n", float("nan"))),
                    engine=float(st_i.get("beta_n", float("nan")))),
        W_MHD_MJ=dict(input=W_in,
                      engine=float(st_i.get("W_MHD", float("nan"))) / 1e6),
        beta_p=dict(input=float(betas.get("beta_p", float("nan"))),
                    engine=float(st_i.get("beta_pol", float("nan"))) / 100.0),
        lcfs_mm=dict(rms=rms, max=mx),
        achieved_form="TokaMaker_interface._corrective_output_jphi (the "
                      "comparison report's 'achieved')",
        input_form="|g-file j_tor_averaged_direct| (bouquet's reader)")


def _distance_ids(b, bl, psi_pad):
    import numpy as np
    from bouquet.engine import _lcfs_deviation_mm
    from bouquet.io.imas import _nearest_index, read_imas_geometry
    from bouquet.TokaMaker_interface import _corrective_output_jphi
    mygs = b.mygs
    with open(b.config.source.ids_path) as fh:
        dd = json.load(fh)
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], _TIME, "equilibrium")
    p1 = eq["time_slice"][ie]["profiles_1d"]
    gq = eq["time_slice"][ie]["global_quantities"]
    psq = np.asarray(p1["psi"], float)
    psn = (psq - psq[0]) / (psq[-1] - psq[0])
    psi = np.asarray(bl.psi_N, float)
    qin = np.abs(np.interp(psi, psn, np.asarray(p1["q"], float)))
    jin = np.interp(psi, psn, np.asarray(p1["j_tor"], float))
    peak = float(np.max(np.abs(jin)))
    A = np.asarray(_corrective_output_jphi(mygs, psi, psi_pad), float)
    R = np.asarray(bl.j_phi, float)
    st_i = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    qp = np.abs(_q_profile(mygs, psi))
    m = (psi >= 0.05) & (psi <= 0.95)
    rel = qp[m] / qin[m] - 1.0
    core, edge = psi < 0.8, psi >= 0.8
    _F0, bnd = read_imas_geometry(b.config.source)
    rms, mx = _lcfs_deviation_mm(mygs, bnd)
    return dict(
        li3=dict(input=float(gq["li_3"]), engine=float(st_i["l_i"]),
                 delta=float(st_i["l_i"]) - float(gq["li_3"])),
        q_axis=dict(input=float(qin[0]), engine=float(qp[0]),
                    delta=float(qp[0] - qin[0])),
        q95=dict(input=float(np.interp(0.95, psi, qin)),
                 engine=float(st_i["q_95"])),
        q_profile_rel_pct=dict(max=100.0 * float(np.max(np.abs(rel))),
                               rms=100.0 * float(np.sqrt(np.mean(rel ** 2)))),
        jphi_vs_equilibrium_jtor_pct_of_peak=dict(
            core=_pct_stats(A - jin, peak, core),
            edge=_pct_stats(A - jin, peak, edge),
            note=("the synthetic OMAS j_tor was written as a jphi-linterp "
                  "input (<j_phi>), so this is like for like on THIS file "
                  "only (jphi convention experiment, claim (d))")),
        requested_minus_achieved_pct_of_peak=dict(
            core=_pct_stats(R - A, peak, core),
            edge=_pct_stats(R - A, peak, edge)),
        lcfs_mm=dict(rms=rms, max=mx))


def _resolve_self(b, bl, psi_pad):
    """Re-solve the delivered state's own stored request, once, from it."""
    import numpy as np
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend
    mygs = b.mygs
    st = bl.engine["state"]
    snap = mygs.copy_eq()
    psi = np.asarray(bl.psi_N, float)
    before = TokaMakerBackend(mygs, SimpleNamespace(
        psi_N=psi, pressure=np.asarray(st["pressure"], float),
        Ip=float(st["Ip"]), kinetics=_kin(bl, psi)), psi_pad=psi_pad)
    m0 = before.measure(final=True)
    before.solve(np.asarray(st["request"], float), n_passes=1)
    m1 = before.measure(final=True)
    mygs.replace_eq(source_eq=snap)
    a0, a1 = np.asarray(m0["achieved"]), np.asarray(m1["achieved"])
    return dict(dl_i=float(m1["li"] - m0["li"]),
                dq0=float(m1["q_row"] - m0["q_row"]),
                dq95=float(m1["stats"].get("q_95", float("nan"))
                           - m0["stats"].get("q_95", float("nan"))),
                dj_rel=float(np.linalg.norm(a1 - a0) / np.linalg.norm(a0)),
                dj_max_rel=float(np.max(np.abs(a1 - a0))
                                 / np.max(np.abs(a0))))


def _kin(bl, psi):
    import numpy as np
    from bouquet.utils import pchip_interp
    k = lambda a: pchip_interp(bl.psi_N_kinetic, a, psi)  # noqa: E731
    return dict(ne=k(bl.ne), te=k(bl.te), ni=k(bl.ni), ti=k(bl.ti),
                zeff=np.clip(k(bl.Zeff), 1.0, None))


def child(part, outdir):
    import bouquet as bq
    _harness.assert_bouquet_is_repo_local()
    src, extra = PARTS[part]
    out = dict(part=part, source=src, settings=dict(extra), stages={})
    path = os.path.join(outdir, f"engine_{part}.json")

    def _stage(name, fn):
        try:
            out[name] = fn()
            out["stages"][name] = "ok"
        except Exception as e:                    # recorded, never lost
            out["stages"][name] = (f"{type(e).__name__}: {e}\n"
                                   + traceback.format_exc()[-3000:])

    try:
        if src == "recon":
            b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                       nthreads=1, n_draws=1,
                                       header=os.path.join(outdir, part))
            psi_pad = float(b.config.source.psi_pad)
        else:
            b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME,
                                     n_draws=1, nthreads=1,
                                     header=os.path.join(outdir, part))
            psi_pad = 1e-3
        g = b.config.generation
        g.reconstruction_engine = "unified"
        for k, v in extra.items():
            setattr(g, k, v)
        b.setup_solver()
        t0 = time.perf_counter()
        holder = {}

        def _build():
            bl = b.prepare_baseline()
            holder["bl"] = bl
            rec = bl.engine
            return dict(
                wall_s=float(time.perf_counter() - t0),
                converged=bool(rec["converged"]),
                loop_converged=bool(rec["loop_converged"]),
                n_passes={ph["name"]: int(ph["record"]["n_passes"])
                          for ph in rec["phases"]},
                max_passes=int(rec["settings"]["loop"]["max_passes"]),
                solves=rec["solves"], delivered=rec["delivered"],
                l_i_target=float(bl.l_i_target),
                delivered_state={k: v for k, v in
                                 (bl.delivered_state or {}).items()
                                 if k != "j_phi_achieved"},
                engine_record=rec)
        _stage("build", _build)
        bl = holder.get("bl")
        if bl is not None:
            _stage("distance", lambda: (_distance_gfile if src == "recon"
                                        else _distance_ids)(b, bl, psi_pad))
            _stage("resolve_self", lambda: _resolve_self(b, bl, psi_pad))
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    finally:
        from bouquet.jbs_loop import jsonable
        with open(path, "w") as fh:
            json.dump(jsonable(out), fh, default=float)
        keep = os.environ.get("BQ_ENGINE_PROBE_OUT")
        if keep:
            os.makedirs(keep, exist_ok=True)
            shutil.copy(path, os.path.join(keep, os.path.basename(path)))
    return path


def run_part(part, outdir, timeout=None):
    """Run one part in its own interpreter; return its JSON (with ``_rc``)."""
    os.makedirs(outdir, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), outdir, "--child", part],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=timeout)
    p = os.path.join(outdir, f"engine_{part}.json")
    with open(os.path.join(outdir, f"engine_{part}.log"), "w") as fh:
        fh.write(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if not os.path.exists(p):
        return dict(part=part, stages={}, fatal=(
            f"no JSON (rc={proc.returncode}):\n{proc.stdout[-3000:]}\n"
            f"{proc.stderr[-3000:]}"), _rc=proc.returncode)
    with open(p) as fh:
        d = json.load(fh)
    d["_rc"] = proc.returncode
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-timeout", type=float, default=None)
    a = ap.parse_args(argv)
    oft_importable()
    if a.child:
        child(a.child, a.outdir)
        return 0
    res = {}
    for part in [p for p in a.parts.split(",") if p]:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; known: {list(PARTS)}")
        res[part] = run_part(part, a.outdir, timeout=a.child_timeout)
        print(f"[measure_engine] {part}: stages {res[part].get('stages')}",
              flush=True)
    with open(os.path.join(a.outdir, "engine_measurement.json"), "w") as fh:
        json.dump(res, fh, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
