"""Why hold IMAS profiles on Φ_N: psi_n vs phi_n on a FUSE dd.

A σ=0 IMAS forward solve (``prepare()``) is run with the profiles held on ψ_N
(``coord="psi_n"``) and on Φ_N (``coord="phi_n"``). A bootstrap (SWB) A/B is
then run from one fixed equilibrium state to isolate the mechanism:

  psi_pinned      kinetics pinned to ψ_N
  phi_pinned      the same kinetics pinned to Φ_N, labelled by that state's own map
  phi_consistent  Φ_N labels taken from psi_pinned's end state (control: must
                  reproduce psi_pinned, so it checks the toolkit's Φ_N path)

Inside SWB the current is the generic seed plus bootstrap, so the equilibrium
relaxes away from the dd's. ψ_N surfaces move in real space with it and
ψ_N-pinned kinetics move with them; Φ_N-pinned kinetics stay close to the
source's placement (OFT tests/physics/tokamaker_torflux_motivation.py).

Usage (one dd, e.g. DIII-D shot A FUSE dd_sim.json, put in ./data/):
    python torflux_imas_effect.py [--dd data/dd_sim.json] [--time 2.0]
        [--gfile G] [--out out] [--zip]
Needs OFT with toroidal-flux support (OFT_PYTHONPATH), ~16 GB RAM, ~5 min.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PSI_GRID = np.linspace(0.02, 0.98, 97)      # common ψ_N readback grid
PED_LEVELS = np.linspace(0.80, 0.995, 40)   # ψ_N levels traced for R_out
# Entity colours (fixed across figures): ψ_N-pinned, Φ_N-pinned, control, source.
C_PSI, C_PHI, C_CTL, C_SRC = "#eb6834", "#2a78d6", "#1baf7a", "#52514e"


# --------------------------------------------------------------------------- cases
def _bouquet(a, coord):
    sys.path.insert(0, REPO)
    import matplotlib
    matplotlib.use("Agg")
    import bouquet as bq
    bq.add_oft_to_path()
    run = bq.Bouquet.from_imas(
        a.dd, mesh=a.mesh, time=a.time, n_draws=1, header=f"torflux_{coord}",
        LCFS_geqdsk=a.gfile, saddle_targets=a.saddle, saddle_weights=[300.0])
    run.source.coord = coord
    g = run.generation
    # The operational shot A setup: FUSE ohmic + SWB bootstrap (ohmic split).
    g.seed, g.isolate_edge_jBS, g.jBS_baseline_mode = 42, False, "ohmic"
    g.closure_channel, g.structured_preset = "structured", "li_soft_onesided"
    g.perturb_jind_in_anchor, g.workflow = True, "custom"
    return run


def _capture_source():
    """Keep the dd baseline (FUSE j_BS, equilibrium j_tor) the reader returns."""
    from bouquet.io import imas
    held = {}
    orig = imas.read_imas_baseline

    def wrap(*args, **kw):
        bl = orig(*args, **kw)
        held.update(x=np.array(bl.psi_N), j_BS=np.array(bl.j_BS), j_phi=np.array(bl.j_phi),
                    eq_jtor=(None if bl.jphi_diff is None
                             else np.array(bl.jphi_diff + bl.j_phi)),
                    li3_ids=(bl.li_metrics or {}).get("ids_li_3"),
                    li1_ids=(bl.li_metrics or {}).get("ids_li_1"))
        return bl
    imas.read_imas_baseline = wrap
    return held


def _state(mygs):
    """Readbacks of the current equilibrium on PSI_GRID, plus R_out(ψ_N), p(ψ_N)."""
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
    from bouquet.physics import q_ravg
    _, f, fp, p, pp = mygs.get_profiles(psi=PSI_GRID.copy())
    _, q, rav, _, _, _ = mygs.get_q(psi=PSI_GRID.copy())
    j = get_jphi_from_GS(f * fp, pp, q_ravg(rav, "<R>"), q_ravg(rav, "<1/R>"))
    p_ped = mygs.get_profiles(psi=PED_LEVELS.copy())[3]
    r_out = []
    for s in PED_LEVELS:
        try:
            r_out.append(float(np.asarray(mygs.trace_surf(float(s)), float)[:, 0].max()))
        except Exception:
            r_out.append(float("nan"))
    st = mygs.get_stats(li_normalization="iter")
    pp_ped = np.asarray(mygs.get_profiles(psi=PED_LEVELS.copy())[4], float)
    q_ped = np.abs(np.asarray(mygs.get_q(psi=PED_LEVELS.copy())[1], float))
    ip = int(np.argmax(np.abs(pp_ped)))
    return dict(psi_range=float(mygs.psi_bounds[1] - mygs.psi_bounds[0]),
                pp_peak=float(np.abs(pp_ped[ip])), q_at_pp_peak=float(q_ped[ip]),
                j=np.asarray(j, float).tolist(), q=np.abs(np.asarray(q, float)).tolist(),
                p=np.asarray(p, float).tolist(), p_ped=np.asarray(p_ped, float).tolist(),
                r_out=r_out, li3=float(st["l_i"]), Ip=float(st["Ip"]),
                li1=float(mygs.get_stats(li_normalization="std")["l_i"]))


def case_prepare(a, coord):
    run = _bouquet(a, coord)
    src = _capture_source()
    run.prepare()
    bl, mygs = run.baseline, run.mygs
    q0, q95 = np.abs(mygs.get_q(psi=np.array([1e-3, 0.95]))[1])
    out = dict(coord=coord, q0=float(q0), q95=float(q95), state=_state(mygs),
               j_BS_swb=np.asarray(bl.j_BS, float).tolist(),
               source={k: (None if v is None else np.asarray(v).tolist()) for k, v in src.items()})
    if coord == "psi_n":
        out["ab"] = _swb_ab(run, src)
    return out


def _swb_ab(run, src):
    """SWB from the prepared psi_n state: ψ_N- vs Φ_N-pinned kinetics."""
    from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
    from bouquet import coords
    bl, mygs = run.baseline, run.mygs
    snap = mygs.copy_eq()
    x = np.asarray(bl.psi_N, float)
    kin = [np.asarray(getattr(bl, k), float) for k in ("ne", "te", "ni", "ti", "Zeff")]
    seeds = {"": np.power(1.0 - np.power(np.clip(x, 0.0, 1.0), 1.5), 1.5),  # generic, in ψ_N
             "_ohm": np.clip(np.asarray(bl.j_inductive, float), 0.0, None)}   # the run's ohmic shape
    seeds["_ohm"] = seeds["_ohm"] / seeds["_ohm"].max()

    def phi_labels():
        q = np.abs(np.asarray(mygs.get_q(psi=np.clip(x, 1e-3, 1 - 1e-3))[1], float))
        return coords.phi_n_from_q(x, q)[1]

    def swb(xg, coord, sd=""):
        mygs.replace_eq(source_eq=snap)
        r = solve_with_bootstrap(mygs, *kin, float(bl.Ip_target), seeds[sd].copy(),
                                 isolate_edge_jBS=False, **coords.swb_grid_kwargs(xg, coord))
        return dict(psi=np.asarray(r["psi_n"], float).tolist(),
                    j_BS=np.asarray(r["j_BS"], float).tolist(), state=_state(mygs))

    mygs.replace_eq(source_eq=snap)
    start = _state(mygs)
    # the start state's thermal pressure (what SWB is given) on its own surfaces
    p_th = 1.602176634e-19 * (kin[0] * kin[1] + kin[2] * kin[3])
    start["p_ped"] = np.interp(PED_LEVELS, x, p_th).tolist()
    x_phi_start = phi_labels()
    ab = {"start": dict(state=start),
          "psi_pinned": swb(x, "psi_n")}
    x_phi_end = phi_labels()                  # labels of psi_pinned's end state
    ab["phi_pinned"] = swb(x_phi_start, "phi_n")
    ab["phi_consistent"] = swb(x_phi_end, "phi_n")
    # With the run's own ohmic shape as the seed, SWB stays near the start state.
    ab["psi_pinned_ohm"] = swb(x, "psi_n", "_ohm")
    ab["phi_pinned_ohm"] = swb(x_phi_start, "phi_n", "_ohm")
    ab["x"] = x.tolist()
    return ab


# --------------------------------------------------------------------------- plots
def _peak(psi, j, lo=0.85):
    psi, j = np.asarray(psi), np.asarray(j)
    i = int(np.argmax(np.where(psi > lo, j, -np.inf)))
    return float(j[i]), float(psi[i])


def analyse(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P, F = res["psi_n"], res["phi_n"]
    src, ab = P["source"], P["ab"]
    xs = np.asarray(src["x"])
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2})

    # Fig 1: the σ=0 forward solves against the dd.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    if src["eq_jtor"] is not None:
        ax[0].plot(xs, np.asarray(src["eq_jtor"]) / 1e6, c=C_SRC, ls=":", label="dd equilibrium j_tor")
    for r, c, lab in ((P, C_PSI, "held on ψ_N"), (F, C_PHI, "held on Φ_N")):
        ax[0].plot(PSI_GRID, np.asarray(r["state"]["j"]) / 1e6, c=c, label=lab)
        ax[1].plot(PSI_GRID, r["state"]["q"], c=c, label=lab)
    ax[0].set(xlabel="ψ_N", ylabel="achieved j_φ [MA/m²]", title="Current after the σ=0 solve")
    ax[1].set(xlabel="ψ_N", ylabel="|q|", title="Safety factor")
    ax[2].plot(xs, np.asarray(src["j_BS"]) / 1e6, c=C_SRC, ls=":", label="FUSE j_BS (dd)")
    ax[2].plot(xs, np.asarray(P["j_BS_swb"]) / 1e6, c=C_PSI, label="SWB, held on ψ_N")
    # the phi run's SWB j_BS sits on its own nodes (same dd nodes, Φ_N labels)
    ax[2].plot(xs, np.asarray(F["j_BS_swb"]) / 1e6, c=C_PHI, label="SWB, held on Φ_N")
    ax[2].set(xlabel="ψ_N of the dd node", ylabel="j_BS [MA/m²]", title="Bootstrap used in the solve",
              xlim=(0.8, 1.0))
    for a_ in ax:
        a_.legend(fontsize=8, frameon=False)
    fig.suptitle("σ=0 IMAS forward solve: profiles held on ψ_N vs Φ_N", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_forward_solve.png"), dpi=150)
    plt.close(fig)

    # Fig 2: the SWB A/B from one state, generic seed (top) and ohmic seed (bottom).
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    rows = ((("psi_pinned", C_PSI, "-", "kinetics on ψ_N"), ("phi_pinned", C_PHI, "-", "kinetics on Φ_N"),
             ("phi_consistent", C_CTL, "--", "control: Φ_N labels of the ψ_N end state")),
            (("psi_pinned_ohm", C_PSI, "-", "kinetics on ψ_N"), ("phi_pinned_ohm", C_PHI, "-", "kinetics on Φ_N")))
    for ax, runs, seed in zip(axs, rows, ("generic seed (1-ψ_N^1.5)^1.5", "seed = the run's ohmic shape")):
        for k, c, ls, lab in runs:
            s_ = ab[k]["state"]
            ax[0].plot(s_["r_out"], np.asarray(s_["p_ped"]) / 1e3, c=c, ls=ls, label=lab)
            ax[1].plot(ab[k]["psi"], np.asarray(ab[k]["j_BS"]) / 1e6, c=c, ls=ls, label=lab)
            ax[2].plot(PSI_GRID, s_["q"], c=c, ls=ls, label=lab)
        ax[0].plot(ab["start"]["state"]["r_out"], np.asarray(ab["start"]["state"]["p_ped"]) / 1e3,
                   c=C_SRC, ls=":", label="start state")
        ax[1].plot(xs, np.asarray(src["j_BS"]) / 1e6, c=C_SRC, ls=":", label="FUSE j_BS (dd)")
        ax[2].plot(PSI_GRID, ab["start"]["state"]["q"], c=C_SRC, ls=":", label="start state")
        ax[0].set(xlabel="outboard R of the ψ_N surface [m]", ylabel="thermal p [kPa]",
                  title=f"Pedestal p in real space\n{seed}")
        ax[1].set(xlabel="ψ_N", ylabel="j_BS [MA/m²]", title="SWB bootstrap", xlim=(0.8, 1.0))
        ax[2].set(xlabel="ψ_N", ylabel="|q|", title="q after SWB")
        for a_ in ax:
            a_.legend(fontsize=8, frameon=False)
    fig.suptitle("SWB from one equilibrium state: kinetics pinned to ψ_N vs Φ_N", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_swb_ab.png"), dpi=150)
    plt.close(fig)

    # Summary numbers.
    fuse_pk = _peak(xs, src["j_BS"])[0]
    rows = {c: dict(li3=r["state"]["li3"], li1=r["state"]["li1"], q0=r["q0"], q95=r["q95"],
                    Ip=r["state"]["Ip"], jBS_peak_over_FUSE=_peak(xs, r["j_BS_swb"])[0] / fuse_pk)
            for c, r in (("psi_n", P), ("phi_n", F))}
    abrows = {k: dict(jBS_peak=_peak(ab[k]["psi"], ab[k]["j_BS"])[0],
                      psi_peak=_peak(ab[k]["psi"], ab[k]["j_BS"])[1],
                      li3=ab[k]["state"]["li3"], psi_range=ab[k]["state"]["psi_range"],
                      pp_peak=ab[k]["state"]["pp_peak"], q_ped=ab[k]["state"]["q_at_pp_peak"],
                      R_out_97=float(np.interp(0.97, PED_LEVELS, ab[k]["state"]["r_out"])))
              for k in ("psi_pinned", "phi_pinned", "phi_consistent", "psi_pinned_ohm", "phi_pinned_ohm")}
    summ = dict(ids=dict(li3=src["li3_ids"], li1=src["li1_ids"]), FUSE_jBS_peak=fuse_pk,
                forward=rows, swb_ab=abrows)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    L = ["| run | l_i(3) | l_i(1) | q0 | q95 | SWB/FUSE j_BS peak |", "|---|---|---|---|---|---|",
         f"| IDS (dd) | {summ['ids']['li3']:.4f} | {summ['ids']['li1']:.4f} | | | |"]
    L += [f"| {c} | {r['li3']:.4f} | {r['li1']:.4f} | {r['q0']:.3f} | {r['q95']:.3f} | "
          f"{r['jBS_peak_over_FUSE']:.3f} |" for c, r in rows.items()]
    L += ["", f"SWB A/B from one state (FUSE pedestal j_BS peak {fuse_pk/1e6:.3f} MA/m²):", "",
          "| case | j_BS peak [MA/m²] | at ψ_N | l_i(3) | ψ_b-ψ_a [Wb/rad] | peak dp/dψ [Pa/(Wb/rad)] | q there | R_out(ψ_N=0.97) [m] |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {r['jBS_peak']/1e6:.4f} | {r['psi_peak']:.4f} | {r['li3']:.4f} | {r['psi_range']:.4f} | "
          f"{r['pp_peak']:.4g} | {r['q_ped']:.3f} | {r['R_out_97']:.4f} |"
          for k, r in abrows.items()]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dd", default=os.path.join(HERE, "data", "dd_sim.json"))
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--gfile", default=None, help="optional LCFS g-file (else the dd boundary)")
    ap.add_argument("--mesh", default=os.path.join(REPO, "examples", "D3D-like", "DIIID_mesh.h5"))
    ap.add_argument("--saddle", type=json.loads, default=[[1.26296234, 1.13728905]],
                    help="saddle targets, JSON (default: DIII-D lower X-point)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    ap.add_argument("--replot", action="store_true", help="re-plot existing <out> results only")
    ap.add_argument("--case", choices=("psi_n", "phi_n"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.case:                                    # worker: one TokaMaker per process
        os.chdir(os.path.join(a.out, a.case))
        json.dump(case_prepare(a, a.case), open("result.json", "w"))
        return
    if a.replot:
        analyse({c: json.load(open(os.path.join(a.out, c, "result.json"))) for c in ("psi_n", "phi_n")}, a.out)
        return
    if not os.path.isfile(a.dd):
        sys.exit(f"dd not found: {a.dd} (put the FUSE dd_sim.json there or pass --dd)")
    procs = {}  # one TokaMaker per process
    for c in ("psi_n", "phi_n"):
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
        log = open(os.path.join(a.out, c, "log.txt"), "w")
        procs[c] = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--case", c] + [
            v for k in ("dd", "mesh", "out") for v in (f"--{k}", getattr(a, k))] + [
            "--time", str(a.time), "--saddle", json.dumps(a.saddle)]
            + (["--gfile", a.gfile] if a.gfile else []), stdout=log, stderr=subprocess.STDOUT)
    for c, p in procs.items():
        if p.wait():
            sys.exit(f"{c} failed: see {os.path.join(a.out, c, 'log.txt')}")
    res = {c: json.load(open(os.path.join(a.out, c, "result.json"))) for c in procs}
    json.dump(dict(dd=a.dd, time=a.time, gfile=a.gfile), open(os.path.join(a.out, "inputs.json"), "w"))
    analyse(res, a.out)
    if a.zip:
        import shutil
        print("zip:", shutil.make_archive(a.out, "zip", os.path.dirname(a.out), os.path.basename(a.out)))


if __name__ == "__main__":
    main()
