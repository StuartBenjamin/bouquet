"""Live-solver halves of the reproducibility contract and the R2 Ip invariant.

Two things can only be checked with the real GS solver:

  * **The seed contract end to end.** Two seeded ensembles built from the same
    config must produce bitwise-identical archives -- every drawn profile,
    ``j_BS``, ``j_inductive`` and scalar.  ``jBS_scale_range`` and
    ``l_i_uncertainty`` are both switched on so the two streams that WERE
    seeded before this work (``np.random.uniform`` for ``scale_jBS``,
    ``np.random.normal`` for the per-draw ``l_i`` target) are exercised
    alongside the GPR and cannot regress while the GPR is fixed.

    The two ensembles run in **separate processes**.  That is how a user
    actually reproduces a run, and it is the only way to compare the *solved*
    equilibria: ``OFT_env`` is a per-process singleton, and a second
    ``generate_bouquet`` in one interpreter does not start from the same
    solver state -- the run installs a strong coil soft-reg and drift bounds
    that ``replace_eq`` does not undo (they live on the TokaMaker object, not
    the equilibrium).  Measured in-process, back-to-back seeded runs agree
    bitwise on every DRAWN quantity but land coil currents 1.4e-3 apart in
    relative terms and l_i ~0.6% apart.  So the in-process check below pins
    the draw stream (what the seed governs) and the cross-process check pins
    the whole archive (what the user gets).

  * **The R2 golden invariant.** At :math:`\\sigma=0` route R2's inductive
    :math:`I_p` renormalisation must return 1.000, because the archived split
    already IS the answer.  It returned 0.8373 while the root ran on
    ``solve_with_bootstrap``'s landed equilibrium against an uncalibrated
    ``Ip_target``; see ``TokaMaker_interface._AnchorIpRenorm``.

    The same probe A/Bs the FSA-measure mode (``BOUQUET_R2_IP_MODE=exact``)
    against the default ratio calibration.  Its acceptance is separate and
    looser BY DERIVATION, not by concession -- read ``_S_ATOL_EXACT`` before
    touching either number.

Runs on the synthetic D3D-like example (no proprietary data).  Marked
``solver`` like ``test_systematics.py``; deselect with ``pytest -m "not
solver"``.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")

_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))


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
        reason="needs OFT + the D3D-like mesh/baseline; skipped when "
               "unavailable"),
]

_SEED = 20260804
_N_DRAWS = 2                 # each draw is a full solve + SWB + homotopy

# Acceptance bars for the sigma=0 R2 invariant.  Measured values on this case
# are quoted in the assertions; none of these may be loosened to make a run
# pass -- a miss is a regression to report, not a tolerance to widen.
_S_ATOL = 1e-3               # |s - 1| ; measured 8.5e-4
_LI_REL = 0.005              # l_i vs recon ; measured +0.100%
_JBS_FRAC = 0.02             # sigma0 j_BS vs baseline split, fraction of peak
                             # (the sigma0-guard bar) ; measured 0.265%

# The 'exact' measure (BOUQUET_R2_IP_MODE=exact) has its OWN acceptance, and it
# is deliberately not _S_ATOL.  The ratio calibration is exact at sigma=0 by
# construction; the FSA measure is not, because it also sees the two ways the
# archived split fails to be the anchor -- the reconstruction's own j_phi
# residual (+0.193% of Ip at the R2 state anchor, i.e. -0.25% of inductive
# amplitude) and the sigma=0 SWB residual (-0.085%).  Measured |s-1| = 3.25e-3
# against that -0.335% budget.  This bar is a NEW pin on NEW behaviour, not a
# widening of _S_ATOL, which still governs the default path below.
_S_ATOL_EXACT = 5e-3         # |s - 1| in exact mode ; measured 3.25e-3


def _draw_groups(path):
    """{draw index: {dataset name: ndarray}} plus the scalar attrs."""
    out = {}
    with h5py.File(path, "r") as hf:
        for sk in sorted(hf["scan"].keys()):
            g = hf[f"scan/{sk}"]
            for k in sorted(g.keys()):
                if not str(k).lstrip("-").isdigit():
                    continue
                grp = g[k]
                rec = {n: np.asarray(grp[n][()]) for n in sorted(grp.keys())
                       if isinstance(grp[n], h5py.Dataset)}
                rec["_attrs"] = {a: grp.attrs[a] for a in sorted(grp.attrs)}
                out[f"{sk}/{k}"] = rec
    return out


# ---------------------------------------------------------------------------
#  the seed contract, end to end (two INDEPENDENT processes)
# ---------------------------------------------------------------------------
def _generate_one_ensemble(header):
    """Baseline + one seeded ensemble, in THIS process.

    The subprocess entry point (see ``__main__`` at the bottom) and therefore
    also the definition of "a run" for the twin comparison.

    Driven through ``generate_bouquet`` rather than ``Bouquet.generate()``
    because ``l_i_uncertainty`` -- one of the two streams that WAS seeded
    before this work -- has no ``GenerationConfig`` field, so the class API
    cannot switch it on.  Everything else mirrors what ``Bouquet.generate()``
    passes, in particular the production coil-drift / homotopy schedule: the
    bare ``generate_bouquet`` defaults are a single +/-1 % pass, which this
    case cannot meet under a 5 % j_phi envelope, and every draw is then
    discarded as infeasible.

    Writes ``<header>.h5`` and ``<header>_diag.json`` (the per-draw sampler
    inputs, which are not archived).
    """
    import numpy as np
    import bouquet as bq
    from bouquet import generate_bouquet
    from bouquet.baseline import resolve_uncertainty
    from bouquet.utils import initialize_equilibrium_database, pchip_interp

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=header, n_draws=_N_DRAWS)
    b.setup_solver()
    bl = b.prepare_baseline()
    gc, fc = b.config.generation, b.config.filtering
    b.config.uncertainty.jphi_scalar_sigma = 0.05
    env = resolve_uncertainty(b.config, bl)
    psi_N = np.asarray(bl.psi_N, dtype=float)
    Zeff_eq = np.clip(
        pchip_interp(np.asarray(bl.psi_N_kinetic, dtype=float),
                     np.asarray(bl.Zeff, dtype=float), psi_N), 1.0, None)

    if os.path.exists(header + ".h5"):
        os.remove(header + ".h5")
    initialize_equilibrium_database(header)
    diags = generate_bouquet(
        b.mygs, psi_N, _N_DRAWS, header,
        np.asarray(bl.j_phi, dtype=float),
        bl.ne, bl.te, bl.ni, bl.ti,
        env["sigma_ne"], env["sigma_te"], env["sigma_ni"], env["sigma_ti"],
        env["sigma_jphi"], env["n_ls"], env["t_ls"], env["j_ls"],
        float(bl.Ip_target), float(bl.l_i_target), Zeff_eq,
        input_jinductive=np.asarray(bl.j_inductive, dtype=float),
        l_i_tolerance=gc.l_i_tolerance,
        psi_pad=float(getattr(b.config.source, "psi_pad", 1e-3)),
        constrain_sawteeth=gc.constrain_sawteeth, recalculate_j_BS=True,
        isolate_edge_jBS=gc.isolate_edge_jBS, floor_j_BS=gc.floor_j_BS,
        scan_key=0,
        psi_N_kinetic=np.asarray(bl.psi_N_kinetic, dtype=float),
        # production coil schedule (what Bouquet.generate() passes)
        coil_drift=gc.coil_drift,
        coil_drift_hard_factor=gc.coil_drift_hard_factor,
        homotopy_passes=gc.homotopy_passes,
        inspec_F_max=fc.inspec_F_max, inspec_VSC_max=fc.inspec_VSC_max,
        # both previously-seeded streams ON, alongside the GPR
        jBS_scale_range=(0.97, 1.03), l_i_uncertainty=0.05,
        diagnostic_plots=False, seed=_SEED,
    )
    with open(header + "_diag.json", "w") as fh:
        json.dump([{k: float(d[k]) for k in ("scale_jBS", "l_i_target_used")}
                   for d in diags], fh)


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory):
    """The same seeded ensemble generated twice, in two fresh interpreters."""
    work = tmp_path_factory.mktemp("twin")
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    out = []
    for tag in ("a", "b"):
        header = str(work / f"twin_{tag}")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                               "ensemble", header], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"seeded ensemble {tag} failed "
                        f"(rc={proc.returncode}):\n{proc.stderr[-4000:]}")
        with open(header + "_diag.json") as fh:
            out.append((_draw_groups(header + ".h5"), json.load(fh)))
    return out


def test_same_seed_reproduces_every_draw_dataset_bitwise(twin_runs):
    """Same seed -> bitwise-identical archives.  This is the whole contract."""
    (a, _), (b, _) = twin_runs
    assert a, "the seeded run produced no draws"
    assert sorted(a) == sorted(b), "the two runs kept different draws"
    for key in sorted(a):
        for name in sorted(a[key]):
            if name == "_attrs":
                continue
            np.testing.assert_array_equal(
                a[key][name], b[key][name],
                err_msg=f"draw {key} dataset {name!r} is not reproducible")


def test_same_seed_reproduces_the_drawn_kinetics_and_currents(twin_runs):
    """Name the datasets that matter, so a silent rename cannot hollow out the
    test above into a no-op."""
    (a, _), (b, _) = twin_runs
    required = ("n_e", "T_e", "n_i", "T_i", "j_phi", "j_inductive", "j_BS")
    for key in sorted(a):
        missing = sorted(set(required) - set(a[key]))
        assert not missing, f"draw {key}: missing {missing}"
        for name in required:
            np.testing.assert_array_equal(a[key][name], b[key][name],
                                          err_msg=f"{key}/{name}")


def test_same_seed_reproduces_the_previously_seeded_streams(twin_runs):
    """scale_jBS and the per-draw l_i target were the ONLY seeded pieces before
    the GPR was fixed; migrating them onto the Generator must not regress them.
    """
    (a, da), (b, db) = twin_runs
    assert len(da) == len(db) and da, "no per-draw diagnostics returned"
    for i, (x, y) in enumerate(zip(da, db)):
        for k in ("scale_jBS", "l_i_target_used"):
            assert x[k] == y[k], \
                f"draw {i}: {k} is not reproducible ({x[k]} vs {y[k]})"
    for key in sorted(a):
        for attr in ("l_i(1)", "l_i(3)", "Ip", "l_i_target_used"):
            if attr in a[key]["_attrs"]:
                assert a[key]["_attrs"][attr] == b[key]["_attrs"][attr], \
                    f"{key}/{attr} not reproducible"


def test_the_previously_seeded_streams_actually_varied(twin_runs):
    """Guard the guard: if jBS_scale_range/l_i_uncertainty had collapsed to a
    constant, the reproducibility assertions above would be vacuous."""
    (_a, da), _b = twin_runs
    if len(da) < 2:
        pytest.skip("only one draw completed; nothing to compare")
    assert len({d["scale_jBS"] for d in da}) > 1, \
        "scale_jBS is constant across draws -- the uniform draw is not firing"
    assert len({d["l_i_target_used"] for d in da}) > 1, \
        "l_i target is constant across draws -- the normal draw is not firing"


def test_seeded_draws_are_not_all_identical(twin_runs):
    """A run whose sigmas collapsed to zero would reproduce trivially."""
    (a, _), _ = twin_runs
    keys = sorted(a)
    if len(keys) < 2:
        pytest.skip("only one draw survived; nothing to compare")
    assert not np.array_equal(a[keys[0]]["n_e"], a[keys[1]]["n_e"]), \
        "the two draws are identical -- the sampler is not perturbing"


# ---------------------------------------------------------------------------
#  the R2 golden invariant: sigma=0 -> the archived split, s == 1.000
# ---------------------------------------------------------------------------
def _run_r2_probe(outdir):
    """sigma=0 route-R2 draws through ``perturb_kinetic_equilibrium``.

    Runs the default ('ratio') path TWICE -- so the pair also tests that the
    anchor snapshot/restore injects no state drift -- the 'exact' FSA measure
    twice, and the 'legacy' path once for the before/after contrast.
    Subprocess entry point; results land in ``<outdir>/r2.npz``.
    """
    import numpy as np
    import bouquet as bq
    from bouquet import perturb_kinetic_equilibrium
    from bouquet.utils import pchip_interp

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "r2"), n_draws=1)
    b.setup_solver()
    bl = b.prepare_baseline()
    gc = b.config.generation
    psi_N = np.asarray(bl.psi_N, dtype=float)
    psi_kin = np.asarray(bl.psi_N_kinetic, dtype=float)
    psi_pad = float(getattr(b.config.source, "psi_pad", 1e-3))
    EC = 1.6022e-19

    def _k2e(a):
        return pchip_interp(psi_kin, np.asarray(a, dtype=float), psi_N)

    pressure = EC * (_k2e(bl.ne) * _k2e(bl.te) + _k2e(bl.ni) * _k2e(bl.ti))
    Zeff_eq = np.clip(_k2e(bl.Zeff), 1.0, None)
    zk, zj = np.zeros_like(psi_kin), np.zeros_like(psi_N)
    snapshot = b.mygs.copy_eq()

    def _once(mode):
        os.environ["BOUQUET_R2_IP_MODE"] = mode
        b.mygs.replace_eq(source_eq=snapshot)
        out = perturb_kinetic_equilibrium(
            b.mygs, psi_N, pressure,
            bl.ne, bl.te, bl.ni, bl.ti, np.asarray(bl.j_phi, dtype=float),
            zk, zk, zk, zk, zj,                     # sigma = 0 everywhere
            0.5, 0.4, 0.25,
            float(bl.Ip_target), float(bl.l_i_target), Zeff_eq, len(psi_N),
            input_jinductive=np.asarray(bl.j_inductive, dtype=float),
            l_i_tolerance=gc.l_i_tolerance, psi_pad=psi_pad,
            constrain_sawteeth=False, recalculate_j_BS=True,
            isolate_edge_jBS=gc.isolate_edge_jBS, floor_j_BS=gc.floor_j_BS,
            scale_jBS=float(getattr(bl, "bs_scale", 1.0)),
            perturb_jind_in_anchor=True, accept_anchor_inband=False,
            psi_N_kinetic=psi_kin, p_thresh=0.05, rng=_SEED,
        )
        d = out[6]
        return (float(d["r2_ip_scale"]),
                # Measure l_i on the SAME estimator bl.l_i_target lives on
                # ('iter' == li(3)); get_stats' default is 'std' == li(1),
                # which is a different functional (~25% apart) and would make
                # the recon-li assertions below compare apples to oranges.
                # Issue #20 -- estimator consistency, not a tolerance change.
                float(b.mygs.get_stats(lcfs_pad=psi_pad,
                                       li_normalization="iter")["l_i"]),
                np.asarray(d["j_BS"], dtype=float),
                np.asarray(d["j_inductive"], dtype=float))

    s1, li1, jbs1, jind1 = _once("ratio")
    s2, li2, jbs2, jind2 = _once("ratio")
    s_ex1, li_ex1, jbs_ex1, jind_ex1 = _once("exact")
    s_ex2, li_ex2, jbs_ex2, jind_ex2 = _once("exact")
    s_leg, li_leg, _, _ = _once("legacy")
    np.savez(
        os.path.join(outdir, "r2.npz"),
        s=np.array([s1, s2]), li=np.array([li1, li2]),
        s_exact=np.array([s_ex1, s_ex2]), li_exact=np.array([li_ex1, li_ex2]),
        jbs_exact1=jbs_ex1, jbs_exact2=jbs_ex2,
        jind_exact1=jind_ex1, jind_exact2=jind_ex2,
        s_legacy=np.array([s_leg]), li_legacy=np.array([li_leg]),
        jbs1=jbs1, jbs2=jbs2, jind1=jind1, jind2=jind2,
        jbs_baseline=np.asarray(bl.j_BS, dtype=float),
        l_i_target=np.array([float(bl.l_i_target)]),
    )


@pytest.fixture(scope="module")
def sigma0_anchor(tmp_path_factory):
    """The R2 probe, in its own interpreter.

    A subprocess for the same reason the twins are: ``OFT_env`` is a
    per-process singleton, so a module that builds a solver in the pytest
    process makes ``pytest -m solver`` unrunnable alongside
    ``test_systematics.py``.  Keeping every solver call behind a subprocess
    means this module never instantiates one.
    """
    work = tmp_path_factory.mktemp("r2")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "r2", str(work)],
        env=dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"R2 probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    return np.load(str(work / "r2.npz"))


def test_sigma0_r2_ip_scale_is_unity(sigma0_anchor):
    """The author-requested golden: at sigma=0 the archived split sums to Ip,
    so the inductive renormalisation must return 1.000.

    Pre-fix this was 0.8043 on the operational case and 0.8373 here, because
    the root ran in solve_with_bootstrap's landed geometry against an
    uncalibrated Ip_target.
    """
    s = float(sigma0_anchor["s"][0])
    assert abs(s - 1.0) <= _S_ATOL, (
        f"sigma=0 R2 Ip scale is {s:.6f}, off unity by {abs(s - 1.0):.3e} "
        f"(bar {_S_ATOL:.0e}); the archived split is not being reproduced")


def test_sigma0_r2_recovers_the_recon_li(sigma0_anchor):
    """The downstream consequence the scale error used to cause."""
    target = float(sigma0_anchor["l_i_target"][0])
    got = float(sigma0_anchor["li"][0])
    assert abs(got - target) / target <= _LI_REL, (
        f"sigma=0 R2 l_i = {got:.5f} vs recon {target:.5f} "
        f"({100 * (got / target - 1):+.3f}%, bar {100 * _LI_REL:.1f}%)")


def test_sigma0_r2_reproduces_the_baseline_jbs(sigma0_anchor):
    """The premise of the invariant: at sigma=0 SWB must return the baseline
    bootstrap, to the same bar the sigma0 guard uses."""
    jbs_bl = np.asarray(sigma0_anchor["jbs_baseline"], dtype=float)
    peak = float(np.max(np.abs(jbs_bl)))
    dev = float(np.max(np.abs(
        np.asarray(sigma0_anchor["jbs1"], dtype=float) - jbs_bl)))
    assert dev / peak <= _JBS_FRAC, (
        f"sigma=0 j_BS is {100 * dev / peak:.3f}% of peak from the baseline "
        f"split (bar {100 * _JBS_FRAC:.1f}%)")


def test_sigma0_r2_is_bit_reproducible(sigma0_anchor):
    """Two identical sigma=0 R2 calls must agree to the bit -- the anchor
    snapshot/restore must not inject any state drift."""
    d = sigma0_anchor
    assert float(d["s"][0]) == float(d["s"][1])
    assert float(d["li"][0]) == float(d["li"][1])
    np.testing.assert_array_equal(d["jbs1"], d["jbs2"],
                                  err_msg="sigma=0 R2 j_BS is not bit-reproducible")
    np.testing.assert_array_equal(d["jind1"], d["jind2"],
                                  err_msg="sigma=0 R2 j_inductive is not bit-reproducible")


def test_sigma0_r2_exact_measure_lands_in_its_own_budget(sigma0_anchor):
    """``BOUQUET_R2_IP_MODE=exact``: the FSA current integral instead of the
    ratio calibration (``utils.Ip_fsa_integral``).

    It does NOT land closer to 1.000 -- 3.25e-3 against the calibration's
    8.5e-4 -- and that is the expected, understood result: the calibration
    cancels every representation error by construction, while the measure
    additionally charges the draw for the reconstruction's own j_phi residual
    (+0.193% of Ip at the R2 state anchor) on top of the sigma=0 SWB residual
    (-0.085%).  -0.335% of inductive amplitude predicted, -0.325% measured.
    """
    s = float(sigma0_anchor["s_exact"][0])
    assert abs(s - 1.0) <= _S_ATOL_EXACT, (
        f"exact-mode sigma=0 Ip scale is {s:.6f}, off unity by "
        f"{abs(s - 1.0):.3e} (bar {_S_ATOL_EXACT:.0e}) -- larger than the "
        f"residual budget accounts for")
    assert abs(s - 1.0) > _S_ATOL, (
        f"exact mode returned {s:.6f}, inside the ratio mode's bar -- if the "
        f"archived split has become self-consistent with the anchor, the "
        f"default should be revisited (see _AnchorIpRenorm)")


def test_sigma0_r2_exact_measure_still_recovers_the_recon_li(sigma0_anchor):
    """The physics acceptance is unchanged by the change of measure: same 0.5%
    bar as the default path.  Measured +0.130% (default: +0.100%)."""
    target = float(sigma0_anchor["l_i_target"][0])
    got = float(sigma0_anchor["li_exact"][0])
    assert abs(got - target) / target <= _LI_REL, (
        f"exact-mode sigma=0 l_i = {got:.5f} vs recon {target:.5f} "
        f"({100 * (got / target - 1):+.3f}%, bar {100 * _LI_REL:.1f}%)")


def test_sigma0_r2_exact_measure_is_bit_reproducible(sigma0_anchor):
    """The FSA weights are captured off a ``copy_eq`` snapshot; two identical
    calls must agree to the bit, exactly as the default path does."""
    d = sigma0_anchor
    assert float(d["s_exact"][0]) == float(d["s_exact"][1])
    assert float(d["li_exact"][0]) == float(d["li_exact"][1])
    np.testing.assert_array_equal(d["jbs_exact1"], d["jbs_exact2"])
    np.testing.assert_array_equal(d["jind_exact1"], d["jind_exact2"])


def test_sigma0_r2_exact_measure_leaves_the_bootstrap_alone(sigma0_anchor):
    """Changing the measure must not move j_BS: route R2 holds the bootstrap
    fixed and moves only the ohmic drive."""
    np.testing.assert_array_equal(
        sigma0_anchor["jbs_exact1"], sigma0_anchor["jbs1"],
        err_msg="the exact measure changed the sigma=0 bootstrap")


def test_legacy_mode_still_shows_the_defect(sigma0_anchor):
    """Documents what was fixed: BOUQUET_R2_IP_MODE=legacy reproduces the old
    geometry error, so the acceptance above is not vacuous."""
    s_legacy = float(sigma0_anchor["s_legacy"][0])
    assert abs(s_legacy - 1.0) > 10 * _S_ATOL, (
        f"legacy mode returned {s_legacy:.6f}, which is already at unity -- "
        f"the fix may no longer be doing anything on this case")


# ---------------------------------------------------------------------------
#  subprocess entry point for the twin ensembles
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Every solver call in this module runs here, in a fresh interpreter:
    #   python tests/test_seeded_reproducibility.py ensemble <header>
    #   python tests/test_seeded_reproducibility.py r2       <outdir>
    # OFT_env is a per-process singleton, so keeping the pytest process free
    # of solvers is what lets this module coexist with test_systematics.py in
    # one `pytest -m solver` run -- and, for the twins, it is also the only
    # honest way to compare two runs' solved equilibria.  pytest never
    # executes this block.
    _oft_importable()
    _WHAT, _ARG = sys.argv[1], sys.argv[2]
    if _WHAT == "ensemble":
        _generate_one_ensemble(_ARG)
    elif _WHAT == "r2":
        _run_r2_probe(_ARG)
    else:
        raise SystemExit(f"unknown subcommand {_WHAT!r}")
