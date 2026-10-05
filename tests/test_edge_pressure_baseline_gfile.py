"""Every g-file bouquet writes carries the FULL pressure under
``separatrix_pressure="offset"`` -- the RECONSTRUCTION's own included.

The defect this pins: the archive's ``_baseline`` g-file and every draw's
g-file (``generate()``) were written with ``lcfs_pressure = p_sep``, but the
package offered no writer for the reconstruction's own delivered state, so a
caller wrote it with a bare ``mygs.save_eqdsk``: the SOLVER frame, ``PRES``
lower by ``p_sep`` everywhere than the archive baseline's for the same
state.  ``Bouquet.save_baseline_eqdsk`` (through
``bouquet.edge_pressure.save_full_pressure_eqdsk``) is that writer.

* the reconstruction's written g-file and the archive baseline's are the
  same bytes for the same state, under ``"offset"`` and ``"legacy"`` (the
  same save call: grid, padding, truncation and ``lcfs_pressure``);
* under ``"offset"`` their ``PRES`` edge value is the baseline's ``p_sep``
  and ``PPRIME`` is untouched; each draw's g-file carries that draw's OWN
  ``p_sep``; under ``"legacy"`` nothing is added (the call is a bare save);
* no double counting: the writer refuses an explicit ``lcfs_pressure``, the
  records are not touched, the archived record adds ``p_sep`` once;
* the writer refuses a missing record and a solver state that a later solve
  has moved; ``prepare_baseline`` arms it on the engine path.

WHAT THIS FILE CAN AND CANNOT TEST.  The stand-in solver writes the
repository's synthetic g-file with its ``PRES`` shifted to the solver frame
plus the requested ``lcfs_pressure`` -- i.e. the stand-in itself implements
the solver's ``save_eqdsk(lcfs_pressure=)`` contract.  A ``PRES`` /
``PPRIME`` read back from its files would therefore test the STAND-IN, not
bouquet.  So the assertions here are on what BOUQUET decides and computes:
the keyword arguments of every ``save_eqdsk`` call (above all the
``lcfs_pressure`` value: the baseline's ``p_sep_applied``, each draw's own
``p_sep``, or absent under "legacy"), that the reconstruction's save is the
SAME call as the archive baseline's (same bytes for the same state), the
records, and the refusals.  What the solver then writes -- the edge ``PRES``
of a real g-file, ``PPRIME`` untouched -- is checked ONLY by the live-solver
probe ``tests/probes/probe_baseline_gfile_frame.py``, which has NOT been run
since it was written (owed on the cluster).
Solver-free; synthetic inputs only.
"""
import contextlib
import copy
import io
import os
import types
import warnings

import h5py
import numpy as np
import pytest

import _engine_toy as T
import test_engine_draws as TD
from _engine_fake_gs import EXAMPLE_GEQDSK, FakeTokaMaker
from bouquet import edge_pressure as EP
from bouquet import engine_draws as ED
from bouquet.io.geqdsk import _read_geqdsk, _write_geqdsk

SEPS = ("legacy", "offset")


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _write_frame_gfile(filename, lcfs_pressure=0.0):
    """TokaMaker's save_eqdsk contract on the synthetic g-file: the
    solver's pressure is zero at the boundary; ``lcfs_pressure`` is added
    to ``PRES``; ``PPRIME`` is not touched."""
    raw = _read_geqdsk(EXAMPLE_GEQDSK)
    pres = np.asarray(raw["PRES"], dtype=float)
    raw["PRES"] = pres - pres[-1] + float(lcfs_pressure)
    _write_geqdsk(raw, filename)


class FrameFake(FakeTokaMaker):
    """FakeTokaMaker whose g-file carries the requested pressure frame and
    whose saves are recorded."""

    def __init__(self, toy, **kw):
        super().__init__(toy, **kw)
        self.saves = []

    def save_eqdsk(self, filename, **kw):
        self.saves.append(dict(kw))
        _write_frame_gfile(filename, kw.get("lcfs_pressure", 0.0))


def _bouquet(fake, edge_record, psi_pad=T.PAD):
    """The Bouquet surface the writer reads: a baseline carrying the
    engine record's edge-pressure block (``_gfile_baseline`` /
    ``_ids_baseline`` set ``Baseline.edge_pressure`` to exactly that), the
    live solver and the source's psi_pad; armed as prepare_baseline arms
    it."""
    from bouquet.run import Bouquet
    b = Bouquet.__new__(Bouquet)
    b.config = types.SimpleNamespace(
        source=types.SimpleNamespace(psi_pad=psi_pad))
    b.mygs = fake
    b.baseline = types.SimpleNamespace(edge_pressure=copy.deepcopy(
        edge_record))
    b._remember_baseline_state()
    return b


def _generate(fake, eng, res, h, n=3):
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    ctx = TD._ctx(eng, res)
    unc = TD._unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD)
    initialize_equilibrium_database(h)
    k = ctx.native
    return _q(
        generate_bouquet, fake, T.PSI, n, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=0.05,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=12345,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=[], engine_draw=G,
        coil_filter="legacy")


@pytest.mark.parametrize("sep", SEPS)
def test_the_reconstruction_gfile_is_the_archive_baselines_pressure_frame(
        tmp_path, monkeypatch, sep):
    eng, res, rec, toy = TD._recon(separatrix_pressure=sep)
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = FrameFake(toy)
    p_sep = float(np.asarray(eng.c.pressure, dtype=float)[-1])
    assert p_sep > 0.0          # the case exercises a non-zero offset
    rec_before = copy.deepcopy(rec["edge_pressure"])

    # the reconstruction's own g-file, written as a harness would right
    # after prepare_baseline() (before generate() moves the solver)
    b = _bouquet(fake, rec["edge_pressure"])
    recon_path = str(tmp_path / "recon.geqdsk")
    assert _q(b.save_baseline_eqdsk, recon_path) == recon_path
    assert len(fake.saves) == 1

    h = str(tmp_path / "arch")
    diags = _generate(fake, eng, res, h)
    assert len(diags) == 3
    # the baseline re-save of generate(), then one per draw
    assert len(fake.saves) == 5
    # the SAME save call as the archive's _baseline g-file
    assert fake.saves[0] == fake.saves[1]
    assert fake.saves[0] == dict(
        nr=257, nz=257, truncate_eq=True, lcfs_pad=T.PAD,
        **({} if sep == "legacy" else {"lcfs_pressure": p_sep}))
    if sep == "legacy":
        # legacy: nothing added, anywhere -- exactly a bare save
        assert all("lcfs_pressure" not in kw for kw in fake.saves)

    with h5py.File(h + ".h5", "r") as hf:
        root = hf["scan/0"] if "scan" in hf else hf
        arch_bytes = bytes(root["_baseline"]["eqdsk"][()])
        draw_bytes = [bytes(root[str(i)]["eqdsk"][()]) for i in range(3)]
        p_draw = [float(np.asarray(root[str(i)]["pressure"])[-1])
                  for i in range(3)]
    with open(recon_path, "rb") as fh:
        recon_bytes = fh.read()
    # same state, same frame: the same bytes
    assert recon_bytes == arch_bytes

    # each draw's save asks for ITS OWN p_sep -- the edge value of the
    # pressure the draw archived -- (offset) or for nothing (legacy).  (What
    # the solver then writes into PRES is the live probe's to check.)
    draw_saves = fake.saves[2:]
    assert len(draw_saves) == 3 and len(draw_bytes) == 3
    for kw, pd in zip(draw_saves, p_draw):
        if sep == "offset":
            assert kw["lcfs_pressure"] == pd
        else:
            assert "lcfs_pressure" not in kw
    if sep == "offset":
        assert len({p_sep, *p_draw}) == 4

    # no double counting: the writer leaves the records alone, and the
    # archived baseline record adds p_sep once
    assert b.baseline.edge_pressure == rec_before
    arc = EP.load_record(h)
    assert arc["p_sep_applied"] == (p_sep if sep == "offset" else 0.0)
    assert rec["edge_pressure"]["p_sep_applied"] == arc["p_sep_applied"]
    fr = arc.get("frames")
    if fr is not None:
        assert fr["full"]["P_ax"] - fr["solver"]["P_ax"] == pytest.approx(
            arc["p_sep_applied"], rel=0, abs=1e-9 * max(1.0, p_sep))


def test_the_writer_asks_for_the_record_offset_and_nothing_else():
    eng, res, rec, toy = TD._recon(separatrix_pressure="offset")
    fake = FrameFake(toy)
    p = rec["edge_pressure"]["p_sep_applied"]
    assert p == float(np.asarray(eng.c.pressure, dtype=float)[-1]) > 0.0
    EP.save_full_pressure_eqdsk(fake, os.devnull, p, nr=33, nz=33)
    EP.save_full_pressure_eqdsk(fake, os.devnull, 0.0, nr=33, nz=33)
    assert fake.saves == [dict(nr=33, nz=33, lcfs_pressure=p),
                          dict(nr=33, nz=33)]
    with pytest.raises(ValueError, match="lcfs_pressure"):
        EP.save_full_pressure_eqdsk(fake, os.devnull, p, lcfs_pressure=p)


def test_the_delivered_offset_is_read_from_the_record_or_refused():
    assert EP.delivered_p_sep({"p_sep_applied": 640.5}) == 640.5
    assert EP.delivered_p_sep({"p_sep_applied": 0.0}) == 0.0
    for bad in (None, {}, {"p_sep_applied": None}):
        with pytest.raises(ValueError, match="p_sep_applied"):
            EP.delivered_p_sep(bad)
    with pytest.raises(ValueError, match="not finite"):
        EP.delivered_p_sep({"p_sep_applied": float("nan")})


def test_the_writer_refuses_a_moved_solver_or_no_baseline(tmp_path):
    eng, res, rec, toy = TD._recon(separatrix_pressure="offset")
    fake = FrameFake(toy)
    b = _bouquet(fake, rec["edge_pressure"])
    # a later solve moved the state the baseline left
    fake.get_psi = lambda normalized=True: np.ones(16)
    with pytest.raises(RuntimeError, match="no longer holds"):
        b.save_baseline_eqdsk(str(tmp_path / "x.geqdsk"))
    assert fake.saves == []
    # never armed (no prepare_baseline)
    b2 = _bouquet(FrameFake(toy), rec["edge_pressure"])
    b2._baseline_psi = None
    with pytest.raises(RuntimeError, match="no longer holds"):
        b2.save_baseline_eqdsk(str(tmp_path / "y.geqdsk"))
    b2.baseline = None
    with pytest.raises(RuntimeError, match="prepare_baseline"):
        b2.save_baseline_eqdsk(str(tmp_path / "z.geqdsk"))
    # a baseline without an edge-pressure record: no frame is guessed
    b3 = _bouquet(FrameFake(toy), None)
    with pytest.raises(ValueError, match="p_sep_applied"):
        b3.save_baseline_eqdsk(str(tmp_path / "w.geqdsk"))
    assert b3.mygs.saves == []


@pytest.mark.parametrize("sep", SEPS)
def test_an_engine_prepare_baseline_arms_the_writer_with_its_own_p_sep(
        tmp_path, monkeypatch, sep):
    """The real wiring (adapter -> engine -> Baseline) on the synthetic
    g-file + p-file, with the toy backend: the Baseline's record is the
    contract's p_sep and the reconstruction's g-file is asked for it."""
    import bouquet as bq
    import bouquet.engine as be
    import test_engine_wiring as TW
    made = {}

    def _backend(mygs, contract, **kw):
        made["c"] = contract
        made["edge"] = kw.get("edge_pressure")
        return T.ToyGS(psi=contract.psi_N, Ip=contract.Ip)

    monkeypatch.setattr(be, "TokaMakerBackend", _backend)
    monkeypatch.setattr(be, "_lcfs_deviation_mm",
                        lambda mygs, pts: (2.5, 7.0))
    b = bq.Bouquet.from_geqdsk(TW._GEQ, profiles=TW._PF, mesh=TW._MESH,
                               n_draws=1, reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip"]
    g.separatrix_pressure = sep

    class _GS(TW._FakeGS):
        def __init__(self):
            super().__init__()
            self.saves = []

        def save_eqdsk(self, filename, **kw):
            self.saves.append(dict(kw))
            _write_frame_gfile(filename, kw.get("lcfs_pressure", 0.0))

    b.mygs = _GS()
    bl = _q(b.prepare_baseline)
    p_sep = float(np.asarray(made["c"].pressure, dtype=float)[-1])
    assert p_sep > 0.0
    want = p_sep if sep == "offset" else 0.0
    assert bl.edge_pressure["p_sep"] == p_sep
    assert bl.edge_pressure["p_sep_applied"] == want
    path = str(tmp_path / "recon.geqdsk")
    b.save_baseline_eqdsk(path)
    assert b.mygs.saves == [dict(
        nr=257, nz=257, truncate_eq=True,
        lcfs_pad=float(b.config.source.psi_pad),
        **({"lcfs_pressure": p_sep} if sep == "offset" else {}))]
