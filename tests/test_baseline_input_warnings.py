"""The baseline's input warnings reach the user on the unified engine route.

``prepare_engine_baseline`` reads the source inside the captured solver
output (``capture_native_output``: stderr redirected into
``Baseline.reconstruction_log``), so a warning issued while the dd or the
p-file is read was never seen -- and Python's per-location registry then
marked it as shown for the session.  The warnings are now collected there
(``bouquet.utils.deferred_warnings``) and issued once per run where the
reconstruction summary is printed.

Fast: the engine's solver-facing steps are stubbed; the warnings come from
the real reading functions (``_dd_total_zeff``, ``resolve_p_fast_input``)
called inside the real capture.
"""

import os
import sys
import types
import warnings

import numpy as np
import pytest

from bouquet import baseline as B
from bouquet.config import FixedComponentsConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_GEQ = os.path.join(_DATA, "g_synthetic_pfast.geqdsk")
_PFILE = os.path.join(_DATA, "p_synthetic.peqdsk")
_MESH = os.path.join(_HERE, os.pardir, "examples", "D3D-like",
                     "DIIID_mesh.h5")

ZEFF_MSG = "closer to the THERMAL-only numerator"
PB_MSG = "two sources of p_fast"


def _cfg(kind):
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                ReconstructionSource, SolverConfig)
    src = (ImasSource(ids_path="dd.json", time=1.0) if kind == "ids" else
           ReconstructionSource(geqdsk_path=_GEQ, profiles_path=_PFILE))
    return BouquetConfig(
        source=src, solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))


def _zeff_read():
    """The dd's stored zeff is the thermal numerator of a beam dd."""
    from bouquet.io.imas import _dd_total_zeff
    psi = np.linspace(0.0, 1.0, 21)
    ne = 5e19 * (1.0 - 0.8 * psi ** 2)
    zth = 1.2 + 0.0 * psi
    z2 = 0.2 * ne * (1.0 - psi)
    _dd_total_zeff({"zeff": zth}, zth, z2, ne, psi)


def _pfast_read():
    """fixed_components.p_fast beside a p-file carrying pb."""
    fc = FixedComponentsConfig()
    fc.p_fast = np.full(5, 1.0e3)
    fc.psi_N = np.linspace(0.0, 1.0, 5)
    kin = B._load_kinetic_profiles(types.SimpleNamespace(profiles_path=_PFILE))
    B.resolve_p_fast_input(fc, kin, _PFILE, None)


def _stub_engine(monkeypatch, kind, read_hook):
    """Stub every solver-facing step of prepare_engine_baseline; the source
    read calls *read_hook* inside the real capture."""
    import bouquet.adapters as A
    import bouquet.engine as E
    import bouquet.solver_geometry as SG
    import bouquet.utils as U

    class _Adapter:
        psi_pad = 0.01

        def __init__(self, *a, **k):
            g = dict(R=1.7, Z=0.0, a=0.6, kappa=1.7, delta=0.3)
            self.eqdsk = types.SimpleNamespace(
                geometry={k_: [v] for k_, v in g.items()})

        def read(self):
            read_hook()
            return types.SimpleNamespace(
                rows={}, pressure=None,
                kind="gfile" if kind == "gfile" else "ids")

    def _bl(*a, **k):
        return types.SimpleNamespace(reconstruction_metrics=None, engine=None,
                                     l_i_target=1.0, reconstruction_log=None)

    def _resolve_baseline(cfg, mygs):
        return object()

    monkeypatch.setattr(A, "GFileAdapter", _Adapter)
    monkeypatch.setattr(A, "IdsAdapter", _Adapter)
    monkeypatch.setattr(B, "resolve_baseline", _resolve_baseline)
    monkeypatch.setattr(U, "_shape_from_boundary",
                        lambda _rz: (1.7, 0.0, 0.6, 1.7, 0.3))
    monkeypatch.setattr(E, "_geometry_isoflux", lambda _g: dict(
        pts=np.zeros((4, 2)), w=np.ones(4), origin="stub"))
    monkeypatch.setattr(SG, "isoflux_stamp", lambda *a: {})
    monkeypatch.setattr(E, "TokaMakerBackend", lambda *a, **k: object())
    monkeypatch.setattr(E, "reconstruction_coil_reg",
                        lambda _gs: {"record": {}, "terms": None})
    monkeypatch.setattr(E, "reconstruct", lambda *a, **k: (
        types.SimpleNamespace(delivered_meas=None), {"converged": True},
        {"solves": 1}))
    monkeypatch.setattr(E, "_gfile_baseline", _bl)
    monkeypatch.setattr(E, "_ids_baseline", _bl)
    monkeypatch.setattr(E, "mse_out", lambda _bl: None)
    monkeypatch.setattr(E, "delivered_axis_reference", lambda *a, **k: {})
    monkeypatch.setattr(E, "bistability_check_on", lambda _gc: False)


def _run(kind):
    from bouquet.engine import prepare_engine_baseline
    gs = types.SimpleNamespace(set_isoflux=lambda *a, **k: None,
                               init_psi=lambda *a, **k: None)
    bq = types.SimpleNamespace(
        config=_cfg(kind), mygs=gs,
        _geom=types.SimpleNamespace(boundary_RZ=None, iso_balance=None),
        _seed_coil_init=lambda _gs: None)
    return prepare_engine_baseline(bq)


@pytest.mark.skipif(not os.path.exists(_MESH), reason="example mesh missing")
@pytest.mark.parametrize("kind, hook, msg", [
    ("ids", _zeff_read, ZEFF_MSG),
    ("gfile", _pfast_read, PB_MSG),
])
def test_the_input_warning_reaches_the_user_once_per_run(
        monkeypatch, capfd, kind, hook, msg):
    _check_once_per_run(monkeypatch, capfd, kind, hook, msg)


NI_WINDOW_MSG = "floored at 0 on"


@pytest.mark.skipif(not os.path.exists(_MESH), reason="example mesh missing")
def test_the_sub_route_n_i_window_warning_reaches_the_user(
        monkeypatch, capfd, tmp_path):
    """ida_hybrid ni_source="standard": the n_i floor/cap warning of
    _merge_ida_kinetics, raised inside the captured dd read, is issued after
    the reconstruction -- once per run, again on a second run."""
    import test_ni_fast_subtraction as NF
    from bouquet.io.imas import _merge_ida_kinetics
    cdf = str(tmp_path / "ida.cdf")
    ne, *_ = NF._ida_cdf(cdf)
    psi = np.linspace(0.0, 1.0, NF.N)
    zf = 0.2 * ne
    zth = np.full(NF.N, 1.3)
    zf[:5], zth[:5] = 0.1 * ne[:5], 0.5           # n_z < 0: capped

    def _hook():
        _merge_ida_kinetics(psi, ne, ne, zth, cdf, 1.0, NF.Z_IMP,
                            ni_source="standard", z_fast=zf, z2_fast=zf,
                            zeff_for_ni=zth)

    _check_once_per_run(monkeypatch, capfd, "ids", _hook, NI_WINDOW_MSG)


def _check_once_per_run(monkeypatch, capfd, kind, hook, msg):
    _stub_engine(monkeypatch, kind, hook)
    # exactly once per run (no duplicate from inside the capture)
    with pytest.warns(UserWarning) as rec:
        _run(kind)
    assert sum(msg in str(w.message) for w in rec) == 1
    # under the default filter: on stderr OUTSIDE the captured output, and
    # once per run -- a second run of the same input warns again
    capfd.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        # Python's own display (pytest records warnings instead): written
        # to whatever sys.stderr is when the warning is shown
        warnings.showwarning = (
            lambda m, c, f, n, file=None, line=None: sys.stderr.write(
                warnings.formatwarning(m, c, f, n, line)))
        bl1 = _run(kind)
        bl2 = _run(kind)
    err = capfd.readouterr().err
    assert err.count(msg) == 2
    for bl in (bl1, bl2):
        assert msg not in (bl.reconstruction_log or "")


def test_outside_a_run_the_reading_functions_warn_at_once():
    with pytest.warns(UserWarning, match=ZEFF_MSG):
        _zeff_read()
    from bouquet.utils import deferred_warnings, emit_deferred_warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with deferred_warnings() as got:
            _zeff_read()
            _zeff_read()                  # collected once
    assert len(got) == 1 and ZEFF_MSG in got[0][0]
    with pytest.warns(UserWarning, match=ZEFF_MSG):
        emit_deferred_warnings(got)


class _ProbeFailed(RuntimeError):
    pass


def _raise(*a, **k):
    raise _ProbeFailed("probe failed")


@pytest.mark.skipif(not os.path.exists(_MESH), reason="example mesh missing")
@pytest.mark.parametrize("step", ["bistability", "partition"])
def test_the_input_warning_survives_a_failure_after_the_capture(
        monkeypatch, step):
    """Review F5: the bistability check's cold solve or the partition
    probe's re-solve runs after the collector has closed; when it raises,
    the collected input warnings are still issued (once), then the error
    propagates."""
    import bouquet.boundary_balance as BB
    import bouquet.engine as E
    _stub_engine(monkeypatch, "ids", _zeff_read)
    geom = types.SimpleNamespace(boundary_RZ=None, iso_balance=None)
    if step == "bistability":
        monkeypatch.setattr(E, "bistability_check_on", lambda _gc: True)
        monkeypatch.setattr(E, "check_bistability", _raise)
    else:
        geom.iso_balance = {"total": 1.0}
        geom.record = lambda: {}
        monkeypatch.setattr(E, "partition_probe_on", lambda _bq: True)
        monkeypatch.setattr(E, "reconstruct", lambda *a, **k: (
            types.SimpleNamespace(delivered_meas=None,
                                  state=types.SimpleNamespace(request=None)),
            {"converged": True}, {"solves": 1}))
        monkeypatch.setattr(BB, "measure_partition", _raise)
    from bouquet.engine import prepare_engine_baseline
    gs = types.SimpleNamespace(set_isoflux=lambda *a, **k: None,
                               init_psi=lambda *a, **k: None)
    bq = types.SimpleNamespace(config=_cfg("ids"), mygs=gs, _geom=geom,
                               _seed_coil_init=lambda _gs: None)
    with pytest.warns(UserWarning) as rec:
        with pytest.raises(_ProbeFailed):
            prepare_engine_baseline(bq)
    assert sum(ZEFF_MSG in str(w.message) for w in rec) == 1
