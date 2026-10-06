"""What a g-file written by the LIVE solver contains (``pytest -m solver``).

The solver-marked twin of
``tests/test_edge_pressure_baseline_gfile.py::
test_a_written_gfile_parses_back_to_the_delivered_frame`` (finding 6 of the
2026-10-06 second-pass review: nothing asserted the CONTENTS of a written
g-file, so a change in the solver's ``save_eqdsk(lcfs_pressure=)`` would
have passed every suite).

Each arm runs ``tests/probes/probe_baseline_gfile_frame.py`` in its own
interpreter (``OFT_env`` is a per-process singleton) on the synthetic g-file
+ p-file example.  The probe builds the baseline, writes the
reconstruction's g-file with ``Bouquet.save_baseline_eqdsk``, a BARE
``save_eqdsk`` of the same state (the solver frame), and runs ``generate()``
with one draw.  This test parses the written files back with the
repository's reader (``bouquet.io.geqdsk._read_geqdsk``) and asserts:

* under ``"offset"``: ``PRES`` of the reconstruction's file is the bare
  file's plus the DELIVERED ``p_sep`` (the record's ``p_sep_applied``) at
  every point, the edge included; under ``"legacy"``: ``p_sep_applied`` is
  0 and ``PRES`` is the bare file's.  The bare file's own edge value is the
  solver's pressure on the last (``lcfs_pad``-truncated) surface, not zero
  (4.8 Pa at d874822 on this example, SOLVER_RESULTS figure 10), so the edge
  is compared as ``PRES_edge = p_sep + PRES_bare_edge``;
* ``PPRIME``, ``QPSI``, ``FPOL``, ``FFPRIM`` and the boundary of the two
  files agree within the format's precision (the offset reaches ``PRES``
  alone) -- the q, F and boundary of the same solver state through two
  saves;
(That the archive's ``_baseline`` g-file is written by the same save call
as the reconstruction's is pinned solver-free in
``tests/test_edge_pressure_baseline_gfile.py``; the archive is a re-solved
state, so its ``PRES`` is not compared point by point here.)

NOT run when this test was written (2026-10-06: no solver runs in that
task); its numbers are owed on the next solver-suite run.  Synthetic inputs
only; one thread.
"""
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "probes", "probe_baseline_gfile_frame.py")
sys.path.insert(0, os.path.join(_HERE, "probes"))
import measure_engine as ME  # noqa: E402

pytestmark = pytest.mark.solver

_files_ok = all(os.path.isfile(p) for p in (ME._GEQ, ME._PF, ME._MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and ME.oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")

ARMS = [("legacy", "offset"), ("legacy", "legacy"),
        ("unified", "offset"), ("unified", "legacy")]


def _parse_tol(a):
    """The g-file's own precision: 16.9E fields, ten significant digits."""
    return 1e-9 * max(1.0, float(np.max(np.abs(np.asarray(a, dtype=float)))))


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    import json
    out = {}
    for eng, sep in ARMS:
        work = str(tmp_path_factory.mktemp(f"gfile_{eng}_{sep}"))
        proc = subprocess.run(
            [sys.executable, _PROBE, work, "--source", "recon",
             "--engine", eng, "--sep", sep],
            env=_harness.subprocess_env(OMP_NUM_THREADS="1",
                                        MKL_NUM_THREADS="1",
                                        OPENBLAS_NUM_THREADS="1",
                                        MPLBACKEND="Agg"),
            capture_output=True, text=True)
        tag = f"recon_{eng}_{sep}"
        js = os.path.join(work, f"baseline_gfile_frame_{tag}.json")
        rec = None
        if os.path.exists(js):
            with open(js) as fh:
                rec = json.load(fh)
        out[(eng, sep)] = dict(
            rc=proc.returncode, rec=rec,
            log=proc.stdout[-3000:] + "\n" + proc.stderr[-3000:],
            files={k: os.path.join(work, f"frame_{tag}_{k}.geqdsk")
                   for k in ("recon", "bare")})
    return out


def _parsed(written, eng, sep):
    from bouquet.io.geqdsk import _read_geqdsk
    w = written[(eng, sep)]
    assert w["rc"] == 0 and w["rec"] is not None, w["log"]
    return w["rec"], {k: _read_geqdsk(p) for k, p in w["files"].items()}


@solver_only
@pytest.mark.parametrize("eng, sep", ARMS)
def test_the_written_pres_is_the_solver_frame_plus_the_delivered_p_sep(
        written, eng, sep):
    rec, g = _parsed(written, eng, sep)
    p_sep = float(rec["p_sep_applied"])
    if sep == "offset":
        assert p_sep > 0.0
    else:
        assert p_sep == 0.0
    pr, pb = (np.asarray(g[k]["PRES"], dtype=float) for k in ("recon",
                                                                "bare"))
    tol = _parse_tol(pr)
    np.testing.assert_allclose(pr - pb, p_sep, rtol=0, atol=2 * tol)
    assert abs((pr[-1] - pb[-1]) - p_sep) <= 2 * tol


@solver_only
@pytest.mark.parametrize("eng, sep", ARMS)
def test_everything_but_pres_round_trips_unchanged(written, eng, sep):
    _, g = _parsed(written, eng, sep)
    r, b = g["recon"], g["bare"]
    for name in ("PPRIME", "QPSI", "FPOL", "FFPRIM", "RBBBS", "ZBBBS"):
        np.testing.assert_allclose(r[name], b[name], rtol=0,
                                   atol=_parse_tol(b[name]), err_msg=name)
    for name in ("SIMAG", "SIBRY", "CURRENT", "BCENTR", "RMAXIS"):
        assert float(r[name]) == pytest.approx(float(b[name]), rel=1e-9,
                                               abs=0.0), name
