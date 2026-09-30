"""The unified reconstruction engine -- live-solver half (``pytest -m solver``).

On both synthetic examples (the D3D-like g-file + p-file, and the OMAS
modelling source at t = 2.3043 s), with ``reconstruction_engine="unified"``:

* the reconstruction converges within the unchanged pass ceiling, and every
  row holds on the DELIVERED equilibrium (the loop residuals re-checked with
  ``check_delivered``, the g-file's hard l_i at the re-match secant's 1e-3,
  the soft rows' discrepancies at ``structured_li_tol``, q0 at ``q0_tol``);
* the distance-to-input table (l_i matched and free, q at like radii, q95,
  the q profile, the core / edge current, beta, W, the boundary, requested -
  achieved) is written to the probe's JSON, in the units of the three-state
  comparison report, for the orchestrator to compare -- no fidelity bar is
  asserted here;
* the delivered state re-solves to itself from its own stored request
  (``|dl_i| <= jbs_tol_li``, ``|dq0|, |dq95| <= jbs_tol_q0``, current
  ``<= jbs_rtol_j``);
* solve counts and wall time are recorded.

Every solver call runs in a subprocess of ``tests/probes/measure_engine.py``
(``OFT_env`` is a per-process singleton); the probe writes every number, pass
or fail.  Set ``BQ_ENGINE_PROBE_OUT=<dir>`` to keep the JSON.  Synthetic
inputs only.
"""
import os
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "probes"))
import measure_engine as ME  # noqa: E402

_files_ok = all(os.path.isfile(p) for p in (ME._OMAS, ME._GEQ, ME._PF,
                                            ME._MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and ME.oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")

#: the loop's own tolerances (GenerationConfig defaults) -- read, not tuned
_S = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3, tol_q0=2e-3, q0_tol=0.01,
          structured_li_tol=0.005)

_PARTS = ("recon", "imas", "imas_q0")


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("engine"))
    return {p: ME.run_part(p, work) for p in _PARTS}


def _need(d, stage):
    st = d["stages"].get(stage)
    assert st == "ok", (f"{d['part']} stage {stage!r} failed: {st}"
                        + (f"\nfatal: {d.get('fatal')}" if d.get("fatal")
                           else ""))
    return d[stage]


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("part", _PARTS)
def test_the_engine_reconstruction_converges_within_the_ceiling(parts, part):
    b = _need(parts[part], "build")
    assert b["converged"] and b["loop_converged"], b.get("delivered")
    for name, n in b["n_passes"].items():
        assert n <= b["max_passes"], (name, n)
    s = b["solves"]
    assert s["anchor"] == 2 and s["delivery"] == 2, s
    assert s["total"] == sum(v for k, v in s.items() if k != "total"), s
    assert b["wall_s"] > 0.0


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("part", _PARTS)
def test_every_row_holds_on_the_delivered_equilibrium(parts, part):
    b = _need(parts[part], "build")
    d = b["delivered"]
    assert d["ok"], d["misses"]
    c = d["checks"]
    assert c["loop"]["r_j"] <= _S["rtol_j"] and c["loop"]["r_I"] \
        <= _S["rtol_Ip"], c["loop"]
    assert c["dl_i"]["value"] <= _S["tol_li"]
    assert c["current_residual"]["value"] <= _S["rtol_j"]
    li = c["l_i"]
    if part == "recon":
        assert li["hard"] and abs(li["delivered"] - li["target"]) < 1e-3, li
    else:
        assert not li["hard"] and abs(li["error"]) \
            <= _S["structured_li_tol"], li
    if part == "imas_q0":
        q = c.get("q0")
        rows = b["engine_record"]["settings"]["rows_active"]
        if "q0" in rows:
            assert q is not None and abs(q["residual"]) <= _S["q0_tol"], q
    # l_i_target is the delivered state's l_i
    assert b["l_i_target"] == li["delivered"]


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("part", _PARTS)
def test_the_delivered_state_resolves_to_itself(parts, part):
    r = _need(parts[part], "resolve_self")
    assert abs(r["dl_i"]) <= _S["tol_li"], r
    assert abs(r["dq0"]) <= _S["tol_q0"], r
    assert abs(r["dq95"]) <= _S["tol_q0"], r
    assert r["dj_rel"] <= _S["rtol_j"], r


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("part", _PARTS)
def test_the_distance_to_input_table_is_written(parts, part):
    t = _need(parts[part], "distance")
    keys = (("li3_matched", "li1_free", "q_axis", "q_002", "q95",
             "q_profile_rel_pct", "jphi_vs_input_pct_of_peak",
             "requested_minus_achieved_pct_of_peak", "beta_n", "W_MHD_MJ",
             "beta_p", "lcfs_mm") if part.startswith("recon") else
            ("li3", "q_axis", "q95", "q_profile_rel_pct",
             "jphi_vs_equilibrium_jtor_pct_of_peak",
             "requested_minus_achieved_pct_of_peak", "lcfs_mm"))
    for k in keys:
        assert k in t, k
