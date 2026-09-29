"""Baseline health under the self-consistent loop -- fast half (no solver).

* A geqdsk baseline whose loop did not converge under
  ``jbs_loop_on_fail="flag"`` is delivered, so it must carry the same
  ``closure_limited`` flag (+ the loop's reason) the IMAS path sets.
* A baseline build that fails part-way leaves NO usable baseline (and never a
  previous slice's), so ``generate()`` refuses instead of drawing from a
  half-built one.

Synthetic inputs only; no device data.
"""
import inspect
from types import SimpleNamespace

import pytest


def _bq(metrics):
    from bouquet.run import Bouquet
    b = Bouquet.__new__(Bouquet)
    b.baseline = SimpleNamespace(reconstruction_metrics=metrics)
    return b


def _rec(converged):
    return dict(converged=converged, stop_reason="pass ceiling 8 reached",
                final=dict(r_j=3.2e-3, r_I=4.0e-5))


def test_a_flag_mode_nonconverged_recon_baseline_is_closure_limited():
    b = _bq(dict(li=0.8, closure_limited_reasons=("an earlier reason",),
                 jbs_loop=_rec(False)))
    with pytest.warns(RuntimeWarning, match="did NOT converge"):
        b._flag_nonconverged_recon_loop()
    m = b.baseline.reconstruction_metrics
    assert m["closure_limited"] is True and m["jbs_converged"] is False
    rs = m["closure_limited_reasons"]
    assert rs[0] == "an earlier reason"
    assert any(r.startswith("j_BS loop: did not converge") for r in rs), rs
    # idempotent: a second call adds nothing
    with pytest.warns(RuntimeWarning):
        b._flag_nonconverged_recon_loop()
    assert b.baseline.reconstruction_metrics["closure_limited_reasons"] == rs


@pytest.mark.parametrize("metrics", [
    dict(li=0.8, jbs_loop=_rec(True)),       # converged: untouched
    dict(li=0.8),                            # legacy (no loop): untouched
    None])
def test_a_converged_or_legacy_recon_baseline_is_untouched(metrics):
    b = _bq(None if metrics is None else dict(metrics))
    b._flag_nonconverged_recon_loop()
    assert b.baseline.reconstruction_metrics == metrics


def test_prepare_baseline_applies_the_flag():
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet.prepare_baseline)
    assert "self._flag_nonconverged_recon_loop()" in src


# ---------------------------------------------------------------------------
#  a failed baseline build leaves no usable baseline
# ---------------------------------------------------------------------------
def _bq_for_prepare(monkeypatch, resolve, fail_in=None):
    import bouquet.baseline as B
    from bouquet.run import Bouquet
    b = Bouquet.__new__(Bouquet)
    b.mygs = object()
    b.baseline = "a previous slice's baseline"
    b._failed_baseline = None
    b.config = SimpleNamespace(
        source=SimpleNamespace(),
        generation=SimpleNamespace(single_profile_jphi=False))
    monkeypatch.setattr(b, "_check_jbs_loop_workflow", lambda gc: None)
    monkeypatch.setattr(b, "_check_structured_mse_reachable", lambda c: None)
    monkeypatch.setattr(B, "resolve_baseline", resolve)
    if fail_in is not None:
        monkeypatch.setattr(b, fail_in, _boom)
    return b


def _boom(*a, **k):
    from bouquet.jbs_loop import JBSNotConverged
    raise JBSNotConverged("self-consistent j_BS loop did not converge", {})


def test_a_failure_after_the_baseline_object_exists_leaves_none(monkeypatch):
    from bouquet.jbs_loop import JBSNotConverged
    half = SimpleNamespace(reconstruction_metrics=None, j_phi="pass-3 state")
    b = _bq_for_prepare(monkeypatch, lambda cfg, gs: half,
                        fail_in="_flag_nonconverged_recon_loop")
    with pytest.raises(JBSNotConverged):
        b.prepare_baseline()
    assert b.baseline is None
    assert b._failed_baseline is half          # kept for debugging only
    # and generate() refuses rather than running on the half-built object
    with pytest.raises(ValueError, match="prepare_baseline"):
        b.generate()


def test_a_failed_rebuild_does_not_leave_the_previous_baseline(monkeypatch):
    def _resolve_fails(cfg, gs):
        raise RuntimeError("reconstruction failed")
    b = _bq_for_prepare(monkeypatch, _resolve_fails)
    with pytest.raises(RuntimeError):
        b.prepare_baseline()
    assert b.baseline is None


def test_a_successful_build_is_delivered(monkeypatch):
    ok = SimpleNamespace(reconstruction_metrics=None)
    b = _bq_for_prepare(monkeypatch, lambda cfg, gs: ok)
    assert b.prepare_baseline() is ok and b.baseline is ok
    assert b._failed_baseline is None
