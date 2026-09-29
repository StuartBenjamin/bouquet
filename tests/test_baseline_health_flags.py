"""Baseline health under the self-consistent loop -- fast half (no solver).

* A geqdsk baseline whose loop did not converge under
  ``jbs_loop_on_fail="flag"`` is delivered, so it must carry the same
  ``closure_limited`` flag (+ the loop's reason) the IMAS path sets.

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
