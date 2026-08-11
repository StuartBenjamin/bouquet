"""Contract tests for the ``jphi_baseline`` psi re-init and its counted
fallback -- issue #24's THIRD converged-on-entry / masked-failure site.

Split out of the step-7 corrective-iteration package (#25, PR #27) because
this is the one piece of it that moves a repo golden: re-initialising psi
lands the baseline solve on a different member of the free-boundary coil
degeneracy, which shifts the mode-1 pinned max coil drift.  The golden was
regenerated for it under an explicit approval; see the PR and the commit that
carries the regenerated fixture.

Two things are pinned here, both solve-free:

* the fallback is COUNTED (``ANCHOR_MASKED_FAILURES["jphi_baseline"]``) and
  says what it reverts to, instead of printing one line and moving on; and
* the psi re-init happens, uses the landed LCFS, and precedes the solve it is
  meant to protect.

They cannot catch a numerical regression inside a GS solve -- the seeded runs
and the regenerated golden do that.  What they catch is that the guard is
still wired in at all.
"""
from __future__ import annotations

import inspect
import re


def test_jphi_baseline_is_a_counted_masked_failure_site():
    from bouquet.TokaMaker_interface import (ANCHOR_MASKED_FAILURES,
                                             _count_masked_anchor_failure)

    assert "jphi_baseline" in ANCHOR_MASKED_FAILURES, (
        "the jphi-linterp baseline solve is #24's third masked site and must "
        "have its own counter key")
    before = ANCHOR_MASKED_FAILURES["jphi_baseline"]
    try:
        _count_masked_anchor_failure("jphi_baseline",
                                     RuntimeError('Exceeded "maxits"'))
        assert ANCHOR_MASKED_FAILURES["jphi_baseline"] == before + 1
    finally:
        ANCHOR_MASKED_FAILURES["jphi_baseline"] = before


def test_the_jphi_baseline_fallback_counts_and_names_what_reverts():
    """The old handler printed one line and moved on, so an archive could not
    say whether its per-draw diagnostics were referenced to the jphi-linterp
    baseline or to recon's inverse-mode LCFS -- references ~1.6 mm / ~0.9 % in
    l_i(3) apart.  Now it is counted, and the revert is spelled out."""
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    blk = re.search(r"\[jphi-baseline\] solve failed.*?\)\n", src, re.S)
    assert blk, "the jphi-baseline failure handler is gone"
    handler = src[max(0, blk.start() - 600):blk.end()]
    assert '_count_masked_anchor_failure("jphi_baseline"' in handler, (
        "the jphi-baseline fallback is still an uncounted mask (issue #24)")
    assert "REVERTING" in handler, \
        "the fallback no longer says what it reverts to"


def test_the_jphi_baseline_solve_reinitialises_psi():
    """#24's converged-on-entry guard, same treatment as the sigma=0 anchor got
    in #22: re-init psi from the landed LCFS before the forward solve, and do
    not make a failed trace fatal."""
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    blk = re.search(r"if jphi_baseline:(.*?)# ---- Boundary-shift diagnostic",
                    src, re.S)
    assert blk, "could not locate the jphi_baseline block"
    body = blk.group(1)
    assert "init_psi(" in body, (
        "the jphi-linterp baseline solve no longer re-initialises psi; it is "
        "the third converged-on-entry site (issue #24)")
    assert "_shape_from_boundary" in body and "safe_trace_surf" in body
    # the re-init must precede the solve it protects
    assert body.index("init_psi(") < body.index("mygs.solve()"), \
        "psi is re-initialised AFTER the solve it is supposed to protect"
