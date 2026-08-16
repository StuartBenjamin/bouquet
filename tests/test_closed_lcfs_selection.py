"""LCFS contour selection must test closedness, not length (issue #33).

On a diverted equilibrium the psi = psi_LCFS level set carries the open
separatrix branch running to the divertor as well as the closed boundary.
The open branch spans the full vessel height, so it can be LONGER than the
LCFS -- and the old ``max(segs, key=len)`` then measured the boundary
against it, reporting 891.86 mm where the truth was 2.06 mm.

These tests are deliberately solve-free: they pin the selector's contract on
synthetic geometry that reproduces the failure in miniature, plus the
structural guard that neither call site has drifted back to a bare
length-max.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from bouquet.utils import (
    OpenLCFSContourWarning,
    select_closed_lcfs,
    _LCFS_CLOSURE_RTOL,
)

_SRC = Path(__file__).resolve().parents[1] / "bouquet"


def _closed_loop(n=64, r=0.6, cx=1.7, cy=0.0):
    """A closed contour, first vertex repeated exactly as matplotlib does."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
    return np.vstack([pts, pts[:1]])


def _open_leg(n=200, cx=1.7):
    """An open branch spanning the full vessel height -- ends far apart."""
    z = np.linspace(-1.2, 1.2, n)
    return np.column_stack([cx + 0.25 * np.sin(2.0 * z), z])


class TestClosednessBeatsLength:
    def test_the_longer_open_branch_does_not_win(self):
        closed, open_ = _closed_loop(n=64), _open_leg(n=200)
        assert len(open_) > len(closed), "fixture must exercise the defect"

        got = select_closed_lcfs([open_, closed])

        assert got is not None
        assert len(got) == len(closed)
        np.testing.assert_allclose(got, closed)

    def test_negative_control_length_max_picks_the_wrong_one(self):
        """Prove the fixture discriminates: the OLD rule fails on it."""
        closed, open_ = _closed_loop(n=64), _open_leg(n=200)

        legacy = max([open_, closed], key=len)

        assert len(legacy) == len(open_), (
            "fixture does not reproduce the defect, so the positive test "
            "above would pass even with the bug present"
        )

    def test_longest_of_several_closed_segments_wins(self):
        small, big = _closed_loop(n=32, r=0.3), _closed_loop(n=128, r=0.6)

        got = select_closed_lcfs([small, big, _open_leg()])

        assert len(got) == len(big)

    def test_all_open_falls_back_loudly(self):
        legs = [_open_leg(n=50), _open_leg(n=90)]

        with pytest.warns(OpenLCFSContourWarning, match="issue #33"):
            got = select_closed_lcfs(legs, context="unit test")

        assert len(got) == len(legs[1]), "fallback is still the longest"

    def test_a_closed_segment_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenLCFSContourWarning)
            select_closed_lcfs([_open_leg(), _closed_loop()])


class TestContract:
    def test_empty_and_degenerate_input_returns_none(self):
        assert select_closed_lcfs([]) is None
        # <= 4 vertices is discarded upstream and here alike
        assert select_closed_lcfs([np.zeros((3, 2))]) is None

    def test_closure_tolerance_scales_with_the_segment(self):
        """A gap just inside/outside the bar decides closedness."""
        loop = _closed_loop(n=64)
        diag = float(np.hypot(*np.ptp(loop, axis=0)))

        inside = loop.copy()
        inside[-1] = loop[0] + np.array([0.4 * _LCFS_CLOSURE_RTOL * diag, 0.0])
        assert select_closed_lcfs([inside]) is not None

        outside = loop.copy()
        outside[-1] = loop[0] + np.array([40.0 * _LCFS_CLOSURE_RTOL * diag, 0.0])
        with pytest.warns(OpenLCFSContourWarning):
            select_closed_lcfs([outside])

    def test_the_bar_is_tight_enough_to_reject_a_divertor_leg(self):
        leg = _open_leg()
        diag = float(np.hypot(*np.ptp(leg, axis=0)))
        gap = float(np.hypot(*(leg[0] - leg[-1])))

        assert gap > _LCFS_CLOSURE_RTOL * diag


class TestBothCallSitesUseTheHelper:
    """Structural guard: neither site may drift back to a bare length-max."""

    #: Whitespace-tolerant: ``key = len`` must not slip past the guard.
    _BARE_LENGTH_MAX = re.compile(
        r"max\s*\(\s*_?segs\s*,\s*key\s*=\s*len\s*\)"
    )

    @pytest.mark.parametrize("name", ["TokaMaker_interface.py", "plotting.py"])
    def test_no_bare_longest_segment_selection_remains(self, name):
        src = (_SRC / name).read_text()

        offenders = self._BARE_LENGTH_MAX.findall(src)

        assert not offenders, (
            f"{name} still selects the LCFS by raw length: {offenders}. "
            "Use utils.select_closed_lcfs (issue #33)."
        )

    @pytest.mark.parametrize(
        "variant",
        ["max(_segs, key=len)", "max(_segs, key = len)", "max( segs , key =len )"],
    )
    def test_the_guard_regex_catches_spacing_variants(self, variant):
        """Negative control: the guard must not be defeated by whitespace."""
        assert self._BARE_LENGTH_MAX.search(variant)

    @pytest.mark.parametrize("name", ["TokaMaker_interface.py", "plotting.py"])
    def test_site_actually_calls_the_shared_helper(self, name):
        """A mention in a comment or docstring must not satisfy this."""
        src = (_SRC / name).read_text()
        code = "\n".join(
            ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
        )

        assert re.search(r"\bselect_closed_lcfs\s*\(", code), (
            f"{name} names select_closed_lcfs but never calls it."
        )
