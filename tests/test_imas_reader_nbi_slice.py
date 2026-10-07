"""The legacy IMAS reader reads each NBI entry AT the slice time.

Owner-approved fix (2026-10-05), the rule of the sawteeth entry next to it
(``tests/test_imas_reader_sawtooth_slice.py``) and of the engine IDS
adapter: a ``core_sources`` beam entry (identifier 2) is read at the
core_sources slice TIME, not at its list index.  An entry that starts one
slice after the IDS time base was read one slice late, and past its last
slice from its FIRST slice.  An entry with no per-slice time and a different
slice count cannot be aligned and is refused -- never its first slice in
place of the missing one.

The time match (owner-approved 2026-10-06, replacing the 1e-6 s absolute
match, under which a beam entry a few microseconds off the time base was
dropped to ZERO with a warning): each entry is matched to its NEAREST own
slice and accepted within HALF its local time-step (the core_profiles step
for a single-time entry; float precision when neither grid has a step);
otherwise the read is REFUSED -- a beam is never silently zeroed.

The slice read is visible in ``Baseline.j_NBI``: the beam entry's
j_parallel at core_sources time t_k is the constant 1e3*(k+1) A/m^2, and the
reference is the same entry on the full time base (the parallel -> toroidal
ratio depends only on the slice's core_profiles, so a correct read is
bit-identical to the reference).

Synthetic inputs only (the shipped D3D-like OMAS example, modified here).
"""
import copy
import json
import os
import warnings

import numpy as np
import pytest

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "examples", "D3D-like")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")


def _example():
    with open(_OMAS) as fh:
        return json.load(fh)


def _dd_with_tagged_nbi(with_times=True, drop_first=False):
    """The example dd whose beam entry carries 1e3*(k+1) at time t_k;
    *drop_first* makes it start one slice after the IDS time base."""
    dd = _example()
    t = list(dd["core_sources"]["time"])
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    n = len(nbi["profiles_1d"][0]["j_parallel"])
    for k, q in enumerate(nbi["profiles_1d"]):
        q["j_parallel"] = [1.0e3 * (k + 1)] * n
        if with_times:
            q["time"] = t[k]
        else:
            q.pop("time", None)
    if drop_first:
        nbi["profiles_1d"] = nbi["profiles_1d"][1:]
    return dd, t


def _read(tmp_path, dd, time, name, record=None):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    p = tmp_path / name
    with open(p, "w") as fh:
        json.dump(dd, fh)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(ImasSource(ids_path=str(p), time=time))
    if record is not None:
        record.extend(str(x.message) for x in w)
    return bl


def _reference(tmp_path, t, k):
    dd, _ = _dd_with_tagged_nbi()
    return np.asarray(_read(tmp_path, dd, t[k], f"ref{k}.json").j_NBI,
                      dtype=float)


def test_the_last_slice_reads_its_own_nbi_slice(tmp_path):
    """At the last slice the entry's OWN last slice is read, not its first
    (the list-index rule's fallback past the end of a late entry)."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    k = len(t) - 1
    got = np.asarray(_read(tmp_path, dd, t[k], "last.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
    assert np.max(np.abs(got)) > 0.0


def test_a_middle_slice_is_not_read_one_slice_late(tmp_path):
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    got = np.asarray(_read(tmp_path, dd, t[1], "mid.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, 1))


def test_a_slice_before_the_entry_starts_is_refused(tmp_path):
    """At the first time the late entry's nearest own slice (t_1) is more
    than half its local time-step away: REFUSED, naming the entry and the
    times (it was zeroed with a warning before 2026-10-06; the list-index
    rule read the NEXT time's)."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    with pytest.raises(ValueError, match=r"IMAS reader: core_sources "
                       r"'nbi_synthetic' \(index 2\) carries a non-zero "
                       r"j_parallel but has no profiles_1d slice within half "
                       r"a time-step of t = 2\.1 s.*Refusing"):
        _read(tmp_path, dd, t[0], "first.json")


def _shifted(dd, shift):
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    for q in nbi["profiles_1d"]:
        q["time"] = q["time"] + shift
    return dd


def test_a_two_microsecond_offset_reads_the_nearest_slice_not_zero(
        tmp_path):
    """The 2026-10-06 review's case: the entry's own times 2 us off the
    core_sources base.  Under the 1e-6 s match the beam was ZERO (one
    warning); under the half-step rule each slice reads its nearest own
    slice, bit-identical to the on-grid read, and nothing is warned."""
    for k in range(3):
        dd, t = _dd_with_tagged_nbi()
        msgs = []
        got = np.asarray(_read(tmp_path, _shifted(dd, 2e-6), t[k],
                               f"us{k}.json", msgs).j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
        assert np.max(np.abs(got)) > 0.0
        assert not any("NBI" in m for m in msgs)


def test_a_two_microsecond_offset_with_no_time_step_is_refused(tmp_path):
    """With no step on either grid (a single-time IDS and a single-time beam
    entry) the window is float precision: a 2 us offset is REFUSED -- never
    a zero beam."""
    dd, t = _dd_with_tagged_nbi()
    k = 1
    for blk in ("core_sources", "core_profiles", "equilibrium"):
        dd[blk]["time"] = [t[k]]
    dd["core_profiles"]["profiles_1d"] = [dd["core_profiles"]["profiles_1d"][k]]
    dd["equilibrium"]["time_slice"] = [dd["equilibrium"]["time_slice"][k]]
    for blk in ("vacuum_toroidal_field",):
        vt = dd["equilibrium"].get(blk)
        if vt and "b0" in vt:
            vt["b0"] = [vt["b0"][k]]
    for src in dd["core_sources"]["source"]:
        src["profiles_1d"] = [src["profiles_1d"][k]]
    # on the time: read
    got = np.asarray(_read(tmp_path, copy.deepcopy(dd), t[k],
                           "one.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
    with pytest.raises(ValueError, match="within half a time-step.*Refusing"):
        _read(tmp_path, _shifted(dd, 2e-6), t[k], "one_us.json")


def test_an_entry_exactly_on_a_slice_time_matches(tmp_path):
    dd, t = _dd_with_tagged_nbi()
    for k, tk in enumerate(t):
        got = np.asarray(_read(tmp_path, copy.deepcopy(dd), tk,
                               f"on{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
        assert np.max(np.abs(got)) > 0.0


def test_a_coarser_entry_grid_matches_its_nearest_slice(tmp_path):
    """An entry on a coarser grid (its own times t_0 and t_2 only): the
    middle slice t_1 reads the entry's NEAREST own slice -- within half the
    entry's local step -- and the end slices their own."""
    dd, t = _dd_with_tagged_nbi()
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nbi["profiles_1d"] = [nbi["profiles_1d"][0], nbi["profiles_1d"][2]]
    near = 0 if abs(t[1] - t[0]) <= abs(t[2] - t[1]) else 2
    got = np.asarray(_read(tmp_path, dd, t[1], "coarse.json").j_NBI)
    # the slice-t_1 conversion of the entry's nearest own slice's current
    want = _dd_with_tagged_nbi()[0]
    nb2 = next(s for s in want["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nb2["profiles_1d"][1]["j_parallel"] = nb2["profiles_1d"][near][
        "j_parallel"]
    ref = np.asarray(_read(tmp_path, want, t[1], "coarse_ref.json").j_NBI)
    np.testing.assert_array_equal(got, ref)
    for k in (0, 2):
        got = np.asarray(_read(tmp_path, copy.deepcopy(dd), t[k],
                               f"coarse{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))


def test_past_the_end_by_more_than_half_a_step_is_refused(tmp_path):
    """The entry stops one slice early: at the last time its nearest own
    slice is a full step away -- refused (never read from another time)."""
    dd, t = _dd_with_tagged_nbi()
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nbi["profiles_1d"] = nbi["profiles_1d"][:-1]
    with pytest.raises(ValueError, match="within half a time-step.*Refusing"):
        _read(tmp_path, dd, t[-1], "early_end.json")
    # the window edge: the entry's two own times (step 0.1 s) shifted so
    # the last slice time is 0.049 s (inside half a step) or 0.051 s
    # (outside) past its last own time
    for off, ok in ((0.049, True), (0.051, False)):
        dd, t = _dd_with_tagged_nbi()
        nbi = next(s for s in dd["core_sources"]["source"]
                   if s["identifier"]["index"] == 2)
        nbi["profiles_1d"] = nbi["profiles_1d"][:-1]
        last = t[-1] - off
        nbi["profiles_1d"][0]["time"] = last - 0.1
        nbi["profiles_1d"][1]["time"] = last
        if ok:
            got = np.asarray(_read(tmp_path, dd, t[-1], "in.json").j_NBI)
            assert np.max(np.abs(got)) > 0.0
        else:
            with pytest.raises(ValueError, match="within half a time-step"):
                _read(tmp_path, dd, t[-1], "out.json")


def test_an_nbi_entry_that_cannot_be_aligned_is_refused(tmp_path):
    dd, t = _dd_with_tagged_nbi(with_times=False, drop_first=True)
    with pytest.raises(ValueError, match="cannot be aligned"):
        _read(tmp_path, dd, t[-1], "notime.json")


def test_an_nbi_entry_on_the_full_time_base_without_times_reads_by_index(
        tmp_path):
    """The common case -- every slice present, no per-slice time -- is read
    by index, exactly as before (= the time-tagged reference)."""
    dd, t = _dd_with_tagged_nbi(with_times=False)
    for k, tk in enumerate(t):
        got = np.asarray(_read(tmp_path, dd, tk, f"full{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))


def test_the_shipped_example_reads_the_same_nbi_as_the_index_rule(tmp_path):
    """Nothing moves for the shipped example (its beam entry is on the full
    time base, without per-slice times): at every slice the reader's j_NBI
    is the entry's slice k converted, i.e. what the list-index rule read."""
    dd = _example()
    t = list(dd["core_sources"]["time"])
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    assert len(nbi["profiles_1d"]) == len(t)
    assert all(q.get("time") is None for q in nbi["profiles_1d"])
    for k, tk in enumerate(t):
        msgs = []
        bl = _read(tmp_path, dd, tk, f"ex{k}.json", msgs)
        # the same entry, slice k pinned explicitly by time
        dd2 = copy.deepcopy(dd)
        nb2 = next(s for s in dd2["core_sources"]["source"]
                   if s["identifier"]["index"] == 2)
        for j, q in enumerate(nb2["profiles_1d"]):
            q["time"] = t[j]
        bl2 = _read(tmp_path, dd2, tk, f"ex_t{k}.json")
        np.testing.assert_array_equal(np.asarray(bl.j_NBI),
                                      np.asarray(bl2.j_NBI))
        assert not any("NBI entry" in m for m in msgs)
