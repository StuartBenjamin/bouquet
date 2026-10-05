"""The legacy IMAS reader reads each NBI entry AT the slice time.

Owner-approved fix (2026-10-05), the rule of the sawteeth entry next to it
(``tests/test_imas_reader_sawtooth_slice.py``) and of the engine IDS
adapter: a ``core_sources`` beam entry (identifier 2) is read at the
core_sources slice TIME, not at its list index.  An entry that starts one
slice after the IDS time base was read one slice late, and past its last
slice from its FIRST slice; at a time it does not cover it now carries no
current (warned).  An entry with no per-slice time and a different slice
count cannot be aligned and is refused -- never its first slice in place of
the missing one.

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


def test_a_slice_before_the_entry_starts_carries_no_beam_current(tmp_path):
    """At the first time the late entry has no slice: no NBI current there,
    and a warning says so (the list-index rule read the NEXT time's)."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    msgs = []
    got = np.asarray(_read(tmp_path, dd, t[0], "first.json", msgs).j_NBI)
    assert np.all(got == 0.0)
    assert any("NBI entry" in m and "no profiles_1d slice at t =" in m
               and "carries no current at this slice" in m for m in msgs)


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
