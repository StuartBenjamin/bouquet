"""Readers can exclude draws whose solve was rescued by the opt-in draw-solve
rescue (#75, owner decision D5 condition 3): ``draw_band(rescued=...)`` and
``merge_archives(rescued=...)``.  Default: include, counted and warned.

Hand-built archives only (no solver); synthetic values.
"""
import json
import warnings

import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq


def _archive(path, rescued=()):
    with h5py.File(path, "w") as hf:
        hf.attrs["schema_version"] = 2
        sg = hf.require_group("scan/1")
        sg.create_group("_baseline").attrs["val"] = 5.0
        for d in range(4):
            g = sg.create_group(str(d))
            g.attrs["val"] = float(d)
            if rescued:
                g.attrs["solve_recovered"] = d in rescued
                if d in rescued:
                    g.attrs["solve_recovered_by"] = "nl_tol=2e-05"
                    g.attrs["solve_nl_tol_accepted"] = 2e-5
    return str(path)


def _ev(view):
    return {"x": float(view.attrs["val"])}


def _band(path, **kw):
    return bq.draw_band(path, "1", _ev, selection="all", require_filter=False,
                        min_n=1, hard_min=1, **kw)["x"]


def test_by_default_rescued_draws_are_kept_counted_and_warned(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={2})
    with pytest.warns(UserWarning, match=r"1 draw\(s\) whose solve was RESCUED"):
        r = _band(p)
    assert sorted(r.values) == [0, 1, 2, 3]
    assert r.provenance["rescued"] == "include"
    assert r.provenance["rescued_draws"] == {2: "nl_tol=2e-05"}


def test_excluded_on_request_with_the_reason(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _band(p, rescued="exclude")
    assert sorted(r.values) == [0, 1, 3]
    assert (2, "rescued:nl_tol=2e-05") in r.dropped


def test_an_archive_without_rescued_draws_is_untouched(tmp_path):
    p = _archive(tmp_path / "a.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _band(p)
    assert sorted(r.values) == [0, 1, 2, 3]
    assert "rescued" not in r.provenance and "rescued_draws" not in r.provenance
    with pytest.raises(ValueError, match="rescued must be one of"):
        _band(p, rescued="drop")


def test_draw_bands_passes_the_policy_through(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={0, 3})
    t = bq.draw_bands([(p, "1")], _ev, selection="all", require_filter=False,
                      min_n=1, hard_min=1, rescued="exclude")
    assert sorted(t.records[0].values) == [1, 2]


def _shard(path, worker, rescued=()):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["l_i_target"] = 0.9
        g.attrs["Ip_target"] = 1.0e6
        for i in range(2):
            d = hf.create_group(f"scan/0/{i}")
            d.attrs["count"] = i
            if rescued:
                d.attrs["solve_recovered"] = i in rescued
                if i in rescued:
                    d.attrs["solve_recovered_by"] = "urf=0.1"
        hf["scan/0"].attrs["parallel_worker_json"] = json.dumps(
            dict(worker_id=worker, n=2, n_attempts=2, seed=7 + worker))
    return str(path)


def test_merge_includes_by_default_and_records_it(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0, rescued={1})
    s1 = _shard(tmp_path / "w1.h5", 1)
    with pytest.warns(UserWarning, match=r"1 merged draw\(s\) had their solve"):
        out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    assert n == 4
    with h5py.File(out, "r") as hf:
        rec = json.loads(hf["scan/0"].attrs["merge_rescued_json"])
        assert bool(hf["scan/0/1"].attrs["solve_recovered"])
    assert rec["policy"] == "include" and rec["n_rescued"] == 1
    assert rec["draws"] == [dict(shard="w0.h5", index=1,
                                 recovered_by="urf=0.1", merged_as=1)]


def test_merge_excludes_on_request(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0, rescued={1})
    s1 = _shard(tmp_path / "w1.h5", 1, rescued={0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0,
                                rescued="exclude")
    assert n == 2
    with h5py.File(out, "r") as hf:
        rec = json.loads(hf["scan/0"].attrs["merge_rescued_json"])
        assert not any(bool(hf[f"scan/0/{k}"].attrs.get("solve_recovered"))
                       for k in ("0", "1"))
    assert rec["policy"] == "exclude" and rec["n_rescued"] == 2
    assert all(d["merged_as"] is None for d in rec["draws"])


def test_merge_without_rescued_draws_writes_nothing_new(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, n = merge_archives([s0], str(tmp_path / "m"), scan_key=0)
    with h5py.File(out, "r") as hf:
        assert "merge_rescued_json" not in hf["scan/0"].attrs
    with pytest.raises(ValueError, match="rescued must be"):
        merge_archives([s0], str(tmp_path / "m2"), scan_key=0, rescued="no")


def _until_n_shard(path, worker, inspec, rescued):
    """A shared until-N shard: draw i is in spec by the loop's verdict
    (``until_n_inspec``) when i in *inspec*; the worker record counted them."""
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["l_i_target"] = 0.9
        g.attrs["Ip_target"] = 1.0e6
        for i in range(3):
            d = hf.create_group(f"scan/0/{i}")
            d.attrs["count"] = i
            d.attrs["solve_recovered"] = i in rescued
            if i in rescued:
                d.attrs["solve_recovered_by"] = "loose_tol=1e-5"
            if inspec is not None:
                d.attrs["until_n_inspec"] = i in inspec
        hf["scan/0"].attrs["parallel_worker_json"] = json.dumps(
            dict(worker_id=worker, n=3, n_attempts=3, seed=7 + worker,
                 n_inspec=len(inspec or ()), shared_target=4, total_cap=10))
    return str(path)


def test_exclude_takes_the_dropped_inspec_draws_off_the_until_n_count(
        tmp_path):
    """Review of #78 (A8): the shards counted each in-spec rescued draw when
    the loop accepted it; a rescued='exclude' merge drops the draw, so the
    manifest's n_inspec_recorded must drop it too, and say so."""
    from bouquet.parallel import merge_archives
    # w0: in spec 0, 1 (1 rescued); w1: in spec 0, 2 (2 rescued), 1 rescued
    # but out of spec -- 4 recorded by the shards, 2 of them excluded
    s0 = _until_n_shard(tmp_path / "w0.h5", 0, inspec={0, 1}, rescued={1})
    s1 = _until_n_shard(tmp_path / "w1.h5", 1, inspec={0, 2},
                        rescued={1, 2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0,
                                rescued="exclude")
    assert n == 3
    with h5py.File(out, "r") as hf:
        man = json.loads(hf["scan/0"].attrs["parallel_manifest_json"])
    u = man["until_n"]
    assert u["n_inspec_recorded_by_shards"] == 4
    assert u["n_inspec_excluded_rescued"] == 2
    assert u["n_inspec_excluded_rescued_unknown"] == 0
    assert u["n_inspec_recorded"] == 2 and u["target"] == 4
    # include: nothing is subtracted, nothing extra is recorded
    s0 = _until_n_shard(tmp_path / "x0.h5", 0, inspec={0, 1}, rescued={1})
    with pytest.warns(UserWarning, match="RESCUED"):
        out, n = merge_archives([s0], str(tmp_path / "mi"), scan_key=0)
    with h5py.File(out, "r") as hf:
        u = json.loads(hf["scan/0"].attrs["parallel_manifest_json"])["until_n"]
    assert u["n_inspec_recorded"] == 2
    assert "n_inspec_excluded_rescued" not in u


def test_exclude_without_the_loops_verdict_is_counted_unknown_and_warned(
        tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _until_n_shard(tmp_path / "w0.h5", 0, inspec=None, rescued={1})
    with pytest.warns(UserWarning, match="no in-loop until-N verdict"):
        out, n = merge_archives([s0], str(tmp_path / "m"), scan_key=0,
                                rescued="exclude")
    with h5py.File(out, "r") as hf:
        u = json.loads(hf["scan/0"].attrs["parallel_manifest_json"])["until_n"]
    assert u["n_inspec_excluded_rescued_unknown"] == 1
    assert u["n_inspec_excluded_rescued"] == 0


def test_the_draw_loop_archives_its_until_n_verdict():
    """The verdict the loop counted is stamped on the archived group, where
    merge_archives reads it back."""
    import inspect

    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.schema import UNTIL_N_INSPEC_ATTR
    src = inspect.getsource(generate_bouquet)
    i_ok = src.index("diagnostics['until_n_inspec'] = bool(_ok)")
    i_st = src.index("{UNTIL_N_INSPEC_ATTR: bool(_ok)}")
    i_cnt = src.index("_n_inspec_seen += 1")
    assert i_ok < i_st < i_cnt
    assert UNTIL_N_INSPEC_ATTR == "until_n_inspec"
