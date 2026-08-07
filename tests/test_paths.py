"""Resolver hardening for :func:`bouquet.paths.add_oft_to_path`.

The regression guarded here: a directory that exists on disk but holds no
``OpenFUSIONToolkit`` package (e.g. a stale ``/Applications`` leftover) must be
SKIPPED, not selected -- and the returned path must be where the package
actually loads from.
"""

import os
import pytest

from bouquet.paths import _has_oft_package, find_ida


def test_has_oft_package(tmp_path):
    # empty directory -> not a package
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not _has_oft_package(str(empty))

    # a bare OpenFUSIONToolkit/ dir with no __init__.py -> still not a package
    partial = tmp_path / "partial"
    (partial / "OpenFUSIONToolkit").mkdir(parents=True)
    assert not _has_oft_package(str(partial))

    # a real package layout -> yes
    good = tmp_path / "good"
    pkg = good / "OpenFUSIONToolkit"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    assert _has_oft_package(str(good))


def test_add_oft_skips_packageless_extra(tmp_path):
    # Needs a real OFT importable in the env to confirm the resolution outcome.
    pytest.importorskip("OpenFUSIONToolkit")
    import bouquet as bq

    nopkg = tmp_path / "nopkg"
    nopkg.mkdir()                                   # exists, but no OFT package
    chosen = bq.add_oft_to_path(extra=str(nopkg))

    # the package-less hint must NOT be what we selected ...
    assert os.path.abspath(chosen) != os.path.abspath(str(nopkg))
    # ... and whatever path is returned must actually contain the package.
    assert _has_oft_package(chosen)


# ---------------------------------------------------------------------------
#  find_ida -- IDA .cdf files live OUTSIDE the analysis repo
# ---------------------------------------------------------------------------
class TestFindIda:
    """IDA files are machine data (often >100 MB, and re-generated as IDA-lite
    evolves), so they are not tracked alongside the notebooks that read them.
    The notebook names the file; this resolves the machine."""

    NAME = "IDA_123456_.cdf"

    def _make(self, d, name=None):
        d.mkdir(parents=True, exist_ok=True)
        f = d / (name or self.NAME)
        f.write_bytes(b"CDF\x00")
        return f

    def test_env_var_pointing_at_the_file(self, tmp_path, monkeypatch):
        f = self._make(tmp_path / "data")
        monkeypatch.setenv("BOUQUET_IDA", str(f))
        assert find_ida(self.NAME, start=str(tmp_path)) == str(f)

    def test_env_var_pointing_at_a_directory_searches_below_it(
            self, tmp_path, monkeypatch):
        """One env var must cover a whole shot tree, not one file per shot."""
        f = self._make(tmp_path / "data" / "shots" / "123456_CTM")
        monkeypatch.setenv("BOUQUET_IDA", str(tmp_path / "data"))
        assert find_ida(self.NAME) == str(f)

    def test_walk_up_finds_a_sibling_and_an_IDA_subdir(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.delenv("BOUQUET_IDA", raising=False)
        f = self._make(tmp_path / "IDA")
        deep = tmp_path / "shots" / "123456"
        deep.mkdir(parents=True)
        assert find_ida(self.NAME, start=str(deep)) == str(f)

    def test_extra_beats_the_walk_up(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BOUQUET_IDA", raising=False)
        near = self._make(tmp_path / "here")
        far = self._make(tmp_path / "elsewhere")
        assert find_ida(self.NAME, start=str(near.parent),
                        extra=str(far.parent)) == str(far)

    def test_missing_file_names_every_place_it_looked(self, tmp_path,
                                                      monkeypatch):
        """Two vintages of a shot are common, so a silent wrong pick is worse
        than a miss -- the error must be actionable."""
        monkeypatch.delenv("BOUQUET_IDA", raising=False)
        with pytest.raises(FileNotFoundError) as e:
            find_ida(self.NAME, start=str(tmp_path))
        msg = str(e.value)
        assert "BOUQUET_IDA" in msg and self.NAME in msg
        assert str(tmp_path) in msg

    def test_local_sibling_beats_a_shared_env_directory(self, tmp_path,
                                                        monkeypatch):
        """The copy sitting next to the notebook is almost always the intended
        vintage; a shared BOUQUET_IDA tree must only be the fallback."""
        local = self._make(tmp_path / "shots" / "123456")
        shared = self._make(tmp_path / "warehouse" / "deep")
        monkeypatch.setenv("BOUQUET_IDA", str(tmp_path / "warehouse"))
        assert find_ida(self.NAME, start=str(local.parent)) == str(local)
        # and with no local copy, the shared tree is still reachable
        local.unlink()
        assert find_ida(self.NAME, start=str(tmp_path / "shots" / "123456")) \
            == str(shared)

    def test_ambiguous_vintages_raise_instead_of_guessing(self, tmp_path,
                                                          monkeypatch):
        """Two same-named files with different sizes are different vintages;
        silently picking one is the failure mode this resolver exists to
        prevent."""
        a = self._make(tmp_path / "data" / "old")
        b = (tmp_path / "data" / "new"); b.mkdir(parents=True)
        (b / self.NAME).write_bytes(b"CDF\x00" * 100)      # different size
        monkeypatch.setenv("BOUQUET_IDA", str(tmp_path / "data"))
        with pytest.raises(FileNotFoundError) as e:
            find_ida(self.NAME, start=str(tmp_path))
        assert "DIFFERING sizes" in str(e.value)
        assert str(a) in str(e.value)

    def test_identical_duplicates_resolve_deterministically(self, tmp_path,
                                                            monkeypatch):
        """Same size in two places = same vintage copied around; take the
        lexicographically first so every machine picks the same one."""
        self._make(tmp_path / "data" / "b_dir")
        first = self._make(tmp_path / "data" / "a_dir")
        monkeypatch.setenv("BOUQUET_IDA", str(tmp_path / "data"))
        assert find_ida(self.NAME, start=str(tmp_path)) == str(first)

    def test_the_walk_is_depth_capped(self, tmp_path, monkeypatch):
        """A typo'd BOUQUET_IDA=$HOME must fail fast, not crawl the disk."""
        from bouquet.paths import _IDA_WALK_MAX_DEPTH
        deep = tmp_path / "data"
        for i in range(_IDA_WALK_MAX_DEPTH + 2):
            deep = deep / f"lvl{i}"
        self._make(deep)                                    # below the cap
        monkeypatch.setenv("BOUQUET_IDA", str(tmp_path / "data"))
        with pytest.raises(FileNotFoundError):
            find_ida(self.NAME, start=str(tmp_path))
