"""Resolver hardening for :func:`bouquet.paths.add_oft_to_path`.

The regression guarded here: a directory that exists on disk but holds no
``OpenFUSIONToolkit`` package (e.g. a stale ``/Applications`` leftover) must be
SKIPPED, not selected -- and the returned path must be where the package
actually loads from.
"""

import os
import pytest

from bouquet.paths import _has_oft_package


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
