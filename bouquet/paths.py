"""Machine-portable resolution of the OFT install, the mesh, and IDA data.

bouquet itself imports headless -- OFT is only loaded lazily inside
``Bouquet.setup_solver`` -- so notebooks and driver scripts are free to call
:func:`add_oft_to_path` *after* ``import bouquet`` to make OFT importable, and
:func:`find_mesh` to locate the TokaMaker mesh, without hardcoding either path.

Both helpers follow the same precedence: an explicit environment variable wins,
then a caller-supplied hint, then a built-in candidate list / walk-up.  This is
the generalisation of the resolver already used in ``tests/test_systematics.py``
(``OFT_PYTHONPATH`` env -> repo-relative fallback -> import check), extended with
the install locations seen across machines (a system-wide ``/Applications`` build
and a per-user ``~/Desktop/plasma`` build).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

# Candidate OFT *python* directories, tried in order after OFT_PYTHONPATH and any
# caller hint.  Add a new machine's path here (or just export OFT_PYTHONPATH).
_OFT_CANDIDATES: List[str] = [
    "/Applications/OpenFUSIONToolkit/python",                       # system-wide app build
    "~/Desktop/plasma/OpenFUSIONToolkit/build_release/python",      # per-user source build
]

# Repo-relative walk-up: from a starting dir, look this many levels up for a
# sibling OpenFUSIONToolkit/build_release/python (covers a checkout that sits
# next to the OFT source tree).
_OFT_RELATIVE = os.path.join("OpenFUSIONToolkit", "build_release", "python")


def _expand(p: Optional[str]) -> Optional[str]:
    return os.path.abspath(os.path.expanduser(p)) if p else None


def _on_path(directory: str) -> bool:
    ap = os.path.abspath(directory)
    return any(ap == os.path.abspath(p) for p in sys.path)


def _has_oft_package(directory: str) -> bool:
    """True iff *directory* actually holds an importable ``OpenFUSIONToolkit``
    package (an ``OpenFUSIONToolkit/__init__.py``), not merely an empty or stale
    directory of that name.  Guards against selecting a leftover install dir that
    exists on disk but no longer contains the package."""
    return os.path.isfile(os.path.join(directory, "OpenFUSIONToolkit", "__init__.py"))


def _oft_module_dir() -> str:
    """The directory ``OpenFUSIONToolkit`` actually loaded from (parent of the
    package dir), or ``""`` if it is not importable.  Used so the returned path
    is honest even when OFT is already importable via a ``.pth`` / editable
    install and no candidate directory needed to be added."""
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return os.path.dirname(os.path.dirname(
            os.path.abspath(OpenFUSIONToolkit.__file__)))
    except Exception:
        return ""


def add_oft_to_path(extra: Optional[str] = None, *, verbose: bool = False) -> str:
    """Ensure ``OpenFUSIONToolkit`` is importable; return the directory it loads
    from.

    Resolution order:

    1. ``OFT_PYTHONPATH`` environment variable, if set and a directory;
    2. the caller-supplied ``extra`` directory;
    3. each entry of :data:`_OFT_CANDIDATES` (``~`` expanded);
    4. a walk-up from the current working directory for a sibling
       ``OpenFUSIONToolkit/build_release/python``.

    A candidate is only selected if it actually **contains** the
    ``OpenFUSIONToolkit`` package -- a directory that exists but holds no package
    (e.g. a stale ``/Applications`` leftover) is skipped, not chosen.  The first
    valid directory is prepended to ``sys.path``; if OFT is already importable
    (e.g. via a ``.pth`` / editable install) no directory is added.  Either way
    the returned path is where the package *actually* loads from, and it is
    confirmed with ``import OpenFUSIONToolkit``.  Raises ``ModuleNotFoundError``
    listing every location tried (and any skipped as package-less) if none work.
    """
    tried: List[str] = []
    skipped_no_pkg: List[str] = []

    def _candidates():
        yield _expand(os.environ.get("OFT_PYTHONPATH"))
        yield _expand(extra)
        for c in _OFT_CANDIDATES:
            yield _expand(c)
        # walk up from CWD looking for a sibling OFT build
        here = os.getcwd()
        for _ in range(6):
            yield os.path.join(here, _OFT_RELATIVE)
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent

    chosen = None
    for cand in _candidates():
        if not cand:
            continue
        tried.append(cand)
        if not os.path.isdir(cand):
            continue
        if not _has_oft_package(cand):
            skipped_no_pkg.append(cand)   # exists but no OpenFUSIONToolkit package
            continue
        if not _on_path(cand):
            sys.path.insert(0, cand)
        chosen = cand
        break

    try:
        import OpenFUSIONToolkit  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        msg = ["Could not import OpenFUSIONToolkit. Set OFT_PYTHONPATH to its "
               "'python' directory, or add the install to "
               "bouquet.paths._OFT_CANDIDATES."]
        if skipped_no_pkg:
            msg.append("Skipped (directory exists but has no OpenFUSIONToolkit "
                       "package):\n  " + "\n  ".join(skipped_no_pkg))
        msg.append("Tried:\n  " + "\n  ".join(tried or ["<no candidates>"]))
        raise ModuleNotFoundError("\n".join(msg)) from exc

    # Report where the package ACTUALLY loaded from -- honest whether we added
    # `chosen` or it was already importable via a .pth. (These can differ if OFT
    # was imported earlier in the session and is cached in sys.modules.)
    actual = _oft_module_dir() or chosen or ""
    if verbose:
        note = ""
        if chosen and os.path.abspath(chosen) != os.path.abspath(actual or chosen):
            note = f" (added {chosen}, but sys.modules already had it)"
        print(f"[bouquet.paths] OFT from: {actual or '(already importable)'}{note}")
    return actual


def find_mesh(
    name: str = "DIIID_mesh.h5",
    start: Optional[str] = None,
    extra: Optional[str] = None,
    *,
    verbose: bool = False,
) -> str:
    """Locate a TokaMaker mesh file; return its absolute path.

    Resolution order:

    1. ``BOUQUET_MESH`` environment variable (an explicit file);
    2. the caller-supplied ``extra`` (file, or a directory containing ``name``);
    3. a walk-up from ``start`` (default: current working directory) checking
       ``<dir>/name`` and ``<dir>/meshes/name`` at each level;
    4. the mesh committed alongside the bouquet examples
       (``examples/D3D-like/<name>``).

    Raises ``FileNotFoundError`` listing every location tried otherwise.
    """
    tried: List[str] = []

    def _file(p: Optional[str]) -> Optional[str]:
        ap = _expand(p)
        if ap:
            tried.append(ap)
            if os.path.isfile(ap):
                return ap
        return None

    # 1. env var
    hit = _file(os.environ.get("BOUQUET_MESH"))
    if hit:
        return _report(hit, verbose)

    # 2. caller hint (file or directory)
    if extra:
        ex = _expand(extra)
        if ex and os.path.isdir(ex):
            hit = _file(os.path.join(ex, name))
        else:
            hit = _file(extra)
        if hit:
            return _report(hit, verbose)

    # 3. walk up from start
    here = _expand(start) or os.getcwd()
    for _ in range(8):
        for cand in (os.path.join(here, name), os.path.join(here, "meshes", name)):
            hit = _file(cand)
            if hit:
                return _report(hit, verbose)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent

    # 4. bouquet's own example mesh
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hit = _file(os.path.join(pkg_root, "examples", "D3D-like", name))
    if hit:
        return _report(hit, verbose)

    raise FileNotFoundError(
        f"Could not locate mesh {name!r}. Set BOUQUET_MESH to the file, pass "
        f"extra=<dir-or-file>, or place it on the walk-up path. Tried:\n  "
        + "\n  ".join(tried or ["<no candidates>"])
    )


# how deep the BOUQUET_IDA directory search will descend.  A data tree is a
# few levels of shot directories; without a cap a typo'd BOUQUET_IDA=$HOME
# walks the whole disk (measured: 77 s) before raising.
_IDA_WALK_MAX_DEPTH = 6


def find_ida(
    name: str,
    start: Optional[str] = None,
    extra: Optional[str] = None,
    *,
    verbose: bool = False,
) -> str:
    """Locate an IDA ``.cdf`` kinetic-profile file; return its absolute path.

    The mesh twin of this (:func:`find_mesh`) exists because the mesh ships
    with bouquet.  This one exists for the opposite reason: IDA files are
    machine data, frequently re-generated as the IDA-lite software evolves and
    up to hundreds of MB, so they are kept **outside** the analysis repo and
    located at run time.  A notebook then names the file it wants without
    naming the machine it is on.

    Resolution order:

    1. ``BOUQUET_IDA`` naming a **file** -- the explicit override;
    2. the caller-supplied ``extra`` (a file, or a directory to search);
    3. a walk-up from ``start`` (default: cwd) checking ``<dir>/name`` and
       ``<dir>/IDA/name`` at each level -- so the copy sitting next to the
       notebook wins over any shared data tree;
    4. ``BOUQUET_IDA`` naming a **directory** -- searched recursively (to
       depth ``_IDA_WALK_MAX_DEPTH``) as the shared-data fallback.

    Multiple vintages of one shot commonly share a basename and load without
    complaint, so a silent wrong pick is worse than a miss.  Hence: a
    directory search that finds the name more than once raises unless every
    copy has identical size; and a miss raises ``FileNotFoundError`` listing
    every location tried.
    """
    tried: List[str] = []

    def _file(p: Optional[str]) -> Optional[str]:
        ap = _expand(p)
        if ap:
            tried.append(ap)
            if os.path.isfile(ap):
                return ap
        return None

    def _dir(d: Optional[str]) -> Optional[str]:
        """Search a directory tree for *name*, refusing to guess between
        differing copies."""
        ad = _expand(d)
        if not ad or not os.path.isdir(ad):
            return None
        hit = _file(os.path.join(ad, name))
        if hit:
            return hit
        tried.append(os.path.join(ad, "**", name) + f" (depth<={_IDA_WALK_MAX_DEPTH})")
        base_depth = ad.rstrip(os.sep).count(os.sep)
        matches = []
        for root, dirs, files in os.walk(ad):
            if root.rstrip(os.sep).count(os.sep) - base_depth >= _IDA_WALK_MAX_DEPTH:
                dirs[:] = []                     # do not descend further
                continue
            dirs.sort()                          # deterministic on every filesystem
            if name in files:
                matches.append(os.path.join(root, name))
        if not matches:
            return None
        matches.sort()
        sizes = {m: os.path.getsize(m) for m in matches}
        if len(set(sizes.values())) > 1:
            listing = "\n  ".join(f"{m}  ({sizes[m]} bytes)" for m in matches)
            raise FileNotFoundError(
                f"{name!r} found more than once under {ad} with DIFFERING sizes "
                f"-- likely different IDA vintages, and picking one silently "
                f"would be worse than failing.  Point BOUQUET_IDA (or extra=) "
                f"at the file you mean:\n  {listing}"
            )
        return matches[0]

    # 1. env var naming a file: explicit override
    env = _expand(os.environ.get("BOUQUET_IDA"))
    if env and os.path.isfile(env):
        tried.append(env)
        return _report(env, verbose, kind="IDA")

    # 2. caller hint
    if extra:
        ex = _expand(extra)
        hit = _dir(ex) if (ex and os.path.isdir(ex)) else _file(extra)
        if hit:
            return _report(hit, verbose, kind="IDA")

    # 3. walk up from start: the file next to the notebook wins over a shared tree
    here = _expand(start) or os.getcwd()
    for _ in range(8):
        for cand in (os.path.join(here, name), os.path.join(here, "IDA", name)):
            hit = _file(cand)
            if hit:
                return _report(hit, verbose, kind="IDA")
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent

    # 4. env var naming a directory: the shared-data fallback
    if env and os.path.isdir(env):
        hit = _dir(env)
        if hit:
            return _report(hit, verbose, kind="IDA")

    raise FileNotFoundError(
        f"Could not locate IDA file {name!r}. Set BOUQUET_IDA to the file or to "
        f"the directory holding your IDA data, pass extra=<dir-or-file>, or "
        f"place it on the walk-up path. Tried:\n  "
        + "\n  ".join(tried or ["<no candidates>"])
    )


def _report(path: str, verbose: bool, kind: str = "mesh") -> str:
    if verbose:
        print(f"[bouquet.paths] {kind}: {path}")
    return path
