"""Machine-portable resolution of the OpenFUSIONToolkit install and the mesh.

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


def add_oft_to_path(extra: Optional[str] = None, *, verbose: bool = False) -> str:
    """Ensure ``OpenFUSIONToolkit`` is importable; return the directory used.

    Resolution order:

    1. ``OFT_PYTHONPATH`` environment variable, if set and a directory;
    2. the caller-supplied ``extra`` directory;
    3. each entry of :data:`_OFT_CANDIDATES` (``~`` expanded);
    4. a walk-up from the current working directory for a sibling
       ``OpenFUSIONToolkit/build_release/python``.

    The first existing directory is prepended to ``sys.path`` and confirmed with
    ``import OpenFUSIONToolkit``.  Raises ``ModuleNotFoundError`` listing every
    location tried if none work, so the failure is actionable on a new machine.
    """
    tried: List[str] = []

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
        if os.path.isdir(cand):
            if not _on_path(cand):
                sys.path.insert(0, cand)
            chosen = cand
            break

    try:
        import OpenFUSIONToolkit  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Could not import OpenFUSIONToolkit. Set OFT_PYTHONPATH to its "
            "'python' directory, or add the install to bouquet.paths._OFT_CANDIDATES. "
            "Tried:\n  " + "\n  ".join(tried or ["<no candidates>"])
        ) from exc

    if verbose:
        print(f"[bouquet.paths] OFT from: {chosen or '(already importable)'}")
    return chosen or ""


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


def _report(path: str, verbose: bool) -> str:
    if verbose:
        print(f"[bouquet.paths] mesh: {path}")
    return path
