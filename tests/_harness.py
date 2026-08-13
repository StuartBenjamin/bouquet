"""Shared plumbing for the solver tests that run their probes in a subprocess.

WHY THIS EXISTS
---------------
Several solver tests keep every live-solver call behind a subprocess, because
``OFT_env`` is a per-process singleton and a module that builds a solver in the
pytest process makes ``pytest -m solver`` unrunnable alongside
``test_systematics.py``.  They launch it as::

    subprocess.run([sys.executable, os.path.abspath(__file__), ...])

Python sets ``sys.path[0]`` of a *script* to the SCRIPT'S OWN DIRECTORY -- here
``<repo>/tests`` -- not to the repository root.  There is no ``bouquet`` package
in ``tests/``, so ``import bouquet`` inside the probe falls through to whatever
is installed.  When ``bouquet`` is installed in editable/development mode, that
resolves to the checkout the install points at, which is NOT necessarily the
tree under test.

The failure mode is silent and severe: the probe solves happily against a
DIFFERENT revision of the library, and the assertions in the parent process
then describe code that was never exercised.  It has already bitten twice:

  * a branch adding ``diagnostics['r2_f_ind']`` saw ``f_ind = nan`` in every
    mode, because the probe imported an older tree with no such key -- read at
    the time as a defect in the feature itself; and
  * a four-branch integration run reported two solver failures that did not
    exist in the code being validated, costing a full re-run to disprove.

Both were the same bug, and neither announced itself.  So this module does two
things, and the second matters more than the first:

  1. :func:`subprocess_env` prepends the repo root to ``PYTHONPATH`` for the
     child, so the probe imports the tree it was launched from.
  2. :func:`assert_bouquet_is_repo_local` runs INSIDE the probe and fails
     loudly, naming both paths, if ``bouquet`` still resolved somewhere else.

(1) alone would fix today's symptom while leaving the class of bug silent; (2)
converts any future recurrence -- a launch site added without the helper, a
``PYTHONPATH`` stripped by a runner, an ``import`` that happens before the path
is set -- into an immediate, self-describing failure.
"""
from __future__ import annotations

import os
import sys

#: Repository root, derived from THIS file's location (``<repo>/tests``).
#: Never hardcoded: the tests must work from any checkout, worktree or clone.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def subprocess_env(**extra):
    """Environment for a probe subprocess, with the repo root importable.

    Prepends :data:`REPO_ROOT` to ``PYTHONPATH`` (preserving any existing
    value) so ``import bouquet`` in the child resolves to the tree under test
    rather than to an editable install pointing elsewhere.

    ``extra`` is merged last, exactly like the ``dict(os.environ, ...)`` idiom
    it replaces.
    """
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = (REPO_ROOT + os.pathsep + existing) if existing else REPO_ROOT
    return dict(os.environ, PYTHONPATH=pythonpath, **extra)


def assert_bouquet_is_repo_local():
    """Fail the probe unless ``bouquet`` was imported from this repository.

    Call at the top of every ``__main__`` probe entry point, BEFORE any solver
    work.  Raises :class:`SystemExit` with both paths in the message, so the
    parent's ``proc.stderr`` tail shows exactly what went wrong instead of a
    downstream ``nan`` or a missing diagnostics key.
    """
    import bouquet

    got = os.path.abspath(bouquet.__file__)
    expected_root = os.path.join(REPO_ROOT, "")
    if not got.startswith(expected_root):
        raise SystemExit(
            "\n".join((
                "",
                "=" * 72,
                "HARNESS ERROR: the probe subprocess imported the WRONG bouquet.",
                "",
                f"  imported from : {got}",
                f"  expected under: {REPO_ROOT}",
                f"  PYTHONPATH    : {os.environ.get('PYTHONPATH', '(unset)')}",
                "",
                "The probe would have solved against a different revision of the",
                "library than the one under test, and every assertion in the",
                "parent process would describe code that was never exercised.",
                "",
                "Usual cause: the subprocess was launched without",
                "tests/_harness.subprocess_env(), so sys.path[0] was <repo>/tests",
                "and `import bouquet` fell through to an editable install that",
                "points at another checkout.",
                "=" * 72,
            ))
        )
    return got


def ensure_repo_on_syspath():
    """Belt-and-braces for the child: put the repo root on ``sys.path`` too.

    ``subprocess_env`` is the primary mechanism; this covers a probe invoked by
    hand (``python tests/test_x.py ...``) without the helper, so running one
    directly during debugging still exercises the local tree.  Must run before
    the first ``import bouquet``.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT
