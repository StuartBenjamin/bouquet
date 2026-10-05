"""Snapshot and restore of the solver state bouquet changes temporarily.

A TokaMaker object carries its state in three places, and a restore has to
cover all of them:

* **the equilibrium object** (``copy_eq`` / ``replace_eq``): psi, the coil
  currents, the coil regularisation (matrix, targets, weights), the global
  targets (Ip, pax, ...), the flux-function profiles, and the isoflux /
  saddle / flux / Mirnov constraints;
* **the device** (NOT in the equilibrium object): the hard coil bounds
  (``set_coil_bounds``), the VSC gains (``set_coil_vsc``), the Vcoils
  (``set_vcoils``) and the settings pushed by ``update_settings`` (maxits,
  nl_tol, urf, ...);
* **Python attributes bouquet publishes on the solver object**
  (:data:`BOUQUET_SOLVER_ATTRS`): the strong / weak coil-regularisation
  stashes and the hard-bound stash ``generate_bouquet`` and
  ``Bouquet._apply_coil_reg`` leave there, which later draw-path code reads.

**The coil-bound mode is one-way.**  Until ``set_coil_bounds`` is first
called, OpenFUSIONToolkit solves the coil least-squares problem by the normal
equations; from the first call on (``set_coil_bounds(None)`` included, which
installs +/-1e98) it solves it by bounded least squares (BVLS), and no call
returns it to the unbounded solve.  The two agree to round-off per solve but
not bit for bit, and a converged Picard iteration carries the difference
(measured on the synthetic g-file example: 3.8e-7 in the inductive amplitude
of the engine's zero-perturbation draw).  Every ``generate()`` enters the
bounded mode (its homotopy installs and then releases bounds), so a restore
cannot undo it; :meth:`SolverState.capture` therefore ENTERS it first --
re-installing the bounds bouquet has on record (``_coil_drift_bounds``, or
none: +/-1e98) -- so the state captured is already in the mode the guarded
code will leave, and the restore puts back exactly what was captured.  That
installs no constraint: +/-1e98 never binds, and a recorded hard bound is
re-installed as recorded.

Not covered (no bouquet code changes them): the isoflux gradient-weight limit
(bouquet always uses the default), the mesh, the coil and conductor
definitions.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

#: Python attributes bouquet publishes on the TokaMaker object (stashes read
#: by later draw-path code); restored as they were (absent stays absent).
BOUQUET_SOLVER_ATTRS = ("_strong_coil_reg", "_weak_coil_reg",
                        "_coil_drift_bounds")


def _settings_values(mygs):
    s = getattr(mygs, "settings", None)
    if s is None:
        return None
    out = {}
    for k in dir(s):
        if k.startswith("_"):
            continue
        v = getattr(s, k)
        if callable(v):
            continue
        out[k] = copy.deepcopy(v)
    return out


def _vsc_facs(mygs):
    vc = getattr(mygs, "_virtual_coils", None)
    if not isinstance(vc, dict):
        return None
    return copy.deepcopy((vc.get("#VSC") or {}).get("facs"))


class SolverState:
    """Everything bouquet may change on a TokaMaker object, captured so that
    :meth:`restore` puts it back.  Build with :meth:`capture`."""

    def __init__(self, mygs):
        self.mygs = mygs
        self.bounds = None
        self.eq = None
        self.settings = None
        self.attrs = {}
        self.vsc = None
        self.vcoils = None

    @classmethod
    def capture(cls, mygs, enter_bounded_mode=True):
        """Capture *mygs*'s state.  With *enter_bounded_mode* (default) the
        one-way coil-bound mode is entered first, re-installing the bounds
        bouquet has on record (see the module docstring)."""
        self = cls(mygs)
        rec = getattr(mygs, "_coil_drift_bounds", None)
        self.bounds = copy.deepcopy(rec)
        self.enter_bounded_mode = bool(enter_bounded_mode)
        if self.enter_bounded_mode and hasattr(mygs, "set_coil_bounds"):
            mygs.set_coil_bounds(copy.deepcopy(rec))
        if hasattr(mygs, "copy_eq") and hasattr(mygs, "replace_eq"):
            self.eq = mygs.copy_eq()
        self.settings = _settings_values(mygs)
        d = getattr(mygs, "__dict__", {})
        self.attrs = {k: copy.deepcopy(d[k]) for k in BOUQUET_SOLVER_ATTRS
                      if k in d}
        self.vsc = _vsc_facs(mygs)
        self.vcoils = copy.deepcopy(getattr(mygs, "_vcoils", None))
        return self

    def restore(self):
        """Put every captured piece back: the equilibrium object, the
        settings (pushed with ``update_settings``), the VSC gains and Vcoils
        when they changed, the coil bounds on record, and the Python
        attributes (an attribute absent at capture is removed)."""
        mygs = self.mygs
        if self.eq is not None:
            mygs.replace_eq(source_eq=self.eq)
        if self.settings is not None:
            s = mygs.settings
            for k, v in self.settings.items():
                if getattr(s, k, None) != v:
                    setattr(s, k, copy.deepcopy(v))
            if hasattr(mygs, "update_settings"):
                mygs.update_settings()
        if self.vsc is not None and _vsc_facs(mygs) != self.vsc \
                and hasattr(mygs, "set_coil_vsc"):
            mygs.set_coil_vsc(copy.deepcopy(self.vsc))
        if (getattr(mygs, "_vcoils", None) != self.vcoils
                and hasattr(mygs, "set_vcoils")):
            mygs.set_vcoils(copy.deepcopy(self.vcoils or {}))
        if self.enter_bounded_mode and hasattr(mygs, "set_coil_bounds"):
            mygs.set_coil_bounds(copy.deepcopy(self.bounds))
        d = getattr(mygs, "__dict__", None)
        if d is not None:
            for k in BOUQUET_SOLVER_ATTRS:
                if k in self.attrs:
                    d[k] = copy.deepcopy(self.attrs[k])
                else:
                    d.pop(k, None)


@contextmanager
def preserved_solver_state(mygs, enter_bounded_mode=True):
    """``with preserved_solver_state(mygs): ...`` -- :meth:`SolverState.
    capture` on entry, :meth:`SolverState.restore` on exit (also on an
    exception)."""
    st = SolverState.capture(mygs, enter_bounded_mode=enter_bounded_mode)
    try:
        yield st
    finally:
        st.restore()
