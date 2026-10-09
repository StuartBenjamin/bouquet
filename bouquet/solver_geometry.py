"""The per-baseline solver geometry: vacuum F0, LCFS boundary, isoflux and saddle constraints.

The mesh, regions, coil mode and VSC are fixed when the solver is set up; everything here belongs to one
baseline, so one solver can run many baselines (:meth:`bouquet.run.Bouquet.point_solver`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

ISOFLUX_WEIGHT = 500.0


@dataclass(frozen=True)
class CaseGeometry:
    F0: float
    boundary_RZ: Optional[np.ndarray]
    iso_pts: Optional[np.ndarray]
    iso_w: Optional[np.ndarray]
    saddle_pts: Optional[np.ndarray] = None
    saddle_w: Optional[np.ndarray] = None

    def apply(self, mygs):
        """Install this geometry on ``mygs`` (after a reset to the clean equilibrium)."""
        mygs.set_profiles(foffset=float(self.F0))
        if self.iso_pts is not None:
            mygs.set_isoflux(self.iso_pts, weights=self.iso_w)
        if self.saddle_pts is not None:
            mygs.set_saddle_constraints(self.saddle_pts, weights=self.saddle_w)


def case_geometry(source, solver) -> CaseGeometry:
    """The :class:`CaseGeometry` of ``source`` (g-file: R_center*B_center and its boundary; IDS: |r0*b0(t)|
    and the boundary outline or ``LCFS_geqdsk``), with ``solver.F0`` / ``isoflux_pts`` / ``saddle_targets``
    overriding."""
    from .config import ImasSource, ReconstructionSource
    if isinstance(source, ReconstructionSource):
        from .io.geqdsk import read_geqdsk
        g = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
        F0_src = abs(g.R_center * g.B_center)
        boundary_RZ = np.column_stack([g.boundary_R, g.boundary_Z])
    elif isinstance(source, ImasSource):
        from .io.imas import read_imas_geometry
        F0_src, boundary_RZ = read_imas_geometry(source)
    else:
        raise TypeError(f"unknown baseline source type {type(source).__name__}")
    F0 = F0_src if solver.F0 is None else float(solver.F0)
    if abs(F0 - F0_src) > 1e-3 * abs(F0_src):
        warnings.warn(f"SolverConfig.F0={F0:.4f} overrides the source's F0={F0_src:.4f}; the engine's IDS "
                      "contract still uses the source's vacuum field", UserWarning, stacklevel=2)
    iso_pts, iso_w = solver.isoflux_pts, solver.isoflux_weights
    if iso_pts is None:
        iso_pts, iso_w = boundary_RZ, np.full(len(boundary_RZ), ISOFLUX_WEIGHT)
    sad = sad_w = None
    if solver.saddle_targets is not None:
        sad = np.asarray(solver.saddle_targets, dtype=np.float64).reshape(-1, 2)
        sad_w = None if solver.saddle_weights is None else np.asarray(solver.saddle_weights, dtype=np.float64)
    return CaseGeometry(F0=F0, boundary_RZ=boundary_RZ, iso_pts=iso_pts, iso_w=iso_w,
                        saddle_pts=sad, saddle_w=sad_w)
