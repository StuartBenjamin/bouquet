"""The :class:`Baseline` -- the common product every source resolves to.

``generate()`` consumes a :class:`Baseline` and never depends on reconstruction
directly. Both :class:`~bouquet.config.ReconstructionSource` and
:class:`~bouquet.config.ImasSource` resolve to this same structure, so adding a
new input pipeline means adding a resolver, not touching generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .config import BaselineSource, SolverConfig


@dataclass
class Baseline:
    """Separated baseline currents + targets + kinetic profiles.

    The currents are split into ohmic + bootstrap + driven (j_phi =
    j_inductive + j_BS + j_NBI + j_RF) regardless of provenance: the
    reconstruction source *produces* the split, the IMAS source *reads* it
    pre-separated.

    CURRENT CONVENTION: every current component here is a flux-surface-averaged
    *toroidal* current density j_phi [A/m^2]. IMAS/neoclassical inputs are
    parallel (<j.B>/B0) and are converted on read via
    :func:`bouquet.physics.parallel_to_toroidal`, so downstream code never mixes
    conventions.
    """

    # --- required fields (no defaults) ---------------------------------
    # current-density grid + separated currents
    psi_N: "np.ndarray"
    j_phi: "np.ndarray"            # total [A/m^2] = j_inductive + j_BS + j_NBI + j_RF
    j_inductive: "np.ndarray"     # ohmic part [A/m^2]   (perturbed via l_i matching)
    j_BS: "np.ndarray"            # bootstrap part [A/m^2] (recomputed per draw)

    # baseline kinetic profiles (SI: m^-3, eV) on `psi_N_kinetic`
    psi_N_kinetic: "np.ndarray"
    ne: "np.ndarray"
    te: "np.ndarray"
    ni: "np.ndarray"
    ti: "np.ndarray"
    Zeff: "np.ndarray"

    # targets held fixed across all perturbed draws
    Ip_target: float
    l_i_target: float

    provenance: str               # "reconstruction" | "imas"

    # --- optional / defaulted fields -----------------------------------
    # Fixed additive components -- summed into EVERY draw, never GPR-perturbed.
    # None is treated as zeros. See FixedComponentsConfig for the contract:
    #   j_phi_total = j_inductive + j_BS + j_NBI + j_RF
    #   p_total     = p_thermal(perturbed) + p_fast
    j_NBI: Optional["np.ndarray"] = None    # beam-driven current [A/m^2]
    j_RF: Optional["np.ndarray"] = None     # RF-driven current [A/m^2]
    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure [Pa]

    # raw bytes preserved for archival into the HDF5 (optional)
    eqdsk_bytes: Optional[bytes] = None
    pfile_bytes: Optional[bytes] = None

    # full reconstruction diagnostics, when provenance == "reconstruction"
    recon: Optional[dict] = None


def resolve_baseline(config: "BouquetConfig", mygs=None) -> Baseline:
    """Dispatch on ``config.source`` and return a populated :class:`Baseline`.

    * :class:`ImasSource` -> delegate to
      :func:`bouquet.io.imas.read_imas_baseline`: take j_ohmic / j_BS / Ip / l_i
      and core_profiles directly (no GS reconstruction), convert parallel->
      toroidal, isotropize p_fast. ``mygs`` is unused.
    * :class:`ReconstructionSource` -> read g-file + profiles (IDA .cdf or
      p-file), run :func:`reconstruct_equilibrium` on ``mygs``, package the
      fitted split currents (converted to toroidal) plus any user-supplied
      fixed components. (Implemented in phase 2 step 3.)

    Both paths return a :class:`Baseline` with every current as toroidal j_phi
    and the fixed additive components (j_NBI, j_RF, p_fast) attached.

    Implemented as a free function so sources stay declarative (plain config)
    and the resolution logic lives in one place.
    """
    from .config import ImasSource, ReconstructionSource

    source = config.source

    if isinstance(source, ImasSource):
        from .io.imas import read_imas_baseline
        return read_imas_baseline(
            source,
            fixed=config.fixed_components,
            p_fast_reduction=config.fixed_components.p_fast_reduction,
        )

    if isinstance(source, ReconstructionSource):
        raise NotImplementedError(
            "ReconstructionSource baseline (g-file + IDA/p-file -> GS recon) "
            "lands in phase 2 step 3"
        )

    raise TypeError(f"unknown baseline source type: {type(source).__name__}")
