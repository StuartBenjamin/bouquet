"""Typed configuration for the Bouquet orchestrator.

The configuration is split along two independent axes so the same generation
machinery serves multiple input pipelines:

  * **Baseline source** -- where the baseline current density (j_phi / j_ohmic /
    j_BS), the targets (Ip, l_i) and the baseline kinetic profiles come from.
    Either reconstruct them from a g-file (:class:`ReconstructionSource`) or read
    them pre-separated from a FUSE IMAS/OMAS IDS (:class:`ImasSource`).

  * **Uncertainty envelope** (:class:`UncertaintyConfig`) -- the kinetic sigma
    profiles and the j_phi sigma, plus the GPR correlation lengths that define
    how profiles are perturbed. Orthogonal to the baseline source.

Pass a fully-populated :class:`BouquetConfig` to ``bq.Bouquet(config)``.

Using dataclasses (rather than a raw dict) buys validation, IDE autocomplete,
and a documented home for every knob -- a typo fails immediately in
``__post_init__`` instead of deep inside a GS solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Solver (common to every baseline source -- perturbed draws are always solved
# with TokaMaker, regardless of where the baseline came from)
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """TokaMaker setup -- everything needed to stand up ``mygs``.

    ``cond_dict`` / ``coil_dict`` normally come straight from the mesh file and
    need no adjustment; they are intentionally *not* surfaced here. Pass
    ``region_overrides`` only in the special cases where you must tweak them.
    """

    mesh_path: str
    nthreads: int = 4
    order: int = 3
    F0: Optional[float] = None                       # vacuum R*Bt; default from g-file/IDS
    isoflux_pts: Optional["np.ndarray"] = None       # (N, 2) boundary constraints
    isoflux_weights: Optional["np.ndarray"] = None   # (N,)
    coil_vsc: dict = field(default_factory=lambda: {"F9A": 1.0, "F9B": -1.0})
    coil_reg: list = field(default_factory=list)
    region_overrides: Optional[dict] = None          # special-case cond/coil dict edits


# ---------------------------------------------------------------------------
# Baseline sources (discriminated union via `BouquetConfig.source`)
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionSource:
    """Baseline obtained by reconstructing a GS equilibrium from a g-file.

    Runs :func:`reconstruct_equilibrium`: fits a smooth inductive profile and
    matches l_i(1) via secant iteration, producing the separated
    j_phi / j_inductive / j_BS that generation needs.

    Baseline kinetic profiles come from ``profiles_path``. The reader dispatches
    on file type -- an IDA ``.cdf`` is used if given (and is then also the
    default sigma source for :class:`UncertaintyConfig`), otherwise a p-file.
    No mixing by default: one file supplies the full profile set. Individual
    profiles can still be replaced via ``profile_overrides`` (e.g.
    ``{"ti": my_ti_array}``) on the kinetic psi_N grid.
    """

    geqdsk_path: str
    profiles_path: str                 # IDA .cdf OR p-file (auto-detected by extension)
    cocos: int = 1
    time: Optional[float] = None       # IDA time slice [s] (multi-time .cdf files)
    profile_overrides: dict = field(default_factory=dict)  # name -> array, manual override
    # reconstruction knobs
    psi_pad: float = 1e-3
    n_k: int = 5                       # inductive-spline knots
    psi_bridge: float = 0.99          # Hermite edge-bridge location
    rescale_j_BS: bool = False
    shelf_psi_N: float = 0.0
    # guess_jinductive is derived from the g-file j_phi when None


@dataclass
class ImasSource:
    """Baseline read directly from a FUSE IMAS/OMAS IDS -- no reconstruction.

    The IDS already carries j_ohmic and j_BS separated (j_phi = j_ohmic + j_BS),
    along with the equilibrium (Ip, l_i) and core_profiles (ne/Te/ni/Ti/Zeff),
    so the reconstruction step is skipped entirely. This is expected to become
    the primary input pipeline.
    """

    ids_path: str                      # IMAS/OMAS file (FUSE output)
    time: Optional[float] = None       # time slice [s]; None -> single/first slice
    occurrence: int = 0


BaselineSource = Union[ReconstructionSource, ImasSource]


# ---------------------------------------------------------------------------
# Fixed additive components (NEVER perturbed by GPR draws)
# ---------------------------------------------------------------------------
@dataclass
class FixedComponentsConfig:
    """Externally-driven current + fast-ion pressure, held fixed across draws.

    These are summed into the baseline *and* every perturbed equilibrium,
    untouched by the GPR perturbation::

        j_phi_total = j_inductive + j_BS + j_NBI + j_RF
        p_total     = p_thermal(perturbed) + p_fast

    Any component may simply be handed in as a 1-D array over ``psi_N`` -- this is
    a first-class input path for *both* sources, not just a fallback. Supplying an
    array always overrides whatever a source would otherwise derive.

    Provenance / defaults:
      * ``p_fast`` -- :class:`ImasSource` builds it from per-species
        ``pressure_fast_{perpendicular,parallel}`` (reduced via
        ``p_fast_reduction``); :class:`ReconstructionSource` defaults to zero.
        Either way an explicit array here wins (e.g. from TRANSP/ONETWO).
      * ``j_NBI`` -- :class:`ImasSource` sums beam-source ``j_parallel``;
        :class:`ReconstructionSource` defaults to zero. Explicit array wins.
      * ``j_RF`` -- **never computed internally** (RF is the least-common input).
        Always zeros unless the user supplies an array here.

    All arrays are on ``psi_N`` (kinetic grid), SI units, toroidal current
    convention for j_*. ``None`` -> zeros.
    """

    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure [Pa]
    j_NBI: Optional["np.ndarray"] = None    # beam-driven TOROIDAL current density [A/m^2]
    j_RF: Optional["np.ndarray"] = None     # RF-driven TOROIDAL current density [A/m^2]
    psi_N: Optional["np.ndarray"] = None    # grid for the above (if arrays given)

    # How to collapse anisotropic fast-ion pressure (p_perp, p_par) to the scalar
    # p_fast that a scalar-pressure GS solver needs. See
    # bouquet.physics.isotropize_fast_pressure.
    #   "trace" -> (2*p_perp + p_par)/3   [DEFAULT; tr(P)/3, preserves fast energy]
    #   "mean"  -> (p_perp + p_par)/2
    #   "perp"  -> p_perp                 (diamagnetic-dominant)
    p_fast_reduction: str = "trace"


# ---------------------------------------------------------------------------
# Uncertainty envelope (orthogonal to the baseline source)
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyConfig:
    """Kinetic sigma profiles + j_phi sigma + GPR correlation lengths.

    Kinetic sigmas are sourced from an IDA ``.cdf`` (``ida_path``). Two modes
    match the operational notebook:

      * ``"ensemble"``  -- reduce the (n_samples, n_radial) posterior to a band
        via ``sigma_method`` (``"percentile"`` -> (p84 - p16)/2, or ``"std"``).
      * ``"direct"``    -- read the ``*_err`` datasets (n_e_err / T_e_err /
        T_12C6_err) directly.

    When ``ida_path`` is None the kinetic sigmas fall back to ``fallback_frac``
    times each baseline profile. Ion density sigma defaults to the electron
    value (quasi-neutrality).
    """

    # Kinetic sigma source. None -> if the baseline source's profiles come from
    # an IDA .cdf, reuse that file (read once); otherwise fall back to
    # `fallback_frac` * baseline profile. Set explicitly to force a sigma file.
    ida_path: Optional[str] = None
    sigma_mode: str = "direct"             # "direct" (*_err datasets) | "ensemble"
    sigma_method: str = "percentile"       # "percentile" | "std"  (ensemble only)
    sigma_ni_from_ne: bool = True          # quasi-neutrality: sigma_ni = sigma_ne
    fallback_frac: float = 0.10            # used when ida_path is None

    # j_phi uncertainty: flat fractional envelope on |j_phi_baseline|
    sigma_jphi_frac: float = 0.10

    # GPR correlation length scales (psi_N units) -- define the perturbation
    n_ls: float = 0.5                      # density
    t_ls: float = 0.4                      # temperature
    j_ls: float = 0.25                     # current density


# ---------------------------------------------------------------------------
# Generation + filtering
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Perturbed-bouquet sampling over the uncertainty neighborhood."""

    n_equils: int = 20
    seed: Optional[int] = None
    l_i_tolerance: float = 0.01
    constrain_sawteeth: bool = False
    # When True, recompute bootstrap each draw via TokaMaker solve_with_bootstrap
    # and convert its parallel output to toroidal (see physics.parallel_to_toroidal),
    # overriding the baseline/FUSE j_BS. When False, keep the baseline j_BS.
    recalculate_j_BS: bool = True
    jBS_scale_range: Optional[tuple] = None
    # coil handling
    lock_coils: bool = True
    lock_coils_weight: float = 1.0e4
    coil_drift_threshold_A: Optional[float] = None
    homotopy_passes: Optional[list] = None   # list of (F_tol, VSC_tol)
    diagnostic_plots: bool = False


@dataclass
class FilterConfig:
    """Postprocessing selection of the machine-realizable subset."""

    rms_max_mm: float = 5.0
    inspec_F_max: float = 0.02      # +/-2% coil-current spec (DIII-D)
    inspec_VSC_max: float = 0.02


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class BouquetConfig:
    """Top-level configuration. Pass to ``bq.Bouquet(config)``."""

    source: BaselineSource
    solver: SolverConfig
    output_header: str                              # HDF5 written to f"{header}.h5"
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    fixed_components: FixedComponentsConfig = field(default_factory=FixedComponentsConfig)

    def __post_init__(self):
        """Validate cross-field invariants early (before any GS solve)."""
        if not self.output_header:
            raise ValueError("output_header must be a non-empty string")

        src = self.source
        if isinstance(src, ReconstructionSource):
            if not src.geqdsk_path or not src.profiles_path:
                raise ValueError(
                    "ReconstructionSource requires geqdsk_path and profiles_path"
                )
        elif isinstance(src, ImasSource):
            if not src.ids_path:
                raise ValueError("ImasSource requires ids_path")
        else:
            raise TypeError(
                "source must be a ReconstructionSource or ImasSource, got "
                f"{type(src).__name__}"
            )

        if self.fixed_components.p_fast_reduction not in ("trace", "mean", "perp"):
            raise ValueError(
                "fixed_components.p_fast_reduction must be 'trace', 'mean', or 'perp'"
            )
        if self.uncertainty.sigma_mode not in ("direct", "ensemble"):
            raise ValueError("uncertainty.sigma_mode must be 'direct' or 'ensemble'")
