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
    # Default nthreads=1: GS solves are then bit-reproducible. nthreads>1 uses
    # OpenMP whose reduction order is non-deterministic -- which the LCFS-normalised
    # li_1 amplifies to ~±1% run-to-run, AND (at stiff slices, e.g. beam-heavy
    # equilibria with a sharp pedestal current) can occasionally tip the GS DLSODE
    # solve into a non-convergence loop on the unlucky thread schedule (a hard hang).
    # nthreads=1 removes both at ~no speed cost here (the solve barely thread-scales);
    # for throughput, run multiple single-threaded PROCESSES in parallel, one per
    # slice, each BLAS-pinned (VECLIB_MAXIMUM_THREADS=1 etc.), NOT nthreads>1.
    nthreads: int = 1
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

    ``impurity_Z`` is the **machine impurity charge** used to map between Z_eff
    and the main-ion density under single-impurity quasineutrality. It is the
    controlling input for the IDA path (which measures ne + Z_eff and derives
    ni = ne (Z_imp - Zeff)/(Z_imp - 1)); the default 6.0 is **carbon**, correct
    for DIII-D and other carbon-wall machines. **Set it for your device** --
    e.g. tungsten ~ 74 (use the effective radiating charge if W is not fully
    stripped), beryllium 4, neon 10. For a p-file this is informational: the
    Osborne ``N Z A`` footer carries the species directly and is authoritative.
    """

    geqdsk_path: str
    profiles_path: str                 # IDA .cdf OR p-file (auto-detected by extension)
    cocos: int = 1
    time: Optional[float] = None       # IDA time slice [s] (multi-time .cdf files)
    impurity_Z: float = 6.0            # effective impurity charge (carbon); set per machine
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
    # --- IDA-hybrid kinetics (GenerationConfig.kinetic_source = "ida_hybrid") ---
    # When set, the baseline ne/Te/Ti/omega_tor are taken from this IDA .cdf
    # (externally fit, smoother across time than FUSE's per-slice profile fits),
    # resampled onto the FUSE core_profiles psi_N grid. Z_eff / Z_imp / the ni
    # dilution stay FUSE (IDA's reported Z_eff is unreliable -- internally
    # inconsistent with its own carbon density). Everything else (currents,
    # equilibrium, p_fast, anchors) stays FUSE. Also wire it to
    # UncertaintyConfig.ida_path so the sigma envelopes come from the same IDA.
    ida_path: Optional[str] = None
    impurity_Z: float = 6.0            # machine impurity charge (carbon); ni dilution
    # Magnetics-only EFIT01 gEQDSK whose LCFS replaces the FUSE dd boundary as the
    # isoflux separatrix target (theoretically the most accurate separatrix). One
    # g-file per slice; the driver picks the nearest EFIT01 time.
    efit01_geqdsk: Optional[str] = None


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

    Each channel resolves independently: an explicit ``sigma_profiles`` array
    wins, else an IDA ``.cdf``, else a per-channel flat fraction
    (``ne``/``te``/``ni``/``ti``_scalar_sigma) times the baseline profile.
    """

    # Kinetic sigma resolution, in priority order per channel (ne/te/ni/ti):
    #   1. an explicit psi_N-dependent profile in `sigma_profiles` (absolute
    #      sigma on the kinetic grid), e.g. from `synthetic_ida_sigma` or a real
    #      diagnostic envelope;
    #   2. an IDA `.cdf` (`ida_path`, or the reconstruction source's own .cdf) --
    #      its `*_err` datasets;
    #   3. a flat fractional fallback, `<chan>_scalar_sigma` * baseline profile.
    ida_path: Optional[str] = None
    sigma_mode: str = "direct"             # "direct" (*_err datasets) | "ensemble"
    sigma_method: str = "percentile"       # "percentile" | "std"  (ensemble only)
    sigma_ni_from_ne: bool = True          # IDA path only: sigma_ni = sigma_ne

    # flat fractional kinetic sigma envelopes (per channel). ni/Ti default wider
    # than ne/Te -- ion density and temperature are harder to diagnose.
    ne_scalar_sigma: float = 0.05
    te_scalar_sigma: float = 0.05
    ni_scalar_sigma: float = 0.10
    ti_scalar_sigma: float = 0.10
    # explicit psi_N-dependent ABSOLUTE sigma profiles (on the kinetic grid),
    # keyed by 'ne'/'te'/'ni'/'ti'. Present channels override the scalar/IDA path.
    sigma_profiles: dict = field(default_factory=dict)

    # j_phi uncertainty: flat fractional envelope on |j_phi_baseline|
    jphi_scalar_sigma: float = 0.10

    # Z_eff uncertainty: flat fractional envelope that ENABLES the zeff channel
    # by default for every source (the physically-consistent density scheme --
    # see the aux block below). Each draw perturbs Z_eff within this band and
    # DERIVES the main-ion density ni from (ne, Z_eff) via quasineutrality, so
    # ni/nz/Z_eff stay mutually consistent and ni_scalar_sigma is unused.
    # Real Z_eff (visible bremsstrahlung / CER) is ~10-20% uncertain; the 0.05
    # default is deliberately conservative -- widen it for a realistic envelope.
    # Set 0.0 to disable (Z_eff held at baseline, ni drawn independently).
    # An explicit aux_sigmas['zeff'] always overrides this.
    zeff_scalar_sigma: float = 0.05

    # GPR correlation length scales (psi_N units) -- define the perturbation
    n_ls: float = 0.5                      # density
    t_ls: float = 0.4                      # temperature
    j_ls: float = 0.25                     # current density

    # --- switchboard: auxiliary perturbed profiles (source-decoupled) --------
    # Supplying a sigma profile (on psi_N_kinetic) ENABLES perturbing a named
    # profile. Baselines are auto-filled by the source where available
    # (omega_tor from the IDS, zeff computed); otherwise supply them in
    # aux_baselines (required for chi_e/chi_i and E_r, which production FUSE
    # files lack, and for the reconstruction/geqdsk path). A warning is emitted
    # if a sigma is given for a baseline that is all-zero or absent.
    # Recognised names: 'zeff' (ACTIVE -- re-enters the per-draw SWB bootstrap),
    # 'omega_tor', 'e_r', 'chi_e', 'chi_i' (passive: perturbed + stored, not GS
    # inputs). Works identically for ImasSource and ReconstructionSource.
    aux_sigmas: dict = field(default_factory=dict)         # {name: sigma(psi_N)}
    aux_baselines: dict = field(default_factory=dict)      # {name: baseline(psi_N)}
    aux_length_scales: dict = field(default_factory=dict)  # {name: GPR length} (default 0.4)


# ---------------------------------------------------------------------------
# Generation + filtering
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Perturbed-bouquet sampling over the uncertainty neighborhood."""

    n_equils: int = 20
    seed: Optional[int] = None
    # Label for this bouquet within the HDF5 file: draws are stored under
    # scan/<scan_key>/. Use it to keep several bouquets in one file under a
    # meaningful key (a time in ms, a beta value, ...). Default 0.
    scan_key: float = 0
    l_i_tolerance: float = 0.05            # l_i acceptance band (fraction of target)
    constrain_sawteeth: bool = False
    # When True, recompute bootstrap each draw via TokaMaker solve_with_bootstrap
    # and convert its parallel output to toroidal (see physics.parallel_to_toroidal),
    # overriding the baseline/FUSE j_BS. When False, keep the baseline j_BS.
    recalculate_j_BS: bool = True
    jBS_scale_range: tuple = (0.99, 1.01)  # bootstrap multiplicative spread
    # Bootstrap profile mode in solve_with_bootstrap. True (default) isolates the
    # edge spike, yielding a clean positive bootstrap; False uses the full SWB
    # profile, which for FUSE equilibria carries an unphysical inner negative
    # lobe (must then be floored, leaving kinks -- see baseline_jphi_caseA plots).
    isolate_edge_jBS: bool = True
    # How the SWB bootstrap is reconciled with the FUSE baseline on the IMAS
    # path (see run._forward_solve_imas_baseline):
    #   "diff"    : keep FUSE total; add fixed correction diff = FUSE_jBS - SWB
    #               to baseline and every draw (anchors to FUSE; risks edge
    #               misalignment when the perturbed pedestal moves).
    #   "rescale" : keep FUSE ohmic; rescale SWB by a single factor so l_i
    #               matches the source (fully self-consistent bootstrap).
    jBS_baseline_mode: str = "diff"
    # Fix B: when the recon-anchor's equilibrium l_i is already within the band,
    # accept the anchor and skip find_optimal_scale + the corrective iteration
    # (which otherwise overshoot l_i and drift degenerate coils off baseline).
    accept_anchor_inband: bool = False
    # Fix C: GPR-perturb the inductive in the recon-anchor itself (then accept).
    # This is an ALTERNATIVE to the standard flagship l_i loop, which ALREADY
    # GPR-perturbs j_inductive (rejection sampling, TokaMaker_interface.py ~1842);
    # BOTH consume sigma_jphi/j_ls and BOTH satisfy the "perturb all profiles incl.
    # j_inductive" rule. Difference: Fix C accepts the perturbed-in-anchor draw and
    # skips find_optimal_scale + the corrective (which can homogenize draws).
    # Default False = the standard flagship loop -- the mechanism geqdsks used
    # successfully (high draw completion). Set True for FUSE/IMAS diff mode (avoids
    # the matching-loop homogenization: 148798 diff+C = 15/15); but it DROPS draws
    # on stiff high-l_i geqdsks (204441) via band-conditioning rejection, so it is
    # NOT a good geqdsk default. FUSE runs opt in explicitly. (Only the PIN_JPHI /
    # DIFF_BS-at-sigma=0 diagnostic modes actually freeze j_ind -- those are the
    # backend-test exceptions to the rule.)
    perturb_jind_in_anchor: bool = False
    # Escape hatch for the per-path workflow guard (run._validate_workflow):
    # from_imas/from_geqdsk auto-apply their validated workflow (IMAS=diff+C,
    # geqdsk=standard l_i loop) and generate() raises on a known-bad combo
    # (geqdsk+Fix C, IMAS without Fix C, or jphi_scalar_sigma<=0 which freezes
    # j_inductive). Set True ONLY for deliberate backend tests / experiments to
    # bypass the guard (it will warn instead of raise).
    allow_unsafe_workflow: bool = False
    # Fail-fast (IMAS path) if the source carries more thermal pressure than the
    # reconstructed total used by the solve -- a dropped impurity species or a
    # fast-ion channel left out. io.imas.read_imas_baseline raises naming the
    # missing component + magnitude; the diff anchor then absorbs only the small
    # residual. Set True ONLY for backend tests to bypass (warns instead).
    allow_incomplete_pressure: bool = False
    # Anchor the total j_phi to equilibrium.profiles_1d.j_tor (the GS-consistent
    # current GPEC reads, carrying the pedestal current) instead of
    # core_profiles.j_tor (the transport parallel-current sum, which differs from
    # ~q=2 outward). Adds a FIXED jphi_diff = equilibrium.j_tor - core_profiles.j_tor
    # to the baseline AND every draw (mirroring jBS_diff/p_diff); the SWB bootstrap
    # + perturbed j_ind ride underneath so the edge current still carries Sauter UQ.
    # IMAS path only (geqdsk reads its total from the g-file). Default True.
    anchor_jtor_to_equilibrium: bool = True
    # Source of the baseline kinetic profiles on the IMAS path:
    #   "fuse"       -> FUSE core_profiles ne/Te/Ti (default; original behaviour)
    #   "ida_hybrid" -> ne/Te/Ti/omega_tor from ImasSource.ida_path (resampled onto
    #                   the FUSE psi_N grid); Z_eff/Z_imp/ni-dilution stay FUSE;
    #                   currents/equilibrium/p_fast/anchors stay FUSE.
    kinetic_source: str = "fuse"
    # Anchor the solve thermal pressure to equilibrium.pressure via the fixed
    # p_diff = equilibrium.pressure - p_reconstructed offset. With FUSE kinetics
    # this re-adds the carbon/fast/GS residual the recon omits. With IDA-hybrid
    # kinetics it would force the (trusted) IDA pressure back onto the FUSE total,
    # which we do NOT want -- so default False. When False, p_diff is None and the
    # solve pressure is e(ne*Te+ni*Ti) + impurity + p_fast with no equilibrium anchor.
    anchor_pressure_to_equilibrium: bool = False
    # Floor the SWB bootstrap at 0 (drop unphysical negative excursions) before
    # it enters j_phi, in both the baseline and every draw. Default False: only
    # needed for the isolate_edge_jBS=False full-profile mode (which carries an
    # inner negative lobe). With the default isolate_edge_jBS=True the spike is
    # already ~clean, so flooring is redundant -- and it REGRESSED 204441
    # (clipping its high-l_i isolate-edge spike drove yield to 0; off -> restored).
    floor_j_BS: bool = False
    # solve_with_bootstrap H-mode self-consistency iterations per draw (default
    # 3); lowering to 2 trades a little accuracy for speed on large bouquets.
    swb_iterations: int = 3
    # Coil handling (homotopy-based). The inverse solve drifts coils within
    # coil_drift, stepped through homotopy_passes = list of (F_tol, VSC_tol)
    # stages that tighten loose->tight (each warm-starts the next). A single
    # direct solve at the tight target is usually infeasible -- the schedule is
    # what makes ±1% coils reachable.
    coil_drift: float = 0.01
    # Optional HARD inequality bounds at +/- coil_drift_hard_factor*coil_drift,
    # enforced in every solve (not just the soft homotopy). None = soft only.
    # Backstop for degenerate-coil drift (e.g. F9B) at the cost of possible
    # boundary error when the reshaped equilibrium genuinely wants the coil off.
    coil_drift_hard_factor: float = None
    homotopy_passes: list = field(
        default_factory=lambda: [(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]
    )
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
    # When False (default) the verbose TokaMaker solver chatter emitted during
    # baseline reconstruction (DLSODE / gs_get_qprof / li-match iteration) is
    # captured to baseline.reconstruction_log and only a curated quality summary
    # is printed. Set True to stream the full solver output for debugging.
    verbose: bool = False

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
        if self.uncertainty.sigma_method not in ("percentile", "std"):
            raise ValueError(
                "uncertainty.sigma_method must be 'percentile' or 'std'")
        if self.generation.n_equils < 1:
            raise ValueError("generation.n_equils must be >= 1")
