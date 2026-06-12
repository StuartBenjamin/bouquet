from . import uncertainties
from . import sampling
from . import TokaMaker_interface
from . import plotting
from . import utils
from . import io
from . import filtering

# gui is NOT imported eagerly to avoid pulling in matplotlib.pyplot
# at package load time (breaks headless / server environments).
# Use:  from bouquet import gui

# Public API re-exports
from .sampling import (
    GPRProfilePerturber,
    generate_perturbed_GPR,
    sigmoid_length_scale,
    verify_gpr_statistics,
    calc_cylindrical_li_proxy,
)

from .TokaMaker_interface import (
    classify_jphi_profile,
    fit_inductive_profile,
    perturb_kinetic_equilibrium,
    generate_bouquet,
    reconstruct_equilibrium,
)

from .uncertainties import (
    new_uncertainty_profiles,
    synthetic_ida_sigma,
)

from .plotting import (
    draw_kinetic_profiles,
    draw_pressure_profiles,
    draw_jphi_total,
    draw_jphi_components,
    draw_jphi_profiles,
    set_plot_style,
    WONG,
    plot_bouquet,
    plot_imas_bouquet,
    plot_transport_profiles,
    plot_extra_profiles,
    plot_tokamaker_comparison,
    plot_geqdsk_bouquet,
    plot_pfile_bouquet,
    plot_coil_currents,
    plot_kinetic_profiles,
    plot_jphi_profiles,
    plot_traces,
    plot_boundary_point_traces,
)

from .io import (
    GEQDSKEquilibrium,
    read_geqdsk,
    PFile,
    read_pfile,
)

from .filtering import (
    filter_coil_currents,
    filter_boundaries,
    read_filter_flags,
    select_indices,
    export_filtered,
)

from .utils import (
    Hmode_profiles,
    Ip_flux_integral_vs_target,
    initialize_equilibrium_database,
    store_equilibrium,
    load_equilibrium,
    load_equilibrium_by_path,
    store_baseline_profiles,
    load_baseline_profiles,
    discover_scan_values,
    count_equilibria,
    list_equilibrium_indices,
    read_eqdsk_from_bytes,
)

# ---- Class-based orchestrator API ----
# (run.py imports TokaMaker/OFT lazily inside methods, so this stays import-safe
# in headless environments.)
from .config import (
    BouquetConfig,
    SolverConfig,
    ReconstructionSource,
    ImasSource,
    FixedComponentsConfig,
    UncertaintyConfig,
    GenerationConfig,
    FilterConfig,
)
from .baseline import Baseline, resolve_baseline, resolve_uncertainty
from .physics import isotropize_fast_pressure, parallel_to_toroidal
from .io.ida import read_ida, IDAProfiles
from .io.imas import read_imas_baseline, read_imas_geometry
from .run import Bouquet
