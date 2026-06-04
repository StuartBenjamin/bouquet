"""Reader for FUSE IMAS/OMAS data-dictionary files (``dd_sim.json``).

FUSE writes the IMAS data dictionary as a plain JSON dump, so this reads with the
stdlib ``json`` module -- no IMAS/OMAS/OMFIT install required. It returns a
fully-separated baseline (no GS reconstruction needed): the IDS already carries
j_ohmic, j_bootstrap and the driven currents split apart.

Field mapping (verified against a D3D FUSE run)::

    equilibrium.time_slice[t].global_quantities.ip            -> Ip_target
    equilibrium.time_slice[t].global_quantities.li_3          -> l_i_target
    core_profiles.profiles_1d[t].grid.{psi,rho_tor_norm}      -> radial grid
    core_profiles.profiles_1d[t].j_ohmic                      -> inductive (parallel)
    core_profiles.profiles_1d[t].j_bootstrap                  -> bootstrap (parallel)
    core_profiles.profiles_1d[t].j_total / j_tor              -> parallel/toroidal totals
    core_profiles.profiles_1d[t].electrons.{density_thermal,temperature}
    core_profiles.profiles_1d[t].ion[*].{density_thermal,temperature,label}
    core_profiles.profiles_1d[t].{electrons,ion[*]}.pressure_fast_{perpendicular,parallel}
    core_sources.source[*].profiles_1d[t].j_parallel          -> beam-source j_NBI only

Currents are converted parallel->toroidal (see :func:`bouquet.physics.parallel_to_toroidal`)
and fast pressure is isotropized (see :func:`bouquet.physics.isotropize_fast_pressure`)
before being packaged into a :class:`~bouquet.baseline.Baseline`.

Note: ``j_BS`` read here is the FUSE bootstrap baseline, but it is *overridden*
when ``GenerationConfig.recalculate_j_BS`` is True -- bouquet then recomputes
bootstrap per draw via TokaMaker ``solve_with_bootstrap`` (whose output is also
parallel and must be converted to toroidal; see ``parallel_to_toroidal``).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ImasSource, FixedComponentsConfig
    from ..baseline import Baseline


# Core-source identifier index for neutral-beam current drive.
NBI_SOURCE_INDEX = 2          # neutral beam injection -> summed into j_NBI
# NOTE: j_RF is NOT computed internally (RF is the least-common input). It is
# left as zeros and accepted as a user-supplied array via
# FixedComponentsConfig.j_RF. See the "revisit RF" flag in the project notes
# if/when internal EC/IC/LH summation is wanted.


def read_imas_baseline(
    source: "ImasSource",
    fixed: Optional["FixedComponentsConfig"] = None,
    p_fast_reduction: str = "trace",
) -> "Baseline":
    """Read a FUSE ``dd_sim.json`` IDS and return a separated :class:`Baseline`.

    Steps:
      1. ``json.load`` the file; select the time slice nearest ``source.time``.
      2. Pull Ip / l_i from ``equilibrium...global_quantities``.
      3. Pull j_ohmic / j_bootstrap (parallel) and the total parallel/toroidal
         currents from ``core_profiles``.
      4. Sum ``core_sources`` ``j_parallel`` over beam sources into j_NBI.
         j_RF is left as zeros (no internal RF calculation).
      5. Convert every parallel current to toroidal via the j_tor/j_total ratio.
      6. Build p_fast by isotropizing per-species fast pressure
         (``isotropize_fast_pressure``, ``method=p_fast_reduction``) and summing.
      7. Arrays in ``fixed`` override the IDS-derived components when provided
         (this is the only way to supply j_RF).

    No Grad-Shafranov reconstruction is performed -- the provenance is "imas".
    """
    raise NotImplementedError
