"""Reader for FUSE IMAS/OMAS data-dictionary files (``dd_sim.json``).

FUSE writes the IMAS data dictionary as a plain JSON dump, so this reads with the
stdlib ``json`` module -- no IMAS/OMAS/OMFIT install required. It returns a
fully-separated baseline (no GS reconstruction needed): the IDS already carries
j_ohmic, j_bootstrap and the driven currents split apart.

Field mapping (verified against a D3D FUSE run)::

    equilibrium.time_slice[t].global_quantities.ip            -> Ip_target
    equilibrium.time_slice[t].global_quantities.li_3          -> l_i_target
    core_profiles.profiles_1d[t].grid.psi                     -> normalised -> psi_N
    core_profiles.profiles_1d[t].j_total                      -> total PARALLEL current
    core_profiles.profiles_1d[t].j_tor                        -> total TOROIDAL current
    core_profiles.profiles_1d[t].j_ohmic                      -> inductive (parallel)
    core_profiles.profiles_1d[t].j_bootstrap                  -> bootstrap (parallel)
    core_profiles.profiles_1d[t].electrons.{density_thermal,temperature}
    core_profiles.profiles_1d[t].ion[*].{density_thermal,temperature,element[].z_n}
    core_profiles.profiles_1d[t].{electrons,ion[*]}.pressure_fast_{perpendicular,parallel}
    core_sources.source[*].profiles_1d[t].j_parallel          -> beam-source j_NBI only

Currents are converted parallel->toroidal (see :func:`bouquet.physics.parallel_to_toroidal`)
via the per-surface factor c = j_tor/j_total, and fast pressure is isotropized
(see :func:`bouquet.physics.isotropize_fast_pressure`). The total j_phi is set to
the authoritative toroidal ``j_tor`` and the inductive component is taken as the
residual ``j_phi - j_BS - j_NBI - j_RF`` so the decomposition sums exactly and Ip
is preserved.

Note: ``j_BS`` read here is the FUSE bootstrap baseline, but it is *overridden*
when ``GenerationConfig.recalculate_j_BS`` is True -- bouquet then recomputes
bootstrap per draw via TokaMaker ``solve_with_bootstrap`` (whose output is also
parallel and must be converted to toroidal; see ``parallel_to_toroidal``).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ..physics import isotropize_fast_pressure, parallel_to_toroidal

if TYPE_CHECKING:
    from ..config import ImasSource, FixedComponentsConfig
    from ..baseline import Baseline

# Core-source identifier index for neutral-beam current drive.
NBI_SOURCE_INDEX = 2          # neutral beam injection -> summed into j_NBI
# NOTE: j_RF is NOT computed internally (RF is the least-common input). It is
# left as zeros and accepted as a user-supplied array via
# FixedComponentsConfig.j_RF. See the "revisit RF" flag in the project notes
# if/when internal EC/IC/LH summation is wanted.


def _nearest_index(time_array, t: Optional[float], what: str) -> int:
    """Index of the slice nearest ``t`` (seconds) in ``time_array``."""
    ta = np.asarray(time_array, dtype=float)
    if ta.size == 1:
        return 0
    if t is None:
        raise ValueError(
            f"{what} has {ta.size} time slices; set ImasSource.time (seconds). "
            f"Range [s]: [{ta.min():.4f}, {ta.max():.4f}]"
        )
    return int(np.argmin(np.abs(ta - t)))


def _isotropic_fast_pressure(species: dict, method: str, n: int):
    """Isotropized fast pressure for one species, or zeros if it carries none."""
    if "pressure_fast_perpendicular" not in species:
        return np.zeros(n)
    p_perp = np.asarray(species["pressure_fast_perpendicular"], dtype=float)
    p_par = np.asarray(species.get("pressure_fast_parallel", p_perp), dtype=float)
    return isotropize_fast_pressure(p_perp, p_par, method=method)


def _override(arr, src_psi, dst_psi):
    """Resample a user-supplied fixed-component array onto the baseline grid."""
    arr = np.asarray(arr, dtype=float)
    if src_psi is None:
        if arr.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the baseline grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return arr
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), arr)


def read_imas_geometry(source: "ImasSource"):
    """Return ``(F0, boundary_RZ)`` from a FUSE IDS for TokaMaker setup.

    ``F0 = |r0 * b0(t)|`` from ``equilibrium.vacuum_toroidal_field`` and the LCFS
    isoflux points from ``equilibrium.time_slice[t].boundary.outline``. Used by
    :meth:`Bouquet.setup_solver` when the source is an :class:`ImasSource`
    (replacing the g-file that the reconstruction path reads F0/boundary from).
    """
    import json

    with open(source.ids_path) as fh:
        dd = json.load(fh)
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], source.time, "equilibrium")
    vtf = eq["vacuum_toroidal_field"]
    r0 = float(vtf["r0"])
    b0 = vtf["b0"]
    b0v = float(b0[ie]) if isinstance(b0, list) else float(b0)
    F0 = abs(r0 * b0v)
    out = eq["time_slice"][ie]["boundary"]["outline"]
    boundary_RZ = np.column_stack([
        np.asarray(out["r"], dtype=float), np.asarray(out["z"], dtype=float),
    ])
    return F0, boundary_RZ


def read_imas_baseline(
    source: "ImasSource",
    fixed: Optional["FixedComponentsConfig"] = None,
    p_fast_reduction: str = "trace",
) -> "Baseline":
    """Read a FUSE ``dd_sim.json`` IDS and return a separated :class:`Baseline`.

    No Grad-Shafranov reconstruction is performed -- provenance is "imas".
    """
    import json
    from ..baseline import Baseline

    with open(source.ids_path, "rb") as fh:
        raw_bytes = fh.read()
    dd = json.loads(raw_bytes)
    T = source.time

    # --- targets from the equilibrium IDS ---
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], T, "equilibrium")
    gq = eq["time_slice"][ie]["global_quantities"]
    Ip_target = abs(float(gq["ip"]))
    ids_li_1 = float(gq["li_1"]) if "li_1" in gq else None
    ids_li_3 = float(gq["li_3"]) if "li_3" in gq else None
    # Provisional l_i_target = IDS li_3; the IMAS forward-solve (Bouquet) replaces
    # this with the TokaMaker-solved li_1 and records the IDS values for sanity.
    l_i_target = ids_li_3 if ids_li_3 is not None else 0.0

    # --- profiles + currents from core_profiles ---
    cp_ids = dd["core_profiles"]
    ic = _nearest_index(cp_ids["time"], T, "core_profiles")
    cp = cp_ids["profiles_1d"][ic]

    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])   # 0 (axis) -> 1 (boundary)
    n = psi_N.size

    j_total = np.asarray(cp["j_total"], dtype=float)   # total parallel
    j_tor = np.asarray(cp["j_tor"], dtype=float)       # total toroidal (authoritative)
    j_ohmic = np.asarray(cp["j_ohmic"], dtype=float)   # parallel (unused: inductive = residual)
    j_boot = np.asarray(cp["j_bootstrap"], dtype=float)  # parallel

    def to_toroidal(j_par):
        return parallel_to_toroidal(j_par, j_parallel_total=j_total, j_tor_total=j_tor)

    j_BS = to_toroidal(j_boot)

    # --- NBI: sum beam-source parallel currents, then convert ---
    src_ids = dd.get("core_sources", {})
    isrc = _nearest_index(src_ids["time"], T, "core_sources") if src_ids.get("time") else ic
    jnbi_par = np.zeros(n)
    for s in src_ids.get("source", []):
        if s.get("identifier", {}).get("index") == NBI_SOURCE_INDEX:
            pr = s.get("profiles_1d", [])
            if pr:
                idx = isrc if len(pr) > isrc else 0
                jnbi_par = jnbi_par + np.asarray(pr[idx]["j_parallel"], dtype=float)
    j_NBI = to_toroidal(jnbi_par)
    j_RF = np.zeros(n)   # never computed internally; user-supplied only

    # --- kinetic profiles + Zeff + fast pressure ---
    el = cp["electrons"]
    ne = np.asarray(el["density_thermal"], dtype=float)
    te = np.asarray(el["temperature"], dtype=float)   # eV
    p_fast = _isotropic_fast_pressure(el, p_fast_reduction, n)

    ni = None
    ti = None
    main_ion = None
    zeff_num = np.zeros(n)
    for ion in cp["ion"]:
        Z = float(ion["element"][0]["z_n"])
        n_s = np.asarray(ion["density_thermal"], dtype=float)
        zeff_num += n_s * Z * Z
        p_fast = p_fast + _isotropic_fast_pressure(ion, p_fast_reduction, n)
        if Z == 1.0 and ni is None:        # main (hydrogenic) ion
            ni = n_s
            ti = np.asarray(ion["temperature"], dtype=float)
            main_ion = ion
    if ni is None:
        raise ValueError("no hydrogenic (Z=1) main ion found in core_profiles.ion")
    Zeff = zeff_num / ne

    # --- extra source-provided profiles for the switchboard perturbation ------
    # Read whatever this source carries (production FUSE files have rotation;
    # chi/E_r are typically absent and supplied via extra_baseline). All on the
    # core_profiles grid (== psi_N_kinetic for IMAS).
    extras = {"zeff": np.asarray(cp["zeff"], dtype=float) if "zeff" in cp else Zeff}
    if main_ion is not None and "rotation_frequency_tor" in main_ion:
        extras["omega_tor"] = np.asarray(main_ion["rotation_frequency_tor"], dtype=float)
    if "e_field" in cp and "radial" in cp["e_field"]:
        extras["e_r"] = np.asarray(cp["e_field"]["radial"], dtype=float)
    ctids = dd.get("core_transport")
    if ctids and ctids.get("model"):
        cpr = ctids["model"][0].get("profiles_1d", [])
        ict = (_nearest_index(ctids["time"], T, "core_transport")
               if ctids.get("time") else ic)
        if cpr:
            ctsl = cpr[ict if len(cpr) > ict else 0]
            ce = ctsl.get("electrons", {}).get("energy", {})
            if "d" in ce:
                extras["chi_e"] = np.asarray(ce["d"], dtype=float)
            if "d" in ctsl.get("total_ion_energy", {}):
                extras["chi_i"] = np.asarray(ctsl["total_ion_energy"]["d"], dtype=float)

    # --- user overrides for fixed additive components ---
    if fixed is not None:
        if fixed.p_fast is not None:
            p_fast = _override(fixed.p_fast, fixed.psi_N, psi_N)
        if fixed.j_NBI is not None:
            j_NBI = _override(fixed.j_NBI, fixed.psi_N, psi_N)
        if fixed.j_RF is not None:
            j_RF = _override(fixed.j_RF, fixed.psi_N, psi_N)

    # Authoritative toroidal total; inductive absorbs the residual so the
    # decomposition sums exactly and Ip is preserved.
    j_phi = j_tor.copy()
    j_inductive = j_phi - j_BS - j_NBI - j_RF

    return Baseline(
        psi_N=psi_N,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="imas",
        j_NBI=j_NBI,
        j_RF=j_RF,
        p_fast=p_fast,
        eqdsk_bytes=None,
        # The OMAS JSON is NOT an Osborne p-file; don't pass it as pfile_bytes
        # (generate_bouquet would try to parse it). Per-draw p-files are built
        # from the kinetic profiles.
        pfile_bytes=None,
        li_metrics={"ids_li_1": ids_li_1, "ids_li_3": ids_li_3},
        extras=extras,
    )
