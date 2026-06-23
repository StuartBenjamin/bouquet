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

    # --- auxiliary source-provided profiles for the switchboard ---------------
    # Read whatever this source carries (production FUSE files have rotation;
    # chi/E_r are typically absent and supplied via aux_baselines). All on the
    # core_profiles grid (== psi_N_kinetic for IMAS).
    aux = {"zeff": np.asarray(cp["zeff"], dtype=float) if "zeff" in cp else Zeff}
    if main_ion is not None and "rotation_frequency_tor" in main_ion:
        aux["omega_tor"] = np.asarray(main_ion["rotation_frequency_tor"], dtype=float)
    if "e_field" in cp and "radial" in cp["e_field"]:
        aux["e_r"] = np.asarray(cp["e_field"]["radial"], dtype=float)
    ctids = dd.get("core_transport")
    if ctids and ctids.get("model"):
        cpr = ctids["model"][0].get("profiles_1d", [])
        ict = (_nearest_index(ctids["time"], T, "core_transport")
               if ctids.get("time") else ic)
        if cpr:
            ctsl = cpr[ict if len(cpr) > ict else 0]
            ce = ctsl.get("electrons", {}).get("energy", {})
            if "d" in ce:
                aux["chi_e"] = np.asarray(ce["d"], dtype=float)
            if "d" in ctsl.get("total_ion_energy", {}):
                aux["chi_i"] = np.asarray(ctsl["total_ion_energy"]["d"], dtype=float)

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
        aux=aux,
    )


# ===========================================================================
#  Perturbed-draw IMAS/OMAS write-back
#
#  TODO(backend): the FINAL implementation should obtain the equilibrium IDS
#  DIRECTLY from the live TokaMaker equilibrium object at generate time (full
#  finite-element fidelity, exact FSA metrics for the toroidal<->parallel
#  current conversion), once OFT exposes an IMAS/ODS export. The eqdsk-based
#  reconstruction below is an INTERIM bridge: the equilibrium IDS is lossless
#  to the archived 257^2 eqdsk (machine-precision GS for stability codes) and
#  j_tor is exact, but the parallel-current split (j_ohmic/j_bootstrap/j_total)
#  uses the baseline IDS factor c(psi)=j_tor/j_total -- exact only when the
#  draw's flux geometry matches the baseline. Do NOT treat this as final.
# ===========================================================================
def write_imas_draw(h5path_or_header, draw_index, template_ids_path, out_path,
                    scan_key=None, time=None):
    """Reconstruct a perturbed IMAS/OMAS IDS for one draw from the bouquet HDF5.

    INTERIM (see module TODO): maps the draw's archived eqdsk to the
    ``equilibrium`` IDS (``profiles_1d`` / ``profiles_2d`` / ``global_quantities``
    / ``boundary`` -- lossless to the eqdsk grid, machine-precision GS) and the
    draw's ``.h5`` kinetics/currents to ``core_profiles``. ``j_tor`` is exact;
    the parallel split is reconstructed with the baseline ratio (interim).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.
    draw_index : int
        Per-draw index within the scan group.
    template_ids_path : str
        The source IMAS/OMAS JSON, used as the structural template.
    out_path : str
        Where to write the perturbed IDS JSON.
    scan_key : str, float, or None
        Scan value selecting the ``scan/<scan_key>`` group.  The default
        ``None`` auto-resolves a single-scan archive (and is the flat layout
        for flat files); pass the explicit value only when the archive holds
        more than one scan.
    time : float, optional
        Time slice [s]; defaults to the template's nearest single slice.

    Returns
    -------
    str
        ``out_path``.
    """
    import json
    import copy
    import h5py
    from .geqdsk import read_geqdsk
    from ..utils import read_eqdsk_from_bytes, _group_path, _resolve_h5

    with open(template_ids_path) as fh:
        out = json.load(fh)

    eq_ids = out["equilibrium"]
    ie = _nearest_index(eq_ids["time"], time, "equilibrium")
    cp_ids = out["core_profiles"]
    ic = _nearest_index(cp_ids["time"], time, "core_profiles")

    h5 = _resolve_h5(h5path_or_header)
    if scan_key is None:
        # Convenience: a single-scan archive resolves unambiguously, so the
        # caller need not echo the generation scan_key.  Flat-layout files
        # (discover_scan_keys -> None) keep scan_key=None.
        from ..utils import discover_scan_keys
        keys = discover_scan_keys(h5)
        if keys:
            if len(keys) != 1:
                raise ValueError(
                    f"{h5} holds {len(keys)} scans {keys}; pass an explicit "
                    "scan_key to write_imas_draw().")
            scan_key = keys[0]
    gp = _group_path(scan_key, draw_index)
    with h5py.File(h5, "r") as hf:
        if gp not in hf:
            raise KeyError(f"draw {draw_index} (scan {scan_key}) not in {h5}")
        g = hf[gp]
        ne = np.asarray(g["n_e [m^-3]"][()]); te = np.asarray(g["T_e [eV]"][()])
        ni = np.asarray(g["n_i [m^-3]"][()]); ti = np.asarray(g["T_i [eV]"][()])
        pkin = np.asarray(g["psi_N_kinetic"][()]); peq = np.asarray(g["psi_N"][()])
        j_tor = np.asarray(g["j_phi [A m^-2]"][()])
        j_ind = np.asarray(g["j_inductive [A m^-2]"][()])
        j_bs = np.asarray(g["j_BS [A m^-2]"][()])
        zeff = np.asarray(g["aux_zeff"][()]) if "aux_zeff" in g else None
        li1 = float(g.attrs.get("l_i(1)", np.nan))
        li3 = float(g.attrs.get("l_i(3)", np.nan))
        eqk = [k for k in g.keys() if k.endswith(".eqdsk")]
        if not eqk:
            raise KeyError(f"draw {draw_index} has no archived eqdsk")
        geq = read_eqdsk_from_bytes(bytes(g[eqk[0]][()]), read_geqdsk)

    # --- equilibrium IDS from the eqdsk (lossless to the eqdsk grid) ---------
    ts = eq_ids["time_slice"][ie]
    psi1d = geq.psi_axis + geq.psi_N * (geq.psi_boundary - geq.psi_axis)
    q95 = float(np.interp(0.95, geq.psi_N, geq.qpsi))
    ts["profiles_1d"] = {
        "psi": psi1d.tolist(),
        "q": geq.qpsi.tolist(),
        "pressure": geq.pres.tolist(),
        "f": geq.fpol.tolist(),
        "dpressure_dpsi": geq.pprime.tolist(),
        "f_df_dpsi": geq.ffprim.tolist(),
    }
    ts["profiles_2d"] = [{
        "grid_type": {"name": "rectangular", "index": 1},
        "grid": {"dim1": geq.R_grid.tolist(), "dim2": geq.Z_grid.tolist()},
        # psi_RZ is indexed [R][Z], matching IMAS dim1=R, dim2=Z
        "psi": geq.psi_RZ.tolist(),
    }]
    gq = dict(ts.get("global_quantities", {}))
    gq.update(
        ip=float(geq.Ip), psi_axis=float(geq.psi_axis),
        psi_boundary=float(geq.psi_boundary),
        magnetic_axis={"r": float(geq.R_mag), "z": float(geq.Z_mag)},
        q_axis=float(geq.qpsi[0]), q_95=q95,
        li_3=li3 if np.isfinite(li3) else geq.li.get("li(3)"),
        beta_normal=geq.betas.get("beta_n"), beta_pol=geq.betas.get("beta_p"),
        beta_tor=geq.betas.get("beta_t"),
    )
    if np.isfinite(li1):
        gq["li_1"] = li1
    ts["global_quantities"] = gq
    ts["boundary"] = {"outline": {"r": geq.boundary_R.tolist(),
                                  "z": geq.boundary_Z.tolist()}}

    # --- core_profiles from the draw (kinetics + currents) ------------------
    cp = cp_ids["profiles_1d"][ic]
    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])

    def to_t(arr, src):     # interp draw array (on src grid) -> template psi grid
        return np.interp(psiN_t, src, arr)

    cp["electrons"]["density_thermal"] = to_t(ne, pkin).tolist()
    cp["electrons"]["temperature"] = to_t(te, pkin).tolist()
    for ion in cp["ion"]:
        if float(ion["element"][0]["z_n"]) == 1.0:
            ion["density_thermal"] = to_t(ni, pkin).tolist()
            ion["temperature"] = to_t(ti, pkin).tolist()
            break
    if zeff is not None:
        cp["zeff"] = to_t(zeff, pkin).tolist()

    # j_tor exact; parallel split via the baseline factor c=j_tor/j_total
    # (INTERIM -- see module TODO). Guard near-axis where j_total -> 0.
    jt_t = to_t(j_tor, peq)
    cp["j_tor"] = jt_t.tolist()
    if "j_total" in cp and "j_tor" in cp:
        base_jtot = np.asarray(cp["j_total"], dtype=float)
        base_jtor = np.asarray(cp["j_tor"], dtype=float)
        eps = 1e-9 * np.nanmax(np.abs(base_jtot)) if base_jtot.size else 0.0
        good = np.abs(base_jtot) > eps
        c = np.ones_like(base_jtot)
        c[good] = base_jtor[good] / base_jtot[good]
        if not np.all(good):
            idx = np.arange(c.size)
            c[~good] = np.interp(idx[~good], idx[good], c[good])
        with np.errstate(divide="ignore", invalid="ignore"):
            cp["j_total"] = (jt_t / c).tolist()
            cp["j_ohmic"] = (to_t(j_ind, peq) / c).tolist()
            cp["j_bootstrap"] = (to_t(j_bs, peq) / c).tolist()

    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out_path


def export_imas_drawset(h5path_or_header, template_ids_path, out_dir,
                        scan_key=None, time=None, selection="selected"):
    """Write one perturbed IMAS/OMAS IDS per draw (see :func:`write_imas_draw`).

    INTERIM (see module TODO). Files are ``{out_dir}/{header}_draw{idx}.json``.
    ``selection`` is ``"selected"`` (in-spec only) or ``"all"``.

    Operates on a single scan.  Pass ``scan_key`` to select it; the default
    ``scan_key=None`` is the flat layout, and is only unambiguous when the
    file holds exactly one scan (otherwise raises -- pass an explicit key).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.

    Returns
    -------
    list of str
        The written IDS paths.
    """
    import os
    from ..filtering import select_indices
    from ..utils import _resolve_h5

    os.makedirs(out_dir, exist_ok=True)
    h5 = _resolve_h5(h5path_or_header)
    base = os.path.basename(h5).replace(".h5", "")
    idxs = select_indices(h5, scan_key=scan_key, selection=selection)
    if isinstance(idxs, dict):
        # scan_key=None over a scan-layout file -> ambiguous unless single.
        if len(idxs) != 1:
            raise ValueError(
                f"{h5} holds {len(idxs)} scans {sorted(idxs)}; pass an "
                "explicit scan_key to export_imas_drawset().")
        scan_key, idxs = next(iter(idxs.items()))
    paths = []
    for i in idxs:
        out = os.path.join(out_dir, f"{base}_draw{i}.json")
        paths.append(write_imas_draw(h5, i, template_ids_path, out,
                                     scan_key=scan_key, time=time))
    return paths
