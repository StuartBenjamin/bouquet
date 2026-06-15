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

    # l_i sanity metrics (IMAS path): IDS-reported li_1/li_3 vs the
    # TokaMaker-solved li_1/li_3 from the forward-solve. l_i_target is set to
    # the TokaMaker li_1; the IDS values are kept here for comparison/plots.
    li_metrics: Optional[dict] = None

    # auxiliary source-provided profiles available to the perturbation
    # switchboard -- rotation ('omega_tor', 'e_r'), transport ('chi_e',
    # 'chi_i') and impurity ('zeff') channels --
    # on psi_N_kinetic. Populated by the reader where the source has them; the
    # user may add/override via UncertaintyConfig.aux_baselines.
    aux: Optional[dict] = None

    # curated reconstruction quality metrics (reconstruction path only): a flat
    # dict consumed by Bouquet's reconstruction summary -- Ip/l_i target-vs-
    # achieved, boundary RMS/max, q0/q95, shape, and a pass/fail verdict.
    reconstruction_metrics: Optional[dict] = None
    # full captured solver chatter from the reconstruction (when verbose=False),
    # kept available for debugging without cluttering the notebook output.
    reconstruction_log: Optional[str] = None

    def __repr__(self):
        # concise summary -- the default dataclass repr dumps every numpy array,
        # which floods a notebook when `reconstruct()`/`prepare_baseline()` is the
        # last expression in a cell.
        import numpy as np
        ng = len(np.atleast_1d(self.psi_N))
        nk = len(np.atleast_1d(self.psi_N_kinetic))
        aux = list((self.aux or {}).keys())
        return (f"Baseline(provenance={self.provenance!r}, "
                f"Ip={self.Ip_target/1e6:.3f} MA, l_i={self.l_i_target:.3f}, "
                f"grids: {ng} eq / {nk} kinetic"
                + (f", aux={aux}" if aux else "") + ")")


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
        return _resolve_reconstruction(source, config, mygs)

    raise TypeError(f"unknown baseline source type: {type(source).__name__}")


def resolve_uncertainty(config, baseline) -> dict:
    """Resolve the perturbation envelope for :func:`generate_bouquet`.

    Returns kinetic sigmas (on ``baseline.psi_N_kinetic``), ``sigma_jphi`` (a
    fractional envelope on ``|baseline.j_phi|``, on ``baseline.psi_N``), and the
    GPR correlation length scales.

    Kinetic sigma source precedence:
      1. ``UncertaintyConfig.ida_path`` if set;
      2. else the reconstruction source's own IDA ``.cdf`` (sigmas read once);
      3. else a per-channel flat fraction ``<chan>_scalar_sigma * |profile|``.
    An explicit ``UncertaintyConfig.sigma_profiles[chan]`` array overrides all.
    """
    import numpy as np
    from .config import ReconstructionSource

    unc = config.uncertainty
    src = config.source
    psi_kin = np.asarray(baseline.psi_N_kinetic, dtype=float)

    ida_path = unc.ida_path
    if ida_path is None and isinstance(src, ReconstructionSource) \
            and src.profiles_path.endswith(".cdf"):
        ida_path = src.profiles_path

    # IDA arrays (read once) available as a fallback below
    ida_sig = None
    if ida_path is not None:
        from .io.ida import read_ida
        ida = read_ida(
            ida_path, time=getattr(src, "time", None),
            sigma_mode=unc.sigma_mode, sigma_method=unc.sigma_method,
            sigma_ni_from_ne=unc.sigma_ni_from_ne,
        )

        def _to_kin(arr):
            return np.interp(psi_kin, ida.psi_N, np.asarray(arr, dtype=float))

        ida_sig = {"ne": _to_kin(ida.sigma_ne), "te": _to_kin(ida.sigma_te),
                   "ni": _to_kin(ida.sigma_ni), "ti": _to_kin(ida.sigma_ti)}

    # Per-channel resolution: explicit profile > IDA > flat scalar fraction.
    _profiles = unc.sigma_profiles or {}
    _scalars = {"ne": unc.ne_scalar_sigma, "te": unc.te_scalar_sigma,
                "ni": unc.ni_scalar_sigma, "ti": unc.ti_scalar_sigma}
    _baseprof = {"ne": baseline.ne, "te": baseline.te,
                 "ni": baseline.ni, "ti": baseline.ti}
    out = {}
    for _ch in ("ne", "te", "ni", "ti"):
        if _ch in _profiles and _profiles[_ch] is not None:
            _arr = np.asarray(_profiles[_ch], dtype=float)
            if _arr.shape != psi_kin.shape:
                raise ValueError(
                    f"sigma_profiles['{_ch}'] has shape {_arr.shape}; expected "
                    f"the kinetic grid {psi_kin.shape} (psi_N_kinetic)")
            out[f"sigma_{_ch}"] = _arr
        elif ida_sig is not None:
            out[f"sigma_{_ch}"] = ida_sig[_ch]
        else:
            out[f"sigma_{_ch}"] = float(_scalars[_ch]) * np.abs(
                np.asarray(_baseprof[_ch], dtype=float))

    out["sigma_jphi"] = unc.jphi_scalar_sigma * np.abs(np.asarray(baseline.j_phi, dtype=float))
    out["n_ls"], out["t_ls"], out["j_ls"] = unc.n_ls, unc.t_ls, unc.j_ls

    # --- switchboard: resolve the auxiliary perturbed profiles ---------------
    # A sigma entry enables a profile. Baseline = manual (aux_baselines) over
    # source-provided (baseline.aux). Warn + skip if the baseline is absent or
    # all-zero (the user supplied a sigma for nothing).
    import warnings
    src_aux = dict(baseline.aux or {})
    man_base = dict(unc.aux_baselines or {})

    # Z_eff channel is enabled by default for EVERY source (the consistent
    # density scheme): unless the user set an explicit aux_sigmas['zeff'], a
    # flat fractional envelope zeff_scalar_sigma * Z_eff_baseline is injected.
    # The baseline Z_eff is source-provided (baseline.aux['zeff'] for IMAS,
    # else baseline.Zeff for the reconstruction path), on the kinetic grid.
    user_sigmas = dict(unc.aux_sigmas or {})
    if "zeff" not in user_sigmas and float(getattr(unc, "zeff_scalar_sigma", 0.0)) > 0:
        base_zeff = src_aux.get("zeff")
        if base_zeff is None:
            base_zeff = np.asarray(baseline.Zeff, dtype=float)
        if "zeff" not in man_base:
            man_base["zeff"] = np.asarray(base_zeff, dtype=float)
        user_sigmas["zeff"] = unc.zeff_scalar_sigma * np.abs(
            np.asarray(man_base["zeff"], dtype=float))

    resolved_sigma, resolved_base = {}, {}
    for name, sig in user_sigmas.items():
        base = man_base.get(name, src_aux.get(name))
        if base is None or not np.any(np.asarray(base, dtype=float)):
            warnings.warn(
                f"aux_sigmas['{name}'] supplied but its baseline is "
                f"{'absent' if base is None else 'all-zero'} in this source; "
                f"perturbation of '{name}' is skipped. Provide it via "
                f"UncertaintyConfig.aux_baselines."
            )
            continue
        resolved_sigma[name] = np.asarray(sig, dtype=float)
        resolved_base[name] = np.asarray(base, dtype=float)
    out["aux_sigmas"] = resolved_sigma
    out["aux_baselines"] = resolved_base
    out["aux_length_scales"] = dict(unc.aux_length_scales or {})
    return out


def _resolve_fixed(comp, src_psi, dst_psi):
    """Fixed additive component onto ``dst_psi`` (zeros if not supplied)."""
    import numpy as np

    if comp is None:
        return np.zeros_like(dst_psi)
    comp = np.asarray(comp, dtype=float)
    if src_psi is None:
        if comp.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the target grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return comp
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), comp)


def _load_kinetic_profiles(source) -> dict:
    """Kinetic profiles (SI) from the reconstruction source's ``profiles_path``.

    Dispatches on file type: an IDA ``.cdf`` (ni = ne, quasi-neutrality) or an
    Osborne p-file (real ni via quasineutrality + Zeff from the ion mix). Returns
    a dict with ``psi_N`` (native kinetic grid), ``ne``/``te``/``ni``/``ti``
    (m^-3, eV), ``Zeff`` (clipped >= 1), and ``raw_bytes`` for archival.
    """
    import numpy as np

    path = source.profiles_path
    if path.endswith(".cdf"):
        from .io.ida import read_ida
        ida = read_ida(path, time=source.time, impurity_Z=source.impurity_Z)
        return dict(
            psi_N=np.asarray(ida.psi_N, dtype=float),
            ne=np.asarray(ida.ne, dtype=float),
            te=np.asarray(ida.te, dtype=float),
            ni=np.asarray(ida.ni, dtype=float),
            ti=np.asarray(ida.ti, dtype=float),
            Zeff=np.clip(np.asarray(ida.Zeff, dtype=float), 1.0, None),
            raw_bytes=ida.raw_bytes,
        )

    # Osborne p-file: ne/ni in 1e20 m^-3, Te/Ti in keV -> SI.
    from .io.pfile import read_pfile
    with open(path, "rb") as fh:
        raw = fh.read()
    pf = read_pfile(path)
    if pf.ion_species is None:
        raise ValueError(
            "p-file carries no ion species in its footer; cannot compute "
            "quasineutrality / Zeff"
        )
    pf.compute_quasineutrality()
    psi_pf, Zeff = pf.compute_zeff()
    return dict(
        psi_N=np.asarray(psi_pf, dtype=float),
        ne=np.asarray(pf.ne, dtype=float) * 1e20,
        te=np.asarray(pf.te, dtype=float) * 1e3,
        ni=np.asarray(pf.ni, dtype=float) * 1e20,
        ti=np.asarray(pf.ti, dtype=float) * 1e3,
        Zeff=np.clip(np.asarray(Zeff, dtype=float), 1.0, None),
        raw_bytes=raw,
    )


def _resolve_reconstruction(source, config, mygs) -> Baseline:
    """Reconstruction-source baseline: GS reconstruct on a live ``mygs``.

    Mirrors the operational notebook: read g-file + IDA profiles, interpolate
    onto the g-file psi_N grid, run :func:`reconstruct_equilibrium`, and package
    the (toroidal) fitted currents. The reconstructed total ``j_phi_fit`` already
    contains all driven current, so fixed components (j_NBI / j_RF) default to
    zero and only re-partition the inductive part if the user supplies them;
    ``p_fast`` (absent from thermal IDA profiles) likewise defaults to zero.
    """
    import numpy as np
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun

    from .io.geqdsk import read_geqdsk
    from .TokaMaker_interface import reconstruct_equilibrium

    if mygs is None:
        raise ValueError(
            "ReconstructionSource requires a live TokaMaker solver; call "
            "setup_solver() before prepare_baseline()"
        )
    if source.profile_overrides:
        raise NotImplementedError("profile_overrides is not yet applied")

    with open(source.geqdsk_path, "rb") as fh:
        eqdsk_bytes = fh.read()
    eqdsk = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
    psi_N = np.asarray(eqdsk.psi_N, dtype=float)

    kin = _load_kinetic_profiles(source)
    psi_N_kin = kin["psi_N"]

    # kinetic profiles (native SI) interpolated onto the equilibrium psi_N grid
    def to_eq(arr):
        return np.interp(psi_N, psi_N_kin, np.asarray(arr, dtype=float))

    ne_eq, te_eq, ni_eq, ti_eq = to_eq(kin["ne"]), to_eq(kin["te"]), to_eq(kin["ni"]), to_eq(kin["ti"])
    Zeff_eq = np.clip(to_eq(kin["Zeff"]), 1.0, None)

    # isoflux from the g-file boundary (matches the recon-stage notebook weight)
    iso_pts = np.column_stack([eqdsk.boundary_R, eqdsk.boundary_Z])
    iso_w = np.ones(len(iso_pts)) * 200.0
    mygs.set_isoflux(iso_pts, weights=iso_w)

    guess_jinductive = create_power_flux_fun(len(psi_N), 1.5, 1.5)["y"]

    # Capture the verbose solver chatter (DLSODE / gs_get_qprof / li-match) unless
    # the user asked for it; the curated summary is printed by Bouquet.reconstruct.
    from .utils import capture_native_output
    verbose = bool(getattr(config, "verbose", False))
    with capture_native_output(enabled=not verbose) as _cap:
        result = reconstruct_equilibrium(
            mygs, eqdsk,
            ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
            iso_pts, iso_w, source.psi_pad,
            guess_jinductive=guess_jinductive,
            n_k=source.n_k,
            psi_bridge=source.psi_bridge,
            rescale_j_BS=source.rescale_j_BS,
            shelf_psi_N=source.shelf_psi_N,
            initialize_psi=True,
        )
        # get_stats traces the q-profile and can emit gs_get_qprof warnings, so
        # keep these inside the capture too.
        Ip_target = abs(float(eqdsk.Ip))
        l_i_target = mygs.get_stats(
            lcfs_pad=source.psi_pad, li_normalization="std")["l_i"]
        recon_metrics = _reconstruction_metrics(
            mygs, eqdsk, result, source, l_i_target)

    j_phi = np.asarray(result["j_phi_fit"], dtype=float)
    j_BS = np.asarray(result["j_BS_used"], dtype=float)

    fc = config.fixed_components
    j_NBI = _resolve_fixed(fc.j_NBI, fc.psi_N, psi_N)
    j_RF = _resolve_fixed(fc.j_RF, fc.psi_N, psi_N)
    j_inductive = j_phi - j_BS - j_NBI - j_RF   # == j_inductive_fit when NBI=RF=0

    p_fast = _resolve_fixed(fc.p_fast, fc.psi_N, psi_N_kin)

    return Baseline(
        psi_N=psi_N,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=psi_N_kin,
        ne=kin["ne"],
        te=kin["te"],
        ni=kin["ni"],
        ti=kin["ti"],
        Zeff=kin["Zeff"],
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="reconstruction",
        j_NBI=j_NBI,
        j_RF=j_RF,
        p_fast=p_fast,
        eqdsk_bytes=eqdsk_bytes,
        pfile_bytes=kin["raw_bytes"],
        # expose the baseline Z_eff (kinetic grid) as a switchboard channel,
        # mirroring the IMAS path -- so the zeff channel resolves its baseline
        # for both the default auto-injection and an explicit aux_sigmas['zeff']
        aux={"zeff": np.asarray(kin["Zeff"], dtype=float)},
        recon=result,
        reconstruction_metrics=recon_metrics,
        reconstruction_log=_cap["text"] or None,
    )


def _reconstruction_metrics(mygs, eqdsk, result, source, l_i_achieved) -> dict:
    """Curate a TokaMaker-vs-EFIT reconstruction-fidelity dict for the summary.

    Each global scalar that isn't ~0 by construction (Ip, l_i, q0/q95, beta,
    shape, separatrix current, stored energy) is reported as a % error against
    the g-file's own EFIT value. Geometric residuals that *should* be ~0
    (boundary match, axis offset, j_phi profile RMS) stay absolute. A simple
    pass/fail verdict is applied to the reconstruction targets.
    """
    import numpy as np

    def pct(tok, ref):
        ref = float(ref)
        if not np.isfinite(ref) or abs(ref) < 1e-12:
            return float("nan")
        return float(100.0 * (float(tok) - ref) / abs(ref))

    q = dict(result.get("quality") or {})
    stats = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="std")

    # --- TokaMaker (reconstructed) values ---
    Ip_tok = float(result.get("Ip_tokamaker", float("nan")))
    li_tok = float(l_i_achieved)
    q0_tok = float(stats.get("q_0", float("nan")))
    q95_tok = float(stats.get("q_95", float("nan")))
    betan_tok = float(stats.get("beta_n", float("nan")))
    betap_tok = float(stats.get("beta_pol", float("nan"))) / 100.0  # x100 -> standard
    kappa_tok = float(stats.get("kappa", float("nan")))
    delta_tok = float(stats.get("delta", float("nan")))
    W_tok = float(stats.get("W_MHD", float("nan"))) / 1e6           # MJ
    o_point = getattr(mygs, "o_point", [float("nan"), float("nan")])
    # separatrix current evaluated just inside the LCFS (psi_N=0.99), where the
    # edge current is better-defined than the near-singular psi_N=1 point.
    _psiN_eq = np.asarray(eqdsk.psi_N, dtype=float)
    _jsep_psiN = 0.99
    jphi = np.asarray(result.get("j_phi_fit"), dtype=float)
    jsep_tok = float(np.interp(_jsep_psiN, _psiN_eq, jphi)) / 1e6   # MA/m^2

    # --- EFIT (g-file) reference values; tolerate a tracing failure ---
    efit = {}
    try:
        efit["Ip"] = abs(float(eqdsk.Ip))
        efit["li"] = float(eqdsk.li.get("li(1)", float("nan")))
        qpsi = np.asarray(eqdsk.qpsi, dtype=float)
        psiN = np.asarray(eqdsk.psi_N, dtype=float)
        efit["q0"] = float(qpsi[0])
        efit["q95"] = float(np.interp(0.95, psiN, qpsi))
        betas = eqdsk.betas
        efit["beta_n"] = float(betas.get("beta_n", float("nan")))
        efit["beta_p"] = float(betas.get("beta_p", float("nan")))
        geo = eqdsk.geometry
        efit["kappa"] = float(np.asarray(geo["kappa"])[-1])
        efit["delta"] = float(np.asarray(geo["delta"])[-1])
        efit["R_mag"] = float(eqdsk.R_mag)
        efit["Z_mag"] = float(eqdsk.Z_mag)
        efit["W_MHD"] = 1.5 * float(eqdsk.volume_integral(eqdsk.pres)[-1]) / 1e6
        jtor_efit = np.asarray(result.get("eqdsk_jtor"), dtype=float)
        efit["j_sep"] = float(np.interp(_jsep_psiN, _psiN_eq, jtor_efit)) / 1e6
    except Exception as exc:  # tracing/format issue -> % errors fall back to nan
        import warnings
        warnings.warn(f"EFIT reference metrics unavailable: {exc}")

    g = lambda k: float(efit.get(k, float("nan")))

    bnd_rms = float(q.get("boundary_rms_mm", float("nan")))
    bnd_max = float(q.get("boundary_max_dev_mm", float("nan")))
    axis_off_mm = float(np.hypot(o_point[0] - g("R_mag"),
                                 o_point[1] - g("Z_mag")) * 1e3)

    # Convergence of the final accepted GS state. reconstruct_equilibrium's
    # li-match only ever keeps successful solves (a failed solve restores the
    # last good psi and is retried), and an unrecoverable solver failure
    # raises before reaching here -- so the honest residual check is that the
    # final state produced finite global quantities.
    converged = bool(np.isfinite(Ip_tok) and np.isfinite(li_tok)
                     and np.isfinite(q95_tok))

    # verdict on the reconstruction *targets*: converged, Ip within 0.5%,
    # boundary RMS < 5 mm, l_i within 5%
    ip_err = pct(Ip_tok, g("Ip"))
    li_err = pct(li_tok, g("li"))
    ok = bool(
        converged
        and np.isfinite(ip_err) and abs(ip_err) < 0.5
        and np.isfinite(bnd_rms) and bnd_rms < 5.0
        and np.isfinite(li_err) and abs(li_err) < 5.0
    )
    return {
        "converged": converged,
        "verdict": "PASS" if ok else "CHECK",
        # fidelity: TokaMaker value, EFIT reference, % error
        "Ip_MA": Ip_tok / 1e6, "Ip_efit_MA": g("Ip") / 1e6, "Ip_err_pct": ip_err,
        "li": li_tok, "li_efit": g("li"), "li_err_pct": li_err,
        "q0": q0_tok, "q0_efit": g("q0"), "q0_err_pct": pct(q0_tok, g("q0")),
        "q95": q95_tok, "q95_efit": g("q95"), "q95_err_pct": pct(q95_tok, g("q95")),
        "beta_n": betan_tok, "beta_n_efit": g("beta_n"),
        "beta_n_err_pct": pct(betan_tok, g("beta_n")),
        "beta_p": betap_tok, "beta_p_efit": g("beta_p"),
        "beta_p_err_pct": pct(betap_tok, g("beta_p")),
        "kappa": kappa_tok, "kappa_efit": g("kappa"),
        "kappa_err_pct": pct(kappa_tok, g("kappa")),
        "delta": delta_tok, "delta_efit": g("delta"),
        "delta_err_pct": pct(delta_tok, g("delta")),
        "j_sep_MA": jsep_tok, "j_sep_efit_MA": g("j_sep"),
        "j_sep_err_pct": pct(jsep_tok, g("j_sep")),
        "W_MHD_MJ": W_tok, "W_MHD_efit_MJ": g("W_MHD"),
        "W_MHD_err_pct": pct(W_tok, g("W_MHD")),
        # zero-ideal residuals (absolute)
        "boundary_rms_mm": bnd_rms, "boundary_max_mm": bnd_max,
        "axis_offset_mm": axis_off_mm,
        "jphi_core_rms_MA": float(q.get("jphi_core_rms", float("nan"))) / 1e6,
        "jphi_edge_rms_MA": float(q.get("jphi_edge_rms", float("nan"))) / 1e6,
    }
