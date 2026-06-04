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
      3. else a fractional fallback ``fallback_frac * |profile|``.
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

    if ida_path is not None:
        from .io.ida import read_ida
        ida = read_ida(
            ida_path, time=getattr(src, "time", None),
            sigma_mode=unc.sigma_mode, sigma_method=unc.sigma_method,
            sigma_ni_from_ne=unc.sigma_ni_from_ne,
        )

        def to_kin(arr):
            return np.interp(psi_kin, ida.psi_N, np.asarray(arr, dtype=float))

        out = dict(
            sigma_ne=to_kin(ida.sigma_ne), sigma_te=to_kin(ida.sigma_te),
            sigma_ni=to_kin(ida.sigma_ni), sigma_ti=to_kin(ida.sigma_ti),
        )
    else:
        f = float(unc.fallback_frac)
        out = dict(
            sigma_ne=f * np.abs(baseline.ne), sigma_te=f * np.abs(baseline.te),
            sigma_ni=f * np.abs(baseline.ni), sigma_ti=f * np.abs(baseline.ti),
        )

    out["sigma_jphi"] = unc.sigma_jphi_frac * np.abs(np.asarray(baseline.j_phi, dtype=float))
    out["n_ls"], out["t_ls"], out["j_ls"] = unc.n_ls, unc.t_ls, unc.j_ls
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
        ida = read_ida(path, time=source.time)
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

    Ip_target = abs(float(eqdsk.Ip))
    l_i_target = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="std")["l_i"]

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
        recon=result,
    )
