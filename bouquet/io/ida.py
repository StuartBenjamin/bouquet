"""Reader for IDA integrated-data-analysis files (DIII-D, netCDF ``.cdf``).

The IDA ``.cdf`` is netCDF4 -- i.e. an HDF5 container -- so it is read with
**h5py alone** (already a bouquet dependency); no OMFIT / netCDF4 / OMFITnc
required. Datasets map directly: ``f['n_e'][:]``, ``f['n_e_err'][:]``, etc.

Returns the kinetic profiles together with their uncertainty (sigma) profiles,
on the IDA psi_N grid. Both the baseline reconstruction (profiles) and the
uncertainty envelope (sigmas) draw from this single read.

Operational DIII-D ``IDA_*.cdf`` layout (verified against a real IDA file):
    profiles are 2-D ``(n_time, n_radial)`` with companion ``*_err`` datasets
    (direct 1-sigma); the radial grid is ``psi_n`` (n_radial,), extending past
    the separatrix to ~1.2; ``time`` is in milliseconds. Units are already SI
    (n_e in m^-3; T_e, T_12C6 in eV). There is no stored main-ion density, but
    the file carries ``Zeff`` (visible bremsstrahlung), so ``ni`` is derived
    from ``(ne, Zeff)`` under single-impurity quasineutrality with the machine
    impurity charge ``impurity_Z`` (carbon Z=6 by default; the carbon CER
    ``T_12C6`` is itself the carbon diagnostic). This makes the IDA baseline
    information-equivalent to a p-file's ``(ne, ni)`` and self-consistent
    between its pressure and its Z_eff -- unlike the legacy ``ni = ne``, which
    silently meant Z_eff = 1 for the pressure while feeding the real Z_eff to
    the bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IDAProfiles:
    """Kinetic profiles + sigmas read from an IDA file, at one time slice.

    Values are returned in SI (ne/ni in m^-3, Te/Ti in eV) on ``psi_N``.
    ``psi_N`` typically extends past the separatrix (~1.2) into the SOL; pass it
    through to ``generate_bouquet`` as ``psi_N_kinetic`` rather than truncating.
    """

    psi_N: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    Zeff: np.ndarray

    sigma_ne: np.ndarray
    sigma_te: np.ndarray
    sigma_ni: np.ndarray
    sigma_ti: np.ndarray

    time: float                     # selected slice [s]
    raw_bytes: Optional[bytes] = None   # original file bytes for archival


@dataclass
class IDACERProfiles:
    """Impurity (carbon CER) channels for a radial-field / rotation analysis.

    Everything needed for the impurity radial force balance
    ``E_r = (dp_C/dR)/(Z_C e n_C) - v_pol B_phi + omega R B_pol`` (see
    :func:`bouquet.physics.radial_field_from_cer`), read at one time slice on
    ``psi_N``. SI units: ``n_carbon`` m^-3, ``t_carbon`` eV, ``omega_tor`` rad/s,
    ``v_pol`` m/s, ``Bpol`` T, ``Rmaj`` m, ``dpsiN_dR`` 1/m. The ``sigma_*``
    fields are the measured 1-sigma envelopes (for propagating E_r uncertainty).
    """

    psi_N: np.ndarray
    n_carbon: np.ndarray
    t_carbon: np.ndarray
    omega_tor: np.ndarray
    v_pol: np.ndarray
    Bpol: np.ndarray
    Rmaj: np.ndarray
    dpsiN_dR: np.ndarray

    sigma_n_carbon: np.ndarray
    sigma_t_carbon: np.ndarray
    sigma_omega_tor: np.ndarray
    sigma_v_pol: np.ndarray

    time: float


def _select_time_index(time_ms: np.ndarray, time_s: Optional[float]) -> int:
    """Return the index of the slice nearest ``time_s`` (seconds)."""
    if time_ms.size == 1:
        return 0
    if time_s is None:
        avail = ", ".join(f"{t/1e3:.4f}" for t in time_ms)
        raise ValueError(
            f"IDA file has {time_ms.size} time slices; pass `time` (seconds). "
            f"Available [s]: {avail}"
        )
    return int(np.argmin(np.abs(time_ms / 1e3 - time_s)))


def read_ida(
    path: str,
    time: Optional[float] = None,
    sigma_mode: str = "auto",
    sigma_method: str = "percentile",   # ensemble-layout band estimator
    sigma_ni_from_ne: bool = True,
    impurity_Z: float = 6.0,
) -> IDAProfiles:
    """Read an IDA ``.cdf`` and return profiles + sigmas at ``time``.

    Parameters
    ----------
    path : str
        Path to the IDA netCDF file.
    time : float, optional
        Time slice in seconds. Required when the file holds more than one slice;
        the nearest slice is selected.
    sigma_mode : {"auto", "direct", "ensemble"}
        ``"auto"`` (default) picks the layout from the array dimensionality.
        ``"direct"`` reads the ``*_err`` datasets (2-D operational layout);
        ``"ensemble"`` reduces a 3-D posterior-sample file (mean profile + a
        sample-spread sigma). Passing a mode that contradicts the file raises.
    sigma_method : {"percentile", "std"}
        Ensemble band estimator: ``"percentile"`` -> (p84-p16)/2 (robust),
        ``"std"`` -> sample standard deviation. Unused for the direct layout.
    sigma_ni_from_ne : bool
        Retained for API symmetry. With ``ni = ne`` the ion-density sigma always
        tracks ``sigma_ne``.

    Notes
    -----
    Opens with ``h5py.File(path, "r")`` -- the file is netCDF4/HDF5, so no
    OMFIT or netCDF4 package is needed. Units are already SI; ``T_12C6`` maps to
    Ti, and ``ni`` is reconstructed as ``n_e - Z_imp * n_12C6``.
    """
    import h5py

    if sigma_mode not in ("direct", "ensemble", "auto"):
        raise ValueError(
            f"unknown sigma_mode {sigma_mode!r}; expected 'direct', 'ensemble', or 'auto'")
    if sigma_method not in ("percentile", "std"):
        raise ValueError(
            f"unknown sigma_method {sigma_method!r}; expected 'percentile' or 'std'")

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    from ..physics import main_ion_density_from_zeff

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)

        # Two field-validated layouts, distinguished by dimensionality:
        #   direct   : (n_time, n_radial) profiles + companion *_err datasets;
        #   ensemble : (n_time, n_samples, n_radial) posterior samples, no *_err
        #              -> profile = sample mean, sigma = sample spread.
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)
        if sigma_mode == "direct" and is_ensemble:
            raise ValueError("sigma_mode='direct' but the file is a 3-D posterior "
                             "(ensemble) IDA; use sigma_mode='auto' or 'ensemble'")
        if sigma_mode == "ensemble" and not is_ensemble:
            raise ValueError("sigma_mode='ensemble' but the file is a 2-D direct "
                             "IDA; use sigma_mode='auto' or 'direct'")

        if is_ensemble:
            def _samples(key):  # (n_samples, n_radial) at the selected slice
                return np.asarray(f[key][t_idx], dtype=float)

            def _band(a):       # symmetric 1-sigma-equivalent over the sample axis
                if sigma_method == "std":
                    return np.std(a, axis=0)
                lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
                return 0.5 * (hi - lo)

            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]   # shared radial grid
            ne_s, te_s = _samples("n_e"), _samples("T_e")
            ti_s, zf_s = _samples("T_12C6"), _samples("Zeff")
            ne, te, ti, Zeff = (ne_s.mean(0), te_s.mean(0), ti_s.mean(0), zf_s.mean(0))
            sigma_ne, sigma_te, sigma_ti = _band(ne_s), _band(te_s), _band(ti_s)
        else:
            def col(key):       # one radial profile at the selected time
                return np.asarray(f[key][t_idx], dtype=float)

            psi_N = np.asarray(f["psi_n"][:], dtype=float)
            ne, te = col("n_e"), col("T_e")          # m^-3, eV
            ti, Zeff = col("T_12C6"), col("Zeff")    # eV (carbon CER), dimensionless
            sigma_ne, sigma_te, sigma_ti = col("n_e_err"), col("T_e_err"), col("T_12C6_err")

        # Main-ion density from the measured (ne, Zeff) via single-impurity
        # quasineutrality: ni = ne (Z_imp - Zeff)/(Z_imp - 1). The IDA file
        # carries Z_eff directly (visible bremsstrahlung), so the dilution is
        # measured, not assumed -- equivalent in information to a p-file's
        # (ne, ni). Z_eff is clipped to [1, Z_imp] so 0 <= ni <= ne.
        Zeff_c = np.clip(Zeff, 1.0, impurity_Z)
        ni = main_ion_density_from_zeff(ne, Zeff_c, impurity_Z)
        # sigma_ni is only used in the (now rare) independent-ni fallback;
        # propagate the ne fractional error onto the derived ni.
        with np.errstate(divide="ignore", invalid="ignore"):
            _frac = np.where(ne > 0, sigma_ne / ne, 0.0)
        sigma_ni = np.abs(ni) * _frac

    return IDAProfiles(
        psi_N=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        sigma_ne=sigma_ne, sigma_te=sigma_te, sigma_ni=sigma_ni, sigma_ti=sigma_ti,
        time=t_sel,
        raw_bytes=raw_bytes,
    )


def read_ida_cer(
    path: str,
    time: Optional[float] = None,
    sigma_method: str = "percentile",
) -> IDACERProfiles:
    """Read the carbon-CER channels needed for a radial-field (E_r) analysis.

    Returns the impurity density / temperature, toroidal + poloidal rotation, the
    midplane poloidal field and geometry (``Rmaj``, ``dpsiN_dR``), and their
    measured 1-sigma envelopes, at ``time`` on the IDA ``psi_N`` grid. Handles
    both file layouts like :func:`read_ida`: direct (2-D + ``*_err``) and ensemble
    (3-D posterior samples -> sample mean + ``sigma_method`` band). Feed the
    result to :func:`bouquet.physics.radial_field_from_cer`.
    """
    import h5py

    if sigma_method not in ("percentile", "std"):
        raise ValueError(f"unknown sigma_method {sigma_method!r}")

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)

        def _band(a):
            if sigma_method == "std":
                return np.std(a, axis=0)
            lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
            return 0.5 * (hi - lo)

        def read(key, err_key=None):
            """(value, sigma) for one channel across either layout."""
            if is_ensemble:
                s = np.asarray(f[key][t_idx], dtype=float)      # (n_samples, n_radial)
                return s.mean(0), _band(s)
            val = np.asarray(f[key][t_idx], dtype=float)
            sig = (np.asarray(f[err_key][t_idx], dtype=float)
                   if err_key and err_key in f else np.zeros_like(val))
            return val, sig

        if is_ensemble:
            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]
        else:
            psi_N = np.asarray(f["psi_n"][:], dtype=float)

        n_c, s_nc = read("n_12C6", "n_12C6_err")
        t_c, s_tc = read("T_12C6", "T_12C6_err")
        omg, s_om = read("omega_tor_12C6", "omega_tor_12C6_err")
        vpol, s_vp = read("v_pol", "v_pol_err")
        bpol, _ = read("Bpol_midplane", "Bpol_midplane_err")
        rmaj, _ = read("Rmaj_midplane", "Rmaj_midplane_err")
        dpsidr, _ = read("dPsiN_dR_midplane", "dPsiN_dR_midplane_err")

    return IDACERProfiles(
        psi_N=psi_N, n_carbon=n_c, t_carbon=t_c, omega_tor=omg, v_pol=vpol,
        Bpol=bpol, Rmaj=rmaj, dpsiN_dR=dpsidr,
        sigma_n_carbon=s_nc, sigma_t_carbon=s_tc,
        sigma_omega_tor=s_om, sigma_v_pol=s_vp,
        time=t_sel,
    )
