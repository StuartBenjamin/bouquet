"""Reader for IDA integrated-data-analysis files (DIII-D, netCDF ``.cdf``).

The IDA ``.cdf`` is netCDF4 -- i.e. an HDF5 container -- so it is read with
**h5py alone** (already a bouquet dependency); no OMFIT / netCDF4 / OMFITnc
required. Datasets map directly: ``f['n_e'][:]``, ``f['n_e_err'][:]``, etc.

Returns the kinetic profiles together with their uncertainty (sigma) profiles,
on the IDA psi_N grid. Both the baseline reconstruction (profiles) and the
uncertainty envelope (sigmas) draw from this single read.

Operational DIII-D ``IDA_*.cdf`` layout (verified against IDA_204441_.cdf):
    profiles are 2-D ``(n_time, n_radial)`` with companion ``*_err`` datasets
    (direct 1-sigma); the radial grid is ``psi_n`` (n_radial,), extending past
    the separatrix to ~1.2; ``time`` is in milliseconds. Units are already SI
    (n_e in m^-3; T_e, T_12C6 in eV). There is no stored main-ion density, so
    ``ni`` is taken equal to ``ne`` (quasi-neutrality approximation), matching
    the operational IDA workflow (e.g. 204441_4400_IDA.ipynb feeds ni = ne to
    reconstruct_equilibrium / generate_bouquet).
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
    sigma_mode: str = "direct",
    sigma_method: str = "percentile",   # reserved for the ensemble layout
    sigma_ni_from_ne: bool = True,
) -> IDAProfiles:
    """Read an IDA ``.cdf`` and return profiles + sigmas at ``time``.

    Parameters
    ----------
    path : str
        Path to the IDA netCDF file.
    time : float, optional
        Time slice in seconds. Required when the file holds more than one slice;
        the nearest slice is selected.
    sigma_mode : {"direct", "ensemble"}
        ``"direct"`` reads the ``*_err`` datasets (the operational layout).
        ``"ensemble"`` (posterior-sample files) is not yet implemented.
    sigma_method : {"percentile", "std"}
        Band estimator for the ensemble layout (unused for ``"direct"``).
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

    if sigma_mode == "ensemble":
        raise NotImplementedError(
            "ensemble (posterior-sample) IDA layout not yet supported; "
            "use sigma_mode='direct' for the operational IDA_*.cdf files"
        )
    if sigma_mode != "direct":
        raise ValueError(f"unknown sigma_mode {sigma_mode!r}; expected 'direct' or 'ensemble'")

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    with h5py.File(path, "r") as f:
        def col(key):  # one radial profile at the selected time
            return np.asarray(f[key][t_idx], dtype=float)

        psi_N = np.asarray(f["psi_n"][:], dtype=float)
        time_ms = np.asarray(f["time"][:], dtype=float)
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)

        ne = col("n_e")          # m^-3
        te = col("T_e")          # eV
        ti = col("T_12C6")       # eV (carbon CER temperature)
        Zeff = col("Zeff")
        # ni ~ ne (quasi-neutrality approximation), matching the operational IDA
        # workflow which feeds ni = ne (not an impurity-diluted main-ion density)
        # to reconstruct_equilibrium / generate_bouquet.
        ni = ne.copy()

        sigma_ne = col("n_e_err")
        sigma_te = col("T_e_err")
        sigma_ti = col("T_12C6_err")
        sigma_ni = sigma_ne.copy()   # consistent with ni = ne

    return IDAProfiles(
        psi_N=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        sigma_ne=sigma_ne, sigma_te=sigma_te, sigma_ni=sigma_ni, sigma_ti=sigma_ti,
        time=t_sel,
        raw_bytes=raw_bytes,
    )
