"""Reader for IDA integrated-data-analysis files (DIII-D, netCDF ``.cdf``).

The IDA ``.cdf`` is netCDF4 -- i.e. an HDF5 container -- so it is read with
**h5py alone** (already a bouquet dependency); no OMFIT / netCDF4 / OMFITnc
required. Datasets map directly: ``f['n_e'][:]``, ``f['n_e_err'][:]``, etc.

Returns the kinetic profiles together with their uncertainty (sigma) profiles,
on the IDA psi_N grid. Both the baseline reconstruction (profiles) and the
uncertainty envelope (sigmas) draw from this single read.

IDA dataset keys (DIII-D):
    n_e, T_e, T_12C6 (Ti), psi_n, n_12C6, Zeff, omega_tor_12C6
    n_e_err, T_e_err, T_12C6_err          (direct 1-sigma)
or a per-time posterior of shape (n_samples, n_radial) reduced to a band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class IDAProfiles:
    """Kinetic profiles + sigmas read from an IDA file, at one time slice.

    Values are returned in SI (ne/ni in m^-3, Te/Ti in eV) on ``psi_N``.
    ``psi_N`` typically extends past the separatrix (~1.2) into the SOL; pass it
    through to ``generate_bouquet`` as ``psi_N_kinetic`` rather than truncating.
    """

    psi_N: "np.ndarray"
    ne: "np.ndarray"
    te: "np.ndarray"
    ni: "np.ndarray"
    ti: "np.ndarray"
    Zeff: "np.ndarray"

    sigma_ne: "np.ndarray"
    sigma_te: "np.ndarray"
    sigma_ni: "np.ndarray"
    sigma_ti: "np.ndarray"

    time: float                     # selected slice [s]
    raw_bytes: Optional[bytes] = None   # original file bytes for archival


def read_ida(
    path: str,
    time: Optional[float] = None,
    sigma_mode: str = "ensemble",       # "ensemble" | "direct"
    sigma_method: str = "percentile",   # "percentile" | "std"  (ensemble only)
    sigma_ni_from_ne: bool = True,
) -> IDAProfiles:
    """Read an IDA ``.cdf`` and return profiles + sigmas at ``time``.

    Parameters
    ----------
    path : str
        Path to the IDA netCDF file.
    time : float, optional
        Time slice in seconds. ``None`` selects the single/first slice.
    sigma_mode : {"ensemble", "direct"}
        ``"ensemble"`` reduces the (n_samples, n_radial) posterior to a band;
        ``"direct"`` reads the ``*_err`` datasets.
    sigma_method : {"percentile", "std"}
        Band estimator for ``sigma_mode="ensemble"``: ``"percentile"`` ->
        (p84 - p16)/2, ``"std"`` -> sample standard deviation.
    sigma_ni_from_ne : bool
        If True, set ``sigma_ni = sigma_ne`` (quasi-neutrality).

    Notes
    -----
    Opens with ``h5py.File(path, "r")`` -- the file is netCDF4/HDF5, so no
    OMFIT or netCDF4 package is needed. Applies unit conversions to SI and the
    ``T_12C6`` -> Ti mapping. (netCDF4 dimension-scale datasets, if present,
    are ignored; the named variables are read as plain datasets.)
    """
    raise NotImplementedError
