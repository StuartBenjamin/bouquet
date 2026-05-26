"""
HDF5 archive helpers and eqdsk I/O utilities for perturbed equilibria.
"""

import os
import tempfile

import h5py
import numpy as np


# ====================================================================
#  Internal helpers
# ====================================================================

def safe_trace_surf(mygs, psi):
    r'''Snapshot/restore-wrapped trace_surf.

    The Fortran-side `trace_surf` has been observed to perturb subsequent
    `get_stats` / boundary measurements via mutation of state hanging off
    the `gs_equil` struct (mesh-cell caches, tracer step state, etc.).
    Pre-OFT-PR-248 this was hard to undo cleanly; with PR #248 we now have
    `mygs.copy_eq()` / `mygs.replace_eq()` which atomically swap the
    `gs_equil` pointer.  This wrapper snapshots the equilibrium before
    the trace, copies the returned LCFS into an owned numpy array, then
    restores the original equilibrium.

    Whatever state lives outside `gs_equil` (eg. module-level
    `active_tracer` in `tracing_2d`) is *not* restored — but that state
    is reset at the start of each `trace_surf` call anyway, so the only
    risk surface is mid-trace interaction with concurrent
    `get_stats`-like calls, which bouquet does not do.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ (`copy_eq` /
        `replace_eq` available).
    psi : float
        Normalized psi value to trace (eg. ``1.0 - psi_pad``).

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` array of (R, Z) points along the traced surface,
        owned by the caller (a copy, decoupled from any internal
        TokaMaker buffers).  Returns ``None`` if the trace fails.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        # legacy OFT build: fall through to bare trace_surf (no protection)
        result = mygs.trace_surf(psi)
        return None if result is None else np.asarray(result).copy()
    saved = mygs.copy_eq()
    try:
        result = mygs.trace_surf(psi)
        if result is not None:
            result = np.asarray(result).copy()
        return result
    finally:
        mygs.replace_eq(source_eq=saved)


def Ip_flux_integral_vs_target(alpha, mygs, jtor_prof, spike_profile, psi_N, Ip_target):
    r'''! Compute difference between integrated a*j_tor+j_spike profile and Ip_target

    @param alpha Scaling factor to solve for
    @param jtor_prof Input j_inductive profile
    @param spike_profile Isolated j_bootstrap spike (a Gaussian), 0.0 everywhere else
    @param my_psi_N Local psi_N grid
    @param my_Ip_target Ip target
    '''
    prof = alpha*jtor_prof + spike_profile
    Ip_computed = mygs.flux_integral(psi_N, prof)
    return Ip_computed - Ip_target

def Hmode_profiles(edge=0.08, ped=0.4, core=2.5, rgrid=201, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    r'''! This function generates H-mode density and temperature profiles evenly spaced in your favorite 
    radial coordinate. Copied from https://omfit.io/_modules/omfit_classes/utils_fusion.html

    @param edge Separatrix height (float)
    @param ped Pedestal height (float)
    @param core On-axis profile height (float)
    @param rgrid Number of radial grid points (int)
    @param expin Inner core exponent for H-mode pedestal profile (float)
    @param expout Outer core exponent for H-mode pedestal profile (float)
    @param widthp Width of pedestal (float)
    @param xphalf Position of tanh (float, optional)
    @result H-mode profile array over radial grid
    '''

    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    xpsi = np.linspace(0, 1, rgrid)
    ones = np.ones(rgrid)

    val = 0.5 * a_t * (1.0 - np.tanh((xpsi - xphalf) / w_E1) - pconst) + edge * ones

    xtoped = xpsi / xped
    for i in range(0, rgrid):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout

    return val

def _scan_val_key(scan_val):
    """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *scan_val* is ``None`` (flat layout).
    """
    if scan_val is None:
        return None
    return str(scan_val)


def _group_path(scan_val, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _scan_val_key(scan_val)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))


def _eqdsk_dataset_name(header, scan_val, count):
    """Return the dataset name used for the raw eqdsk bytes."""
    base = os.path.basename(header)
    bkey = _scan_val_key(scan_val)
    if bkey is not None:
        safe_key = bkey.replace("/", "_").replace(" ", "_")
        return f"{base}_{safe_key}_{int(count)}.eqdsk"
    return f"{base}_{int(count)}.eqdsk"


# ====================================================================
#  Database lifecycle
# ====================================================================
def initialize_equilibrium_database(header):
    """
    Create (or open) the top-level HDF5 database file on disk.

    Parameters
    ----------
    header : str
        Base name for the database.  File will be ``<header>.h5``.

    Returns
    -------
    db_path : str
        Absolute path to the HDF5 file.
    """
    db_path = os.path.abspath(f"{header}.h5")
    with h5py.File(db_path, "a"):
        pass
    return db_path


# ====================================================================
#  Per-equilibrium storage
# ====================================================================
_PROFILE_KEYS = [
    "psi_N",
    "j_phi [A m^-2]",
    "j_BS [A m^-2]",
    "j_BS,edge [A m^-2]",
    "j_inductive [A m^-2]",
    "n_e [m^-3]",
    "T_e [eV]",
    "n_i [m^-3]",
    "T_i [eV]",
    "w_ExB [rad/s]",
]


def store_equilibrium(
    header,
    count,
    eqdsk_filepath,
    psi_N,
    j_phi,
    j_BS,
    j_inductive,
    n_e,
    T_e,
    n_i,
    T_i,
    w_ExB,
    li1,
    li3,
    scan_val=None,
    pressure=None,
    j_BS_edge=None,
    pfile_bytes=None,
    Zeff=None,
    coil_currents=None,
    psi_N_kinetic=None,
    homotopy_pass=None,
    homotopy_F_lim=None,
    homotopy_VSC_lim=None,
    max_F_drift_pct=None,
    max_VSC_drift_pct=None,
    in_spec=None,
    inspec_F_max=None,
    inspec_VSC_max=None,
):
    """
    Write one perturbed equilibrium into the HDF5 database.

    Parameters
    ----------
    header : str
        Base name (same string passed to ``initialize_equilibrium_database``).
    count : int
        Perturbation index (typically 0 -- N-1).
    eqdsk_filepath : str
        Path to the ``.geqdsk`` / ``.eqdsk`` file.  Read as raw bytes so
        the Fortran-namelist formatting is preserved exactly.
    psi_N, j_phi, j_BS, j_inductive,
    n_e, T_e, n_i, T_i, w_ExB : array_like, 1-D
        Profile arrays.
    li1 : float
        Internal inductance l_i(1).
    li3 : float
        Internal inductance l_i(3).
    scan_val : str, float, int, or None
        Scan-point label.  When provided, an extra ``scan/{label}/``
        group layer is inserted.  ``None`` gives the flat layout.
    pressure : array_like or None
        1-D total pressure [Pa].
    j_BS_edge : array_like or None
        1-D isolated edge bootstrap current [A m^-2].
    pfile_bytes : bytes or None
        Raw p-file content to store alongside the g-file bytes.
    Zeff : array_like or None
        1-D effective charge profile (dimensionless).
    coil_currents : dict or None
        Coil currents {name: current_A} from TokaMaker.
    """
    db_path = os.path.abspath(f"{header}.h5")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found.  "
            f"Call initialize_equilibrium_database('{header}') first."
        )

    with open(eqdsk_filepath, "rb") as fh:
        eqdsk_bytes = fh.read()

    grp_path = _group_path(scan_val, count)
    ds_name  = _eqdsk_dataset_name(header, scan_val, count)

    with h5py.File(db_path, "a") as hf:
        # clean slate if this entry already exists
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        # ---- raw eqdsk (opaque binary -- bit-perfect) --------------------
        grp.create_dataset(ds_name, data=np.void(eqdsk_bytes))

        # ---- 1-D profiles -----------------------------------------------
        grp.create_dataset("psi_N",               data=np.asarray(psi_N,       dtype=np.float64))
        grp.create_dataset("j_phi [A m^-2]",       data=np.asarray(j_phi,       dtype=np.float64))
        grp.create_dataset("j_BS [A m^-2]",        data=np.asarray(j_BS,        dtype=np.float64))
        grp.create_dataset("j_inductive [A m^-2]", data=np.asarray(j_inductive, dtype=np.float64))

        if j_BS_edge is not None:
            grp.create_dataset("j_BS,edge [A m^-2]", data=np.asarray(j_BS_edge, dtype=np.float64))
        grp.create_dataset("n_e [m^-3]",          data=np.asarray(n_e,         dtype=np.float64))
        grp.create_dataset("T_e [eV]",            data=np.asarray(T_e,         dtype=np.float64))
        grp.create_dataset("n_i [m^-3]",          data=np.asarray(n_i,         dtype=np.float64))
        grp.create_dataset("T_i [eV]",            data=np.asarray(T_i,         dtype=np.float64))
        grp.create_dataset("w_ExB [rad/s]",       data=np.asarray(w_ExB,       dtype=np.float64))

        # ---- optional: kinetic profile grid (when different from psi_N) ----
        if psi_N_kinetic is not None:
            grp.create_dataset("psi_N_kinetic", data=np.asarray(psi_N_kinetic, dtype=np.float64))

        if pressure is not None:
            grp.create_dataset("pressure [Pa]", data=np.asarray(pressure, dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"] = float(li1)
        grp.attrs["l_i(3)"] = float(li3)
        grp.attrs["count"]  = int(count)
        if scan_val is not None:
            grp.attrs["scan_val"] = scan_val

        # ---- optional: p-file bytes ----------------------------------------
        if pfile_bytes is not None:
            pf_ds = ds_name.replace(".eqdsk", ".pfile")
            grp.create_dataset(pf_ds, data=np.void(pfile_bytes))

        # ---- optional: Zeff profile ----------------------------------------
        if Zeff is not None:
            grp.create_dataset("Zeff", data=np.asarray(Zeff, dtype=np.float64))

        # ---- optional: coil currents ---------------------------------------
        if coil_currents is not None:
            import json
            names = list(coil_currents.keys())
            values = np.array([coil_currents[n] for n in names], dtype=np.float64)
            grp.create_dataset("coil_currents [A]", data=values)
            grp.attrs["coil_names"] = json.dumps(names)

        # ---- homotopy / in-spec metadata (per-draw) -----------------------
        if homotopy_pass is not None:
            grp.attrs["homotopy_pass"] = int(homotopy_pass)
        if homotopy_F_lim is not None:
            grp.attrs["homotopy_F_lim"] = float(homotopy_F_lim)
        if homotopy_VSC_lim is not None:
            grp.attrs["homotopy_VSC_lim"] = float(homotopy_VSC_lim)
        if max_F_drift_pct is not None:
            grp.attrs["max_F_drift_pct"] = float(max_F_drift_pct)
        if max_VSC_drift_pct is not None:
            grp.attrs["max_VSC_drift_pct"] = float(max_VSC_drift_pct)
        if in_spec is not None:
            grp.attrs["in_spec"] = bool(in_spec)
        if inspec_F_max is not None:
            grp.attrs["inspec_F_max"] = float(inspec_F_max)
        if inspec_VSC_max is not None:
            grp.attrs["inspec_VSC_max"] = float(inspec_VSC_max)


def load_equilibrium(header, count, scan_val=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    scan_val : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure [Pa]"``, ``"Zeff"``,
        ``"coil_currents"``, ``"pfile_bytes"``.
    """
    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(scan_val, count)
    ds_name  = _eqdsk_dataset_name(header, scan_val, count)

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes -------------------------------------------
        eqdsk_bytes = bytes(grp[ds_name][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            out_path = os.path.join(eqdsk_out_dir, ds_name)
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

        # ---- optional: Zeff -----------------------------------------------
        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # ---- optional: p-file bytes ----------------------------------------
        pf_ds = ds_name.replace(".eqdsk", ".pfile")
        if pf_ds in grp:
            result["pfile_bytes"] = bytes(grp[pf_ds][()])

        # ---- optional: coil currents ---------------------------------------
        if "coil_currents [A]" in grp:
            import json
            values = np.array(grp["coil_currents [A]"])
            names = json.loads(grp.attrs.get("coil_names", "[]"))
            result["coil_currents"] = dict(zip(names, values))

    return result


# ====================================================================
#  Baseline (input) profile storage
# ====================================================================
def store_baseline_profiles(
    header,
    psi_N,
    ne,
    te,
    ni,
    ti,
    pressure,
    j_phi,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    Ip_target,
    l_i_target,
    scan_val=None,
    eqdsk_bytes=None,
    pfile_bytes=None,
    psi_N_kinetic=None,
    coil_currents=None,
    coil_names=None,
    recon_lcfs_ref=None,
):
    """
    Store the input (baseline) profiles and their uncertainties.

    For hierarchical layout (*scan_val* is not ``None``), these are
    stored in ``scan/{label}/_baseline/``.  For flat layout they go
    in ``_baseline/``.

    Parameters
    ----------
    eqdsk_bytes : bytes or None
        Raw baseline geqdsk file content.  Stored so that
        ``plot_geqdsk_bouquet`` can distinguish the true baseline
        from perturbed equilibria.
    pfile_bytes : bytes or None
        Raw baseline p-file content.

    This data is written once per scan-point and is required by the
    plotting GUI to be fully self-contained.
    """
    db_path = os.path.abspath(f"{header}.h5")
    bkey = _scan_val_key(scan_val)

    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    with h5py.File(db_path, "a") as hf:
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        grp.create_dataset("psi_N",              data=np.asarray(psi_N,      dtype=np.float64))
        grp.create_dataset("n_e [m^-3]",         data=np.asarray(ne,         dtype=np.float64))
        grp.create_dataset("T_e [eV]",           data=np.asarray(te,         dtype=np.float64))
        grp.create_dataset("n_i [m^-3]",         data=np.asarray(ni,         dtype=np.float64))
        grp.create_dataset("T_i [eV]",           data=np.asarray(ti,         dtype=np.float64))
        grp.create_dataset("pressure [Pa]",       data=np.asarray(pressure,   dtype=np.float64))
        grp.create_dataset("j_phi [A m^-2]",      data=np.asarray(j_phi,      dtype=np.float64))
        grp.create_dataset("sigma_ne [m^-3]",    data=np.asarray(sigma_ne,   dtype=np.float64))
        grp.create_dataset("sigma_te [eV]",      data=np.asarray(sigma_te,   dtype=np.float64))
        grp.create_dataset("sigma_ni [m^-3]",    data=np.asarray(sigma_ni,   dtype=np.float64))
        grp.create_dataset("sigma_ti [eV]",      data=np.asarray(sigma_ti,   dtype=np.float64))
        grp.create_dataset("sigma_jphi [A m^-2]", data=np.asarray(sigma_jphi, dtype=np.float64))

        if psi_N_kinetic is not None:
            grp.create_dataset("psi_N_kinetic", data=np.asarray(psi_N_kinetic, dtype=np.float64))

        grp.attrs["Ip_target"]  = float(Ip_target)
        grp.attrs["l_i_target"] = float(l_i_target)

        if eqdsk_bytes is not None:
            grp.create_dataset("baseline.eqdsk", data=np.void(eqdsk_bytes))
        if pfile_bytes is not None:
            grp.create_dataset("baseline.pfile", data=np.void(pfile_bytes))

        # ---- Recon-LCFS reference for downstream boundary-deviation
        # measurements.  Captured by the caller via mygs.trace_surf() at
        # the SAME mygs state where the baseline.eqdsk was saved, giving
        # a method-consistent reference (~10000 points) for plot_traces
        # and the per-stage bnd-diag inside generate_bouquet.  Using this
        # instead of the eqdsk's 100-pt boundary eliminates ~2-3 mm of
        # save_eqdsk sampling noise (see Probe 2 in save_eqdsk_probe.py).
        if recon_lcfs_ref is not None:
            grp.create_dataset(
                "recon_lcfs_ref",
                data=np.asarray(recon_lcfs_ref, dtype=np.float64),
            )

        # Recon's converged coil currents (the perturbation reference).
        # Saved alongside profiles so post-processors can compute
        # absolute coil drift per draw without re-running recon.
        if coil_currents is not None:
            if coil_names is not None:
                names = list(coil_names)
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            elif isinstance(coil_currents, dict):
                names = list(coil_currents.keys())
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            else:
                names = [f"coil_{i}" for i in range(len(coil_currents))]
                values = np.asarray(coil_currents, dtype=np.float64)
            grp.create_dataset("coil_currents [A]", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))


# ====================================================================
#  Introspection helpers (used by GUI and notebook API)
# ====================================================================
def discover_scan_values(h5path):
    """
    Discover all scan values in an HDF5 equilibrium database.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.

    Returns
    -------
    scan_values : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
    with h5py.File(h5path, "r") as hf:
        if "scan" not in hf:
            return None
        keys = list(hf["scan"].keys())

    # Sort numerically when all keys look like numbers, otherwise
    # fall back to lexicographic order.
    try:
        return sorted(keys, key=float)
    except (ValueError, TypeError):
        return sorted(keys)


def count_equilibria(h5path, scan_value=None):
    """
    Count the number of perturbed equilibria stored for a scan value.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.
    scan_value : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    n : int
    """
    bkey = _scan_val_key(scan_value)
    with h5py.File(h5path, "r") as hf:
        if bkey is not None:
            parent = hf[f"scan/{bkey}"]
        else:
            parent = hf
        # Count integer-named groups (skip _baseline and other metadata)
        return sum(1 for k in parent.keys()
                   if k not in ("_baseline", "scan"))


def load_baseline_profiles(h5path, scan_value=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.
    scan_value : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    bkey = _scan_val_key(scan_value)
    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Baseline group '{grp_path}' not found in {h5path}.  "
                f"Was store_baseline_profiles() called?"
            )
        grp = hf[grp_path]
        for key in grp.keys():
            result[key] = np.array(grp[key])
        for attr in grp.attrs:
            result[attr] = grp.attrs[attr]

    return result


def load_equilibrium_by_path(h5path, count, scan_value=None):
    """
    Load one perturbed equilibrium from an HDF5 file by path.

    Like :func:`load_equilibrium` but takes a file path instead of a
    header string, and uses *scan_value* instead of *baseline*.  Does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium`
    if you need those).
    """
    bkey = _scan_val_key(scan_value)
    if bkey is not None:
        grp_path = f"scan/{bkey}/{int(count)}"
    else:
        grp_path = str(int(count))

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {h5path}"
            )
        grp = hf[grp_path]

        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        if "psi_N_kinetic" in grp:
            result["psi_N_kinetic"] = np.array(grp["psi_N_kinetic"])

        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        if "coil_currents [A]" in grp:
            import json
            values = np.array(grp["coil_currents [A]"])
            names = json.loads(grp.attrs.get("coil_names", "[]"))
            result["coil_currents"] = dict(zip(names, values))

        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result


# ====================================================================
#  eqdsk byte-stream helper
# ====================================================================
def read_eqdsk_from_bytes(raw_bytes, reader_func):
    """
    Call an existing eqdsk reader that expects a filename,
    but feed it in-memory bytes instead of a file on disk.
    """
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".eqdsk",
        delete=False,
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        result = reader_func(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
