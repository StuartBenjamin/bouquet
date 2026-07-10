"""
HDF5 archive helpers and eqdsk I/O utilities for perturbed equilibria.
"""

import contextlib
import os
import sys
import tempfile

import h5py
import numpy as np

from .schema import write_profile


@contextlib.contextmanager
def capture_native_output(enabled=True):
    """Capture *all* stdout/stderr — both Python-level and OS-level.

    Two layers are needed to fully silence a TokaMaker solve:

    * :func:`contextlib.redirect_stdout`/``redirect_stderr`` catch Python
      ``print()`` (the homotopy / boundary / li-match diagnostics), which under
      a Jupyter kernel go straight to the cell via ``sys.stdout``, bypassing the
      file descriptors.
    * an ``os.dup2`` file-descriptor redirect catches output from the compiled
      extension (the Fortran/C solver: ``DLSODE``, ``gs_get_qprof`` warnings)
      that bypasses Python's ``sys.stdout``.

    Yields a dict whose ``"text"`` key holds the combined captured output once
    the block exits (also populated if the block raises). ``enabled=False`` is a
    no-op pass-through, so callers can gate on a ``verbose`` flag.
    """
    import io

    holder = {"text": ""}
    if not enabled:
        yield holder
        return
    sys.stdout.flush()
    sys.stderr.flush()
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    py_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(py_buf), contextlib.redirect_stderr(py_buf):
            yield holder
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        holder["text"] = py_buf.getvalue() + tmp.read().decode("utf-8", "replace")
        tmp.close()


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


def safe_save_eqdsk(mygs, filename, **kwargs):
    r'''Snapshot/restore-wrapped save_eqdsk.

    `mygs.save_eqdsk` internally re-runs the q-profile tracer (which
    sets `active_tracer` state and may mutate cached `<R>` / `<1/R>`
    geometry on the `gs_equil` struct) and has been empirically
    observed to shift `mygs.get_globals()[0]` (Ip integral) by ~0.5-
    0.8% when called against a converged equilibrium.  This wrapper
    snapshots the equilibrium via `mygs.copy_eq()` before the save,
    then restores via `mygs.replace_eq()` after -- so the .geqdsk
    file is written from the unmodified state AND downstream
    diagnostics see the same state recon converged to.

    On legacy OFT builds without `copy_eq` / `replace_eq` (pre-PR
    #248), falls through to a bare `mygs.save_eqdsk(...)` with no
    protection.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ for the
        protected snapshot/restore path.
    filename : str
        Same as `mygs.save_eqdsk` filename arg.
    **kwargs
        Passed through to `mygs.save_eqdsk(...)`.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        return mygs.save_eqdsk(filename, **kwargs)
    saved = mygs.copy_eq()
    try:
        return mygs.save_eqdsk(filename, **kwargs)
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

def _scan_key(scan_key):
    """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *scan_key* is ``None`` (flat layout).
    """
    if scan_key is None:
        return None
    return str(scan_key)


def _resolve_h5(h5path_or_header):
    """Resolve an archive reference to an absolute ``.h5`` path.

    Accepts (duck-typed, so no import cycle):
      * a ``BouquetArchive`` (uses its ``.path``);
      * a ``Bouquet`` (uses ``config.output_header``);
      * a full path ending in ``.h5`` (returned unchanged);
      * a bare *header* stem (``<header>.h5`` resolved to absolute).

    The single place archive references are normalized, so every reader accepts
    a run object, an archive, a header, or a path interchangeably (F1).
    """
    p = getattr(h5path_or_header, "path", None)          # BouquetArchive
    if isinstance(p, str) and p.endswith(".h5"):
        return p
    cfg = getattr(h5path_or_header, "config", None)      # Bouquet
    ref = getattr(cfg, "output_header", None) if cfg is not None else h5path_or_header
    s = str(ref)
    return s if s.endswith(".h5") else os.path.abspath(f"{s}.h5")


def _default_scan_key(ref, scan_key):
    """Fill a missing ``scan_key`` from a ``Bouquet`` reference's config."""
    if scan_key is None:
        cfg = getattr(ref, "config", None)
        if cfg is not None:
            return getattr(cfg.generation, "scan_key", None)
    return scan_key


def _group_path(scan_key, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _scan_key(scan_key)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))


# ====================================================================
#  Database lifecycle
# ====================================================================
def initialize_equilibrium_database(header):
    """
    Create (or open) the top-level HDF5 database file on disk.

    Stamps ``schema_version`` / ``bouquet_version`` / ``created`` at creation
    (the single chokepoint every archive passes through), so the ABSENCE of
    ``schema_version`` reliably identifies a pre-v2 legacy file to readers.
    ``config_json`` provenance is added separately by :func:`write_provenance`.

    Parameters
    ----------
    header : str
        Base name for the database.  File will be ``<header>.h5``.

    Returns
    -------
    db_path : str
        Absolute path to the HDF5 file.
    """
    import datetime
    from . import __version__
    db_path = os.path.abspath(f"{header}.h5")
    with h5py.File(db_path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = datetime.datetime.now().isoformat(
                timespec="seconds")
    return db_path


# HDF5 archive schema version -- imported from the schema module (now v2:
# bare dataset names + units attrs, fixed eqdsk/pfile names, unified coils).
from .schema import SCHEMA_VERSION


def write_provenance(h5path_or_header, config=None, scan_key=None):
    """Stamp provenance onto an archive: schema/version/timestamp + config JSON.

    File-level attrs ``schema_version``, ``bouquet_version``, ``created`` (set
    once) and ``updated``; and, when a :class:`~bouquet.BouquetConfig` is given,
    a ``config_json`` dataset at the file root mirrored into the
    ``scan/<key>/`` group when ``scan_key`` is provided (each slice can carry its
    own config). The per-scan copies are AUTHORITATIVE; the root copy is only
    the most recent write and is overwritten by each slice of a multi-slice
    sweep -- :func:`load_config` therefore disambiguates by scan and never
    silently serves a stale root copy. Called by ``Bouquet.generate`` /
    ``run_shard`` / ``merge_archives``; safe to call repeatedly.
    """
    import datetime
    from . import __version__
    path = _resolve_h5(h5path_or_header)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with h5py.File(path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = now
        hf.attrs["updated"] = now
        if config is not None:
            cj = config.to_json()
            targets = [hf]
            bkey = _scan_key(scan_key)
            if bkey is not None:
                targets.append(hf.require_group(f"scan/{bkey}"))
            for grp in targets:
                if "config_json" in grp:
                    del grp["config_json"]
                grp.create_dataset("config_json", data=cj)


def load_config(h5path_or_header, scan_key=None):
    """Reconstruct the :class:`~bouquet.BouquetConfig` stored in an archive.

    The per-scan ``config_json`` copies are authoritative (the root copy is
    just the most recent write). With an explicit ``scan_key``, that scan's
    copy is read. With ``scan_key=None``: a single stored per-scan config is
    served directly, but a MULTI-scan archive raises and lists the keys --
    each slice of a sweep carried its own config, so "the" config is
    ambiguous and the (last-writer-wins) root copy would silently describe
    only the final slice. Raises a clear error on pre-provenance archives.
    """
    from .config import BouquetConfig
    path = _resolve_h5(h5path_or_header)
    with h5py.File(path, "r") as hf:
        node = None
        bkey = _scan_key(scan_key)
        scan_cfgs = ([k for k in hf["scan"] if f"scan/{k}/config_json" in hf]
                     if "scan" in hf else [])
        if bkey is not None:
            if f"scan/{bkey}/config_json" in hf:
                node = hf[f"scan/{bkey}/config_json"]
        elif len(scan_cfgs) == 1:
            node = hf[f"scan/{scan_cfgs[0]}/config_json"]
        elif len(scan_cfgs) > 1:
            raise KeyError(
                f"{path} holds per-scan configs for {len(scan_cfgs)} scan keys "
                f"({scan_cfgs}); pass scan_key=<one of these> -- each slice of "
                "a sweep carries its own config, so there is no single file "
                "config to return.")
        elif "config_json" in hf:
            node = hf["config_json"]
        if node is None:
            raise KeyError(
                f"{path} has no stored config for scan_key={scan_key!r} "
                f"(schema_version={hf.attrs.get('schema_version', 'absent')}; "
                f"per-scan configs present: {scan_cfgs or 'none'}). Only files "
                "written by a provenance-aware Bouquet carry a config.")
        raw = node[()]
    return BouquetConfig.from_json(raw.decode() if isinstance(raw, bytes) else str(raw))


# ====================================================================
#  Per-equilibrium storage
# ====================================================================
_PROFILE_KEYS = [
    "psi_N",
    "j_phi",
    "j_BS",
    "j_BS,edge",
    "j_inductive",
    "n_e",
    "T_e",
    "n_i",
    "T_i",
    "w_ExB",
]


def _read_coil_names(grp):
    """Coil-name list from the ``coil_names`` string dataset (schema v2; F15)."""
    if "coil_names" not in grp:
        return []
    return [n.decode() if isinstance(n, bytes) else str(n)
            for n in grp["coil_names"][()]]


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
    scan_key=None,
    pressure=None,
    pressure_thermal=None,
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
    perturbed_lcfs_ref=None,
    l_i_target_used=None,
    l_i_uncertainty=None,
    x_points=None,
    diverted=None,
    aux=None,
    eq_fsa=None,
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
    scan_key : str, float, int, or None
        Scan-point label.  When provided, an extra ``scan/{label}/``
        group layer is inserted.  ``None`` gives the flat layout.
    pressure : array_like or None
        1-D total pressure.
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

    grp_path = _group_path(scan_key, count)

    with h5py.File(db_path, "a") as hf:
        # clean slate if this entry already exists
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        # ---- raw eqdsk (opaque binary -- bit-perfect; schema-v2 fixed
        # name, the group path carries the coordinates) --------------------
        from .schema import EQDSK_DS
        grp.create_dataset(EQDSK_DS, data=np.void(eqdsk_bytes))

        # ---- 1-D profiles -----------------------------------------------
        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "j_phi", j_phi)
        write_profile(grp, "j_BS", j_BS)
        write_profile(grp, "j_inductive", j_inductive)

        if j_BS_edge is not None:
            write_profile(grp, "j_BS,edge", j_BS_edge)
        write_profile(grp, "n_e", n_e)
        write_profile(grp, "T_e", T_e)
        write_profile(grp, "n_i", n_i)
        write_profile(grp, "T_i", T_i)
        write_profile(grp, "w_ExB", w_ExB)

        # ---- optional: kinetic profile grid (when different from psi_N) ----
        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        if pressure is not None:
            write_profile(grp, "pressure", pressure)
        # Thermal (main-ion + electron) pressure alongside the total stored above,
        # so plots can show what the kinetic profiles contribute vs the impurity +
        # fast-ion pressure the GS solve actually used (total - thermal).
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)

        # ---- optional: auxiliary perturbed profiles (the switchboard) ----
        # Stored on psi_N_kinetic. Named "aux_<name>" (e.g. aux_omega_tor,
        # aux_chi_e, aux_zeff) for downstream (transport) consumers.
        if aux:
            for _name, _arr in aux.items():
                grp.create_dataset(f"aux_{_name}",
                                   data=np.asarray(_arr, dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"] = float(li1)
        grp.attrs["l_i(3)"] = float(li3)
        grp.attrs["count"]  = int(count)
        if scan_key is not None:
            grp.attrs["scan_key"] = scan_key

        # ---- optional: p-file bytes ----------------------------------------
        # Per-draw pfile blobs are only stored for TEXT p-files (rewritten with
        # this draw's perturbed kinetics -- genuinely draw-specific data). A
        # binary IDA .cdf source cannot be draw-perturbed and would only be
        # duplicated verbatim into every draw (~190 MB x n_draws with the new
        # IDA-database files); it lives ONCE in _baseline and per-draw readers
        # (BouquetArchive.DrawView.pfile_bytes, load_equilibrium) fall back.
        from .schema import is_binary_profile_source
        if pfile_bytes is not None and not is_binary_profile_source(pfile_bytes):
            pf_ds = "pfile"
            grp.create_dataset(pf_ds, data=np.void(pfile_bytes))

        # ---- optional: Zeff profile ----------------------------------------
        if Zeff is not None:
            write_profile(grp, "Zeff", Zeff)

        # ---- optional: coil currents ---------------------------------------
        if coil_currents is not None:
            import json
            names = list(coil_currents.keys())
            values = np.array([coil_currents[n] for n in names], dtype=np.float64)
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))

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
        # The l_i_target this draw actually converged to.  Equal to the
        # bouquet's l_i_target when l_i_uncertainty=0; otherwise a per-
        # draw sample from N(l_i_target, l_i_uncertainty * l_i_target).
        # Stored so post-run analysis can verify the sampled
        # distribution + diagnose any draw whose realised l_i diverged
        # from its (intentionally perturbed) target.
        if l_i_target_used is not None:
            grp.attrs["l_i_target_used"] = float(l_i_target_used)
        if l_i_uncertainty is not None:
            grp.attrs["l_i_uncertainty"] = float(l_i_uncertainty)

        # ---- High-resolution LCFS reference for this draw.
        # Captured by perturb_kinetic_equilibrium via safe_trace_surf at
        # the same mygs state save_eqdsk was called from -- so this and
        # the baseline.recon_lcfs_ref are method-consistent (both 10k-pt
        # trace_surf at psi = 1 - psi_pad).  Eliminates the ~4 mm
        # sampling-noise floor that the 100-pt eqdsk RBBBS introduces
        # when used as the perturbed-side boundary in plot_traces.
        if perturbed_lcfs_ref is not None:
            grp.create_dataset(
                "perturbed_lcfs_ref",
                data=np.asarray(perturbed_lcfs_ref, dtype=np.float64),
            )

        # ---- TokaMaker-computed X-point(s) for this draw ------------------
        # The poloidal-field nulls returned by mygs.get_xpoints(), captured
        # at the SAME solver state the eqdsk was saved from.  This is the
        # authoritative X-point location (a true B_p=0 saddle), used by
        # plot_boundary_point_traces in place of any geometric corner guess.
        # Shape (N, 2) = [[R, Z], ...]; the active (boundary-defining) null
        # is the last row when ``diverted`` is True.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

        # ---- Live-equilibrium FSA block (optional subgroup) --------------
        # Captured from the converged TokaMaker equilibrium at the same state
        # the eqdsk was saved from, so this draw's own flux geometry enables
        # an exact toroidal<->parallel conversion at IMAS export
        # (physics.capture_equilibrium_fsa -> physics.toroidal_to_parallel).
        if eq_fsa:
            from .schema import EQ_FSA_GROUP, EQ_FSA_UNITS
            fsa_grp = grp.create_group(EQ_FSA_GROUP)
            for _name, _arr in eq_fsa.items():
                if _arr is None:
                    continue
                ds = fsa_grp.create_dataset(
                    _name, data=np.asarray(_arr, dtype=np.float64))
                _u = EQ_FSA_UNITS.get(_name)
                if _u:
                    ds.attrs["units"] = _u


def load_eq_fsa(header, count, scan_key=None):
    """Load one draw's live-equilibrium FSA block, or ``None`` if not captured.

    Returns a dict of 1-D arrays (``psi_N``, ``F``, ``avg_inv_R``,
    ``avg_inv_R2`` (present only when the exact quadrature succeeded),
    ``avg_B2``, ``q``, ``dV_dpsi``, ``f_trap``, ``B_avg``) -- the geometry
    :func:`bouquet.physics.toroidal_to_parallel` needs for an exact IMAS
    write-back. ``None`` for archives written without live capture (fall back
    to the baseline-ratio reconstruction).
    """
    from .schema import EQ_FSA_GROUP
    h5path = _resolve_h5(header)
    gp = _group_path(scan_key, count)
    with h5py.File(h5path, "r") as hf:
        if gp not in hf or EQ_FSA_GROUP not in hf[gp]:
            return None
        g = hf[gp][EQ_FSA_GROUP]
        return {k: np.array(g[k], dtype=float) for k in g}


def load_equilibrium(header, count, scan_key=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    scan_key : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure"``, ``"Zeff"``,
        ``"coil_currents"``, ``"pfile_bytes"``.
    """
    from .schema import EQDSK_DS

    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(scan_key, count)

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes (schema-v2 fixed name) ----------------------
        if EQDSK_DS not in grp:
            legacy = [k for k in grp if k.endswith(".eqdsk")]
            raise KeyError(
                f"no '{EQDSK_DS}' dataset in {grp_path} of {db_path}"
                + (f" -- found legacy-named {legacy}: this is a pre-v2 "
                   "archive; regenerate it with the current bouquet (or "
                   "read it via BouquetArchive, whose suffix scan still "
                   "resolves legacy eqdsk names)." if legacy else "."))
        eqdsk_bytes = bytes(grp[EQDSK_DS][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            # coordinate-carrying FILENAME (the fixed dataset name would
            # make every extracted draw clobber the same 'eqdsk' file)
            bkey = _scan_key(scan_key)
            stem = os.path.basename(header) + (f"_{bkey}" if bkey else "")
            out_path = os.path.join(eqdsk_out_dir, f"{stem}_{int(count)}.eqdsk")
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

        # ---- optional: Zeff -----------------------------------------------
        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # ---- optional: p-file bytes ----------------------------------------
        # Text p-file sources are stored per draw (draw-perturbed); binary IDA
        # .cdf sources live ONCE in _baseline -- fall back there when the draw
        # carries no blob (see store_equilibrium / is_binary_profile_source).
        pf_ds = "pfile"
        if pf_ds in grp:
            result["pfile_bytes"] = bytes(grp[pf_ds][()])
        else:
            _bl = grp.parent.get("_baseline") if grp.parent is not None else None
            if _bl is not None and pf_ds in _bl:
                result["pfile_bytes"] = bytes(_bl[pf_ds][()])

        # ---- optional: coil currents ---------------------------------------
        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
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
    scan_key=None,
    pressure_thermal=None,
    eqdsk_bytes=None,
    pfile_bytes=None,
    psi_N_kinetic=None,
    coil_currents=None,
    coil_names=None,
    recon_lcfs_ref=None,
    x_points=None,
    diverted=None,
    aux_baselines=None,
    aux_sigmas=None,
    j_BS=None,
    j_inductive=None,
    source_kind=None,
):
    """
    Store the input (baseline) profiles and their uncertainties.

    For hierarchical layout (*scan_key* is not ``None``), these are
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
    bkey = _scan_key(scan_key)

    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    with h5py.File(db_path, "a") as hf:
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "n_e", ne)
        write_profile(grp, "T_e", te)
        write_profile(grp, "n_i", ni)
        write_profile(grp, "T_i", ti)
        write_profile(grp, "pressure", pressure)
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)
        write_profile(grp, "j_phi", j_phi)
        if j_BS is not None:
            write_profile(grp, "j_BS", j_BS)
        if j_inductive is not None:
            write_profile(grp, "j_inductive", j_inductive)
        write_profile(grp, "sigma_ne", sigma_ne)
        write_profile(grp, "sigma_te", sigma_te)
        write_profile(grp, "sigma_ni", sigma_ni)
        write_profile(grp, "sigma_ti", sigma_ti)
        write_profile(grp, "sigma_jphi", sigma_jphi)

        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        # ---- auxiliary-profile baselines + sigmas (on the kinetic
        # grid), so plot_transport_profiles can draw input +/- sigma bands
        if aux_baselines:
            for _n, _a in aux_baselines.items():
                grp.create_dataset(f"aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))
        if aux_sigmas:
            for _n, _a in aux_sigmas.items():
                grp.create_dataset(f"sigma_aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))

        grp.attrs["Ip_target"]  = float(Ip_target)
        grp.attrs["l_i_target"] = float(l_i_target)
        # Provenance marker for robust path detection in plotting (independent of
        # the source-decoupled aux switchboard): "imas" or "geqdsk".
        if source_kind is not None:
            grp.attrs["source_kind"] = str(source_kind)

        if eqdsk_bytes is not None:
            grp.create_dataset("eqdsk", data=np.void(eqdsk_bytes))
        if pfile_bytes is not None:
            grp.create_dataset("pfile", data=np.void(pfile_bytes))

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

        # ---- TokaMaker-computed X-point(s) for the recon baseline --------
        # Same provenance as the per-draw ``x_points`` (mygs.get_xpoints()
        # at the recon-converged state).  plot_boundary_point_traces uses
        # this to decide whether the top/bottom traces are tracking a true
        # X-point and to anchor the deviation reference.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

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
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))


# ====================================================================
#  Introspection helpers (used by GUI and notebook API)
# ====================================================================
def discover_scan_keys(h5path_or_header):
    """
    Discover all scan values in an HDF5 equilibrium database.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- either a full ``.h5`` path or a bare header
        stem (``<header>.h5`` is resolved).

    Returns
    -------
    scan_keys : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
    h5path = _resolve_h5(h5path_or_header)
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


def count_equilibria(h5path_or_header, scan_key=None):
    """
    Count the number of perturbed equilibria stored for a scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    n : int
    """
    return len(list_equilibrium_indices(h5path_or_header, scan_key=scan_key))


def list_equilibrium_indices(h5path_or_header, scan_key=None):
    """Return the sorted list of integer draw indices actually stored.

    Band-rejected / failed draws leave GAPS in the index sequence (e.g.
    ``[0, 1, 2, 3, 4, 5, 7, ...]`` with draw 6 missing), so callers must
    iterate these indices rather than ``range(count_equilibria(...))`` --
    the latter assumes a contiguous ``0..n-1`` and KeyErrors on the gap.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    list of int
        Sorted stored draw indices.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    with h5py.File(h5path, "r") as hf:
        parent = hf[f"scan/{bkey}"] if bkey is not None else hf
        return sorted(
            int(k) for k in parent.keys()
            if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit()
        )


def load_baseline_profiles(h5path_or_header, scan_key=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
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


def load_equilibrium_by_path(h5path_or_header, count, scan_key=None):
    """
    Load one perturbed equilibrium (profiles + scalars only).

    Like :func:`load_equilibrium`, but addressed by *scan_key* and does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium` if
    you need those).  Accepts either a full ``.h5`` path or a bare header
    stem for *h5path_or_header*.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
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

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        if "psi_N_kinetic" in grp:
            result["psi_N_kinetic"] = np.array(grp["psi_N_kinetic"])

        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # auxiliary ("switchboard") perturbed profiles -- aux_zeff, aux_omega_tor,
        # aux_chi_e, ... on psi_N_kinetic -- so per-draw plots (draw_zeff, the aux
        # figure) can overlay the draws, not just the baseline.
        for _k in grp.keys():
            if str(_k).startswith("aux_"):
                result[str(_k)] = np.array(grp[_k])

        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
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
