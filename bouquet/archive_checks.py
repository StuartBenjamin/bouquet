"""q and coil checks over bouquet archives: one row per stored equilibrium."""

import json
import os

import h5py
import numpy as np

from .coil_check import DEFAULT_EXCLUDE, CoilCheck
from .io.geqdsk import GEQDSKEquilibrium
from .io.imas import ARCHIVE_EQDSK_COCOS
from .q_check import QCheck, mse_er_at_chords, direct_loglike
from .schema import COIL_VALUES_DS, EQ_FSA_GROUP, find_bytes_dataset
from .utils import _read_coil_names, discover_scan_keys, list_equilibrium_indices

#: Columns of a row; a check that was not asked for, or could not run, is NaN.
ROW_KEYS = ("archive", "scan", "draw", "is_baseline", "selected", "n_coil_reg",
            "chi2_direct", "logL_direct", "nu_direct", "axis_offset_R", "axis_offset_Z",
            "chi2_direct_aligned", "logL_direct_aligned",
            "logL_points", "chi2_points", "nu_points", "axis_error", "logL_gp", "d2_gp", "m_gp",
            "logL_coil", "chi2_coil", "nu_coil", "max_abs_z", "worst_coil",
            "coil_units_suspect", "status")


def _load(obj, cls):
    return cls.load(obj) if isinstance(obj, (str, os.PathLike)) else obj


def _eqdsk(g):
    """The group's archived eqdsk as an equilibrium, or None."""
    name = find_bytes_dataset(g)
    return (None if name is None else
            GEQDSKEquilibrium.from_bytes(bytes(g[name][()]), cocos=ARCHIVE_EQDSK_COCOS))


def _score_q(row, qc, g, eq, gp):
    """Points-alone (and with *gp* the GP) check of the group's eq_fsa q (else its
    eqdsk's), axis from its eqdsk."""
    if EQ_FSA_GROUP in g:
        psi, q = g[EQ_FSA_GROUP]["psi_N"][()], g[EQ_FSA_GROUP]["q"][()]
    elif eq is not None:
        psi, q = np.linspace(0.0, 1.0, eq.qpsi.size), eq.qpsi
    else:
        row["status"].append("no_q")
        return
    R_axis = None if eq is None else eq.R_mag
    try:
        p = qc.points_loglike(psi, q, R_axis=R_axis)
        row.update(logL_points=p["logL"][0], chi2_points=p["chi2"][0], nu_points=p["nu"],
                   axis_error=p["axis_error"][0])
        if gp:
            r = qc.gp_loglike(psi, q, R_axis=R_axis)
            row.update(logL_gp=r["logL"][0], d2_gp=r["d2"][0], m_gp=r["m"])
    except ValueError as e:
        row["status"].append(f"q: {e}")


def _on(eq, mse, shift=(0.0, 0.0)):
    """The MSE block with E_r mapped onto *eq* moved by -*shift* (the chords moved by +*shift*)."""
    block, profile = mse
    block = dict(block, R=np.asarray(block["R"]) + shift[0], Z=np.asarray(block["Z"]) + shift[1])
    return mse_er_at_chords(eq, block, *profile) if profile else block


def _score_direct(row, mse, eq, axis_ref, align):
    """Direct MSE check of the equilibrium; with *axis_ref* its axis offset, and with *align*
    the check again with the equilibrium moved onto that axis (a diagnostic only)."""
    if eq is None:
        row["status"].append("no_eqdsk")
        return
    shifts = [(0.0, 0.0)]
    if axis_ref is not None:
        off = (eq.R_mag - axis_ref[0], eq.Z_mag - axis_ref[1])
        row.update(axis_offset_R=off[0], axis_offset_Z=off[1])
        if align:
            shifts.append(tuple(np.nan_to_num(off)))
    try:
        r = [direct_loglike(eq, _on(eq, mse, sh)) for sh in shifts]
    except (ValueError, RuntimeError) as e:
        row["status"].append(f"direct: {e}")
        return
    row.update(chi2_direct=r[0]["chi2"], logL_direct=r[0]["logL"], nu_direct=r[0]["nu"])
    if len(r) > 1:
        row.update(chi2_direct_aligned=r[1]["chi2"], logL_direct_aligned=r[1]["logL"])


def check_archive(path, q_check=None, coil_check=None, sigma_model=None, mse=None,
                  scan_key=None, exclude_coils=DEFAULT_EXCLUDE, device=None, gp=False,
                  axis_ref=None, align_axis=False):
    """Rows (dicts with :data:`ROW_KEYS`) scoring the baseline and every draw of
    the archive at *path* (all scans, or *scan_key*).

    * *q_check* (:class:`~bouquet.q_check.QCheck` or its npz): ``logL_points``,
      ``chi2_points``, ``nu_points`` of the draw's ``eq_fsa`` q, or of its
      eqdsk's q when it has no ``eq_fsa`` (the baseline), against the q points,
      with the axis error ``axis_error`` set by the equilibrium's own magnetic
      axis (its eqdsk); with *gp* also ``logL_gp``, ``d2_gp``, ``m_gp`` against
      the GP fit to the points.
    * *mse* ``(block, profile)``, *profile* being the arguments of
      :func:`~bouquet.q_check.mse_er_at_chords` after the block (``phi_N,
      E_r[, E_r_err, E_tor, B_theta]``, or empty for no E_r): the direct check
      of each equilibrium's own eqdsk, with E_r mapped onto it
      (``chi2_direct``, ``logL_direct``, ``nu_direct``), the MSE likelihood of
      an equilibrium.  With *axis_ref* ``(R, Z)`` [m], e.g. EFIT02's magnetic
      axis: ``axis_offset_R``, ``axis_offset_Z`` (equilibrium minus *axis_ref*),
      and with *align_axis* ``chi2_direct_aligned``, ``logL_direct_aligned``
      with the equilibrium moved onto *axis_ref*: how much of the misfit is
      the axis position, which stays in the direct columns.
    * *coil_check* (:class:`~bouquet.coil_check.CoilCheck` or its npz) with
      *sigma_model*, which is then required: ``logL_coil``, ``chi2_coil``,
      ``nu_coil``, ``max_abs_z``, ``worst_coil``; ``coil_units_suspect``
      names the coil families whose currents look like a turns mismatch.

    A missing archive, or an equilibrium without the data a check needs, is a
    ``status`` entry, never an exception.  ``n_coil_reg`` counts the run's
    ``solver.coil_reg`` terms: the coil check is a measurement test only for
    runs pinned to the measured currents.
    """
    if not os.path.isfile(path):
        return [dict(dict.fromkeys(ROW_KEYS, np.nan), archive=str(path), status="no_archive")]
    qc, cc = _load(q_check, QCheck), _load(coil_check, CoilCheck)
    if cc is not None and sigma_model is None:
        raise ValueError("check_archive: coil_check needs sigma_model (0 for the "
                         "measurement alone); it has no default")
    keys = [scan_key] if scan_key is not None else (discover_scan_keys(path) or [None])
    rows, coil_rows = [], []
    with h5py.File(path, "r") as hf:
        cfg = json.loads(hf["config_json"][()]) if "config_json" in hf else {}
        n_reg = len((cfg.get("solver") or {}).get("coil_reg") or [])
        for key in keys:
            grp = hf[f"scan/{key}"] if key is not None else hf
            members = (["_baseline"] if "_baseline" in grp else []) + [
                str(i) for i in list_equilibrium_indices(path, key)]
            for k in members:
                g = grp[k]
                row = dict(dict.fromkeys(ROW_KEYS, np.nan), archive=str(path), scan=key,
                           draw=-1 if k == "_baseline" else int(k),
                           is_baseline=k == "_baseline", selected=g.attrs.get("selected"),
                           n_coil_reg=n_reg, status=[])
                rows.append(row)
                eq = _eqdsk(g) if qc is not None or mse is not None else None
                if qc is not None:
                    _score_q(row, qc, g, eq, gp)
                if mse is not None:
                    _score_direct(row, mse, eq, axis_ref, align_axis)
                if cc is not None:
                    names = _read_coil_names(g)
                    if names and COIL_VALUES_DS in g:
                        coil_rows.append((row, tuple(names), g[COIL_VALUES_DS][()]))
                    else:
                        row["status"].append("no_coil_data")
    for names in {n for _, n, _ in coil_rows}:
        sel = [(row, cur) for row, n, cur in coil_rows if n == names]
        try:
            r = cc.loglike(names, [cur for _, cur in sel], sigma_model,
                           exclude=exclude_coils, device=device)
        except ValueError as e:
            for row, _ in sel:
                row["status"].append(f"coil: {e}")
            continue
        for j, (row, _) in enumerate(sel):
            row.update(logL_coil=r["logL"][j], chi2_coil=r["chi2"][j], nu_coil=r["nu"],
                       max_abs_z=r["max_abs_z"][j], worst_coil=r["worst_coil"][j],
                       coil_units_suspect=",".join(r["units_suspect"]))
    for row in rows:
        row["status"] = "; ".join(row["status"]) or "ok"
    return rows


def _check_job(job):
    return check_archive(**job)


def check_archives(jobs, n_workers=1):
    """:func:`check_archive` rows of every job (a dict of its keyword
    arguments), in job order; with *n_workers* > 1 in spawned processes, one
    thread each (:func:`bouquet.threads.worker_environment`), so a calling
    script needs the ``if __name__ == "__main__":`` guard."""
    if n_workers <= 1:
        return [r for job in jobs for r in _check_job(job)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from .threads import worker_environment
    with worker_environment(), ProcessPoolExecutor(
            n_workers, mp_context=mp.get_context("spawn")) as ex:
        return [r for rows in ex.map(_check_job, jobs) for r in rows]
