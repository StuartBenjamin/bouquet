"""q and coil checks over bouquet archives: one row per stored equilibrium."""

import json
import os

import h5py
import numpy as np

from .coil_check import DEFAULT_EXCLUDE, CoilCheck
from .io.geqdsk import GEQDSKEquilibrium
from .io.imas import ARCHIVE_EQDSK_COCOS
from .q_check import QCheck, mse_er_at_chords, mse_loglike
from .schema import EQ_FSA_GROUP
from .utils import _read_coil_names, discover_scan_keys

#: Columns of a row; a check that was not asked for, or could not run, is NaN.
ROW_KEYS = ("archive", "scan", "draw", "is_baseline", "selected", "n_coil_reg",
            "logL_q", "d2_q", "m_q", "axis_dR", "chi2_mse", "logL_mse", "nu_mse",
            "logL_coil", "chi2_coil", "nu_coil", "max_abs_z", "worst_coil",
            "coil_units_suspect", "status")


def _eqdsk_R_axis(raw):
    """RMAXIS [m] from g-file bytes (first field of the third line)."""
    return float(raw.split(b"\n")[2][:16])


def _load(obj, cls):
    return cls.load(obj) if isinstance(obj, (str, os.PathLike)) else obj


def check_archive(path, q_check=None, coil_check=None, sigma_model=None, mse=None,
                  scan_key=None, exclude_coils=DEFAULT_EXCLUDE, device=None):
    """Rows (dicts with :data:`ROW_KEYS`) scoring the baseline and every draw of
    the archive at *path* (all scans, or *scan_key*).

    * *q_check* (:class:`~bouquet.q_check.QCheck` or its npz): ``logL_q``,
      ``d2_q``, ``m_q`` of the draw's ``eq_fsa`` q, or of its eqdsk's q when
      it has no ``eq_fsa`` (the baseline), with the axis error ``axis_dR``
      set by the equilibrium's own magnetic axis (its eqdsk).
    * *mse* ``(block, (phi_N, E_r, E_r_err))``: the direct check of each
      equilibrium's own eqdsk, with E_r mapped onto it (``chi2_mse``,
      ``logL_mse``, ``nu_mse``).
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
            members = (["_baseline"] if "_baseline" in grp else []) + sorted(
                (k for k in grp if k.lstrip("-").isdigit()), key=int)
            for k in members:
                g = grp[k]
                row = dict(dict.fromkeys(ROW_KEYS, np.nan), archive=str(path), scan=key,
                           draw=-1 if k == "_baseline" else int(k),
                           is_baseline=k == "_baseline", selected=g.attrs.get("selected"),
                           n_coil_reg=n_reg, status=[])
                rows.append(row)
                eq = None
                if mse is not None or (qc is not None and EQ_FSA_GROUP not in g):
                    eq = GEQDSKEquilibrium.from_bytes(
                        bytes(g["eqdsk"][()]), cocos=ARCHIVE_EQDSK_COCOS) if "eqdsk" in g else None
                if qc is not None:
                    if EQ_FSA_GROUP in g:
                        psi, q = g[EQ_FSA_GROUP]["psi_N"][()], g[EQ_FSA_GROUP]["q"][()]
                    elif eq is not None:
                        psi, q = np.linspace(0.0, 1.0, eq.qpsi.size), eq.qpsi
                    else:
                        psi = None
                    if psi is None:
                        row["status"].append("no_q")
                    else:
                        R_ax = (eq.R_mag if eq is not None else
                                _eqdsk_R_axis(bytes(g["eqdsk"][()])) if "eqdsk" in g else None)
                        r = qc.loglike(psi, q, R_axis=R_ax)
                        row.update(logL_q=r["logL"][0], d2_q=r["d2"][0], m_q=r["m"],
                                   axis_dR=r["axis_dR"][0])
                if mse is not None and eq is None:
                    row["status"].append("no_eqdsk")
                elif mse is not None:
                    try:
                        r = mse_loglike(eq, mse_er_at_chords(eq, mse[0], *mse[1]))
                        row.update(chi2_mse=r["chi2"], logL_mse=r["logL"], nu_mse=r["nu"])
                    except (ValueError, RuntimeError) as e:
                        row["status"].append(f"mse: {e}")
                if cc is not None:
                    names = _read_coil_names(g)
                    if names and "coil_currents" in g:
                        coil_rows.append((row, tuple(names), g["coil_currents"][()]))
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
