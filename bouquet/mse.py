"""MSE pitch-angle data and forward model, shared by the closures that use it.

Motional-Stark-effect polarimetry measures, per chord, the tangent of the
pitch angle gamma.  In the EFIT k-file convention (standard chords, A6 = A7 =
A8 = 0) the synthetic signal of an equilibrium is

.. code-block:: text

    tan(gamma) = (A1 B_Z + A5 E_r) / (A2 B_phi + A3 B_R + A4 B_Z)

with the A-coefficients the chord's viewing geometry and ``E_r`` the radial
electric field at the chord [V/m].  This module holds that ONE formula, the
data-block schema it is evaluated on, and the helpers every closure that
consumes pitch angles uses -- so the physics is written down once.

**Data block** (a plain dict, JSON-friendly; every array is one entry per
chord):

``R``, ``Z``        chord intersection point [m]
``tgamma``          measured tan(gamma)
``sigma``           1-sigma uncertainty of ``tgamma``
``weight``          the fit weight (EFIT's FWTGAM).  NOT a 0/1 mask: it is
                    folded into an effective uncertainty
                    ``sigma_eff = sigma / sqrt(weight)`` once, and a chord with
                    ``weight <= 0`` is excluded
``A1`` .. ``A4``    viewing-geometry coefficients (required)
``A5``              E_r coefficient (optional; zero when absent)
``Er``              E_r at the chord [V/m] (optional; zero when absent).  When
                    it is supplied the forward model carries ``A5 E_r``
``er_corrected``    optional bool: ``tgamma`` has ALREADY been corrected for
                    E_r upstream (e.g. by the reconstruction that produced it).
                    The forward model then carries no E_r term, and supplying
                    a non-zero ``Er`` as well is refused as a double count

**Field orientation.**  The sign of each field component in the measurement's
convention (plasma current and toroidal-field helicity, COCOS) need not match
the equilibrium's.  :func:`mse_sign_convention` resolves it EMPIRICALLY on one
equilibrium, by evaluating the four ``(+-B_pol, +-B_phi)`` combinations and
keeping the one with the lowest chi^2; the caller then FREEZES it.  This
selects a convention, not a fit to the data: the wrong-helicity branches are
off by orders of magnitude in chi^2, and the table of all four is returned so
the choice can be audited.

Everything here is pure numpy except :func:`mse_field_at`, which reads the
field off a live equilibrium object.
"""

import numpy as np

#: The fewest usable (weighted, finite) chords a block may carry.  Below it a
#: pitch-angle chi^2 is a statement about a handful of points, and the
#: consumers refuse rather than fit it.
MSE_MIN_CHORDS = 4

#: Keys every data block must carry.
MSE_REQUIRED_KEYS = ("R", "Z", "tgamma", "sigma", "weight",
                     "A1", "A2", "A3", "A4")


class MSEDataUnusable(ValueError):
    """The MSE data block is absent, malformed, or carries too few chords."""


def mse_chords(md, min_chords=MSE_MIN_CHORDS, sigma_sys=0.0):
    """Validate an MSE data block and return its USABLE chords.

    Returns a dict of 1-D float arrays over the active chords only (``R``,
    ``Z``, ``tgamma``, ``sigma``, ``weight``, ``sigma_eff``, ``A1`` .. ``A5``,
    ``Er``) plus ``index`` (the active chords' positions in the input),
    ``n_total``, ``n_active``, ``er_applied`` (the forward model carries a
    non-zero ``A5 E_r``), ``er_corrected`` (the caller's flag) and
    ``sigma_sys``.

    A chord is active when its weight is positive and ``tgamma``, ``sigma``,
    the A-coefficients, ``R`` and ``Z`` are all finite with ``sigma > 0``.
    ``sigma_eff = sqrt(sigma^2 / weight + sigma_sys^2)``: the fit weight
    folded in once, and an OPTIONAL caller-stated systematic added in
    quadrature (default 0 -- nothing is inflated unless asked for).

    Raises :class:`MSEDataUnusable` (a ``ValueError``) on a missing block or
    key, arrays of unequal length, a non-finite or negative ``sigma_sys``, a
    non-zero ``Er`` together with ``er_corrected=True``, or fewer than
    *min_chords* active chords.
    """
    if md is None:
        raise MSEDataUnusable("no MSE data block was supplied")
    if not isinstance(md, dict):
        raise MSEDataUnusable(f"the MSE data block must be a dict, got "
                              f"{type(md).__name__}")
    missing = [k for k in MSE_REQUIRED_KEYS if k not in md]
    if missing:
        raise MSEDataUnusable("the MSE data block lacks "
                              + ", ".join(missing)
                              + f" (required: {', '.join(MSE_REQUIRED_KEYS)})")
    try:
        arr = {k: np.atleast_1d(np.asarray(md[k], dtype=float)).ravel()
               for k in MSE_REQUIRED_KEYS}
    except (TypeError, ValueError) as e:
        raise MSEDataUnusable(f"the MSE data block is not numeric ({e})")
    n = arr["tgamma"].size
    for k in ("A5", "Er"):
        if md.get(k) is None:
            arr[k] = np.zeros(n)
        else:
            try:
                arr[k] = np.atleast_1d(np.asarray(md[k], dtype=float)).ravel()
            except (TypeError, ValueError) as e:
                raise MSEDataUnusable(f"MSE '{k}' is not numeric ({e})")
    bad = [k for k, v in arr.items() if v.size != n]
    if bad:
        raise MSEDataUnusable(
            "MSE arrays differ in length: "
            + ", ".join(f"{k}={arr[k].size}" for k in bad)
            + f" against tgamma={n}")
    ss = float(sigma_sys)
    if not (np.isfinite(ss) and ss >= 0.0):
        raise MSEDataUnusable(f"sigma_sys must be finite and >= 0, got "
                              f"{sigma_sys!r}")
    er_corrected = bool(md.get("er_corrected", False))

    act = arr["weight"] > 0.0
    for k in ("R", "Z", "tgamma", "sigma", "weight", "A1", "A2", "A3", "A4",
              "A5", "Er"):
        act &= np.isfinite(arr[k])
    act &= arr["sigma"] > 0.0
    n_act = int(act.sum())
    er_applied = bool(np.any((arr["A5"][act] * arr["Er"][act]) != 0.0))
    if er_corrected and bool(np.any(arr["Er"][act] != 0.0)):
        raise MSEDataUnusable(
            "the MSE block says tgamma is ALREADY E_r-corrected "
            "(er_corrected=True) and also supplies a non-zero Er -- applying "
            "A5*Er to corrected data would count E_r twice")
    if n_act < int(min_chords):
        raise MSEDataUnusable(
            f"only {n_act} of {n} MSE chords are usable (weight > 0, finite, "
            f"sigma > 0); at least {int(min_chords)} are required")
    out = {k: v[act].copy() for k, v in arr.items()}
    out["sigma_eff"] = np.sqrt(out["sigma"] ** 2 / out["weight"] + ss ** 2)
    out.update(index=np.nonzero(act)[0], n_total=int(n), n_active=n_act,
               er_applied=er_applied, er_corrected=er_corrected,
               sigma_sys=ss, min_chords=int(min_chords), excluded=[])
    return out


def mse_er_terms(ch):
    """One-line statement of how E_r entered the forward model."""
    if ch["er_corrected"]:
        return ("tgamma supplied E_r-CORRECTED upstream (er_corrected=True); "
                "forward model carries no E_r term; A6(E_z)=0")
    if ch["er_applied"]:
        return "A5*Er applied (caller-supplied E_r at the chords); A6(E_z)=0"
    return "no E_r: A5*Er omitted (no Er supplied); A6(E_z)=0"


def mse_tan_gamma(B, ch, sign_pol=1.0, sign_tor=1.0):
    """Synthetic tan(gamma) at the chords of *ch* from the field *B*.

    *B* is ``(n_active, 3)`` -- ``(B_R, B_phi, B_Z)`` at each active chord, the
    order a TokaMaker ``get_field_eval("B")`` returns.  ``sign_pol`` multiplies
    the poloidal components (B_R, B_Z) and ``sign_tor`` the toroidal one; see
    :func:`mse_sign_convention`.
    """
    B = np.asarray(B, dtype=float).reshape(-1, 3)
    sp, st = float(sign_pol), float(sign_tor)
    num = ch["A1"] * sp * B[:, 2] + ch["A5"] * ch["Er"]
    den = (ch["A2"] * st * B[:, 1] + ch["A3"] * sp * B[:, 0]
           + ch["A4"] * sp * B[:, 2])
    return num / den


def mse_chi2(tg_pred, ch):
    """``(chi2, z)`` -- ``z = (tg_pred - tgamma) / sigma_eff`` per active chord.

    ``chi2 = sum(z**2)``, i.e. ``sum_k w_k ((pred - meas) / sigma_k)^2`` with
    the fit weight folded into ``sigma_eff`` (see :func:`mse_chords`).
    """
    z = (np.asarray(tg_pred, dtype=float) - ch["tgamma"]) / ch["sigma_eff"]
    return float(np.sum(z ** 2)), z


def mse_sign_convention(B, ch):
    """``(sign_pol, sign_tor, table)`` minimising chi^2 over the 4 orientations.

    ``table`` maps ``"(+1,+1)"``-style keys to each combination's chi^2 so the
    choice is auditable.  A non-finite chi^2 on every combination (e.g. the
    field is zero at the chords) raises :class:`MSEDataUnusable`.
    """
    table = {}
    best = None
    for sp in (1.0, -1.0):
        for st in (1.0, -1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                c2, _ = mse_chi2(mse_tan_gamma(B, ch, sp, st), ch)
            table[f"({sp:+.0f},{st:+.0f})"] = c2
            if np.isfinite(c2) and (best is None or c2 < best[2]):
                best = (sp, st, c2)
    if best is None:
        raise MSEDataUnusable("the synthetic tan(gamma) is non-finite for "
                              "every field orientation -- the field at the "
                              "chords is unusable")
    return best[0], best[1], table


#: Per-chord arrays of a chord dict (:func:`mse_chords`); every one is indexed
#: by the ACTIVE chords, so dropping a chord drops its entry from each.
MSE_CHORD_ARRAYS = ("R", "Z", "tgamma", "sigma", "weight", "A1", "A2", "A3",
                    "A4", "A5", "Er", "sigma_eff", "index")

#: Exclusion reason for a chord the solver's field interpolator cannot place
#: on its mesh (see :func:`mse_field_at`).
MSE_REASON_OFF_MESH = ("off the solver mesh (the field interpolator found no "
                       "cell containing (R, Z)); no field can be read there")


def mse_exclude(ch, drop, reason):
    """A copy of chord dict *ch* without the chords where *drop* is True.

    Every dropped chord is appended to ``ch["excluded"]`` as ``(input index,
    reason)`` -- a chord is never removed without a recorded reason.
    ``n_active`` and ``er_applied`` are recomputed on what is left; the
    caller decides whether that is still enough chords (``ch["min_chords"]``).
    """
    drop = np.asarray(drop, dtype=bool).ravel()
    if drop.shape != (int(ch["n_active"]),):
        raise ValueError(f"mse_exclude: mask of {drop.size} entries for "
                         f"{int(ch['n_active'])} active chords")
    keep = ~drop
    out = dict(ch)
    for k in MSE_CHORD_ARRAYS:
        if k in ch:
            out[k] = np.asarray(ch[k])[keep].copy()
    out["excluded"] = list(ch.get("excluded", ())) + [
        (int(i), str(reason)) for i in np.asarray(ch["index"])[drop]]
    out["n_active"] = int(keep.sum())
    out["er_applied"] = bool(np.any((out["A5"] * out["Er"]) != 0.0))
    return out


def mse_field_at(eq, R, Z):
    """``(B, found)``: the field of a live equilibrium at the chords.

    ``B`` is ``(n, 3)`` -- ``(B_R, B_phi, B_Z)`` -- and ``found`` ``(n,)``
    bool.  A chord the interpolator cannot place on the mesh gets
    ``found=False`` and a row of NaN; its field is NEVER reported.

    Why this has to be explicit: TokaMaker's field interpolator returns its
    own value buffer, and when a point is not inside any mesh cell (or fails
    the barycentric test) the Fortran routine returns WITHOUT writing it -- so
    a plain ``eval`` of an off-mesh point hands back whatever the PREVIOUS
    point left there, finite and wrong.  Two independent guards, either of
    which is sufficient: the buffer is poisoned with NaN before every
    evaluation (a point that writes nothing then reads NaN), and the cell the
    interpolator reports is required to be a real one (``cell > 0``; the
    Fortran sets 0 for not-found and a negative index for a failed
    barycentric test).  An interpolator without those attributes (a test
    double) is judged on the finiteness of what it returns.

    The interpolator is created FRESH on every call: an interpolator made
    before a re-solve or an equilibrium swap is bound to stale state, and
    evaluating it can crash the process.
    """
    Beval = eq.get_field_eval("B")
    R = np.atleast_1d(np.asarray(R, dtype=float)).ravel()
    Z = np.atleast_1d(np.asarray(Z, dtype=float)).ravel()
    B = np.full((R.size, 3), np.nan, dtype=float)
    found = np.zeros(R.size, dtype=bool)
    buf = getattr(Beval, "val", None)
    cell = getattr(Beval, "cell", None)
    for k in range(R.size):
        if not (np.isfinite(R[k]) and np.isfinite(Z[k]) and R[k] > 0.0):
            continue
        if isinstance(buf, np.ndarray):
            buf[:] = np.nan
        if cell is not None and int(getattr(cell, "value", 0)) <= 0:
            # a not-found / failed point leaves 0 or -|cell| behind; start the
            # next search from the interpolator's own "no guess" value
            cell.value = -1
        v = np.asarray(Beval.eval(np.array([R[k], Z[k]], dtype=float)),
                       dtype=float).ravel()
        ok = v.size == 3 and bool(np.all(np.isfinite(v)))
        if cell is not None:
            ok = ok and int(getattr(cell, "value", 0)) > 0
        if ok:
            B[k] = v
            found[k] = True
    return B, found
