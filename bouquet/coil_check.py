"""Measured coil currents over a time window, and candidate coil sets' likelihood.

Part 1, :func:`coil_distribution`: per coil, the window mean of the measured
current and its spread (measurement error and the scatter within the window).
Part 2, :meth:`CoilCheck.loglike`: Gaussian likelihood of candidate coil
currents with ``sigma^2 = sigma_meas^2 + sigma_model^2``.  The model term (how
far a reconstruction may sit from the machine) has no default: the caller
states it.
"""

import numpy as np

from .coil_spec import resolve_coil_sigma

#: Excluded unless asked for: the ohmic coil, which the reconstruction treats as
#: free and whose sections the mesh lumps together.
DEFAULT_EXCLUDE = ("ECOILA", "ECOILB")

#: A coil family whose median |candidate / measured| lies outside
#: (1 / UNITS_RATIO, UNITS_RATIO) is reported as a units mismatch (a turns
#: factor applied on one side only is ~55-61x).
UNITS_RATIO = 8.0


class CoilCheck:
    """Measured coil currents (``mean``, ``sigma_meas``, per name) in the units
    of the candidates; see :func:`coil_distribution`."""

    def __init__(self, names, mean, sigma_meas, meta=None):
        self.names = [str(n) for n in names]
        self.mean = np.asarray(mean, dtype=float)
        self.sigma_meas = np.asarray(sigma_meas, dtype=float)
        self.meta = dict(meta or {})

    def loglike(self, names, currents, sigma_model, exclude=DEFAULT_EXCLUDE, device=None):
        """Score candidate coil sets: rows of *currents*, columns named by *names*.

        *sigma_model* is ``0`` (measurement only) or anything
        :func:`bouquet.coil_spec.resolve_coil_sigma` accepts: a ``{coil:
        sigma}`` table (absent coils are not judged), ``{"floor",
        "fraction"}`` of the measured current, a callable, or a named model of
        *device*.  Judged coils are those both here and in *names*, with a
        model sigma, and not in *exclude*.

        Returns ``dict(logL, chi2, chi2_nu, max_abs_z, worst_coil, z, coils,
        nu, units, units_suspect, sigma_model)``, per row where it varies:
        ``logL = -(chi2 + sum log(2 pi sigma^2)) / 2``; ``units`` is each
        family's median ``|candidate / measured|``.
        """
        names = [str(n) for n in names]
        cur = np.atleast_2d(np.asarray(currents, dtype=float))
        meas = dict(zip(self.names, self.mean))
        if isinstance(sigma_model, (int, float)) and sigma_model == 0:
            sig_m, model = dict.fromkeys(self.names, 0.0), {"kind": "none"}
        else:
            sig_m, model = resolve_coil_sigma(meas, sigma_model, device=device)
        coils = [c for c in names if c in meas and c in sig_m and c not in exclude]
        if not coils:
            raise ValueError("CoilCheck.loglike: no coil to judge (names, measured "
                             "coils, model sigma and exclude leave none)")
        i = [self.names.index(c) for c in coils]
        mu = self.mean[i]
        sig = np.hypot(self.sigma_meas[i], [sig_m[c] for c in coils])
        cand = cur[:, [names.index(c) for c in coils]]
        z = (cand - mu) / sig
        chi2 = np.sum(z ** 2, axis=1)
        ratio = np.abs(cand / mu)
        units = {f: float(np.median(ratio[:, [c[0] == f for c in coils]]))
                 for f in sorted({c[0] for c in coils})}
        return dict(logL=-0.5 * (chi2 + np.log(2 * np.pi * sig ** 2).sum()), chi2=chi2,
                    chi2_nu=chi2 / len(coils), max_abs_z=np.max(np.abs(z), axis=1),
                    worst_coil=[coils[j] for j in np.argmax(np.abs(z), axis=1)], z=z,
                    coils=coils, nu=len(coils), units=units,
                    units_suspect=[f for f, r in units.items()
                                   if not 1 / UNITS_RATIO < r < UNITS_RATIO],
                    sigma_model=model)

    def save(self, path):
        import json
        np.savez(path, names=np.array(self.names), mean=self.mean,
                 sigma_meas=self.sigma_meas, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        import json
        with np.load(path) as f:
            return cls(f["names"], f["mean"], f["sigma_meas"], json.loads(str(f["meta"])))


def coil_distribution(traces, t0, t1, turns, sigma_extra=None, meta=None):
    """:class:`CoilCheck` of the measured currents over the window [t0, t1] [s].

    *traces* is ``{name: (time [s], current [A], 1-sigma [A] or None)}``,
    already in the candidates' current orientation (e.g.
    :func:`bouquet.coil_targets.pf_active_traces`).  Per coil, ``mean`` is the
    window mean and ``sigma_meas`` is the hypot of the window standard deviation,
    the mean 1-sigma, and ``sigma_extra[name]`` [A] (caller-stated terms such as
    calibration or integrator drift).  Both are then multiplied by ``turns[name]``
    (circuit amps to the candidates' units; 1.0 when absent, as in
    :func:`bouquet.coil_targets.coil_reg_from_measured`).  Coils with fewer
    than 2 samples in the window are left out.
    """
    extra = sigma_extra or {}
    names, mean, sig = [], [], []
    for name, (t, d, e) in traces.items():
        w = (np.asarray(t) >= t0) & (np.asarray(t) <= t1)
        if w.sum() < 2:
            continue
        f = abs(float(turns.get(name, 1.0)))
        err = 0.0 if e is None else float(np.mean(np.asarray(e)[w]))
        names.append(name)
        mean.append(f * float(np.mean(d[w])))
        sig.append(f * float(np.sqrt(np.var(d[w], ddof=1) + err ** 2
                                     + float(extra.get(name, 0.0)) ** 2)))
    return CoilCheck(names, mean, sig, meta=dict(meta or {}, t0=float(t0), t1=float(t1),
                                                 turns={k: float(v) for k, v in turns.items()}))
