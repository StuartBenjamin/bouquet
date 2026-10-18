"""Write the sigma envelopes of the synthetic fixtures from the radial templates.

    python tests/data/apply_sigma_templates.py [DIR]      (default: this directory)

The templates are the relative 1-sigma envelopes ``sigma/|x|`` of
``bouquet.uncertainties.SIGMA_TEMPLATES``: the median relative envelopes of
IDA-lite deliveries on DIII-D H-mode flat-top, ramp / early H-mode and L-mode
slices, tabulated on 14 psi_N nodes and rounded to two significant figures.
Each fixture takes the template of its phase (``IDA_REGIME``, ``DD_REGIME``)::

    sigma(psi_N) = max(interp(min(psi_N, 1), GRID, TEMPLATE[q]), FLOOR[q]) * |x(psi_N)|

piecewise linear in psi_N, the psi_N = 1 value held outside the LCFS, x the
synthetic profile itself.  The product is then rounded to 6 significant figures
(``SIGMA_SIG_FIGS``) so the files regenerate bit for bit on every platform (the 2 s.f. are on the
template).  What is written:

* the two IDA-layout files: ``n_e_err``, ``T_e_err``, ``T_12C6_err``,
  ``Zeff_err``, ``n_12C6_err``, overwritten in place (same shape and dtype);
  ``omega_tor_12C6_err`` has no template and is left as it is (it is not read);
* the dd: on every ``core_profiles.profiles_1d`` slice the IMAS error fields
  ``electrons.density_thermal_error_upper``, ``electrons.temperature_error_upper``,
  ``ion[].density_thermal_error_upper`` (main ion: the derived n_i template;
  carbon: the n_C template), ``ion[].temperature_error_upper`` (T_i) and
  ``zeff_error_upper``.  Upper only: the reader takes a symmetric envelope from
  it (``bouquet.io.imas.read_imas_sigmas``).

Nothing else changes.  The script checks that every other dataset of each IDA
file, and every non-error field of the dd, is bit-for-bit what it was, and
refuses (writes nothing) otherwise.  ``make_synthetic_fixtures.py`` calls the
same functions, so a full regeneration gives the same envelopes.
"""
import gzip
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bouquet.uncertainties import sigma_template  # noqa: E402

#: the template of each fixture's phase (FIXTURE_SPEC.md: no pedestal ->
#: L-mode; pedestal still building -> ramp; developed pedestal -> H-mode)
IDA_REGIME = {"diiid_profs_synthetic.cdf": "ramp",       # 1.2 s, pedestal forming
              # the same kinetic fit on the q of g_synthetic_pfast.geqdsk (the hybrid-route pair)
              "diiid_profs_synthetic_pfast.cdf": "ramp",
              "ida_synthetic_lmode.cdf": "lmode"}
DD_REGIME = {0.7: "lmode",              # ramp-up L-mode
             1.1: "ramp",               # L-H transition
             1.2: "ramp",               # pedestal forming
             1.4: "hmode_flattop",      # peak pedestal
             3.0: "hmode_flattop"}      # dense flat top
DD_NAME = "dd_synthetic.json.gz"

#: IDA dataset -> template quantity
IDA_QUANTITY = {"n_e": "ne", "T_e": "te", "T_12C6": "ti", "Zeff": "zeff",
                "n_12C6": "nc"}


def relative_sigma(psi_N, quantity, regime):
    """The template's relative sigma at ``psi_N`` (psi_N = 1 value held beyond 1)."""
    return sigma_template(psi_N, quantity, regime)


#: Every written sigma is rounded to this many significant figures, so the
#: fixtures regenerate bit for bit on any platform (the interpolation and the
#: product differ by an ULP between BLAS / numpy builds; 6 s.f. is far below
#: anything a draw can see and far above the ULP).
SIGMA_SIG_FIGS = 6


def stable(values):
    """``values`` rounded to :data:`SIGMA_SIG_FIGS` significant figures, as float64."""
    v = np.asarray(values, float)
    out = np.array([float(f"{x:.{SIGMA_SIG_FIGS}g}") for x in v.ravel()], dtype=np.float64)
    return out.reshape(v.shape)


def ida_errors(psi_N, data, regime):
    """``{"<name>_err": sigma}`` for the IDA datasets in ``data`` (1-D profiles)."""
    return {k + "_err": stable(relative_sigma(psi_N, q, regime) * np.abs(np.asarray(data[k], float)))
            for k, q in IDA_QUANTITY.items() if k in data}


def dd_regime(t):
    for tt, reg in DD_REGIME.items():
        if abs(float(t) - tt) < 1e-6:
            return reg
    raise KeyError(f"no sigma template assigned to the dd slice at t = {t!r} s")


def _cp_psi_N(cp1):
    psi = np.asarray(cp1["grid"]["psi"], float)
    return (psi - psi[0]) / (psi[-1] - psi[0])


def add_dd_errors(cp1, regime):
    """Add the IMAS ``*_error_upper`` fields to one core_profiles slice (in place)."""
    x = _cp_psi_N(cp1)

    def err(arr, q):
        return stable(relative_sigma(x, q, regime) * np.abs(np.asarray(arr, float))).tolist()

    el = cp1["electrons"]
    el["density_thermal_error_upper"] = err(el["density_thermal"], "ne")
    el["temperature_error_upper"] = err(el["temperature"], "te")
    for ion in cp1["ion"]:
        main = float(ion["element"][0]["z_n"]) == 1.0
        ion["density_thermal_error_upper"] = err(ion["density_thermal"], "ni" if main else "nc")
        ion["temperature_error_upper"] = err(ion["temperature"], "ti")
    if "zeff" in cp1:
        cp1["zeff_error_upper"] = err(cp1["zeff"], "zeff")
    return cp1


def _strip_errors(node):
    """A copy of a JSON tree without any ``*_error_upper`` / ``*_error_lower`` key."""
    if isinstance(node, dict):
        return {k: _strip_errors(v) for k, v in node.items()
                if not (k.endswith("_error_upper") or k.endswith("_error_lower"))}
    if isinstance(node, list):
        return [_strip_errors(v) for v in node]
    return node


def apply_ida(path, regime):
    """Overwrite the five templated ``*_err`` datasets of an IDA-layout file."""
    import h5py
    with h5py.File(path, "r") as f:
        before = {k: (f[k].dtype, f[k].shape, f[k][()].tobytes()) for k in f}
        x = np.asarray(f["psi_n"][()], float)
        data = {k: np.asarray(f[k][()], float)[0] for k in IDA_QUANTITY if k in f}
    new = ida_errors(x, data, regime)
    allowed = {k + "_err" for k in IDA_QUANTITY}
    if not set(new) <= allowed:
        raise ValueError(f"{os.path.basename(path)}: refusing to write {sorted(set(new) - allowed)}; "
                         f"only {sorted(allowed)} are templated")
    for k, v in new.items():
        dt, shp, _ = before[k]
        if shp != (1, v.size):
            raise ValueError(f"{os.path.basename(path)}: {k} has shape {shp}, expected (1, {v.size})")
    with h5py.File(path, "r+") as f:
        for k, v in new.items():
            f[k][...] = v[None, :].astype(before[k][0])
    with h5py.File(path, "r") as f:
        after = {k: (f[k].dtype, f[k].shape, f[k][()].tobytes()) for k in f}
    if set(after) != set(before):
        raise RuntimeError(f"{path}: the dataset list changed")
    for k in before:
        if k in new:
            continue
        if after[k] != before[k]:
            raise RuntimeError(f"{path}: {k} changed; only the templated *_err may")
    return new


def apply_dd(path):
    """Write the templated error fields on every core_profiles slice of the dd."""
    with gzip.open(path, "rt") as fh:
        dd = json.load(fh)
    ref = json.dumps(_strip_errors(dd), sort_keys=True)
    cp = dd["core_profiles"]
    for t, cp1 in zip(cp["time"], cp["profiles_1d"]):
        add_dd_errors(cp1, dd_regime(t))
    if json.dumps(_strip_errors(dd), sort_keys=True) != ref:
        raise RuntimeError(f"{path}: a non-error field changed")
    write_dd(dd, path)
    with gzip.open(path, "rt") as fh:
        back = json.load(fh)
    if json.dumps(_strip_errors(back), sort_keys=True) != ref:
        raise RuntimeError(f"{path}: a non-error field changed on the round trip")
    return dd


def write_dd(dd, path):
    """gzip JSON with a fixed header (no file name, mtime 0): reproducible bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(json.dumps(dd).encode("utf-8"))
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())


def _report(path, regime, quantities, rel_of):
    cols = (0.0, 0.5, 0.9, 1.0)
    print(f"  {os.path.basename(path)} [{regime}]")
    for q in quantities:
        r = rel_of(q)
        print(f"    {q:<8s}" + "  ".join(f"{100 * v:5.1f}%" for v in r(cols)))


def main(d=HERE):
    for name, reg in IDA_REGIME.items():
        p = os.path.join(d, name)
        apply_ida(p, reg)
        _report(p, reg, IDA_QUANTITY,
                lambda k, reg=reg: (lambda xs: [relative_sigma(x, IDA_QUANTITY[k], reg) for x in xs]))
    p = os.path.join(d, DD_NAME)
    dd = apply_dd(p)
    for t in dd["core_profiles"]["time"]:
        print(f"  {DD_NAME} t = {t:g} s -> {dd_regime(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else HERE))
