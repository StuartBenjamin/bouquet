"""The synthetic fixtures carry the radial sigma templates of their phase.

tests/data/apply_sigma_templates.py writes them; these tests check that each
fixture's relative sigma IS its template (interpolated on the file's own grid)
and that re-applying the script to the committed files changes nothing, i.e.
the envelopes are regenerable from tracked code.
"""
import gzip
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest

from bouquet.uncertainties import sigma_template

h5py = pytest.importorskip("h5py")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, DATA)
import apply_sigma_templates as ast  # noqa: E402

#: the assignment (FIXTURE_SPEC.md): phase -> template
IDA_FILES = {"diiid_profs_synthetic.cdf": "ramp", "ida_synthetic_lmode.cdf": "lmode"}
DD_SLICES = {0.7: "lmode", 1.1: "ramp", 1.2: "ramp", 1.4: "hmode_flattop",
             3.0: "hmode_flattop"}
IDA_Q = {"n_e": "ne", "T_e": "te", "T_12C6": "ti", "Zeff": "zeff", "n_12C6": "nc"}


def _rel(sig, x):
    sig, x = np.asarray(sig, float), np.abs(np.asarray(x, float))
    ok = x > 0
    return ok, sig[ok] / x[ok]


def test_the_script_assigns_the_same_templates():
    assert ast.IDA_REGIME == IDA_FILES
    assert ast.DD_REGIME == DD_SLICES


@pytest.mark.parametrize("name", sorted(IDA_FILES))
def test_ida_fixture_relative_sigma_is_its_template(name):
    reg = IDA_FILES[name]
    with h5py.File(os.path.join(DATA, name), "r") as f:
        x = f["psi_n"][()]
        assert x.max() > 1.0                           # the SOL nodes hold the psi_N = 1 value
        for k, q in IDA_Q.items():
            ok, rel = _rel(f[k + "_err"][0], f[k][0])
            assert ok.sum() >= 140, (name, k)
            np.testing.assert_allclose(rel, sigma_template(x[ok], q, reg), rtol=1e-12,
                                       err_msg=f"{name} {k}")
            assert np.ptp(rel) > 0.02, (name, k)       # not a flat fraction any more
        # omega has no template: the unread envelope stays the constant fraction
        ok, rel = _rel(f["omega_tor_12C6_err"][0], f["omega_tor_12C6"][0])
        np.testing.assert_allclose(rel, 0.10, rtol=1e-12)


def _dd():
    with gzip.open(os.path.join(DATA, "dd_synthetic.json.gz"), "rt") as fh:
        return json.load(fh)


def test_dd_slices_carry_their_template_as_imas_error_fields():
    cp = _dd()["core_profiles"]
    assert [round(t, 6) for t in cp["time"]] == sorted(DD_SLICES)
    for t, c in zip(cp["time"], cp["profiles_1d"]):
        reg = DD_SLICES[round(t, 6)]
        psi = np.asarray(c["grid"]["psi"], float)
        x = (psi - psi[0]) / (psi[-1] - psi[0])
        checks = [(c["electrons"], "density_thermal", "ne"),
                  (c["electrons"], "temperature", "te"), (c, "zeff", "zeff")]
        for ion in c["ion"]:
            main = float(ion["element"][0]["z_n"]) == 1.0
            checks += [(ion, "density_thermal", "ni" if main else "nc"),
                       (ion, "temperature", "ti")]
        for node, fld, q in checks:
            ok, rel = _rel(node[fld + "_error_upper"], node[fld])
            assert ok.all(), (t, fld)
            np.testing.assert_allclose(rel, sigma_template(x, q, reg), rtol=1e-12,
                                       err_msg=f"t={t} {fld} ({q}, {reg})")
            assert fld + "_error_lower" not in node     # symmetric: upper only


def test_reapplying_the_script_reproduces_the_committed_fixtures(tmp_path):
    names = list(IDA_FILES) + [ast.DD_NAME]
    for n in names:
        shutil.copy(os.path.join(DATA, n), tmp_path / n)
    ast.main(str(tmp_path))

    def digest(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    for n in IDA_FILES:      # the HDF5 bytes: an in-place rewrite of the same values
        with h5py.File(os.path.join(DATA, n), "r") as a, h5py.File(tmp_path / n, "r") as b:
            assert set(a) == set(b)
            for k in a:
                assert a[k][()].tobytes() == b[k][()].tobytes(), (n, k)
    assert digest(os.path.join(DATA, ast.DD_NAME)) == digest(tmp_path / ast.DD_NAME)


def test_the_script_refuses_a_changed_profile(tmp_path, monkeypatch):
    """Only the five templated envelopes may be written; anything else is refused."""
    n = "diiid_profs_synthetic.cdf"
    shutil.copy(os.path.join(DATA, n), tmp_path / n)
    real = ast.ida_errors

    def bad(x, data, regime):
        out = real(x, data, regime)
        out["n_e"] = out["n_e_err"]                    # a profile, not an envelope
        return out
    before = open(tmp_path / n, "rb").read()
    monkeypatch.setattr(ast, "ida_errors", bad)
    with pytest.raises(ValueError, match="refusing to write"):
        ast.apply_ida(str(tmp_path / n), "ramp")
    assert open(tmp_path / n, "rb").read() == before    # nothing written
