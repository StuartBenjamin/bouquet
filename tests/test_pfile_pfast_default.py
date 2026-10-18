"""Decision B (2026-10-17): the g-file + p-file route takes the p-file's own
``pb`` block as ``p_fast`` by default, and every g-file-route reconstruction
carries a REPORT-ONLY pressure-fidelity stamp.

Fast (no solver): the reader, the input resolution and the stamp's own
arithmetic.  The end-to-end beta_N of the shipped total-pressure pair
(g_synthetic_pfast.geqdsk + p_synthetic.peqdsk) is measured with the solver
(tests/data/FIXTURE_SPEC.md section 3).
"""

import os
import types
import warnings

import numpy as np
import pytest

from bouquet import baseline as B
from bouquet.config import FixedComponentsConfig
from bouquet.io.pfile import read_pfile

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PFILE = os.path.join(_DATA, "p_synthetic.peqdsk")


def _kin(path):
    return B._load_kinetic_profiles(types.SimpleNamespace(profiles_path=path))


def _without_block(src, dst, name):
    """A copy of the p-file *src* with its *name* block removed."""
    lines = open(src).read().splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) >= 3 and head[1] == "psinorm" and head[2].startswith(
                name + "("):
            i += 1 + int(head[0])
            continue
        out.append(lines[i])
        i += 1
    open(dst, "w").write("".join(out))
    return dst


def test_the_shipped_pfile_pb_is_parsed():
    kin = _kin(PFILE)
    pf = read_pfile(PFILE)
    np.testing.assert_array_equal(kin["pb"], np.asarray(pf.pb, dtype=float))
    np.testing.assert_array_equal(kin["pb_psi_N"],
                                  np.asarray(pf["pb"]["psinorm"], dtype=float))
    assert kin["pb"][0] == pytest.approx(58.0, rel=1e-6)      # kPa on axis


def test_p_fast_defaults_to_the_pfile_pb_and_is_stamped():
    kin = _kin(PFILE)
    vals, x, stamp = B.resolve_p_fast_input(FixedComponentsConfig(), kin,
                                            PFILE, None)
    np.testing.assert_array_equal(vals, 1e3 * kin["pb"])
    np.testing.assert_array_equal(x, kin["pb_psi_N"])
    assert stamp["p_fast_source"] == "pfile_pb"
    assert stamp["coord"] == "psi_n" and stamp["pfile_has_pb"] is True
    assert stamp["p_fast_axis_Pa"] == pytest.approx(58.0e3, rel=1e-6)
    # on the kinetic grid, the route's own two-step resolution
    pk = B._resolve_fixed(vals, x, kin["psi_N"])
    assert pk[0] == pytest.approx(58.0e3, rel=1e-6)
    # in a Phi_N run the psinorm grid goes through the map
    pmap = (np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11) ** 2)
    _, xr, _ = B.resolve_p_fast_input(FixedComponentsConfig(), kin, PFILE,
                                      pmap)
    np.testing.assert_allclose(xr, np.interp(kin["pb_psi_N"], *pmap))


def test_an_explicit_p_fast_wins_and_two_sources_warn_once():
    kin = _kin(PFILE)
    B._PB_TWO_SOURCES_WARNED.discard(PFILE)
    fc = FixedComponentsConfig()
    fc.p_fast = np.full(5, 1.0e3)
    fc.psi_N = np.linspace(0.0, 1.0, 5)
    with pytest.warns(UserWarning, match="fixed_components.p_fast"):
        vals, x, stamp = B.resolve_p_fast_input(fc, kin, PFILE, None)
    assert vals is fc.p_fast
    np.testing.assert_array_equal(x, fc.psi_N)
    assert stamp == dict(p_fast_source="fixed_components", pfile_has_pb=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        B.resolve_p_fast_input(fc, kin, PFILE, None)          # once only


def test_a_pfile_without_pb_behaves_as_before(tmp_path):
    p = _without_block(PFILE, str(tmp_path / "p_nopb.peqdsk"), "pb")
    kin = _kin(p)
    assert "pb" not in kin and "pb_psi_N" not in kin
    ref = _kin(PFILE)
    for k in ("ne", "te", "ni", "ti", "Zeff"):
        np.testing.assert_array_equal(kin[k], ref[k])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vals, x, stamp = B.resolve_p_fast_input(FixedComponentsConfig(), kin,
                                                p, None)
    assert vals is None and x is None
    assert stamp == dict(p_fast_source="none", pfile_has_pb=False)
    # an explicit p_fast on a p-file without pb: no two-source warning
    fc = FixedComponentsConfig()
    fc.p_fast = np.full(5, 1.0e3)
    fc.psi_N = np.linspace(0.0, 1.0, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, stamp = B.resolve_p_fast_input(fc, kin, p, None)
    assert stamp == dict(p_fast_source="fixed_components", pfile_has_pb=False)


def test_the_pressure_fidelity_stamp_reports_and_warns_never_gates():
    psi = np.linspace(0.001, 0.999, 65)
    p_in = 9.0e4 * (1.0 - psi ** 2) + 1.0e3
    # a dropped beam: about the -63 % on-axis deficit of the pb-less pair
    with pytest.warns(UserWarning, match="report-only"):
        r = B.pressure_fidelity_record(psi, 0.37 * p_in, p_in)
    assert r["report_only"] is True and r["axis_deficit_flag"] is True
    assert r["axis_rel_err_pct"] == pytest.approx(-63.0, rel=1e-9)
    assert r["rel_rms"] == pytest.approx(0.63, rel=1e-9)
    assert r["warn_band_pct"] == B.PRESSURE_FIDELITY_WARN_PCT == 10.0
    # within the band (and an excess, not a deficit): recorded, no warning
    # warn=False (inside the captured reconstruction): recorded, the warning
    # left to warn_pressure_fidelity where the summary is printed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = B.pressure_fidelity_record(psi, 0.37 * p_in, p_in, warn=False)
    assert r["axis_deficit_flag"] is True
    with pytest.warns(UserWarning, match="fixed_components.p_fast"):
        B.warn_pressure_fidelity(r)
    for f in (0.95, 1.30):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = B.pressure_fidelity_record(psi, f * p_in, p_in)
        assert r["axis_deficit_flag"] is False
        assert r["axis_rel_err_pct"] == pytest.approx(100.0 * (f - 1.0))


def test_the_input_record_has_its_own_baseline_field():
    """Review of B (C4): the g-file route's p_fast INPUT record lives on
    Baseline.p_fast_source, appended with a None default; Baseline.p_fast_meta
    stays the IMAS route's reduction record (rule / basis / requested /
    warned), so a summary over mixed baselines never meets two schemas.  Both
    g-file baselines (legacy and engine) fill the new field, never
    p_fast_meta."""
    import ast
    import dataclasses
    import inspect
    import textwrap
    from bouquet import Baseline
    from bouquet import engine as E
    fields = dataclasses.fields(Baseline)
    assert fields[-1].name == "p_fast_source"
    assert fields[-1].default is None
    for fn in (B._resolve_reconstruction, E._gfile_baseline):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        kws = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "Baseline"
               for k in n.keywords}
        assert "p_fast_source" in kws, fn.__name__
        assert "p_fast_meta" not in kws, fn.__name__
