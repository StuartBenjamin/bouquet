"""The radial sigma templates and ``synthetic_ida_sigma`` (bouquet.uncertainties).

The template tables are the rounded (2 s.f.) median relative envelopes; these
tests pin their structure, the interpolation rule and the helper's values.
"""
import numpy as np
import pytest

from bouquet import sigma_template, synthetic_ida_sigma
from bouquet.uncertainties import (DEFAULT_SIGMA_TEMPLATE, SIGMA_TEMPLATE_FLOOR,
                                   SIGMA_TEMPLATE_GRID, SIGMA_TEMPLATES)

_Q = ("ne", "te", "ti", "zeff", "nc", "ni")


def _two_sf(v):
    return float(f"{v:.2g}") == v


def test_tables_are_14_nodes_two_significant_figures_above_the_floor():
    assert len(SIGMA_TEMPLATE_GRID) == 14
    assert SIGMA_TEMPLATE_GRID[0] == 0.0 and SIGMA_TEMPLATE_GRID[-1] == 1.0
    assert np.all(np.diff(SIGMA_TEMPLATE_GRID) > 0)
    assert set(SIGMA_TEMPLATES) == {"hmode_flattop", "ramp", "lmode"}
    for reg, tab in SIGMA_TEMPLATES.items():
        assert set(tab) == set(_Q), reg
        for q, row in tab.items():
            assert len(row) == 14, (reg, q)
            assert all(_two_sf(v) for v in row), (reg, q)
            assert min(row) > SIGMA_TEMPLATE_FLOOR[q], (reg, q)
    assert DEFAULT_SIGMA_TEMPLATE == "hmode_flattop"


@pytest.mark.parametrize("reg", sorted(SIGMA_TEMPLATES))
@pytest.mark.parametrize("q", _Q)
def test_template_is_the_node_table_linear_between_nodes_held_outside(reg, q):
    row = np.asarray(SIGMA_TEMPLATES[reg][q])
    g = np.asarray(SIGMA_TEMPLATE_GRID)
    np.testing.assert_array_equal(sigma_template(g, q, reg), row)
    mid = 0.5 * (g[:-1] + g[1:])
    np.testing.assert_allclose(sigma_template(mid, q, reg), 0.5 * (row[:-1] + row[1:]),
                               rtol=1e-12)
    np.testing.assert_array_equal(sigma_template([1.0, 1.1, 1.2], q, reg), row[-1])
    assert sigma_template(-0.1, q, reg) == row[0]


def test_aliases_and_refusals():
    x = np.linspace(0, 1, 7)
    for a, k in (("n_e", "ne"), ("T_12C6", "ti"), ("Z_eff", "zeff"), ("n_12C6", "nc"),
                 ("n_i", "ni"), ("density", "ne")):
        np.testing.assert_array_equal(sigma_template(x, a), sigma_template(x, k))
    with pytest.raises(ValueError, match="Unknown quantity"):
        sigma_template(x, "omega")
    with pytest.raises(ValueError, match="Unknown sigma template"):
        sigma_template(x, "ne", "hmode")


def test_synthetic_ida_sigma_is_the_flattop_template():
    """The helper's values (the flat-top template; the sine-basis constants it
    replaced were 1.2-3x lower) at psi_N 0 / 0.5 / 0.9 / 1.0."""
    x = [0.0, 0.5, 0.9, 1.0]
    want = {"ne": [0.057, 0.028, 0.058, 0.15], "te": [0.052, 0.078, 0.13, 0.19],
            "ti": [0.089, 0.087, 0.14, 0.31], "zeff": [0.16, 0.087, 0.12, 0.41],
            "nc": [0.16, 0.097, 0.12, 0.45], "ni": [0.17, 0.081, 0.097, 0.22]}
    for q, v in want.items():
        np.testing.assert_array_equal(synthetic_ida_sigma(x, q), v)
    xs = np.linspace(0.0, 1.2, 50)
    for q in _Q:
        np.testing.assert_array_equal(synthetic_ida_sigma(xs, q),
                                      sigma_template(xs, q, "hmode_flattop"))
    # ni is its own (derived) envelope now, not an alias of ne
    assert not np.array_equal(synthetic_ida_sigma(xs, "ni"), synthetic_ida_sigma(xs, "ne"))
    with pytest.raises(ValueError):
        synthetic_ida_sigma(xs, "q")


def test_flattop_uniform_psi_means():
    """The uniform-in-psi_N means of the default shapes (about the volume
    average; docs/CHANGES_SUMMARY.md): n_e 4.0, T_e 8.4, n_i 9.6, T_i 9.7,
    Z_eff 11 %."""
    x = np.linspace(0.0, 1.0, 20001)
    got = {q: float(np.trapezoid(sigma_template(x, q), x)) for q in ("ne", "te", "ni", "ti", "zeff")}
    want = {"ne": 0.040, "te": 0.084, "ni": 0.096, "ti": 0.097, "zeff": 0.11}
    for q, v in want.items():
        assert float(f"{got[q]:.2g}") == v, (q, got[q])
