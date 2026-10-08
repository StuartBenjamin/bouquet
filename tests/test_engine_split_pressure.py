"""engine_split_pressure: which archived component carries the
pressure-driven current p'(<R> - F^2<1/R>/<B^2>).  Solver-free."""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.config import GenerationConfig
from bouquet.engine import (ENGINE_SPLIT_CONVENTIONS, _split,
                            composed_factor, engine_settings,
                            split_pressure_term)


def _geom(n=21):
    psi = np.linspace(0.0, 1.0, n)
    return dict(psi_N=psi, F=np.full(n, 3.4), inv_R=np.full(n, 0.6),
                B2=np.full(n, 4.0), R_avg=np.full(n, 1.7),
                pprime=-1e4 * (1.0 - psi))


def _eng(mode, n=21):
    g = _geom(n)
    st = SimpleNamespace(geom=g, request=np.linspace(2e6, 1e5, n),
                         lambda_bs=np.full(n, 0.3))
    c = SimpleNamespace(jB_fix_parts=dict(nbi=np.full(n, 0.1),
                                          rf=np.full(n, 0.05)))
    eng = SimpleNamespace(c=c, s=dict(split_pressure=mode),
                          delivered_closure=dict(out=dict(s_bs=1.2)))
    return eng, dict(state=st)


def test_inductive_is_the_original_split():
    eng, res = _eng("inductive")
    R, j_ind, j_bs, j_nbi, j_rf = _split(eng, res)
    g = res["state"].geom
    kap = composed_factor(g)
    np.testing.assert_array_equal(j_bs, 1.2 * kap * res["state"].lambda_bs)
    np.testing.assert_array_equal(j_ind, R - j_bs - j_nbi - j_rf)


def test_bootstrap_moves_exactly_pG_from_inductive_to_bootstrap():
    (eb, rb), (ei, ri) = _eng("bootstrap"), _eng("inductive")
    _, ind_b, bs_b, *_ = _split(eb, rb)
    _, ind_i, bs_i, *_ = _split(ei, ri)
    pg = split_pressure_term(rb["state"].geom)
    np.testing.assert_allclose(bs_b - bs_i, pg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ind_i - ind_b, pg, rtol=0, atol=1e-9)
    assert np.max(np.abs(pg)) > 0.0


def test_settings_default_and_refusal():
    s = engine_settings(GenerationConfig(reconstruction_engine="unified"))
    assert s["split_pressure"] == "bootstrap"
    s = engine_settings(GenerationConfig(reconstruction_engine="unified",
                                         engine_split_pressure="inductive"))
    assert s["split_pressure"] == "inductive"
    with pytest.raises(ValueError):
        engine_settings(GenerationConfig(reconstruction_engine="unified",
                                         engine_split_pressure="ohmic"))
    assert split_pressure_term(_geom(), "inductive") == 0.0
    assert set(ENGINE_SPLIT_CONVENTIONS) == {"bootstrap", "inductive"}
