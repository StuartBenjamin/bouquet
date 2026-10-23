"""Replaying archived swb draws (Bouquet.replay_swb): the kinetic noise
inversion (kinetic_sampler.kinetic_noise + sampling.ReplayNoise) and the
replay draw method (swb_draws.SwbReplayDraws).  Solver-free."""
import json
import types

import h5py
import numpy as np
import pytest
from scipy.linalg import solve_triangular

from bouquet import swb_draws as SD
from bouquet.kinetic_sampler import KineticBase, sample_kinetics, kinetic_noise
from bouquet.physics import ELEMENTARY_CHARGE as EC
from bouquet.sampling import (GPRProfilePerturber, ReplayExhausted, ReplayNoise,
                              generate_perturbed_GPR, make_rng)

PSI = np.linspace(0.0, 1.0, 65)
NE = 5e19 * (1.0 - 0.8 * PSI)          # monotonic with a margin: few redraws
TE = 3e3 * (1.0 - 0.9 * PSI)
TI = 2.5e3 * (1.0 - 0.9 * PSI)
NI = 0.8 * NE


def _thermal(d):
    return np.trapezoid(EC * (d.ne * d.te + d.ni * d.ti), PSI)


def _base(shift=0.0, wiggle=0.0, zeff_dne=None):
    """A base; ``shift`` moves the mean, ``wiggle`` the sigma (as a regrid would)."""
    f = 0.03 * (1.0 + wiggle * np.sin(23.0 * PSI) ** 2)
    ne, ti = NE * (1.0 + shift * PSI), TI * (1.0 - shift * PSI)
    return KineticBase(
        psi_kin=PSI, ne=ne, te=TE, ni=NI, ti=ti, sigma_ne=f * ne, sigma_te=f * TE,
        sigma_ni=f * NI, sigma_ti=f * ti, n_ls=0.5, t_ls=0.5,
        aux_sigmas={"zeff": 0.1 + 0.0 * PSI, "e_r": 1e3 + 0.0 * PSI},
        aux_baselines={"zeff": np.full(PSI.size, 1.8), "e_r": 1e4 * (1.0 - PSI)},
        aux_length_scales={"zeff": 0.4}, Z_imp=6.0, zeff_dne=zeff_dne, clips=True)


class _Counting(np.random.Generator):
    """Counts the normals a draw consumed (one per GPR channel when nothing
    was redrawn)."""

    def __init__(self, seed):
        super().__init__(np.random.PCG64(seed))
        self.n = 0

    def standard_normal(self, *a, **k):
        self.n += 1
        return super().standard_normal(*a, **k)


def _draws_equal(a, b, rtol):
    for k in ("ne", "te", "ni", "ti"):
        np.testing.assert_allclose(getattr(a, k), getattr(b, k), rtol=rtol, atol=0.0)
    assert a.aux.keys() == b.aux.keys()
    for k in b.aux:
        np.testing.assert_allclose(a.aux[k], b.aux[k], rtol=rtol,
                                   atol=rtol * np.max(np.abs(b.aux[k])))


def test_a_draw_is_the_mean_plus_the_factor_times_its_normals():
    sig = 0.05 * (1.0 - 0.5 * PSI)
    got = generate_perturbed_GPR(PSI, TE / TE[0], sig, length_scale=0.3, rng=make_rng(4))
    z = make_rng(4).standard_normal((PSI.size, 1))
    P = GPRProfilePerturber(length_scale=0.3)
    L = P.factor(PSI, sig)
    d = L @ z - P.jitter(sig) * solve_triangular(L, z, lower=True, trans="T", check_finite=False)
    np.testing.assert_array_equal(got, TE / TE[0] + d[:, 0])
    assert GPRProfilePerturber(length_scale=0.3).factor(PSI, 0.0 * PSI) is None


@pytest.mark.parametrize("zeff_dne", [None, 2e-21 * np.ones_like(PSI)])
def test_the_noise_of_a_draw_remakes_it(zeff_dne):
    b = _base(zeff_dne=zeff_dne)
    for seed in range(6):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b), p_thresh=0.5)
        noise = ReplayNoise(kinetic_noise(b, d))      # a filtered draw: to ~sqrt(eps)
        _draws_equal(sample_kinetics(b, noise, _thermal, _thermal(b), p_thresh=0.5), d, 1e-5)
        assert noise.remaining == 0


def test_on_a_new_base_the_noise_gives_that_base_s_own_draw():
    old, new = _base(), _base(shift=0.02, wiggle=0.1)
    n_ch = 5                                    # ne, te, zeff, ti, e_r: no redraws
    checked = 0
    for seed in range(20):
        r_old, r_new = _Counting(seed), _Counting(seed)
        d_old = sample_kinetics(old, r_old, _thermal, _thermal(old), p_thresh=0.5)
        d_new = sample_kinetics(new, r_new, _thermal, _thermal(new), p_thresh=0.5)
        if r_old.n == r_new.n == n_ch:          # the same normals made both
            replay = ReplayNoise(kinetic_noise(old, d_old))
            _draws_equal(sample_kinetics(new, replay, _thermal, _thermal(new),
                                         p_thresh=0.5), d_new, 1e-3)
            checked += 1
    assert checked >= 10


def test_replay_noise_refuses_past_the_record():
    noise = ReplayNoise([np.zeros((3, 1))])
    with pytest.raises(ReplayExhausted):
        noise.standard_normal((4, 1))           # another shape
    noise.standard_normal((3, 1))
    with pytest.raises(ReplayExhausted):
        noise.standard_normal((3, 1))           # past the record


# ---- SwbReplayDraws on a hand-built archive --------------------------------
_MYGS = types.SimpleNamespace(flux_integral=lambda x, y: np.trapezoid(y, x))
SEED = 1e6 * (1.0 - PSI ** 1.5) ** 1.5


def _inputs(b):
    return dict(
        psi_N=PSI, pressure=EC * (b.ne * b.te + b.ni * b.ti), ne=b.ne, te=b.te,
        ni=b.ni, ti=b.ti, sigma_ne=b.sigma_ne, sigma_te=b.sigma_te,
        sigma_ni=b.sigma_ni, sigma_ti=b.sigma_ti, n_ls=b.n_ls, t_ls=b.t_ls,
        j_ls=0.25, Zeff=np.full(PSI.size, 1.8), coord="psi_n", p_thresh=0.5,
        psi_N_kinetic=PSI, p_fast=None, z_fast=None, z2_fast=None,
        zeff_includes_fast=False, Z_imp=6.0, aux_sigmas=b.aux_sigmas,
        aux_baselines=b.aux_baselines, aux_length_scales=b.aux_length_scales,
        ni_from_zeff=True, zeff_dne=b.zeff_dne, isolate_edge_jbs=False,
        kinetic_clips=True)


def _recipe(log):
    def recipe(kin, j_seed, warm_eq=None):
        log.append((kin, np.array(j_seed)))
        j_ind = 1.04 * np.asarray(j_seed)
        return {"jphi_ind": j_ind, "jphi_bs": 1e5 + 0.0 * j_ind, "jphi_total": j_ind + 1e5}
    return recipe


def _archive(path, b, outs):
    """Draws ``{count: swb_draw output}`` archived as an swb run writes them."""
    with h5py.File(path, "w") as hf:
        hf.attrs["schema_version"] = 3
        g = hf.create_group("scan/1235/_baseline")
        for k, v in dict(psi_N_kinetic=PSI, n_e=b.ne, T_e=b.te, n_i=b.ni, T_i=b.ti,
                         sigma_ne=b.sigma_ne, sigma_te=b.sigma_te, sigma_ni=b.sigma_ni,
                         sigma_ti=b.sigma_ti).items():
            g[k] = v
        for k in b.aux_sigmas:
            g[f"aux_{k}"], g[f"sigma_aux_{k}"] = b.aux_baselines[k], b.aux_sigmas[k]
        for count, out in outs.items():
            g = hf.create_group(f"scan/1235/{count}")
            for k, v in zip(("n_e", "T_e", "n_i", "T_i"), out[:4]):
                g[k] = v
            for k, v in out[6]["aux"].items():
                g[f"aux_{k}"] = v
            g["jphi_ind"] = out[6]["jphi_ind"]
            g.attrs["swb_alpha"] = out[6]["swb_alpha"]
            g.attrs["kinetic_sampler_json"] = json.dumps(out[6]["kinetic_sampler"])
            g.attrs["swb_jind_resamples"] = out[6]["jind_resamples"]
            g.attrs["swb_jind_clipped"] = out[6]["jind_clipped"]


def test_a_replayed_draw_is_the_archived_draw(tmp_path):
    b, log = _base(), []
    draws = SD.SwbDraws(_recipe(log), SEED, 0.1 * SEED)
    outs = {c: draws.draw(_MYGS, make_rng(c), 1.0, c, inputs=_inputs(b)) for c in (0, 2)}
    _archive(tmp_path / "src.h5", b, outs)
    replay = SD.SwbReplayDraws(_recipe(log), SEED, 0.1 * SEED,
                               source=str(tmp_path / "src.h5"), scan_key="1235", warm=False)
    assert replay.stamp["swb_replay_source"] == str(tmp_path / "src.h5")
    for c in (0, 2):
        got = replay.draw(_MYGS, None, 1.0, c, inputs=_inputs(b))
        for i in range(4):
            np.testing.assert_allclose(got[i], outs[c][i], rtol=1e-5, atol=0.0)
        np.testing.assert_allclose(log[-1][1], log[c // 2][1], rtol=1e-13)   # its j_seed
        assert got[6]["jind_resamples"] == outs[c][6]["jind_resamples"]
    with pytest.raises(SD.SwbReplayRefused) as e:
        replay.draw(_MYGS, None, 1.0, 1, inputs=_inputs(b))        # not archived
    assert replay.rejection_reason(e.value, "perturb") == SD.SWB_REPLAY_REJECTION


def _source(tmp_path):
    """A hand-built swb archive with draws 0 and 2 (count 1 refused)."""
    b = _base()
    draws = SD.SwbDraws(_recipe([]), SEED, 0.1 * SEED)
    _archive(tmp_path / "src.h5", b,
             {c: draws.draw(_MYGS, make_rng(c), 1.0, c, inputs=_inputs(b)) for c in (0, 2)})
    return str(tmp_path / "src.h5")


def test_a_replay_starts_its_draws_from_solve_b_as_prepared(tmp_path, monkeypatch):
    from _swb import fake_oft, swb_cfg
    from bouquet.swb import build_swb_context
    fake_oft(monkeypatch)
    coils = {"F1A": 1.0}
    bl = types.SimpleNamespace(
        coil_reg_target=dict(coils), ip_closure={}, ohm_scale=1.0, swb_seed_profile=SEED,
        jphi_total=SEED + 1e5, psi_N=PSI, jphi_pd=None, jphi_saw_out=None, jphi_saw_input=None,
        swb_baseline=dict(coils=dict(coils), swb_jbs_convention="c", swb_jbs_conversion="v"))
    mygs = types.SimpleNamespace(get_coil_currents=lambda: (dict(coils), None))

    def no_solve(*a, **k):
        raise AssertionError("a replay re-solved solve B")
    bq = types.SimpleNamespace(baseline=bl, config=swb_cfg(), mygs=mygs, _swb_solve=no_solve)
    env = {"sigma_jphi": 0.1 * np.abs(bl.jphi_total)}
    replay = dict(source=_source(tmp_path), scan_key="1235", warm=True)
    m = build_swb_context(bq, env, replay)
    assert isinstance(m, SD.SwbReplayDraws) and m.counts == {0, 2}
    assert m.stamp["swb_jbs_convention"] == "c" and m.stamp["swb_replay_warm"] is True
    mygs.get_coil_currents = lambda: ({"F1A": 2.0}, None)        # mygs moved off solve B
    with pytest.raises(RuntimeError, match="not on solve B"):
        build_swb_context(bq, env, replay)


def test_replay_swb_warm_starts_the_baseline_only_while_it_runs(tmp_path, monkeypatch):
    from bouquet import archive as AR
    from bouquet.swb import SWB_REPLAY_URF, SwbBaseline
    src = _source(tmp_path)
    monkeypatch.setattr(AR.BaselineView, "load_state",
                        lambda self, mygs=None: types.SimpleNamespace(copy_eq=lambda: "archived B"))
    seen = {}
    ns = types.SimpleNamespace(mygs=types.SimpleNamespace(settings=types.SimpleNamespace(urf=0.2)),
                               config=types.SimpleNamespace(generation=types.SimpleNamespace(scan_key="1235")))
    ns.prepare_replay = types.MethodType(SwbBaseline.prepare_replay, ns)
    ns.prepare_baseline = lambda: seen.update(prep=(ns.mygs.settings.urf, getattr(ns, "_swb_warm_eq", None)))
    ns.generate = lambda n, replay: seen.update(gen=(n, replay, ns.mygs.settings.urf)) or "generated"
    assert SwbBaseline.replay_swb(ns, src) == "generated"
    assert seen["prep"] == (SWB_REPLAY_URF, "archived B")
    assert seen["gen"] == (3, dict(source=src, scan_key="1235", warm=True), SWB_REPLAY_URF)
    assert ns.mygs.settings.urf == 0.2 and ns._swb_warm_eq is None
    SwbBaseline.replay_swb(ns, src, warm=False)                  # cold: the run's own settings
    assert seen["prep"] == (0.2, None)
    seen.clear()                                                 # prepared: no second baseline solve
    assert SwbBaseline.replay_swb(ns, src, prepared=True) == "generated"
    assert "prep" not in seen and seen["gen"][2] == SWB_REPLAY_URF and ns.mygs.settings.urf == 0.2
