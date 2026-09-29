"""closure_channel="structured" with MSE pitch angles as a third measurement.

No solver and no machine data anywhere.  The "equilibrium" is an analytic
large-aspect-ratio cylinder: a current profile j(psi_N) on psi_N = (r/a)^2 is
renormalised to the target Ip (as the real forward solve renormalises the
shape it is handed), its enclosed current gives B_theta = mu0 I(r) / (2 pi r),
and B_phi = B0 R0 / R.  tan(gamma) at outboard-midplane chords is then a
smooth NONLINEAR function of the structured coefficients -- nonlinear through
the Ip renormalisation, the 1/r and the denominator -- which is what the
finite-difference linearisation has to cope with.

What is proved here:

* the data block (``bouquet.mse``): validation, the fit-weight fold, the
  E_r double-count refusal, the forward formula and the orientation choice;
* the solvers: with an MSE linear model they return the EXACT minimiser of
  prior + chi2 under their constraints (checked against an independent
  closed-form KKT solve), Ip stays exact on the hard channel, the one-sided
  prior still settles, a vanishing MSE weight recovers the no-MSE answer, and
  ``mse_lin=None`` changes nothing;
* the linearisation: the forward-difference Jacobian is first-order accurate
  (error halves with the step) against a central-difference reference, the
  outer loop's recorded linearisation residual is the true one, a LINEAR
  forward model is solved exactly, and on the nonlinear cylinder the achieved
  objective and chi2 both fall;
* the run.py stage end to end on a stubbed solver: records, solve count,
  refusal and fallback behaviour, the required flag, and a JSON-serialisable
  ip_closure record.
"""
import json
import types
import warnings

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from bouquet.config import GenerationConfig
from bouquet.mse import (MSE_ORIENTATION_DCHI2, MSEDataUnusable, mse_chi2,
                         mse_chords, mse_equilibrium_orientation,
                         mse_field_at, mse_orientation, mse_orientation_check,
                         mse_tan_gamma)
from bouquet.utils import (MSE_FLAG_PREFIX, Ip_fsa_weights,
                           close_ip_structured, close_ip_structured_soft,
                           structured_basis_eval, structured_mse_jacobian,
                           structured_mse_linear_model, structured_mse_outer,
                           structured_objective_no_mse,
                           STRUCTURED_WEIGHTS_PHYSICS)

_N = 201
_MU0 = 4.0e-7 * np.pi
R0, AMINOR, B0 = 1.7, 0.6, 2.0


# ---------------------------------------------------------------------------
#  fixtures: the same closable hybrid the structured-closure tests use
# ---------------------------------------------------------------------------
def _geom():
    psi = np.linspace(0.01, 0.99, _N)
    r = AMINOR * np.sqrt(psi)
    R_avg = R0 + 0.1 * r ** 2 / AMINOR
    inv_R = (1.0 / R0) * (1.0 + 0.05 * (r / AMINOR) ** 2)
    inv_R2 = inv_R ** 2 * (1.0 + 0.02 * (r / AMINOR) ** 2)
    dV_dpsi = 4.0 * np.pi ** 2 * R0 * r * (AMINOR / (2.0 * np.sqrt(psi)))
    pprime = -8.0e3 * (1.0 - psi)
    return {"psi_N": psi, "psi_q": psi, "R_avg": R_avg, "inv_R": inv_R,
            "inv_R2": inv_R2, "dV_dpsi": dV_dpsi, "dpsi_dpsiN": 0.9,
            "pprime": pprime}


def _parts():
    g = _geom()
    w, c = Ip_fsa_weights(g, convention="jphi-linterp")
    psi = g["psi_N"]
    j_ind = 8.0e5 * (1.0 - psi) ** 1.5
    j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2) + 2.0e4 * (1.0 - psi)
    j_fix = 1.0e5 * (1.0 - psi) ** 3
    lin = lambda j: float(trapezoid(w * np.asarray(j, float), psi))
    Ip_s = 1.04 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
    return psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s


class _Cylinder:
    """The analytic stand-in for a free-boundary solve (see module doc)."""

    def __init__(self, psi, Ip):
        self.psi = np.asarray(psi, float)
        self.Ip = float(Ip)
        self.I = None
        self.n_solves = 0

    def solve(self, j):
        I = np.pi * AMINOR ** 2 * cumulative_trapezoid(
            np.asarray(j, float), self.psi, initial=0.0)
        # the core below psi[0] carries the axis value of j
        I = I + np.pi * AMINOR ** 2 * float(j[0]) * self.psi[0]
        self.I = I * (self.Ip / I[-1])          # Ip imposed, as a solve does
        self.n_solves += 1
        return 7

    def field(self, R, Z):
        R = np.asarray(R, float)
        r = np.abs(R - R0)
        pn = (r / AMINOR) ** 2
        I_r = np.interp(pn, self.psi, self.I)
        Bth = _MU0 * I_r / (2.0 * np.pi * r)
        return np.column_stack([np.zeros_like(R), B0 * R0 / R, Bth])


#: the fake solver mesh: a box around the cylinder (R_min, R_max, Z_min, Z_max)
_MESH = (R0 - 0.95, R0 + 0.95, -1.2, 1.2)


class _Eval:
    """Mimics TokaMaker's field interpolator, INCLUDING its failure mode.

    It returns a copy of its own value buffer; for a point outside the mesh it
    writes nothing and sets the cell to 0 -- so a caller that does not check
    reads back the PREVIOUS point's field.  ``with_cell=False`` drops the cell
    attribute (only the buffer poisoning can then catch it).
    """

    def __init__(self, cyl, with_cell=True):
        self.cyl = cyl
        self.val = np.zeros(3)
        if with_cell:
            self.cell = types.SimpleNamespace(value=-1)

    def eval(self, pt):
        r, z = float(pt[0]), float(pt[1])
        inside = _MESH[0] <= r <= _MESH[1] and _MESH[2] <= z <= _MESH[3]
        if hasattr(self, "cell"):
            self.cell.value = 7 if inside else 0
        if inside:
            self.val[:] = self.cyl.field(np.array([r]), np.array([z]))[0]
        return self.val[:3].copy()


class _FakeGS:
    """``get_field_eval`` + ``o_point``: what the MSE stage reads."""

    #: the cylinder's magnetic axis
    o_point = np.array([R0, 0.0])

    def __init__(self, cyl, with_cell=True):
        self.cyl = cyl
        self.with_cell = with_cell

    def get_field_eval(self, name):
        assert name == "B"
        return _Eval(self.cyl, with_cell=self.with_cell)

    def copy_eq(self):
        return _Snap()


def _chord_geometry(n=12):
    r = AMINOR * np.sqrt(np.linspace(0.06, 0.9, n))
    return R0 + r, np.zeros(n)


def _hybrid(x, Phi, j_ind, j_bs, j_fix):
    K = Phi.shape[0]
    return (1.0 + x[:K] @ Phi) * j_ind + (1.0 + x[K:] @ Phi) * j_bs + j_fix


#: The analytic cylinder puts +B_theta along +Z at the outboard midplane,
#: which in a right-handed (R, phi, Z) frame is a plasma current along -phi;
#: its B_phi = B0 R0 / R is along +phi.  A block measured on THAT field (the
#: synthetic data below) therefore states ip_sign=-1, bt_sign=+1.
_CYL_IP, _CYL_BT = -1.0, 1.0


def _mse_block(tg, R, Z, sigma=3.0e-3, weight=None, **kw):
    n = len(tg)
    md = dict(R=list(R), Z=list(Z), tgamma=list(tg),
              sigma=[sigma] * n,
              weight=([1.0] * n if weight is None else list(weight)),
              A1=[1.0] * n, A2=[1.0] * n, A3=[0.1] * n, A4=[0.05] * n,
              ip_sign=_CYL_IP, bt_sign=_CYL_BT)
    md.update(kw)
    return md


def _synthetic_world(x_true=None, K=4):
    """(parts, cyl, Phi, chords) with data generated at x_true."""
    psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
    Phi = structured_basis_eval(None, psi)
    if x_true is None:
        x_true = np.array([0.0, -0.12, -0.05, 0.0, 0.0, 0.0, 0.08, 0.0])
    cyl = _Cylinder(psi, Ip_s)
    cyl.solve(_hybrid(x_true, Phi, j_ind, j_bs, j_fix))
    R, Z = _chord_geometry()
    B = cyl.field(R, Z)
    ch0 = mse_chords(_mse_block(np.zeros(R.size), R, Z))
    tg_true = mse_tan_gamma(B, ch0)
    ch = mse_chords(_mse_block(tg_true, R, Z))
    return (psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s), cyl, Phi, ch


# ---------------------------------------------------------------------------
class TestDataBlock:
    def test_weight_is_folded_into_sigma_once(self):
        R, Z = _chord_geometry(6)
        md = _mse_block([0.1] * 6, R, Z, sigma=0.01,
                        weight=[1.0, 4.0, 0.25, 1.0, 1.0, 1.0])
        ch = mse_chords(md)
        np.testing.assert_allclose(ch["sigma_eff"][:3], [0.01, 0.005, 0.02])

    def test_inactive_chords_are_dropped_and_indexed(self):
        R, Z = _chord_geometry(8)
        tg = [0.1] * 8
        tg[2] = float("nan")
        md = _mse_block(tg, R, Z, weight=[1, 1, 1, 0, 1, 1, -1, 1])
        ch = mse_chords(md)
        assert ch["n_total"] == 8 and ch["n_active"] == 5
        assert list(ch["index"]) == [0, 1, 4, 5, 7]

    def test_sigma_sys_adds_in_quadrature_and_defaults_to_zero(self):
        R, Z = _chord_geometry(5)
        md = _mse_block([0.1] * 5, R, Z, sigma=0.003)
        np.testing.assert_allclose(mse_chords(md)["sigma_eff"], 0.003)
        np.testing.assert_allclose(mse_chords(md, sigma_sys=0.004)["sigma_eff"],
                                   0.005)

    @pytest.mark.parametrize("mut, match", [
        (lambda md: md.pop("A2"), "lacks A2"),
        (lambda md: md.update(R=md["R"][:-1]), "differ in length"),
        (lambda md: md.update(weight=[0.0] * 6), "only 0 of 6"),
        (lambda md: md.update(Er=[1e4] * 6, A5=[1e-6] * 6,
                              er_corrected=True), "count E_r twice"),
        (lambda md: md.pop("ip_sign"), "lacks ip_sign"),
        (lambda md: md.pop("bt_sign"), "lacks bt_sign"),
        (lambda md: md.update(ip_sign=0), "'ip_sign' must be \\+1 or -1"),
        (lambda md: md.update(bt_sign=2.0), "'bt_sign' must be \\+1 or -1"),
        (lambda md: md.update(ip_sign=float("nan")), "must be \\+1 or -1"),
        (lambda md: md.update(ip_sign="1"), "must be \\+1 or -1"),
        (lambda md: md.update(bt_sign=True), "must be \\+1 or -1"),
    ])
    def test_refusals(self, mut, match):
        R, Z = _chord_geometry(6)
        md = _mse_block([0.1] * 6, R, Z)
        mut(md)
        with pytest.raises(MSEDataUnusable, match=match):
            mse_chords(md)

    def test_none_is_unusable(self):
        with pytest.raises(MSEDataUnusable, match="no MSE data block"):
            mse_chords(None)

    def test_min_chords(self):
        R, Z = _chord_geometry(4)
        mse_chords(_mse_block([0.1] * 4, R, Z))
        with pytest.raises(MSEDataUnusable, match="at least 5"):
            mse_chords(_mse_block([0.1] * 4, R, Z), min_chords=5)


class TestForwardModel:
    def test_formula_including_the_er_term(self):
        R, Z = _chord_geometry(4)
        md = _mse_block([0.0] * 4, R, Z, A5=[2e-6] * 4, Er=[3e4] * 4)
        ch = mse_chords(md)
        assert ch["er_applied"] and not ch["er_corrected"]
        B = np.array([[0.01, 2.0, 0.3]] * 4)
        exp = (1.0 * 0.3 + 2e-6 * 3e4) / (1.0 * 2.0 + 0.1 * 0.01 + 0.05 * 0.3)
        np.testing.assert_allclose(mse_tan_gamma(B, ch), exp, rtol=1e-14)
        # orientation: poloidal components flip together, toroidal alone
        exp2 = (-0.3 + 2e-6 * 3e4) / (-2.0 - 0.001 - 0.015)
        np.testing.assert_allclose(mse_tan_gamma(B, ch, -1.0, -1.0), exp2,
                                   rtol=1e-14)

    def test_equilibrium_orientation_is_read_off_the_field(self):
        _parts_, cyl, Phi, ch = _synthetic_world()
        B = cyl.field(ch["R"], ch["Z"])
        o = mse_equilibrium_orientation(B, ch["R"], ch["Z"], (R0, 0.0))
        assert (o["ip"], o["bt"]) == (_CYL_IP, _CYL_BT)
        assert o["n_ip_agree"] == o["n_ip_used"] == ch["n_active"]
        # the same current along +phi: B_Z < 0 outboard, and the circulation
        # about an axis ABOVE/BELOW the chord uses B_R too
        Bm = B * np.array([1.0, 1.0, -1.0])
        assert mse_equilibrium_orientation(Bm, ch["R"], ch["Z"],
                                           (R0, 0.0))["ip"] == 1.0
        Rv, Zv = np.array([R0, R0]), np.array([0.3, -0.3])
        # a current along +phi = y-hat (R = x-hat, Z = z-hat, right-handed):
        # at +Z from it Biot-Savart gives y x z = +x, i.e. B_R > 0 above the
        # axis and B_R < 0 below it
        Bv = np.array([[0.2, 2.0, 0.0], [-0.2, 2.0, 0.0]])
        assert mse_equilibrium_orientation(Bv, Rv, Zv, (R0, 0.0))["ip"] == 1.0
        Bmix = B.copy()
        Bmix[1, 1] *= -1.0
        with pytest.raises(RuntimeError, match="one sign"):
            mse_equilibrium_orientation(Bmix, ch["R"], ch["Z"], (R0, 0.0))
        with pytest.raises(RuntimeError, match="axis"):
            mse_equilibrium_orientation(B, ch["R"], ch["Z"], (-1.0, 0.0))

    def test_orientation_is_stated_not_fitted(self):
        """The data measured in the OPPOSITE toroidal-field direction: the
        stated bt_sign alone decides; nothing is chosen by chi^2, so there is
        no loop order to depend on."""
        _parts_, cyl, Phi, ch = _synthetic_world()
        B = cyl.field(ch["R"], ch["Z"])
        o = mse_equilibrium_orientation(B, ch["R"], ch["Z"], (R0, 0.0))
        md = _mse_block(list(mse_tan_gamma(B, ch, 1.0, -1.0)), ch["R"],
                        ch["Z"], bt_sign=-_CYL_BT)
        chf = mse_chords(md)
        sp, st = mse_orientation(chf, o)
        assert (sp, st) == (1.0, -1.0)
        chk = mse_orientation_check(B, chf, sp, st)
        assert chk["chi2_stated"] < 1e-20 and not chk["disagrees"]
        # its exact twin (-1,+1) fits identically: the data cannot tell them
        # apart, which is WHY the orientation must be stated
        assert chk["twin"] == "(-1,+1)" and chk["twin_delta_chi2"] == 0.0
        assert "indistinguishable" in chk["note"]
        # the SAME data with the wrong statement is reported, never switched
        chw = mse_chords(dict(md, bt_sign=_CYL_BT))
        spw, stw = mse_orientation(chw, o)
        assert (spw, stw) == (1.0, 1.0)
        bad = mse_orientation_check(B, chw, spw, stw)
        assert bad["disagrees"] and bad["delta_chi2"] > 1e3
        assert bad["best_other"] in ("(+1,-1)", "(-1,+1)")

    @staticmethod
    def _tg_by_hand(B, ip_rel, bt_rel, A1, A2, A3, A4, A5=0.0, Er=0.0):
        """tan(gamma) of the DISCHARGE, written out term by term."""
        BR, Bphi, BZ = ip_rel * B[:, 0], bt_rel * B[:, 1], ip_rel * B[:, 2]
        return (A1 * BZ + A5 * Er) / (A2 * Bphi + A3 * BR + A4 * BZ)

    @pytest.mark.parametrize("with_er", [False, True])
    @pytest.mark.parametrize("ip_data, bt_data", [(1, 1), (1, -1), (-1, 1),
                                                  (-1, -1)])
    def test_reversed_ip_and_bt_with_and_without_er(self, ip_data, bt_data,
                                                    with_er):
        _parts_, cyl, Phi, ch = _synthetic_world()
        R, Z = ch["R"], ch["Z"]
        n = R.size
        B = cyl.field(R, Z)
        o = mse_equilibrium_orientation(B, R, Z, (R0, 0.0))
        A5, Er = (2e-6, 2.5e4) if with_er else (0.0, 0.0)
        tg = self._tg_by_hand(B, ip_data * o["ip"], bt_data * o["bt"],
                              1.0, 1.0, 0.1, 0.05, A5, Er)
        kw = dict(A5=[A5] * n, Er=[Er] * n) if with_er else {}
        chd = mse_chords(_mse_block(list(tg), R, Z, ip_sign=ip_data,
                                    bt_sign=bt_data, **kw))
        assert chd["er_applied"] is with_er
        sp, st = mse_orientation(chd, o)
        assert (sp, st) == (ip_data * o["ip"], bt_data * o["bt"])
        chk = mse_orientation_check(B, chd, sp, st)
        assert chk["chi2_stated"] < 1e-18 and not chk["disagrees"]
        # the twin: identical without E_r, separated ONLY by E_r with it
        if with_er:
            assert chk["twin_delta_chi2"] < -1e3
        else:
            assert chk["twin_delta_chi2"] == 0.0
        # a caller who states BOTH signs wrongly: with E_r the data say so
        # (loudly, via `disagrees`); without E_r it is undetectable by
        # construction -- and in neither case is the orientation switched
        chw = mse_chords(_mse_block(list(tg), R, Z, ip_sign=-ip_data,
                                    bt_sign=-bt_data, **kw))
        spw, stw = mse_orientation(chw, o)
        assert (spw, stw) == (-sp, -st)
        w = mse_orientation_check(B, chw, spw, stw)
        assert w["disagrees"] is with_er
        if with_er:
            assert w["delta_chi2"] > MSE_ORIENTATION_DCHI2

    def test_field_read_through_the_equilibrium_interface(self):
        _parts_, cyl, Phi, ch = _synthetic_world()
        B, found = mse_field_at(_FakeGS(cyl), ch["R"], ch["Z"])
        assert found.all()
        np.testing.assert_allclose(B, cyl.field(ch["R"], ch["Z"]), rtol=1e-15)


class TestOffMesh:
    """A chord the interpolator cannot place is never given a stale field."""

    @pytest.mark.parametrize("with_cell", [True, False])
    def test_off_mesh_point_never_reads_the_previous_chord(self, with_cell):
        _parts_, cyl, Phi, ch = _synthetic_world()
        R = np.array([R0 + 0.3, R0 + 3.0, R0 + 0.4, R0 + 0.2])
        Z = np.array([0.0, 0.0, 0.0, 5.0])
        # the raw interpolator really does hand back the previous point
        ev = _Eval(cyl, with_cell=with_cell)
        first = ev.eval([R[0], Z[0]])
        np.testing.assert_array_equal(ev.eval([R[1], Z[1]]), first)
        B, found = mse_field_at(_FakeGS(cyl, with_cell=with_cell), R, Z)
        assert list(found) == [True, False, True, False]
        assert np.all(np.isnan(B[~found]))
        np.testing.assert_allclose(B[found], cyl.field(R[found], Z[found]),
                                   rtol=1e-15)

    def _add_off_mesh_chord(self, g, R_off=R0 + 3.0):
        for k in ("R", "Z", "tgamma", "sigma", "weight", "A1", "A2", "A3",
                  "A4"):
            g.mse_data[k] = list(g.mse_data[k]) + [g.mse_data[k][-1]]
        g.mse_data["R"][-1] = R_off

    def test_stage_excludes_it_with_a_reason_and_matches_the_run_without_it(
            self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl0, *_ = _run_stage(gc)
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(gc,
                                                 tamper=self._add_off_mesh_chord)
        rec = bl.ip_closure
        assert rec["structured_mse_status"] == "applied"
        assert rec["structured_mse_n_chords"] == ch["n_active"]
        assert rec["structured_mse_n_chords_total"] == ch["n_active"] + 1
        (ex,) = rec["structured_mse_excluded_chords"]
        assert ex["index"] == ch["n_active"]
        assert "off the solver mesh" in ex["reason"]
        # excluding the chord is exactly the run that never had it
        np.testing.assert_array_equal(rec["structured_coeffs_a"],
                                      bl0.ip_closure["structured_coeffs_a"])
        np.testing.assert_array_equal(rec["structured_coeffs_b"],
                                      bl0.ip_closure["structured_coeffs_b"])

    def _mostly_off(self, g):
        n = len(g.mse_data["R"])
        g.mse_data["R"] = [R0 + 3.0] * (n - 3) + list(g.mse_data["R"][-3:])

    def test_too_few_chords_on_the_mesh_refuses_loudly(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(gc, tamper=self._mostly_off)
        rec = bl.ip_closure
        assert rec["structured_mse_status"].startswith("refused")
        assert "remain on the solver mesh" in rec["structured_mse_status"]
        assert rec["closure_limited"]
        assert len(rec["structured_mse_excluded_chords"]) == ch["n_active"] - 3
        assert not state.get("mse_applied")

    def test_too_few_chords_on_the_mesh_required_raises(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_mse_required=True)
        with pytest.raises(RuntimeError, match="remain on the solver mesh"):
            _run_stage(gc, tamper=self._mostly_off)


# ---------------------------------------------------------------------------
def _random_lin(K=4, n=10, seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    R, Z = _chord_geometry(n)
    ch = mse_chords(_mse_block(list(0.3 + 0.01 * rng.standard_normal(n)),
                               R, Z, sigma=3e-3))
    J = 0.05 * scale * rng.standard_normal((n, 2 * K))
    tg0 = 0.3 + 0.01 * rng.standard_normal(n)
    x0 = 0.01 * rng.standard_normal(2 * K)
    return structured_mse_linear_model(x0, tg0, J, ch), ch


def _kkt_reference(W, M, m, C=None, d=None):
    """argmin x'Wx + ||Mx - m||^2 (s.t. Cx = d), by the textbook KKT system."""
    H = 2.0 * (np.diag(W) + M.T @ M)
    g = 2.0 * M.T @ m
    if C is None:
        return np.linalg.solve(H, g)
    C = np.atleast_2d(C)
    k = C.shape[0]
    Kmat = np.block([[H, C.T], [C, np.zeros((k, k))]])
    return np.linalg.solve(Kmat, np.concatenate([g, np.atleast_1d(d)]))[:H.shape[0]]


def _Mm(lin):
    sig = lin["sigma_eff"]
    M = lin["J"] / sig[:, None]
    m = (lin["tgamma"] - lin["tg0"] + lin["J"] @ lin["x0"]) / sig
    return M, m


class TestSolversWithMSE:
    def test_hard_is_the_exact_constrained_minimiser_and_keeps_ip(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin()
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  mse_lin=mlin)
        W = np.concatenate([STRUCTURED_WEIGHTS_PHYSICS["ind"],
                            STRUCTURED_WEIGHTS_PHYSICS["bs"]])
        M, m = _Mm(mlin)
        C = np.concatenate([out["A_row"], out["B_row"]])
        x_ref = _kkt_reference(W, M, m, C, out["deficit"])
        x = np.concatenate([out["a"], out["b"]])
        np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-12)
        assert abs(out["ip_residual"]) < 1e-9 * abs(Ip_s)
        z = M @ x - m
        assert out["mse_chi2_model"] == pytest.approx(float(z @ z), rel=1e-12)
        assert out["mse_objective_model"] == pytest.approx(
            float(W @ x ** 2 + z @ z), rel=1e-12)
        assert out["constraints"][-1].startswith("MSE tan(gamma)")

    def test_mse_moves_the_answer_and_lowers_the_model_chi2(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin()
        base = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix)
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  mse_lin=mlin)
        M, m = _Mm(mlin)
        xb = np.concatenate([base["a"], base["b"]])
        zb = M @ xb - m
        assert out["mse_chi2_model"] < float(zb @ zb)
        # and the full objective is below the no-MSE answer's
        assert out["mse_objective_model"] < (
            structured_objective_no_mse(base) + float(zb @ zb))

    def test_soft_quadratic_case_matches_closed_form(self):
        # soft Ip (linear row), no l_i: the whole objective is quadratic
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin()
        sig_ind = np.array([0.1, 0.4, 0.4, 0.4])
        sig_bs = np.array([0.5, 0.3, 0.15, 0.1])
        sIp = 0.005 * abs(Ip_s)
        out = close_ip_structured_soft(psi, w, c, Ip_s, sIp, j_ind, j_bs,
                                       j_fix, sigma_ind=sig_ind,
                                       sigma_bs=sig_bs, mse_lin=mlin)
        W = 1.0 / np.concatenate([sig_ind, sig_bs]) ** 2
        M, m = _Mm(mlin)
        p = np.concatenate([out["A_row"], out["B_row"]]) / sIp
        M2 = np.vstack([M, p])
        m2 = np.concatenate([m, [out["deficit"] / sIp]])
        x_ref = _kkt_reference(W, M2, m2)
        x = np.concatenate([out["a"], out["b"]])
        np.testing.assert_allclose(x, x_ref, rtol=1e-7, atol=1e-10)
        assert out["objective"] == pytest.approx(out["mse_objective_model"])
        assert structured_objective_no_mse(out) == pytest.approx(
            out["objective"] - out["mse_chi2_model"])

    def test_hard_and_soft_agree_when_given_the_same_statement(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin()
        h = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                mse_lin=mlin)
        sig = 1.0 / np.sqrt(np.concatenate([STRUCTURED_WEIGHTS_PHYSICS["ind"],
                                            STRUCTURED_WEIGHTS_PHYSICS["bs"]]))
        s = close_ip_structured_soft(psi, w, c, Ip_s, None, j_ind, j_bs,
                                     j_fix, sigma_ind=sig[:4],
                                     sigma_bs=sig[4:], mse_lin=mlin)
        np.testing.assert_allclose(np.r_[s["a"], s["b"]], np.r_[h["a"], h["b"]],
                                   rtol=1e-7, atol=1e-10)

    def test_one_sided_prior_settles_on_its_exact_pattern(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin(seed=11, scale=3.0)
        up = [0.1, 0.1, 0.1, 0.4]
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  sigma_ind_up=up, mse_lin=mlin)
        a = np.asarray(out["a"])
        W_ind = np.where(a > 0.0, 1.0 / np.asarray(up) ** 2,
                         np.asarray(STRUCTURED_WEIGHTS_PHYSICS["ind"]))
        W = np.concatenate([W_ind, STRUCTURED_WEIGHTS_PHYSICS["bs"]])
        M, m = _Mm(mlin)
        C = np.concatenate([out["A_row"], out["B_row"]])
        x_ref = _kkt_reference(W, M, m, C, out["deficit"])
        np.testing.assert_allclose(np.r_[a, out["b"]], x_ref, rtol=1e-8,
                                   atol=1e-12)
        # the settled pattern is self-consistent
        assert tuple(bool(v) for v in (x_ref[:4] > 0)) == out["sign_pattern"]

    def test_vanishing_mse_weight_recovers_the_no_mse_answer(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin()
        mlin = dict(mlin, sigma_eff=mlin["sigma_eff"] * 1e9)
        for fn, args in ((close_ip_structured, ()),):
            a = fn(psi, w, c, Ip_s, j_ind, j_bs, j_fix)
            b = fn(psi, w, c, Ip_s, j_ind, j_bs, j_fix, mse_lin=mlin)
            np.testing.assert_allclose(np.r_[b["a"], b["b"]],
                                       np.r_[a["a"], a["b"]],
                                       rtol=1e-9, atol=1e-13)

    def test_absent_block_changes_nothing(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        for fn, extra in ((close_ip_structured, {}),
                          (close_ip_structured_soft, None)):
            if extra is None:
                a = fn(psi, w, c, Ip_s, 0.005 * Ip_s, j_ind, j_bs, j_fix)
                b = fn(psi, w, c, Ip_s, 0.005 * Ip_s, j_ind, j_bs, j_fix,
                       mse_lin=None)
            else:
                a = fn(psi, w, c, Ip_s, j_ind, j_bs, j_fix)
                b = fn(psi, w, c, Ip_s, j_ind, j_bs, j_fix, mse_lin=None)
            assert set(a) == set(b)
            assert not any(k.startswith("mse_") for k in a)
            for k in a:
                if isinstance(a[k], np.ndarray):
                    np.testing.assert_array_equal(a[k], b[k])
                else:
                    assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k])

    def test_jacobian_width_must_match_the_basis(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        mlin, _ch = _random_lin(K=3)
        with pytest.raises(ValueError, match="columns"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                mse_lin=mlin)


# ---------------------------------------------------------------------------
class TestLinearisation:
    def _tan_gamma_of(self, parts, cyl, Phi, ch):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = parts

        def f(x):
            cyl.solve(_hybrid(np.asarray(x, float), Phi, j_ind, j_bs, j_fix))
            return mse_tan_gamma(cyl.field(ch["R"], ch["Z"]), ch)
        return f

    def test_forward_difference_jacobian_is_first_order(self):
        parts, cyl, Phi, ch = _synthetic_world()
        f = self._tan_gamma_of(parts, cyl, Phi, ch)
        x0 = np.zeros(8)
        tg0 = f(x0)
        free = np.ones(8, bool)
        # central-difference reference at a tiny step
        Jref = np.column_stack([(f(x0 + 1e-5 * e) - f(x0 - 1e-5 * e)) / 2e-5
                                for e in np.eye(8)])
        errs = []
        for h in (0.04, 0.02, 0.01):
            J = structured_mse_jacobian(f, x0, tg0, free, step=h)
            errs.append(np.max(np.abs(J - Jref)) / np.max(np.abs(Jref)))
        assert errs[0] < 0.05
        # first order: halving the step halves the error
        assert errs[1] / errs[0] == pytest.approx(0.5, abs=0.1)
        assert errs[2] / errs[1] == pytest.approx(0.5, abs=0.1)

    def test_pinned_columns_cost_nothing_and_stay_zero(self):
        parts, cyl, Phi, ch = _synthetic_world()
        f = self._tan_gamma_of(parts, cyl, Phi, ch)
        x0 = np.zeros(8)
        tg0 = f(x0)
        free = np.array([True, True, False, True, False, True, True, True])
        n0 = cyl.n_solves
        J = structured_mse_jacobian(f, x0, tg0, free)
        assert cyl.n_solves - n0 == 6
        assert np.all(J[:, ~free] == 0.0)

    def test_linear_forward_model_is_solved_exactly(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        rng = np.random.default_rng(5)
        R, Z = _chord_geometry(10)
        G = 0.05 * rng.standard_normal((10, 8))
        t0 = 0.3 + 0.01 * rng.standard_normal(10)
        ch = mse_chords(_mse_block(list(t0 + G @ (0.05 * rng.standard_normal(8))),
                                   R, Z))
        f = lambda x: t0 + G @ np.asarray(x, float)
        resolve = lambda l: close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs,
                                                j_fix, mse_lin=l)
        pred = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix)
        xp = np.r_[pred["a"], pred["b"]]
        res = structured_mse_outer(xp, structured_objective_no_mse(pred),
                                   f(xp), f, resolve, ch, np.ones(8, bool))
        st = res["record"]["steps"][-1]
        assert st["linearisation_residual_max_sigma"] < 1e-9
        direct = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix,
            mse_lin=structured_mse_linear_model(np.zeros(8), t0, G, ch))
        np.testing.assert_allclose(res["x"], np.r_[direct["a"], direct["b"]],
                                   rtol=1e-6, atol=1e-10)
        assert not res["flags"]

    @pytest.mark.parametrize("soft", [False, True])
    def test_objective_and_chi2_fall_on_the_nonlinear_cylinder(self, soft):
        parts, cyl, Phi, ch = _synthetic_world()
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = parts
        f = self._tan_gamma_of(parts, cyl, Phi, ch)
        if soft:
            kw = dict(sigma_ind=[0.1, 0.4, 0.4, 0.4],
                      sigma_bs=[0.5, 0.3, 0.15, 0.1])
            solve = lambda l=None: close_ip_structured_soft(
                psi, w, c, Ip_s, 0.005 * Ip_s, j_ind, j_bs, j_fix,
                mse_lin=l, **kw)
        else:
            solve = lambda l=None: close_ip_structured(
                psi, w, c, Ip_s, j_ind, j_bs, j_fix, mse_lin=l)
        pred = solve()
        xp = np.r_[pred["a"], pred["b"]]
        tgp = f(xp)
        res = structured_mse_outer(xp, structured_objective_no_mse(pred),
                                   tgp, f, solve, ch, np.ones(8, bool),
                                   n_steps=2)
        rec = res["record"]
        assert rec["chi2_after"] < 0.5 * rec["chi2_before"]
        assert rec["objective_after"] < rec["objective_before"]
        # each chord-method step is non-increasing in the achieved objective
        o = [rec["objective_before"]] + [s["objective_achieved"]
                                         for s in rec["steps"]]
        assert all(b <= a * (1 + 1e-12) for a, b in zip(o, o[1:]))
        # the recorded linearisation residual is the TRUE one
        s1 = rec["steps"][0]
        assert s1["linearisation_residual_max_sigma"] > 0.0
        assert rec["n_solves"] == 8 + 2
        assert not res["flags"]

    def test_recorded_linearisation_residual_is_recomputable(self):
        parts, cyl, Phi, ch = _synthetic_world()
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = parts
        f = self._tan_gamma_of(parts, cyl, Phi, ch)
        solve = lambda l=None: close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, mse_lin=l)
        pred = solve()
        xp = np.r_[pred["a"], pred["b"]]
        tgp = f(xp)
        res = structured_mse_outer(xp, structured_objective_no_mse(pred), tgp,
                                   f, solve, ch, np.ones(8, bool))
        J = res["record"]["jacobian"]
        lres = (f(res["x"]) - (tgp + J @ (res["x"] - xp))) / ch["sigma_eff"]
        assert res["record"]["steps"][0]["linearisation_residual_max_sigma"] \
            == pytest.approx(float(np.max(np.abs(lres))), rel=1e-10)

    def test_a_failing_linearisation_is_flagged_not_hidden(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        R, Z = _chord_geometry(8)
        solve = lambda l=None: close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, mse_lin=l)
        pred = solve()
        xp = np.r_[pred["a"], pred["b"]]
        g = np.full(8, 0.5)
        # a forward model whose curvature the linear step cannot see: the
        # step aimed at the data (s = +0.01) lands at 0.41, ten times further
        # from it than where it started
        def f(x):
            sx = g @ (np.asarray(x, float) - xp)
            return np.full(8, 0.3 + sx + 1000.0 * sx ** 2)
        ch = mse_chords(_mse_block([0.31] * 8, R, Z, sigma=1e-4))
        res = structured_mse_outer(xp, structured_objective_no_mse(pred),
                                   f(xp), f, solve, ch, np.ones(8, bool),
                                   fd_step=1e-7)
        assert res["record"]["objective_after"] > res["record"]["objective_before"]
        assert res["flags"] and res["flags"][0].startswith(MSE_FLAG_PREFIX)


# ---------------------------------------------------------------------------
#  the run.py stage, end to end on the analytic cylinder
# ---------------------------------------------------------------------------
class _Snap:
    def get_q(self, psi=None, compute_geo=False):
        return (np.asarray(psi, float), np.full(np.size(psi), 1.2), None, None)


def _run_stage(gc, tamper=None):
    """predictor -> (common-tail solve) -> MSE stage -> corrector -> delivered."""
    from bouquet.run import Bouquet

    parts, cyl, Phi, ch = _synthetic_world()
    psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = parts
    gc.mse_data = _mse_block(list(ch["tgamma"]), ch["R"], ch["Z"])
    if tamper is not None:
        tamper(gc)
    geom = dict(psi_N=psi, psi_q=psi)
    probe = j_ind + j_bs + j_fix
    bl = types.SimpleNamespace(sawtooth=None, ip_closure=None)
    s_ind, s_bs, ohm, bs, extra, state = \
        Bouquet._close_ip_structured_predictor(
            gc, bl, _Snap(), geom, probe, psi, j_ind, j_bs, j_fix, probe,
            1.0, abs(Ip_s), w, c, lin(j_ind), lin(j_bs), lin(j_fix))
    bl.j_inductive, bl.j_BS = s_ind * j_ind, s_bs * j_bs
    bl.j_phi = bl.j_inductive + bl.j_BS + j_fix
    bl.ohm_scale, bl.bs_scale = ohm, bs
    bl.ip_closure = dict(closure_limited=False, closure_limited_reasons=(),
                         **extra)
    cyl.solve(bl.j_phi)                           # the common-tail solve
    gs = _FakeGS(cyl)
    n0 = cyl.n_solves
    if state is not None and state.get("mse") is not None:
        Bouquet._close_ip_structured_mse_stage(state, bl, gs, cyl.solve)
    n_stage = cyl.n_solves - n0
    if state is not None:
        Bouquet._close_ip_structured_corrector(
            state, bl, gs, cyl.solve,
            ip_of=lambda j: float(lin(j) + c),
            roundtrip_gate=Bouquet._structured_roundtrip_gate(abs(Ip_s)))
        if state.get("mse_applied"):
            Bouquet._structured_mse_delivered(state, bl, gs)
    return bl, state, n_stage, cyl, ch


class TestRunStage:
    def test_default_preset_end_to_end(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(gc)
        rec = bl.ip_closure
        assert rec["structured_mse"] and rec["structured_mse_status"] == "applied"
        assert rec["structured_mse_n_chords"] == ch["n_active"]
        assert rec["structured_mse_n_solves"] == n_stage == 8 + 1
        assert rec["structured_mse_chi2_after"] < 0.5 * rec[
            "structured_mse_chi2_before"]
        assert rec["structured_mse_objective_after"] < rec[
            "structured_mse_objective_before"]
        assert rec["structured_mse_chi2_delivered"] == pytest.approx(
            rec["structured_mse_chi2_after"], rel=1e-12)
        # every GS solve after the predictor's is an extra solve, and the
        # verdict says where they went
        assert rec["n_extra_solves"] == 9
        assert rec["structured_corrector_n_solves"] == 0
        assert "MSE stage (9 solves)" in rec["sawtooth_verdict"]
        assert "(0 extra solves)" not in rec["sawtooth_verdict"]
        # the delivered equilibrium is judged, and here it is the stage's own
        assert rec["structured_mse_delivered_worse"] is False
        assert rec["structured_mse_objective_delivered_comparable"] is True
        assert rec["structured_mse_objective_delivered"] == pytest.approx(
            rec["structured_mse_objective_after"], rel=1e-12)
        z = np.asarray(rec["structured_mse_residual_sigma_delivered"])
        assert z.size == ch["n_active"]
        assert rec["structured_mse_residual_sigma_delivered_max_abs"] == \
            pytest.approx(float(np.max(np.abs(z))))
        assert all("objective_rose_vs_previous" in st_
                   for st_ in rec["structured_mse_step_log"])
        assert len(rec["structured_mse_residual_sigma_after"]) == ch["n_active"]
        assert rec["structured_mse_er_applied"] is False
        o = rec["structured_mse_orientation"]
        assert (o["pol"], o["tor"]) == (1.0, 1.0)
        assert (o["ip_sign_equilibrium"], o["bt_sign_equilibrium"]) == (
            _CYL_IP, _CYL_BT)
        assert rec["structured_mse_orientation_disagrees"] is False
        # bl carries the delivered closure, and ip_closure describes it
        np.testing.assert_allclose(
            rec["structured_s_ind_profile"],
            bl.j_inductive / state["j_ind"], rtol=1e-12)
        assert state["mse_lin"] is not None
        json.dumps(rec)                              # serialisable as is
        assert not rec["closure_limited"], rec["closure_limited_reasons"]

    def test_no_block_means_no_mse_anything(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(
            gc, tamper=lambda g: setattr(g, "mse_data", None))
        assert state is None and n_stage == 0
        assert not any(k.startswith("structured_mse") for k in bl.ip_closure)

    def test_required_and_absent_raises(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_mse_required=True)
        with pytest.raises(MSEDataUnusable, match="structured_mse_required"):
            _run_stage(gc, tamper=lambda g: setattr(g, "mse_data", None))

    def test_required_and_unusable_raises(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_mse_required=True)

        def _few(g):
            g.mse_data["weight"] = [0.0] * (len(g.mse_data["weight"]) - 3) \
                + [1.0] * 3
        with pytest.raises(MSEDataUnusable, match="only 3 of"):
            _run_stage(gc, tamper=_few)

    def test_unusable_not_required_is_warned_and_recorded(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")

        def _bad(g):
            g.mse_data.pop("A1")
        with pytest.warns(UserWarning, match="UNUSABLE"):
            bl, state, n_stage, cyl, ch = _run_stage(gc, tamper=_bad)
        rec = bl.ip_closure
        assert rec["structured_mse"] is False
        assert rec["structured_mse_status"].startswith("unusable")
        assert n_stage == 0

    def _refusing(self, g):
        # a measurement no in-bounds closure can reach, at a tiny sigma
        g.mse_data["tgamma"] = [3.0 * t for t in g.mse_data["tgamma"]]
        g.mse_data["sigma"] = [1e-7] * len(g.mse_data["sigma"])

    def test_refused_stage_not_required_keeps_predictor_and_flags(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(gc, tamper=self._refusing)
        rec = bl.ip_closure
        assert rec["structured_mse_status"].startswith("refused")
        assert rec["closure_limited"]
        assert any(r.startswith(MSE_FLAG_PREFIX)
                   for r in rec["closure_limited_reasons"])
        assert "scale bounds" in rec["structured_mse_status"] \
            or "outside" in rec["structured_mse_status"]
        # 8 finite-difference probes, then the predictor's hybrid re-solved so
        # mygs holds the equilibrium bl describes
        assert n_stage == 8 + 1
        assert rec["structured_mse_n_solves"] == 9
        assert rec["n_extra_solves"] == 9
        assert "MSE stage refused (9 solves)" in rec["sawtooth_verdict"]
        I_kept = cyl.I.copy()
        cyl.solve(bl.j_phi)
        np.testing.assert_allclose(cyl.I, I_kept, rtol=1e-14)
        assert not state.get("mse_applied")
        assert "structured_mse_chi2_after" not in rec

    def test_refused_stage_required_raises(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_mse_required=True)
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            _run_stage(gc, tamper=self._refusing)

    def test_wrongly_stated_orientation_is_flagged_and_kept(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(
            gc, tamper=lambda g: g.mse_data.update(bt_sign=-_CYL_BT))
        rec = bl.ip_closure
        o = rec["structured_mse_orientation"]
        assert (o["pol"], o["tor"]) == (1.0, -1.0)          # as STATED
        assert rec["structured_mse_orientation_disagrees"] is True
        assert rec["closure_limited"]
        assert any("disagree with the stated field orientation" in r
                   for r in rec["closure_limited_reasons"])

    def test_the_flag_is_judged_on_the_delivered_equilibrium(self):
        """A delivered equilibrium that fits the chords worse than the
        pre-MSE closure is FLAGGED, whatever the MSE stage's own last solve
        looked like.  Reporting only: nothing is re-solved."""
        from bouquet.run import Bouquet
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_preset="none")
        bl, state, n_stage, cyl, ch = _run_stage(gc)
        assert bl.ip_closure["structured_mse_delivered_worse"] is False
        # stand-in for a corrector re-solve that moved the equilibrium away
        # from what the MSE stage delivered
        n0 = cyl.n_solves
        cyl.solve(1.6 * state["j_ind"] + state["j_BS_swb"]
                  + state["j_fixed"])
        state["mse_corrector_resolved"] = True
        Bouquet._structured_mse_delivered(state, bl, _FakeGS(cyl))
        assert cyl.n_solves == n0 + 1        # the judgement solved nothing
        rec = bl.ip_closure
        assert rec["structured_mse_chi2_delivered"] > rec[
            "structured_mse_chi2_before"]
        assert rec["structured_mse_delivered_worse"] is True
        assert rec["closure_limited"]
        assert any("delivered chi2" in r and r.startswith(MSE_FLAG_PREFIX)
                   for r in rec["closure_limited_reasons"])
        # hard channel: the prior-only objective stays comparable
        assert rec["structured_mse_objective_delivered_comparable"] is True

    def test_soft_resolved_objective_is_not_compared(self):
        from bouquet.run import Bouquet
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic")
        bl, state, n_stage, cyl, ch = _run_stage(gc)
        state["mse_corrector_resolved"] = True
        Bouquet._structured_mse_delivered(state, bl, _FakeGS(cyl))
        rec = bl.ip_closure
        assert rec["structured_mse_objective_delivered_comparable"] is False
        assert rec["structured_mse_objective_delivered"] is None
        assert "not comparable" in rec[
            "structured_mse_objective_delivered_note"]

    def test_hard_channel_with_mse(self):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic",
                              structured_preset="none")
        bl, state, n_stage, cyl, ch = _run_stage(gc)
        rec = bl.ip_closure
        assert rec["structured_solver"] == "hard-KKT"
        assert rec["structured_mse_status"] == "applied"
        assert abs(rec["structured_ip_residual"]) < 1e-6
        assert rec["structured_mse_chi2_after"] < rec["structured_mse_chi2_before"]


class TestRequiredReachability:
    def _cfg(self, **kw):
        from bouquet.config import ImasSource
        gc = GenerationConfig(**kw)
        return types.SimpleNamespace(source=ImasSource(ids_path="x.json"),
                                     generation=gc)

    def test_required_on_another_channel_is_refused(self):
        from bouquet.run import Bouquet
        cfg = self._cfg(structured_mse_required=True,
                        closure_channel="bootstrap", jBS_baseline_mode="ohmic")
        with pytest.raises(ValueError, match="will not run"):
            Bouquet._check_structured_mse_reachable(cfg)

    def test_required_on_the_structured_path_passes(self):
        from bouquet.run import Bouquet
        cfg = self._cfg(structured_mse_required=True,
                        closure_channel="structured",
                        jBS_baseline_mode="ohmic", recalculate_j_BS=True)
        Bouquet._check_structured_mse_reachable(cfg)

    def test_not_required_is_never_checked(self):
        from bouquet.run import Bouquet
        Bouquet._check_structured_mse_reachable(types.SimpleNamespace(
            source=None, generation=GenerationConfig()))


class TestConfig:
    def test_defaults_add_nothing(self):
        gc = GenerationConfig()
        assert gc.mse_data is None and gc.structured_mse_required is False
        assert gc.structured_mse_sigma_sys == 0.0
        assert gc.structured_mse_steps == 1
        assert gc.structured_mse_min_chords == 4

    def test_block_roundtrips_through_the_config_json(self):
        from bouquet.config import BouquetConfig
        R, Z = _chord_geometry(5)
        gc = GenerationConfig(closure_channel="structured",
                              mse_data=_mse_block([0.1] * 5, R, Z,
                                                  er_corrected=True),
                              structured_mse_required=True)
        d = json.loads(json.dumps({"g": __import__("bouquet.config", fromlist=[
            "_encode"])._encode(gc)}))["g"]
        assert d["mse_data"]["er_corrected"] is True
        assert d["structured_mse_required"] is True
        assert BouquetConfig  # imported: the dataclass field is serialised


# ---------------------------------------------------------------------------
#  the corrector's entry state after an APPLIED MSE stage
# ---------------------------------------------------------------------------
class _LiSnap:
    def __init__(self, world):
        self.world = world

    def get_q(self, psi=None, compute_geo=False):
        return (np.asarray(psi, float), np.full(np.size(psi), 1.05), None,
                None)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": self.world["li"] * 1.008}


class _LiGS:
    def __init__(self, world):
        self.world = world

    def copy_eq(self):
        return _LiSnap(self.world)


def _corrector_after_mse(monkeypatch, mse_applied):
    """The corrector on a stubbed closure, entered after an MSE stage."""
    from bouquet import utils
    from bouquet.run import Bouquet

    world = {"li": 0.87}          # 3 % below the 0.9 target: one step

    def _solver(*a, **kw):
        return dict(s_ind=np.ones(5), s_bs=np.ones(5), a=[0.0], b=[0.0],
                    ohm_scale_eff=1.0, bs_scale_eff=1.0, structure_ind=0.0,
                    structure_bs=0.0, ip_residual_pct=0.0, Ip_hybrid=1.0e6,
                    ip_residual=0.0, residual_sigma_Ip=None,
                    li_predicted=float(kw["li_target"]))

    monkeypatch.setattr(utils, "close_ip_structured", _solver)
    monkeypatch.setattr(utils, "close_ip_structured_soft", _solver)
    monkeypatch.setattr(utils, "li_achieved",
                        lambda eq, li_kind="li_1", psi_pad=1e-3,
                        perimeter=None: (float(eq.world["li"]), {}))

    def _solve(j):
        world["li"] = 0.9
        return 7

    bl = types.SimpleNamespace(
        ip_closure={"closure_limited": False, "closure_limited_reasons": (),
                    "structured_li_solved_predictor": 0.85,
                    "structured_li_achieved_predictor": 0.85,
                    "structured_li_residual_predictor": -0.05},
        ohm_scale=1.0, bs_scale=1.0)
    state = dict(
        q0_target=1.05, psi_q=np.linspace(0.0, 1.0, 5),
        psi_geom=np.linspace(0.0, 1.0, 5), j_ind=np.ones(5),
        j_BS_swb=np.ones(5), j_fixed=np.zeros(5), axis=None,
        w_lin=np.ones(5), c_signed=0.0, Ip_signed=1.0e6, ip_ind=7.0e5,
        ip_bs=3.0e5, ip_fix=0.0, basis=None, weights=None, q0_tol=0.01,
        gated=False, soft=False, ip_sigma=None, sigma_ind=None,
        sigma_bs=None, li_target=0.9, li_sigma=None, li_kind="li_1",
        li_geom={"perimeter": 4.2}, psi_pad=1e-3, li_tol=0.005,
        li_max_corrector_steps=1,
        mse={"n_active": 6}, mse_applied=mse_applied, mse_n_solves=9)
    Bouquet._close_ip_structured_corrector(state, bl, _LiGS(world), _solve)
    return bl.ip_closure, state


class TestCorrectorAfterMSE:
    def test_entry_readbacks_are_not_called_predictor(self, monkeypatch):
        rec, state = _corrector_after_mse(monkeypatch, mse_applied=True)
        # the predictor's values (recorded by the MSE stage) are untouched
        assert rec["structured_li_achieved_predictor"] == 0.85
        assert rec["structured_li_residual_predictor"] == -0.05
        # the corrector's entry state is named for what it is
        assert rec["structured_li_achieved_mse_stage"] == pytest.approx(0.87)
        assert rec["structured_li_residual_mse_stage"] == pytest.approx(-0.03)
        assert "MSE-stage equilibrium" in rec["structured_corrector_entry"]
        # 9 MSE solves + 1 corrector solve
        assert rec["structured_corrector_n_solves"] == 1
        assert rec["n_extra_solves"] == 10
        assert rec["sawtooth_verdict"].startswith(
            "structured predictor + MSE stage (9 solves) + 1 corrector solve")
        assert state["mse_corrector_resolved"] is True

    def test_without_an_applied_stage_the_names_are_unchanged(self,
                                                             monkeypatch):
        rec, state = _corrector_after_mse(monkeypatch, mse_applied=False)
        assert rec["structured_li_achieved_predictor"] == pytest.approx(0.87)
        assert "structured_li_achieved_mse_stage" not in rec
        assert "structured_corrector_entry" not in rec
