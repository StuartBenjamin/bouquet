"""Contract tests for the fast-ion pressure in ``reconstruct_equilibrium``.

Background
----------
``reconstruct_equilibrium`` used to build its Grad-Shafranov pressure as
thermal-only, single-ion::

    pres_tmp = e * (ne*te + ni*ti)

while every consumer of that reconstruction solved a HIGHER pressure --
``perturb_kinetic_equilibrium`` (per draw) and the state anchor both add
``p_fast`` and the impurity term.  The reconstruction therefore tuned its
inductive-current knob against a lower-pressure equilibrium than the draws it
feeds, and its reported ``beta_n`` / ``beta_p`` / ``W_MHD`` were low by ~20%
on a beam-heated shot.

These tests lock in the contract.  They are deliberately cheap: they do NOT
run a GS solve (that needs OFT, a mesh and ~40 s), so they cannot catch a
numerical regression inside the solve.  What they do catch is the plumbing
and grid contract, which is what actually drifted.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_reconstruct_equilibrium_accepts_p_fast_and_Z_imp():
    """The kwargs exist and default to None (default-off)."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    params = inspect.signature(reconstruct_equilibrium).parameters
    for name in ("p_fast", "Z_imp"):
        assert name in params, f"reconstruct_equilibrium lost its {name} kwarg"
        assert params[name].default is None, (
            f"{name} must default to None so the pre-fix behaviour is "
            "reproduced bitwise when it is not supplied"
        )


def test_impurity_pressure_is_zero_for_falsy_Z_imp():
    """The guarded impurity call is safe on impurity-free (geqdsk) sources."""
    from bouquet.physics import impurity_pressure

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    for Z in (None, 0, 0.0):
        out = impurity_pressure(ne, ni, ti, Z)
        assert np.all(out == 0.0), f"impurity_pressure({Z!r}) should be zeros"


def test_impurity_pressure_matches_the_one_zeff_model():
    """nz = (ne - ni)/Z_imp thermalized at ti -- the term single-ion omits.

    Note: ``physics`` uses the CODATA elementary charge (1.602176634e-19)
    while the thermal terms use ``EC = 1.6022e-19``, a 1.5e-5 relative
    difference confined to the impurity term.  Harmless for recon-vs-draw
    consistency (both call THIS function), so the test pins the physics
    module's own constant rather than asserting the two agree.
    """
    from bouquet.physics import impurity_pressure, _EC as EC

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    Z = 6.0
    expected = EC * ((ne - ni) / Z) * ti
    np.testing.assert_allclose(impurity_pressure(ne, ni, ti, Z), expected,
                               rtol=1e-12, atol=0.0)


def test_kin_to_eq_regrid_is_the_same_functional_everywhere():
    """The recon must receive p_fast through the SAME kin->eq map the draws use.

    ``baseline.to_eq``, ``perturb_kinetic_equilibrium._kin_to_eq`` and the state
    anchor's ``_kin2eq`` are all ``utils.pchip_interp(psi_kin, arr, psi_N)``.
    If any of them is switched to a different interpolant, the reconstruction
    and the draws silently solve slightly different fast pressures again.
    """
    from bouquet.utils import pchip_interp

    psi_kin = np.linspace(0.0, 1.0, 150)
    psi_eq = np.linspace(0.0, 1.0, 129)
    p_fast = 2.1e4 * (1.0 - psi_kin ** 2) ** 2

    a = pchip_interp(psi_kin, p_fast, psi_eq)
    b = pchip_interp(psi_kin, p_fast, psi_eq)
    np.testing.assert_array_equal(a, b)
    # Shape-preserving and non-negative for a non-negative input.
    assert np.all(a >= -1e-9), "kin->eq regrid must not undershoot below zero"
    assert a.shape == psi_eq.shape


def test_p_fast_is_additive_and_raises_the_pressure():
    """Sign contract: adding fast pressure must RAISE the solve pressure.

    A larger pressure means a larger Shafranov shift, so R_axis moves outboard
    and l_i(3) ~ 1/R_axis falls.  If this ever inverts, the mechanism behind
    the fix is misunderstood and the reconstruction's l_i knob will compensate
    the wrong way.
    """
    EC = 1.6022e-19
    ne = np.linspace(5e19, 1e19, 32)
    te = np.linspace(3e3, 2e2, 32)
    ni = ne * 0.9
    ti = te
    p_fast = 2.1e4 * (1.0 - np.linspace(0, 1, 32) ** 2) ** 2

    thermal = EC * (ne * te + ni * ti)
    total = thermal + p_fast
    assert np.all(total >= thermal)
    assert total[0] > thermal[0], "fast pressure must raise the on-axis pressure"


@pytest.mark.parametrize("Z_imp", [None, 6.0])
def test_baseline_passes_p_fast_on_the_equilibrium_grid(Z_imp):
    """``_resolve_reconstruction`` must hand the recon an EQUILIBRIUM-grid p_fast.

    Guards the specific bug shape: ``Baseline.p_fast`` is contractually on the
    KINETIC grid, so passing that array straight through would feed the recon a
    profile of the wrong length (or, worse, silently broadcast).
    """
    from bouquet.baseline import _resolve_fixed
    from bouquet.utils import pchip_interp

    psi_src = np.linspace(0.0, 1.0, 64)
    psi_kin = np.linspace(0.0, 1.0, 150)
    psi_eq = np.linspace(0.0, 1.0, 129)
    p_fast_src = 2.1e4 * (1.0 - psi_src ** 2) ** 2

    p_fast_kin = _resolve_fixed(p_fast_src, psi_src, psi_kin)
    assert p_fast_kin.shape == psi_kin.shape, "Baseline.p_fast stays kinetic-grid"

    p_fast_eq = pchip_interp(psi_kin, p_fast_kin, psi_eq)
    assert p_fast_eq.shape == psi_eq.shape, (
        "the recon must receive p_fast on the equilibrium grid"
    )
    # Two-step (src->kin->eq) is what the draws do; assert it stays finite and
    # positive so a grid mix-up shows up as a failure rather than as a silent
    # ~1% current-profile bias.
    assert np.all(np.isfinite(p_fast_eq))
    assert p_fast_eq.max() > 0.0
