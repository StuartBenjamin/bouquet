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

import ast
import inspect
import textwrap

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


def _materialize_nested(outer, name, env):
    """Compile and return a nested ``def`` out of ``outer``'s REAL source.

    The kin->eq helpers are closures inside multi-hundred-line functions that
    cannot be called without a solve.  Rather than re-implement them in the
    test (which is how the previous version of this contract test ended up
    comparing ``pchip_interp`` to itself and became unfailable), this lifts the
    nested ``def`` verbatim out of the enclosing function's source and executes
    it with its free variables supplied through ``env``.

    So the object under test IS the shipped code: if someone edits
    ``baseline._resolve_reconstruction.to_eq`` to call ``np.interp``, the
    materialized function calls ``np.interp`` too, and the comparison fails.
    """
    src = textwrap.dedent(inspect.getsource(outer))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name == name
                and node is not tree.body[0]):
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            ns = dict(env)
            exec(compile(mod, f"<{outer.__name__}.{name}>", "exec"), ns)
            return ns[name]
    raise LookupError(
        f"nested def {name!r} no longer exists inside {outer.__name__} -- the "
        "kin->eq contract test can no longer see the code it guards; re-point "
        "it rather than deleting it"
    )


def test_recon_and_draw_kin_to_eq_agree_on_the_same_inputs():
    """The recon path's kin->eq map must equal the draw path's, bit for bit.

    Executes the ACTUAL nested helpers -- ``baseline._resolve_reconstruction``'s
    ``to_eq`` (reconstruction path) and ``perturb_kinetic_equilibrium``'s
    ``_kin_to_eq`` (draw path) -- lifted from their real source, and compares
    them on identical inputs.  If either drifts to a different interpolant the
    recon and the draws again solve slightly different fast pressures, which is
    exactly the regression this file exists to prevent.
    """
    import bouquet.baseline as _baseline
    import bouquet.TokaMaker_interface as _tmi
    from bouquet.utils import pchip_interp

    psi_kin = np.linspace(0.0, 1.0, 150)
    psi_eq = np.linspace(0.0, 1.0, 129)

    draw_k2e = _materialize_nested(
        _tmi.perturb_kinetic_equilibrium, "_kin_to_eq",
        dict(np=np, pchip_interp=pchip_interp,
             _dual_grid=True, psi_kin=psi_kin, psi_N=psi_eq),
    )
    recon_k2e = _materialize_nested(
        _baseline._resolve_reconstruction, "to_eq",
        dict(np=np, pchip_interp=pchip_interp,
             psi_N_kin=psi_kin, psi_N=psi_eq),
    )

    # A smooth core profile, a pedestal-like profile (where a linear regrid
    # differs most from PCHIP), and a non-monotone one.
    profiles = {
        "parabolic": 2.1e4 * (1.0 - psi_kin ** 2) ** 2,
        "pedestal": 1.0 / (1.0 + np.exp((psi_kin - 0.95) / 0.02)),
        "wiggly": np.sin(6.0 * np.pi * psi_kin) + 2.0,
    }
    for label, arr in profiles.items():
        a = draw_k2e(arr)
        b = recon_k2e(arr)
        np.testing.assert_array_equal(
            a, b, err_msg=f"recon and draw kin->eq disagree on {label!r}"
        )
        assert a.shape == psi_eq.shape

    # Shape-preserving and non-negative for a non-negative input.
    out = draw_k2e(profiles["parabolic"])
    assert np.all(out >= -1e-9), "kin->eq regrid must not undershoot below zero"


def test_every_kin_to_eq_site_routes_through_pchip_interp():
    """No kin->eq site may use a different interpolant.

    The two helpers exercised above are ``def``s; the remaining sites are
    lambdas/defs inside ``generate_bouquet`` that cannot be materialized as
    cleanly.  Assert structurally that each one calls ``pchip_interp`` and that
    no linear regrid (``np.interp`` / ``interp1d``) has crept into the kin->eq
    role next to it.
    """
    import bouquet.baseline as _baseline
    import bouquet.TokaMaker_interface as _tmi

    # (module, nested-helper name) pairs that perform a kin->eq regrid.
    sites = [
        (_tmi, "_kin_to_eq"), (_tmi, "_kin2eq"), (_tmi, "_k2e"), (_tmi, "_to_eq"),
        (_baseline, "to_eq"),
    ]
    trees = {}
    for mod, _ in sites:
        trees.setdefault(mod, ast.parse(inspect.getsource(mod)))

    seen = set()
    for mod, name in sites:
        bodies = []
        for node in ast.walk(trees[mod]):
            # `def name(...)` form
            if isinstance(node, ast.FunctionDef) and node.name == name:
                bodies.append(node)
            # `name = lambda ...` form
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                if any(isinstance(t, ast.Name) and t.id == name
                       for t in node.targets):
                    bodies.append(node.value)
        assert bodies, (
            f"no kin->eq helper named {name!r} found in {mod.__name__}; the "
            "site was renamed or removed -- re-point this contract test"
        )
        for body in bodies:
            called = {c.func.id for c in ast.walk(body)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            attr_called = {
                f"{c.func.value.id}.{c.func.attr}" for c in ast.walk(body)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
            }
            assert "pchip_interp" in called, (
                f"{mod.__name__}.{name} (line {body.lineno}) no longer regrids "
                f"through pchip_interp; it calls {sorted(called | attr_called)}"
            )
            forbidden = {"np.interp", "interp1d", "_interp1d"}
            assert not (forbidden & (called | attr_called)), (
                f"{mod.__name__}.{name} (line {body.lineno}) uses a LINEAR "
                "regrid in the kin->eq role"
            )
            seen.add((mod.__name__, name))

    assert len(seen) == len(sites), "some kin->eq site went unchecked"


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


class _DummyEqdsk:
    """Just enough of a geqdsk for the p_fast shape guard (no solve)."""

    def __init__(self, n=129):
        self.psi_N = np.linspace(0.0, 1.0, n)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(2.1e4, id="python-scalar"),
        pytest.param(np.float64(2.1e4), id="numpy-scalar"),
        pytest.param(np.array(2.1e4), id="0d-array"),
        pytest.param(np.array([2.1e4]), id="length-1-array"),
        pytest.param(np.zeros(150), id="kinetic-grid-length"),
        pytest.param(np.zeros((129, 1)), id="2d-column"),
    ],
)
def test_reconstruct_equilibrium_rejects_non_equilibrium_grid_p_fast(bad):
    """A scalar/length-1/wrong-length p_fast must raise, not broadcast.

    Before the guard, ``pres_tmp + np.asarray(p_fast)`` silently broadcast a
    scalar into a flat pressure offset, so the recon solved a different
    pressure from the draws (which regrid and would have raised) -- reviving
    the very inconsistency this module fixes.
    """
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    eqdsk = _DummyEqdsk(129)
    with pytest.raises(ValueError, match=r"p_fast must be a 1-D array"):
        # mygs=None is safe: the guard runs before anything touches the solver.
        reconstruct_equilibrium(
            None, eqdsk, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            p_fast=bad,
        )


def test_reconstruct_equilibrium_p_fast_error_names_the_expected_shape():
    """The ValueError must tell the caller what to do, not just that it failed."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    eqdsk = _DummyEqdsk(129)
    with pytest.raises(ValueError) as exc:
        reconstruct_equilibrium(
            None, eqdsk, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            p_fast=np.zeros(150),
        )
    msg = str(exc.value)
    assert "(129,)" in msg and "(150,)" in msg, "error must show both shapes"
    assert "pchip_interp" in msg, "error must point at the kin->eq remedy"


def test_Z_imp_is_scalar_by_design_and_rejects_arrays():
    """Z_imp needs no shape guard -- it is a scalar and coercion fails loudly.

    Documents why the p_fast guard has no Z_imp twin: ``impurity_pressure``
    does ``float(Z_imp)``, so a non-scalar raises rather than broadcasting.
    """
    from bouquet.physics import impurity_pressure

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    with pytest.raises((TypeError, ValueError)):
        impurity_pressure(ne, ni, ti, np.array([6.0, 6.0]))


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
