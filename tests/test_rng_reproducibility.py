"""The reproducibility contract: one seed governs every draw.

``GenerationConfig.seed`` used to reach only NumPy's legacy global RNG, while
every GPR draw site called ``generate_perturbed_GPR(..., rng=None)`` and got a
fresh ``np.random.default_rng()`` -- OS entropy per draw.  Seeded ensembles
were therefore not regenerable, and no draw-level value could be pinned.

These are the solver-free halves of the contract:

  * ``make_rng`` behaves as the single seed -> Generator entry point;
  * the samplers honour an injected Generator (same seed -> bitwise identical,
    different seed -> different, no rng -> independent);
  * every draw call site in the perturbation path passes ``rng=`` -- a source
    check, because the defect was silent at runtime (unseeded draws still
    *look* fine) and only a structural assertion prevents its return;
  * a committed golden pins the seeded draw stream bitwise
    (``tests/golden/rng_stream_manifest.json``).

The live-solver half -- two seeded ``generate()`` runs producing bitwise
identical archives -- is in ``test_seeded_reproducibility.py`` (``solver``).
"""
import ast
import hashlib
import json
import os

import numpy as np
import pytest

from bouquet.sampling import (make_rng, generate_perturbed_GPR,
                              _draw_monotonic_perturbation)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
_SLIM = os.path.join(_GOLDEN_DIR, "D3Dlike_Hmode_golden_slim.h5")
_RNG_MANIFEST = os.path.join(_GOLDEN_DIR, "rng_stream_manifest.json")

_PSI = np.linspace(0.0, 1.0, 65)
_PROFILE = 1.0 - 0.85 * _PSI ** 2
_SIGMA = 0.05 * np.ones_like(_PSI)


# ---------------------------------------------------------------------------
#  make_rng -- the one seed-consumption point
# ---------------------------------------------------------------------------
class TestMakeRng:
    def test_int_seed_is_deterministic(self):
        assert make_rng(7).standard_normal(4).tolist() == \
            make_rng(7).standard_normal(4).tolist()

    def test_distinct_seeds_give_distinct_streams(self):
        assert make_rng(7).standard_normal(8).tolist() != \
            make_rng(8).standard_normal(8).tolist()

    def test_none_is_fresh_entropy(self):
        assert make_rng(None).standard_normal(8).tolist() != \
            make_rng(None).standard_normal(8).tolist()

    def test_generator_passes_through_unchanged(self):
        """A caller that already owns a stream can inject it."""
        g = np.random.default_rng(3)
        assert make_rng(g) is g

    def test_returns_a_generator(self):
        for s in (None, 0, 12345):
            assert isinstance(make_rng(s), np.random.Generator)


# ---------------------------------------------------------------------------
#  the samplers honour an injected Generator
# ---------------------------------------------------------------------------
class TestSamplersHonourRng:
    def test_gpr_same_seed_is_bitwise_identical(self):
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_gpr_different_seed_differs(self):
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(43))
        assert not np.allclose(a, b)

    def test_gpr_without_rng_is_not_reproducible(self):
        """rng=None is the documented OS-entropy mode -- keep it that way."""
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1)
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1)
        assert not np.allclose(a, b)

    def test_monotonic_draw_same_seed_is_bitwise_identical(self):
        """The rejection loop must draw from the RUN's stream, not a new one:
        it consumes a variable number of draws, so a per-attempt generator
        would desynchronise every later channel even if the seed were set."""
        kw = dict(psi_N=_PSI, normalised_profile=_PROFILE,
                  sigma_profile=_SIGMA, length_scale=0.3)
        np.testing.assert_array_equal(
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)),
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)))

    def test_monotonic_draw_different_seed_differs(self):
        kw = dict(psi_N=_PSI, normalised_profile=_PROFILE,
                  sigma_profile=_SIGMA, length_scale=0.3)
        assert not np.allclose(
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)),
            _draw_monotonic_perturbation(**kw, rng=make_rng(12)))

    def test_one_generator_sequences_a_multi_channel_draw(self):
        """Two channels off ONE Generator reproduce as a *sequence*: replaying
        only the second call must NOT match, or the stream is not shared."""
        rng = make_rng(5)
        first = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                       n_samples=1, rng=rng)
        second = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                        n_samples=1, rng=rng)
        assert not np.allclose(first, second)
        rng2 = make_rng(5)
        np.testing.assert_array_equal(
            first, generate_perturbed_GPR(_PSI, _PROFILE,
                                          sigma_profile=_SIGMA, n_samples=1,
                                          rng=rng2))
        np.testing.assert_array_equal(
            second, generate_perturbed_GPR(_PSI, _PROFILE,
                                           sigma_profile=_SIGMA, n_samples=1,
                                           rng=rng2))


# ---------------------------------------------------------------------------
#  structural: no draw site may fall back to OS entropy
# ---------------------------------------------------------------------------
_DRAW_FUNCS = {"generate_perturbed_GPR", "_draw_monotonic_perturbation"}
_PERTURB_MODULE = os.path.join(_HERE, "..", "bouquet",
                               "TokaMaker_interface.py")


def _draw_calls(path):
    """(lineno, func, passes_rng) for every draw call in a module."""
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in _DRAW_FUNCS:
            yield node.lineno, name, any(k.arg == "rng" for k in node.keywords)


def test_every_draw_site_threads_the_generator():
    """The regression guard for the whole bug class.

    An unseeded draw site is invisible at runtime -- it produces perfectly
    good-looking numbers -- so the only durable protection is asserting that
    no call site omits ``rng=``.  Nine sites existed when this was written.
    """
    calls = list(_draw_calls(_PERTURB_MODULE))
    assert calls, "found no draw call sites; did the module move?"
    missing = [(ln, fn) for ln, fn, has_rng in calls if not has_rng]
    assert not missing, (
        f"draw call sites without an explicit rng= (they would fall back to "
        f"OS entropy and break the seed contract): {missing}")


def test_generate_bouquet_consumes_the_seed_once():
    """``seed`` must become a Generator, not just np.random.seed()."""
    with open(_PERTURB_MODULE) as fh:
        src = fh.read()
    assert "rng = make_rng(seed)" in src
    # and the two formerly-legacy draws must come off it
    assert "rng.uniform(" in src, "scale_jBS draw is not on the run Generator"
    assert "rng.normal(" in src, "l_i target draw is not on the run Generator"


# ---------------------------------------------------------------------------
#  committed golden: the seeded draw stream, bitwise
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (os.path.isfile(_SLIM) and os.path.isfile(_RNG_MANIFEST)),
    reason="golden fixture not built; run tests/golden/make_golden_fixture.py")
def test_seeded_draw_stream_matches_golden():
    """Pin the seeded draw stream against the golden baseline profiles.

    Pure NumPy (no solver, no mesh), so this is bitwise-portable and is the
    first draw-level golden the package can hold at all: before the seed
    reached the GPR the stream was OS entropy.  Re-pin deliberately with
    ``python tests/golden/make_golden_fixture.py --rng-stream-only`` and review
    the manifest diff.
    """
    import h5py
    import sys
    sys.path.insert(0, _GOLDEN_DIR)
    from make_golden_fixture import draw_stream

    with open(_RNG_MANIFEST) as fh:
        man = json.load(fh)

    with h5py.File(_SLIM, "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = (hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None
              else hf["_baseline"])
        psi_kin = np.asarray(bl["psi_N_kinetic"][()], dtype=float)
        psi_N = np.asarray(bl["psi_N"][()], dtype=float)
        profiles = {"ne": bl["n_e"][()], "te": bl["T_e"][()],
                    "ni": bl["n_i"][()], "ti": bl["T_i"][()],
                    "jphi": bl["j_phi"][()]}
        sigmas = {"ne": bl["sigma_ne"][()], "te": bl["sigma_te"][()],
                  "ni": bl["sigma_ni"][()], "ti": bl["sigma_ti"][()],
                  "jphi": bl["sigma_jphi"][()]}

    assert psi_kin.size == man["grids"]["psi_N_kinetic"]
    assert psi_N.size == man["grids"]["psi_N"]

    drawn = draw_stream(psi_kin, psi_N, profiles, sigmas, seed=man["seed"])
    assert set(drawn) == set(man["channels"])
    for ch, exp in man["channels"].items():
        a = np.ascontiguousarray(drawn[ch], dtype=np.float64)
        assert a.size == exp["n"], ch
        # readable first: which value moved
        for i, v in zip(exp["sample_indices"], exp["sample_values"]):
            assert float(a[i]) == v, f"{ch}[{i}]"
        assert float(a.min()) == exp["min"], ch
        assert float(a.max()) == exp["max"], ch
        # then exhaustive
        assert hashlib.sha256(a.tobytes()).hexdigest() == exp["sha256"], (
            f"{ch}: draw stream changed bitwise (samples still matched, so the "
            f"change is elsewhere in the profile)")


@pytest.mark.skipif(
    not (os.path.isfile(_SLIM) and os.path.isfile(_RNG_MANIFEST)),
    reason="golden fixture not built; run tests/golden/make_golden_fixture.py")
def test_golden_draw_stream_actually_perturbs():
    """Guard the guard: a golden of an all-zero-sigma stream would pass for
    the wrong reason."""
    import h5py
    with h5py.File(_SLIM, "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = (hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None
              else hf["_baseline"])
        for k in ("sigma_ne", "sigma_te", "sigma_ni", "sigma_ti",
                  "sigma_jphi"):
            assert float(np.max(np.abs(bl[k][()]))) > 0.0, k
