# Physics notes

Behavioural notes on the parts of the pipeline where the physics, not the
plumbing, determines the answer. The full derivations, sign conventions, and
numerical floors live in [`../architecture.md`](../architecture.md); this page
covers the guarantees a user should know about and the knobs that change them.

## Contents

- [What the ensemble is (and isn't)](#what-the-ensemble-is-and-isnt)
- [The σ=0 consistency guard](#the-0-consistency-guard)
- [Bootstrap current treatment](#bootstrap-current-treatment)
- [Differential bootstrap (`jbs_delta_mode`)](#differential-bootstrap-jbs_delta_mode)
- [Kinetics regridding](#kinetics-regridding)
- [Edge-profile classification](#edge-profile-classification)
- [Hybrid kinetics on the IMAS path](#hybrid-kinetics-on-the-imas-path)
- [Z_eff-primary density scheme](#z_eff-primary-density-scheme)
- [Corrective j_phi iteration](#corrective-j_phi-iteration)

---

## What the ensemble is (and isn't)

The selected (in-spec) draws are a **realizability-filtered sensitivity
ensemble**, not a calibrated Bayesian posterior. Read them as *"equilibria
consistent with the stated kinetic-profile uncertainty that remain broadly
machine-realizable"* — useful for sensitivity and what-if analysis. They are
**not** a posterior you should quote calibrated probabilistic confidence
intervals from. Specifically:

- Only the **kinetic profiles** are sampled (the prior); the equilibrium is
  forward-solved with the **boundary held** and the **coils left to drift**.
  Pinning the coils instead (and letting the boundary move) gives a *different*
  ensemble — neither is "the" posterior; the choice privileges the
  magnetics-measured boundary.
- Selection is a **hard threshold** on coil drift and boundary RMS (approximate
  Bayesian computation), **not** a likelihood weighting by the real measurement
  covariances, and there is no joint correlation structure between the
  perturbed quantities and the constraints.
- The coil thresholds (especially the VSC channel metric) are **engineering
  heuristics**, not the true measurement/control uncertainties — see
  [coil-constraints.md](coil-constraints.md) for the specific assumptions and
  their limits.

A genuinely calibrated posterior would replace the hard cut with soft
likelihood weighting using the joint coil/magnetics/kinetics covariances; that
is future work.

## The σ=0 consistency guard

The pipeline is designed so that at σ=0 (no kinetic perturbation) the output
reproduces the reconstructed equilibrium to within reconstruction's own
residual: l_i within ~0.5% of target, X-point within ~2 mm, boundary RMS
~3–4 mm against the input g-file. This requires the recon-anchor solve in
`perturb_kinetic_equilibrium` (replacing the seed-based inductive with
reconstruction's `j_inductive_fit`) and a post-anchor l_i tolerance gate; see
[architecture.md §3.3](../architecture.md#33-li-matching-recon-anchor--adaptive-gate).

`Bouquet.verify_sigma0_consistency()` turns that design intent into a runnable
check. It replays the exact per-draw pre-bootstrap sequence — state-anchor solve
at the baseline j_phi/pressure, `solve_with_bootstrap` on the baseline
kinetics, toroidal conversion, axis-transition smoothing — and compares the
resulting spike to `baseline.j_BS`:

```python
b.reconstruct()                          # or b.prepare()
res = b.verify_sigma0_consistency(tol_frac=0.02)
assert res["passed"], res["max_dev_frac"]
b.generate()
```

It returns `spike0`, `max_dev`, `rms_dev`, `max_dev_frac`, `psi_worst`, and
`passed`, and costs one bootstrap solve (~1 min). Call it after
`reconstruct()` / `prepare_baseline()` and before `generate()`; it leaves the
solver re-anchored on the baseline equilibrium.

**Why it exists.** Any systematic deviation at σ=0 is inherited by *every* draw
as a j_phi target bias. The 2026-07 hollow-core bug was exactly such an
inconsistency: near-axis Gaussian smoothing of j_BS was applied on the
reconstruction path only, so every draw got the raw collapsed innermost-surface
point instead. The resulting grid-point-scale axis deficit hollowed every
draw's core by ~9% and shifted the q0 distribution wholesale by +12% — while
leaving l_i unbiased, and therefore invisible to every existing diagnostic.
The fix mirrors the smoothing into the draw path so both share it; this guard
is what keeps it from silently regressing.

## Bootstrap current treatment

The per-draw bootstrap comes from TokaMaker's Sauter/Redl
`solve_with_bootstrap`, whose parallel output is converted to toroidal with the
flux-surface geometry factor `c = 1/(⟨R⟩⟨1/R⟩)` (`bouquet.physics.parallel_to_toroidal`).

Two composition modes:

- **Shared near-axis smoothing** (default). Every spike — baseline and draws
  alike — passes through the same `smooth_jbs_transition` axis treatment. This
  is the fix for the hollow-core bug above and is what
  `verify_sigma0_consistency` exercises.
- **Differential composition** (`jbs_delta_mode=True`), below.

`recalculate_j_BS=False` skips the per-draw recompute entirely and keeps the
baseline j_BS — useful for isolating pressure-driven from current-driven
responses, not for production.

Both `from_geqdsk` and `from_imas` set `isolate_edge_jBS=False`, i.e. the
unified forward decomposition: `j_inductive` is pure ohmic and `j_BS` carries
the full physical Sauter profile (core hump + edge spike). It closes exactly,
is non-negative, and yields better than the older isolated-edge-spike split.
Set `isolate_edge_jBS=True` only for dedicated edge-spike studies.

## Differential bootstrap (`jbs_delta_mode`)

Opt-in alternative to the shared smoothing. Each draw's spike is composed as

```
j_BS(draw) = baseline_j_BS + [ SWB_raw(perturbed) − SWB_raw(σ=0) ]
```

with both solver terms **raw** (no smoothing of the perturbed profiles). Any
common-mode evaluation artifact — the collapsed innermost-surface point being
the motivating example — cancels exactly, and the per-draw Sauter response
passes through unfiltered. The l_i conditioning machinery is unchanged.

Cost: one extra `solve_with_bootstrap` call per run for the σ=0 reference,
computed in the same pre-draw anchor context. Under this mode the σ=0 draw
reproduces the baseline split exactly *by construction*, so
`verify_sigma0_consistency` becomes a pure bootstrap-context-reproducibility
probe rather than an independent check.

## Kinetics regridding

Measurement-grid kinetic profiles are resampled onto the equilibrium `psi_N`
grid with **knot-passthrough PCHIP**, not linear interpolation. Linear
interpolation put staircase steps into the profiles, which the Sauter model
differentiates — producing a ~10× larger step-energy artifact in j_BS. PCHIP is
shape-preserving (no new extrema) and passes exactly through the measurement
knots, so the gradient inputs are clean without smoothing away real structure.

The same regridding applies wherever a kinetic quantity crosses grids
(`psi_N_kinetic` → `psi_N`), including the dual-grid case where the kinetic
profiles extend past ψ_N = 1 into the SOL: GPR sampling includes the SOL,
equilibrium solving uses the confined region only.

## Edge-profile classification

`classify_jphi_profile` labels the baseline current profile `H_mode`,
`Lmode_like_jphi`, or `L_mode` and reports edge-spike alignment metrics. Peak
detection uses a **two-pass valley height reference** rather than the
innermost-surface value, which is numerically fragile (it is exactly the
collapsed axis point that caused the hollow-core bug). A profile with
significant bootstrap but no detected edge peak now retains the full Sauter
split instead of having it zeroed.

## Hybrid kinetics on the IMAS path

`GenerationConfig.kinetic_source` selects where the IMAS-path baseline kinetics
come from:

| Value | ne / Te / Ti / ω_tor | Z_eff, Z_imp, n_i dilution | Currents, equilibrium, p_fast, anchors |
|---|---|---|---|
| `"fuse"` *(default)* | FUSE `core_profiles` | FUSE | FUSE |
| `"ida_hybrid"` | IDA `.cdf` fits, PCHIP-resampled onto the FUSE ψ_N grid | FUSE | FUSE |

`Bouquet.from_imas(..., ida_path=…)` selects `"ida_hybrid"` automatically and
also points `UncertaintyConfig.ida_path` at the same file, so the sigma
envelopes come from the measured fits rather than flat scalars. The optional
`efit01_geqdsk` argument replaces the FUSE boundary with a magnetics-only EFIT
separatrix.

Note that `anchor_pressure_to_equilibrium` defaults to `False` for exactly this
reason: with IDA-hybrid kinetics, anchoring to `equilibrium.pressure` would
force the trusted IDA pressure back onto the FUSE total.

## Z_eff-primary density scheme

Each draw perturbs {n_e, T_e, T_i, Z_eff} and *derives* the main-ion density
from quasi-neutrality with a single effective impurity charge (`impurity_Z`,
carbon 6.0 by default — set it for your device). n_i, n_z, and Z_eff therefore
stay mutually consistent in every sample, which a naive independent-perturbation
scheme cannot guarantee. One Z_eff value per draw. See
[architecture.md §4](../architecture.md#4-quasi-neutrality-and-impurity-handling).

## Corrective j_phi iteration

TokaMaker's `jphi-linterp` mode imposes the requested `j_phi(psi_N)` using
pre-solve geometry, so the *achieved* flux-surface-averaged j_phi drifts from
the request once ψ converges. On the reconstruction path an adaptive Newton
iteration (2–8 steps) drives the achieved profile back onto the target.

On the IMAS path the same correction is available as
`imas_corrective_jphi=True` but is **off by default** while it is validated:
the single-pass solve there measures a +3.2–3.4% achieved-j_phi offset, which
surfaces as l_i +3.4% and q(ψ_N) −5% against the source IDS. The archived
j_phi is always the *achieved* TokaMaker profile on both paths, not the
requested one.

A related, accepted artifact: a localized ~8–10% dip in core j_phi relative to
the input g-file, which is an l_i-versus-peakedness tradeoff intrinsic to
matching both. Pinning the core has been tried and is unstable. See
[architecture.md §16](../architecture.md#16-known-limitations-and-future-work).

A separate known issue — the small constant boundary offset from `jphi-linterp`
edge/separatrix handling that sets the ~0.5 mm σ=0 floor — is written up in
[ISSUE_jphi_edge_reconstruction.md](ISSUE_jphi_edge_reconstruction.md).
