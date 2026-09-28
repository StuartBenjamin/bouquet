# PR: self-consistent bootstrap current (default ON)

*Draft pull-request description for `feat/jbs-self-consistent-loop`. The PR is
not opened yet; this file is the text it will carry. Stacked on
`feat/structured-mse-term` (the structured closure's MSE term), which it
includes.*

## Summary

Until now bouquet computed the bootstrap current **once** — OFT's
`solve_with_bootstrap` (SWB) on its own auxiliary equilibrium — and then only
rescaled it. This PR replaces that with a Redl bootstrap re-evaluated on the
equilibrium bouquet actually delivers, iterated to self-consistency with the
current closure and the Grad–Shafranov solve, and makes it the **default**.

Two defects of the frozen bootstrap motivate it:

1. **Grid.** SWB assumes an evenly sampled ψ_N. bouquet never passed `psi_N=`,
   so on the IMAS path — whose integrated-modelling grid is uniform in ρ_tor —
   profile gradients were mis-scaled by dψ_N/du (well below 1 near the axis,
   above 1 beyond mid-radius) and the geometry sampled at the wrong surfaces.
   The geqdsk path (uniform grid) is unaffected.
2. **Frozen geometry.** The bootstrap was evaluated on SWB's generic-seed,
   thermal-only equilibrium, never on the delivered one (source inductive
   current, full pressure, closure multipliers, post-homotopy coils). Since
   j_BS ∝ (dp/dψ_N)/Δψ and Δψ scales with √l_i, the amplitude error is
   several percent, and the closures, correctors and draws inherited it.

## What changes

- **`physics.evaluate_jBS`** — a port of SWB's inner Redl evaluation run once
  on the equilibrium it is handed (no solve inside), with geometry sampled on
  the caller's surfaces, gradients on the true grid, and a direct
  ⟨j·B⟩ → toroidal conversion. Bit-identical to SWB's first pass on a uniform
  grid; grid-independent on a non-uniform one (the defect-1 regression test).
- **`bouquet/jbs_loop.py`** — the fixed-point kernel: closure on the current
  geometry → GS solve → Redl, with joint under-relaxation of the bootstrap
  (ω = 0.7) and of the solved current (β = 0.7; damps the closure ↔ geometry
  oscillation), ω halved only on sustained growth, convergence = every active
  residual on **two consecutive passes** (`r_j ≤ 1e-3`, `r_I ≤ 1e-4 I_p`,
  `Δl_i ≤ 1e-3`, `Δq0 ≤ 2e-3`), ceilings 8 (baseline) / 6 (+2 post-homotopy)
  per draw. Non-convergence raises `JBSNotConverged` with the full history,
  or flags the slice (`jbs_loop_on_fail="flag"`); a non-converged draw is a
  failed draw.
- **Everywhere a bootstrap enters j_φ:** the IMAS baseline in every
  `jBS_baseline_mode` and closure channel (structured closure incl. its MSE
  stage: Jacobian once, chord steps with j_BS re-evaluated, one final Jacobian
  refresh), every draw (Fix C and the standard l_i loop, plus a post-homotopy
  check), `verify_sigma0_consistency` (the loop's own σ=0 invariant) and the
  geqdsk reconstruction. In diff mode `jBS_diff` becomes a pure model offset on
  the delivered baseline geometry, so a σ=0 draw still reproduces the source
  bootstrap exactly.
- **Structured soft closure stop test** — called once per loop pass, it now
  accepts an iterate stationary to within the objective's rounding noise
  (`stop_reason="noise_floor"`, recorded) instead of refusing it; inside the
  loop a refusal is retried once from the previous pass's coefficients
  (logged). This extends the solver's acceptance only where it previously
  refused; every result it returned before is bit-identical.
- **Default ON** (`GenerationConfig.jbs_self_consistent=True`).
  `jbs_self_consistent=False` is the legacy frozen path, bit for bit. A stored
  config that predates the field loads with the loop off (and warns), so an
  old archive's `config_json` replays the model it was produced with.
  `single_profile_jphi=True` / `recalculate_j_BS=False` (no bootstrap to
  iterate) are refused unless the legacy flag is set.
- **Archive schema v3** (additive): the `jbs_loop` block
  (`jbs_converged`, `jbs_n_passes`, `jbs_loop_json`) on every loop draw and on
  `_baseline`; readers `DrawView.jbs_loop`, `ScanView.baseline_jbs_loop` /
  `bootstrap_model`. No migration: a v2 archive reads as frozen everywhere.
- **Plots** label the bootstrap "self-consistent Redl bootstrap" or "frozen
  SWB bootstrap (legacy)" from what the archive records.
- **Post-homotopy check of a standard draw** re-solves through the draw's own
  Ip renormalisation + corrective iteration instead of handing the achieved
  current back to one jphi-linterp solve (which exhausted `maxits`).

No existing solver tolerance (`nl_tol`, `maxits`, `structured_li_tol`,
`q0_tol`, the soft solver's `rtol`/`max_iter`, …) and no test bar changed.

## Evidence

- **Legacy flag is the legacy path, bit for bit.** Out-of-tree A/B against the
  pre-loop tree on the synthetic D3D-like example (same machine, same OFT
  build, one thread): geqdsk reconstruction + σ=0 check + two draws, IMAS
  diff baseline + one draw, and the IMAS structured (ohmic) closure — every
  array, attr and archived dataset identical (276 / 156 / 177 compared items
  respectively); the only
  differences are the new closure stop-test bookkeeping keys (additive).
  In-tree, a solver test tripwires the loop kernel and the Redl evaluator on
  the legacy flag and asserts neither is entered while `solve_with_bootstrap`
  is.
- **Grid independence** (fast + live): the same physical profiles on a
  uniform and a strongly non-uniform ψ_N grid give the same j_BS to
  interpolation accuracy; the legacy evenly-sampled reading does not.
- **Fixed point** (fast + live): init-independent (anchor vs legacy init
  converge to the same baseline); a loop started at its fixed point returns
  it in one pass; relaxation changes the path, not the fixed point (tested
  against the closed-form fixed point of a two-state model).
- **Real-data A/B (summarised, device-agnostic).** On a multi-shot
  tokamak campaign (hundreds of slice-channels, two closure channels) the
  loop converged on ≥ 99 % of slices in 4–5 passes at roughly neutral cost;
  the remaining slices failed loudly (flagged, excluded) rather than silently.
  The bootstrap fraction of I_p moved down by a few percent of I_p on IMAS-path
  hybrids and a spurious mid-radius inductive cut the frozen bootstrap forced
  on the closure disappeared; the q = 2 location moved by a few 10⁻³ in ψ_N.

## Golden refresh

**Not done — blocking.** A loop-on regeneration of the geqdsk-path golden
(from the fixture's own stored config, 20 draws, seed 12345, one thread)
converges the reconstruction (4 passes) but rejects essentially every draw:
the standard draw's Gauss–Seidel bootstrap coupling contracts at ≈0.38/pass
from r_j ≈ 2e-2…1e-1 and needs ≈7–9 passes against the per-draw ceiling of 6,
and the post-homotopy check (ceiling 2 with the two-consecutive-pass rule)
cannot accept a draw whose first pass misses (observed: r_I 1.08e-4 then
2.3e-5 → rejected). 0 of the first 12 draws were accepted. The ceilings are
approved convergence settings and are left unchanged; the fixture stays the
frozen-bootstrap one until they are settled (see `tests/golden/README.md`).

## Tests

Fast suite (no solver): 1155 passed, 1 failed on the laptop and on the Linux
production build — the failure is `test_the_fixture_says_what_built_it`,
which requires the provenance stamp the (not yet refreshed) fixture predates.
Solver suites on the Linux production build: fsa 6 passed / 1 skipped
(build-aware collapse demonstration), harness 1, loop solver 19 (incl. the
legacy-flag tripwire test), l_i closure 10, seeded reproducibility 12,
systematics 2 passed / 1 failed (the pre-existing golden l_i(1) miss,
documented in `tests/golden/README.md`).

## Reviewer notes

- The behaviour change is intended: archives made with this release are not
  comparable to earlier ones draw-for-draw; compare them as two bootstrap
  models (`ScanView.bootstrap_model`), or rerun with
  `jbs_self_consistent=False`.
- Drivers that relied on the loop being OFF by default for a "frozen"
  reference arm must now set `jbs_self_consistent=False` explicitly.
- Not in this PR: iterating the diff-mode *baseline* (pinned to the source
  total by design), a ψ_N-label remap of the kinetic profiles, and any
  fallback for a GS failure inside a blended pass (such a pass fails loudly).
