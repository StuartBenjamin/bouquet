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
  `Δl_i ≤ 1e-3`, `Δq0 ≤ 2e-3`). Pass ceilings (limits, not tolerances):
  8 for the baseline / reconstruction, 12 for each loop of a draw
  (`jbs_max_passes_draw`) and 4 post-homotopy passes
  (`jbs_max_passes_post_homotopy`). Non-convergence raises `JBSNotConverged` with the full history,
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
- **Where a draw's loop starts** (recorded per loop as `init_source`): Redl
  at the draw's state anchor on the draw's OWN perturbed kinetics for its
  first loop, warm from its previous converged bootstrap for later ones, and
  a relaxed blend with Redl on the delivered equilibrium post-homotopy --
  never the unperturbed baseline bootstrap. Initialisation only: a fast and a
  solver test start the same draw loop from the baseline bootstrap and reach
  the same fixed point.
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

## Figures

All five figures come from the public synthetic example only: the
D3D-like g-file, p-file and mesh in `examples/D3D-like`, configured from the
golden fixture's stored config (read the way
`tests/golden/regenerate_golden_run.py` reads it). Each is a single-slice
reconstruction on the geqdsk path, run one thread at a time, with no draws and
no IMAS. `docs/figures/make_bootstrap_loop_figures.py` regenerates them. Each
figure's computation is its own subcommand, taking about 1–3 min per
reconstruction; `all` runs every subcommand. The numbers quoted below come
from the development laptop build. The loop-on reconstruction reproduces the
refreshed fixture's baseline j_BS to 5×10⁻⁵ of its peak, and its l_i target
to 2×10⁻⁷. Bootstrap fractions are `I_BS / I_tor`: both are integrals of the
delivered profiles, weighted by the loop's own current measure
(`residual_weights`), so the normalisation of that measure cancels.

**Figure 1 — what the flag changes** (`docs/figures/jbs_loop_fig1_profiles.pdf`).
Solid lines show the delivered reconstruction with the loop on
(self-consistent Redl bootstrap). Dashed lines show the same reconstruction
with `jbs_self_consistent=False` (frozen SWB, legacy). Orange is j_BS, blue is
the inductive current and black is the total j_tor, all against ψ_N. The left
panel covers the full radius; the right panel zooms on the pedestal. The two
totals nearly coincide, because the reconstruction fits j_tor to the g-file
either way. What moves is the split between bootstrap and inductive current.
The self-consistent pedestal bootstrap peak is 5 % higher (0.534 →
0.562 MA m⁻²), and the inductive current gives up the same current (−11 kA) in the
edge, beyond ψ_N ≈ 0.85. I_BS/I_p goes from 0.2293 to 0.2385 (I_BS 284 → 296 kA). The
l_i target barely moves (0.653833 → 0.653864), and neither does q0
(1.2306 → 1.2311).

**Figure 2 — convergence of the loop** (`docs/figures/jbs_loop_fig2_residuals.pdf`).
This plots the per-pass residuals of the loop-on reconstruction on a log
scale:
- r_j (current-weighted L2 distance between the Redl bootstrap of the new
  equilibrium and the one it was solved with);
- r_I (the same distance as a fraction of I_p);
- Δl_i;
- Δq0.

Each has its tolerance as a horizontal line. Dashed means the criterion is
gated; dotted means it is not. Δq0 is only logged here: it is not a
criterion on the reconstruction path. The figure script records q0 per pass
through a logging wrapper and never gates on it. Relaxation is ω = 0.7 on the
bootstrap and β = 0.7 on the solved current, and ω is never halved.

Passes 1–4 are the main loop, where each pass is an inductive fit plus the
l_i secant. It converges in 4 passes: passes 3 and 4 are the two consecutive
"ok" passes. r_j falls by about 0.3 per pass (1 − ω): 4.5×10⁻³ → 1.4×10⁻³ →
4.2×10⁻⁴ → 1.3×10⁻⁴. The corrective iteration then moves the equilibrium. The
open symbols are the post-corrective check, r_j = 1.8×10⁻³ and
r_I = 4.9×10⁻⁴, which is outside tolerance. Three post-corrective passes
(5–7) bring it back to r_j = 5.9×10⁻⁵ and r_I = 1.4×10⁻⁵.

**Figure 3 — defect A, the grid** (`docs/figures/jbs_loop_fig3_grid.pdf`).
Top panel: Redl j_BS on the delivered equilibrium, from the same physical
profiles sampled in three ways:
- black: on the uniform ψ_N grid (257 points, the reference);
- blue dashed: on a strongly non-uniform grid (513 points uniform in √ψ_N,
  ρ-like), with that grid passed to `evaluate_jBS`;
- vermillion: the same non-uniform arrays read as if they were evenly
  sampled, which is the silent assumption SWB made and bouquet never
  corrected.

Bottom panel: the difference from the reference, as a percentage of peak
j_BS. With the grid passed, the result is grid-independent to interpolation
accuracy. The current-weighted error is 0.17 %, and in the shaded mid-radius
band (0.3 ≤ ψ_N ≤ 0.7) it is below 3×10⁻⁵ of the peak. The legacy reading is
wrong everywhere. The gradient is mis-scaled by dψ_N/du = 2√ψ_N and the
geometry is sampled on the wrong surfaces. The result is low inside
ψ_N ≈ 0.14, 8–12 % of peak high across mid-radius, and nearly doubled (+92 %
of peak) at the pedestal. Its weighted error is 79 %.

**Figure 4 — the fixed point does not depend on the start** (`docs/figures/jbs_loop_fig4_init.pdf`).
The same loop-on reconstruction was started from four initial bootstraps:
- the anchor evaluation (the default);
- the legacy SWB profile (`jbs_init="swb"`);
- the anchor evaluation × 0.8;
- the anchor evaluation × 1.2.

The two scaled starts are made by a figure-only wrapper that multiplies the
loop's first iterate.

Left panel: the delivered j_BS of all four, with the pedestal in the inset.
The curves lie on top of each other. The largest pairwise current-weighted
distance is 2.6×10⁻⁵, 40× below r_j's tolerance, and the l_i targets agree to
6×10⁻⁷. I_BS/I_p is 0.2385 in all four cases.

Right panel: r_j per pass, with the main loop filled and the post-corrective
loop open. A worse start costs passes, not accuracy:
- anchor: 4 + 3 passes;
- SWB: 6 + 3;
- ×0.8: 8 + 3;
- ×1.2: 8 + 3.

Every start contracts at the same rate. The ±20 % starts converge exactly at
the baseline ceiling of 8 passes. That is a deliberately bad start, not the
production one, but it shows how much headroom that ceiling leaves.

**Figure 5 — σ = 0 reproduces the baseline** (`docs/figures/jbs_loop_fig5_sigma0.pdf`).
This is `verify_sigma0_consistency` under the loop on the geqdsk path, with
no IMAS. The σ = 0 draw loop runs from the state anchor with the unperturbed
kinetics and converges in 4 passes.
- Top panel: its j_BS (orange dashed) over the baseline's (black).
- Bottom panel: the difference as a percentage of peak.
- Text box: the invariant's own measures.

The differences are:
- r_j against the baseline: 4.3×10⁻⁵ (tolerance 10⁻³);
- r_I: 8.3×10⁻⁶ (tolerance 10⁻⁴);
- |Δl_i|: 2.0×10⁻⁶ (tolerance 10⁻³), against the delivered baseline
  equilibrium's post-corrective l_i of 0.656455;
- largest pointwise difference: 5.4×10⁻⁵ of peak, at ψ_N ≈ 0.11.

The σ = 0 draw therefore lands on the baseline 12–500× inside the loop's own
tolerances.

## Golden refresh

**Done.** The geqdsk-path golden fixture is regenerated loop-on from its own
stored config (20 requested draws, one thread, the current OFT line; the
build identity is stamped into the fixture and manifest):

- reconstruction loop converged in 4 passes;
- 17 draws archived, 10 in spec; one skipped (an l_i-match candidate's solve
  exhausted `maxits`, the pre-existing failure mode of that solve) and two
  rejected at the post-homotopy stage (known limitation below);
- every archived draw's loops converged: anchor loops 3–6 passes, l_i-match
  candidate loops 4–9 (mostly 7–8; 1–5 candidates per draw), post-homotopy
  0 (accepted as delivered, 3 draws) or 2–4 passes; no loop reached its
  ceiling;
- recorded physics: baseline l_i target 0.65384 → 0.65386, baseline
  I_BS/I_p 0.2285 → 0.2380; draw l_i(1) mostly 1–10 % lower and I_BS/I_p
  higher (the draws are different realisations of the same seed, since the
  l_i-match paths differ).

The seeded draw-stream golden (`rng_stream_manifest.json`) is unchanged for
the kinetic channels; only the `jphi` channel's hash moved, because it is
drawn from the baseline j_φ, which now carries the self-consistent bootstrap.

## Known limitation: slow post-homotopy divergence of a standard draw

Two of the 20 golden draws were rejected after ~30–60 min each: the
post-homotopy corrective re-solve (at the homotopy's coil bounds, I_p and
p_axis pinned) diverges -- every flux-surface trace fails, the solve spends
its whole iteration budget -- and the diverged state is refused by the
`get_q` axis-collapse guard. The draw is rejected loudly, never accepted.
The corrective iteration's first input is the achieved-derived target
itself, so routing the pass through the corrective iteration did not change
the request that fails. Options, none implemented (details in
`tests/golden/README.md`): fail fast on a growing solve residual or failed
trace; start the corrective iteration from the draw's last corrective input;
`protect_state` for a clearer failure reason; a recorded failure field;
status quo.

## Tests

Fast suite (no solver) on the refreshed fixture: 1165 passed on the laptop
(the provenance test that needed the refreshed fixture now passes; the new
`test_the_fixture_is_a_self_consistent_bootstrap_run` is included). Golden
tests (`test_golden_bouquet.py`): 21 passed on the laptop and on the Linux
production build. Solver suites on the Linux production build before the
refresh: fsa 6 passed / 1 skipped (build-aware collapse demonstration),
harness 1, loop solver 19 (incl. the legacy-flag tripwire test), l_i
closure 10, seeded reproducibility 12, systematics 2 passed / 1 failed (the
golden l_i(1) miss of the pre-refresh fixture, documented in
`tests/golden/README.md`). Systematics against the refreshed fixture on
the Linux production build: 2 passed / 1 failed. Mode 3 now passes (draw 0
within every bar; draw 3's replay produced no equilibrium -- a `maxits`
failure -- and was skipped by the test). Mode 1 is a NEW failure: max coil
drift 1.2292 % against 0.3 % (maximal on F9B; the whole coil set moves), already present between the
test's class-API loop-on reconstruction and the fixture's loop-on baseline
(the two agreed to 0.006 % with the frozen bootstrap). Diagnosed, not fixed;
no bar changed (`tests/golden/README.md`, "Known limitation: mode-1 coil
drift after the refresh"): the cause is not the loop but the refresh's
archival convention -- regenerated through `Bouquet.generate()`, the fixture
archives the ACHIEVED baseline current, where the earlier goldens' recipe
archived the INPUT current the replay feeds back; fed the generator's input,
the test's own entry point reproduces the refreshed baseline coils to
0.0006 % (mode-1 metric 0.0092 %). Options listed there; a decision is
needed. **Blocking for the PR.**

## Reviewer notes

- The behaviour change is intended: archives made with this release are not
  comparable to earlier ones draw-for-draw; compare them as two bootstrap
  models (`ScanView.bootstrap_model`), or rerun with
  `jbs_self_consistent=False`.
- Drivers that relied on the loop being OFF by default for a "frozen"
  reference arm must now set `jbs_self_consistent=False` explicitly.
- Not in this PR: iterating the diff-mode *baseline* (pinned to the source
  total by design) and a ψ_N-label remap of the kinetic profiles.
- Follow-ups discussed but **not approved or implemented**: a fallback for a
  GS failure inside a blended pass (such a pass fails loudly today); the
  fail-fast guard on a diverging corrective solve (known limitation above);
  and starting each l_i-match candidate's loop from Redl on that candidate's
  own geometry (today later candidates start warm from the draw's previous
  converged bootstrap; their larger first residual is geometric).
