# The unified reconstruction engine (`reconstruction_engine="unified"`)

One reconstruction loop for both input types -- a g-file (+ p-file / IDA
profiles) and a modelling source (IMAS / OMAS IDS) -- behind
`GenerationConfig.reconstruction_engine`. **Default `"legacy"`**: the existing
reconstruction and IMAS baseline paths, bit for bit; nothing in this page
runs unless `"unified"` is set.

**Status (Stage 3).** The engine builds the baseline (`Bouquet.prepare_baseline()`
returns the same `Baseline` the rest of the package consumes, plus
`Baseline.engine`, the full record) and runs the draws: `generate()` and
`verify_sigma0_consistency()` run on the engine when it built the baseline in
the same session ([Draws](#draws) below). A mismatched pair -- `"unified"`
with a baseline the engine did not build, or `"legacy"` with an engine
baseline -- is refused: the legacy draw routes compose the bootstrap with a
different conversion and keep the pressure-driven term frozen in the
inductive, so they would not reproduce an engine reconstruction at zero
perturbation (and vice versa).

Code: `bouquet/engine.py` (the engine, the TokaMaker backend, the wiring),
`bouquet/engine_draws.py` (the draws), `bouquet/adapters.py` (the source
adapters), the kernel `bouquet.jbs_loop.run_jbs_loop` (unchanged except an
opt-in hook for added criteria). Tests: `tests/test_engine.py` (a toy
Grad-Shafranov stand-in), `tests/test_engine_adapters.py`,
`tests/test_engine_wiring.py`, `tests/test_engine_draws.py` (the draws, on
the toy and a TokaMaker stand-in over it), `tests/test_engine_draws_legacy_ast.py`
(the legacy draw path is the frozen code), `tests/test_engine_solver.py`
(`-m solver`), probe `tests/probes/measure_engine.py`.

## The picture

```
 g-file adapter                         IDS adapter
 <j.B>_in - Redl(anchor), smoothed      |B0| j_ohmic, beam j_parallel fixed
 rows: Ip, l_i(3) hard, (q0)            rows: Ip, li_3 soft, (q0), (MSE)
            \                               /
             v                             v
   +---------------------------------------------------------------+
   | compose on G_k (the latest SOLVED geometry):                  |
   |   J = F<1/R>/<B^2> [s_ind <j.B>_ind + s_bs <j.B>_BS + <j.B>_fix]|
   |       + p'(<R> - F^2<1/R>/<B^2>)                  (identity I2)|
   +---------------------------------------------------------------+
             |
             v
   closure (0 solves): rows + discrepancies, prior / preset
             |
             v
   relax (beta) -> ONE GS solve ----------------------+
             |                                        |
             v                                        |
   measure on the solved state: Redl, l_i, Ip and c,  |  pass
   q0, tan(gamma), request - achieved                 |
             |                                        |
             v                                        |
   update lambda_BS (omega), discrepancies, MSE J ----+
   criteria on 2 consecutive passes
             |
             v
   delivery solve (two passes) + checks on every row
   = THE reconstruction (Baseline, metrics, draw reference)
```

## The contract (`adapters.EngineContract`)

| item | g-file adapter | IDS adapter |
|---|---|---|
| kinetics n_e, T_e, n_i, T_i, Z_eff | p-file / IDA, PCHIP onto the g-file ψ_N (as the legacy reconstruction) | core_profiles (or the IDA hybrid), as `read_imas_baseline` resolves them |
| pressure | thermal + impurity + fast (fixed) | thermal + impurity + fast, **no `p_diff`** |
| inductive (parallel) | `<j.B>_in = F p' + F'<B^2>/mu0` on the g-file's own traced surfaces (identity I0), minus Redl `<j.B>` on the anchor, minus the fixed parts; smoothed with a verbatim copy of `fit_inductive_profile`'s basis (spline + PCHIP, zero edge anchor, ≥ 0), **no amplitude search** | `|B0| j_ohmic` (IMAS `<j.B>/B0`); when the source has none, the parallel residual `j_total − j_bootstrap − Σ driven`; refused when neither exists |
| fixed driven (parallel) | user `j_NBI` / `j_RF` (toroidal inputs), converted with the anchor's `F<1/R>/<B^2>` | core_sources beam `j_parallel` × `|B0|`, held fixed (a user override is converted as on the left) |
| boundary | g-file LCFS | equilibrium boundary outline |
| rows | Ip (exact); l_i(3) = the reader's `li(2)` key, **hard**, absolute tolerance 1e-3; q0 (optional) | Ip (soft, σ = 0.5 % of Ip); li_3 (soft, σ = 0.04); q0 (optional); MSE chords (optional) |
| signs | positive frame: `sign(Ip)`, `|F|`; a file whose `<j_phi>` disagrees in sign with its Ip is refused (wrong COCOS) | `source_current_sign`, `|B0|`; `b0_sign` recorded |

The IDS σ values are the `li_soft_onesided` preset's
(`utils.STRUCTURED_PRESETS`); the g-file l_i tolerance is the legacy step-5 /
re-match secant's (`_rematch_li_request`'s `li_tol` default, read from its
signature). The q0 target is the source's own q **at the measurement radius**
(the ψ_pad-clipped axis sample the closure's axis row uses), admitted by
`utils.q0_gate_admits` (sawtooth model active, or `|q0_source| ≤ q0_gate`).
MSE rows accept **E_r-corrected** pitch angles only (`er_corrected=True`, no
`Er`); a raw-E_r modelling request is refused.

Pressures use `physics.ELEMENTARY_CHARGE`. The anchor E_0 is one (two-pass) solve of the source's own total current as
the legacy path solves it (g-file: `|j_tor_averaged_direct|`; IDS: `j_tor`);
it seeds the geometry and the first Redl bootstrap and does not enter the
fixed point.

## Composition (identity I2)

Components are stored as parallel currents `<j.B>` [T A/m²]. The solver's
variable -- the plain flux-surface average `<j_phi>` TokaMaker's
`jphi-linterp` consumes -- is formed in one place, `engine.compose`, on the
geometry of the latest solve:

```
<j_phi> = <j.B> F<1/R>/<B^2>  +  p' (<R> - F^2 <1/R>/<B^2>)
```

The first term is the field-aligned conversion (c) of the verification report
(the legacy `evaluate_jBS` output uses (a), `<j.B>/(F<1/R>)`, about 6 % high
at the bootstrap peak; the engine takes the Redl `<j.B>` from the evaluator's
diagnostics and converts it itself). The second term is the pressure-driven
toroidal current (diamagnetic + Pfirsch–Schlüter, zero `<j.B>`), recomputed
every pass from that equilibrium's own `p'` and geometry (13–22 % of `<j_phi>`
at ψ_N 0.9–0.97 on the synthetic examples). `F`, `<B^2>` come from the
evaluator's surfaces (`sauter_fc`), `<R>`, `<1/R>`, `<1/R^2>`, `V'`, `p'` from
`utils.fsa_current_geometry`. The Redl `<j.B>` receives the shared
innermost-surface repair (`smooth_jbs_transition`) every SWB-derived profile
receives.

## A pass

1. Compose on `G_k` with the bootstrap iterate `lambda_BS,k`.
2. Closure, zero solves: `utils.close_ip_structured` (hard rows: the g-file)
   or `close_ip_structured_soft` via `soft_closure_with_retry` (soft rows:
   the IDS), with the rows and their discrepancies; the Ip round trip checked
   by `ip_roundtrip_gate` (0.05 %). A refusal (scale bounds 0.2 < s < 5, a
   degenerate row, a failed round trip) raises `EngineClosureRefused` with
   its reason.
3. Relax: `js = (1−β) js_prev + β jc` (`CurrentRelaxer`, `jbs_relax_current`).
4. **One GS solve** of `js` (+ the delivery correction, when on).
5. Measure on the solved state: Redl `<j.B>`, l_i (`li_achieved`, li_3), Ip,
   the uniform factor `c = achieved / request` (linear Ip measure), q at the
   row radius, tan γ at the chords, request − achieved (core/edge, % of peak).
6. Update (between passes, never after the last): `lambda_BS` by the kernel
   (ω); the row discrepancies; the MSE linearisation point and Jacobian; the
   delivery correction.

The kernel is `run_jbs_loop` with `step` = 1–5; the engine's rows enter
through its `extra` hook and the existing `AxisRowPin`.

### Rows and their update

| row | model in the closure (on `G_k`) | measurement on `E_k+1` | update |
|---|---|---|---|
| Ip | `Ip_fsa_weights` affine exact measure | the solver imposes Ip; `c` recorded | — |
| l_i | `structured_li_model` (li_3) | `li_achieved` | `d_k = (1−ω) d_k−1 + ω [l_i(E_k+1) − l_i_model(js_k; G_k+1)]`; the closure's target is `T − d` |
| q0 | the axis-current row | q at the row radius | `AxisRowPin`: `j_ref0 ← j0_solved · q0 / q0_target` |
| MSE | `tan γ ≈ tg0 + J (x − x0)` | `mse_tan_gamma` of the solved field (`mse_field_at` -> `(B, found)`) | offset refreshed from every solve; `J` by finite differences once at convergence, then Broyden (`"fd_broyden"`) or held (`"fd_chord"`) |

MSE, as the loop's own chord stage: at the first read (the finite-difference
base) chords OFF the solver mesh are excluded with their reason; fewer than
`structured_mse_min_chords` left leaves the MSE term NOT applied and the
slice flagged (refused with `structured_mse_required=True`); a chord missing
on a later read is a refusal. The field orientation is STATED, never fitted:
`sign_pol = ip_sign(data)·ip_sign(equilibrium)`, `sign_tor =
bt_sign(data)·bt_sign(equilibrium)`, the equilibrium's read off its field;
if another orientation fits the chords better by Δχ² >
`mse.MSE_ORIENTATION_DCHI2` the slice is flagged and the stated one KEPT.
The adapters complete a block that states no `ip_sign`/`bt_sign` from the
source's declared orientation in the (R, φ, Z) frame (g-file: CURRENT and
BCENTR signs times its COCOS's σ_RpZ; IDS: `ip`, `b0` signs, COCOS 11) and
refuse a block the source contradicts. MSE knobs the engine never reads
(any, without the `"mse"` row; `structured_mse_steps` with it) are refused
as on the legacy path.

On a **hard** row the closure imposes `model + d = T`, so at the fixed point
the delivered l_i is on the target. On a **soft** row the fit weighs
`(model + d − T)/σ`, which at the fixed point is the delivered residual
`(l_i − T)/σ` (the `LiRowPin` soft semantics). The l_i discrepancy is taken
against the model of the current that was actually SOLVED, evaluated on the
NEW geometry -- see "Deviations" below.

The MSE linearisation point is the coefficient vector whose current was
solved: with the current relaxed it is the same β-blend of the closure's
coefficients (the composition is affine in them).

### Convergence (existing constants only, two consecutive passes)

`engine.convergence_table()` returns this, with each value read from its home:

| criterion | value | where it lives |
|---|---|---|
| r_j | 1e-3 | `GenerationConfig.jbs_rtol_j` |
| r_I | 1e-4 | `GenerationConfig.jbs_rtol_Ip` |
| \|Δl_i\| | 1e-3 | `GenerationConfig.jbs_tol_li` |
| \|Δq0\| (q0 row) | 2e-3 | `GenerationConfig.jbs_tol_q0` |
| \|q0 − q0_target\| (q0 row) | 0.01 | `GenerationConfig.q0_tol` (via `AxisRowPin`) |
| l_i, g-file hard row | 1e-3 | `TokaMaker_interface._rematch_li_request` `li_tol` default |
| l_i, IDS hard row / soft-row discrepancy | 0.005 | `GenerationConfig.structured_li_tol` |
| MSE tan γ change | 0.1 σ_eff | `jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA` |
| closure-half current residual (standing, decision 9) | 1e-3 | `jbs_rtol_j`, the loop's current gate |
| consecutive passes / ceiling | 2 / 8 | `JBS_REQUIRED_CONSECUTIVE` / `jbs_max_passes` |
| ω floor / growth abort | 0.25 / 3 | `JBS_RELAX_FLOOR` / `JBS_GROWTH_ABORT_PASSES` |
| closure scale bounds | 0.2 < s < 5 | `close_ip_structured` default |
| Ip round trip | 0.05 % | `utils.IP_ROUNDTRIP_TOL_PCT` |

The MSE stage runs the loop again after the Jacobian, with its own ceiling
`jbs_max_passes` (as the legacy chord stage).

### Failure

Exactly the loop's: `JBSNotConverged` raises (or, with
`jbs_loop_on_fail="flag"`, the result is delivered flagged
`closure_limited`); `JBSNonFinite` raises at once whatever the policy; a
closure refusal raises `EngineClosureRefused` with its reason; a failed GS
solve raises `EngineSolveError`. A failed build leaves no baseline
(`Bouquet._failed_baseline` keeps the half-built one).

### Delivery

After the loop (and the MSE stage), the closure is run once more on the last
solved geometry with the last bootstrap and discrepancies, **unrelaxed**, and
solved with two passes. It is checked with `jbs_loop.check_delivered` (r_j,
r_I) and on every row: the l_i row error, q0 against `q0_tol` and Δq0, the MSE
tan γ change, Δl_i and the closure-half current residual against the last
pass. A miss raises `JBSNotConverged` (or flags). That solve, its request,
its geometry snapshot, `x*`, `lambda_BS*` and the discrepancies are the
reconstruction: `EngineState`, recorded in `Baseline.engine["state"]`.

**What a draw inherits.** Composing on the stored geometry snapshot with the
stored components, `x*` and `lambda_BS*` (+ the delivery correction)
reproduces the stored request bit for bit (tested, and re-checked by every
draw context before it draws); the state also carries the q0 row and target,
so a draw can keep the q0 row as an option for sawtoothing discharges.

## Presets

| `engine_preset` | basis / prior | rows admitted |
|---|---|---|
| `"structured"` (default) | 4 Gaussians (ψ_N 0.15/0.45/0.75/0.95, width 0.20), `li_soft_onesided` σ ladders | Ip, l_i, q0, MSE |
| `"bootstrap_scalar"` | constant basis, `s_ind` pinned | Ip |
| `"sawtooth_two_scalar"` | constant basis, two scalars | Ip, q0 (falls back to `"bootstrap_scalar"` with a notice when the gate rejects q0, as the legacy sawtooth channel) |

The scalar presets reduce exactly to `close_ip("bootstrap")` and
`close_ip_q0` (tested).

## The delivery correction (`engine_delivery_correction`, default off)

With the solver's `jphi-linterp` defect present, the achieved current differs
from the request by a local edge footprint (~2.8 % of peak at ψ_N 0.996 on
the synthetic g-file). With the option on, each pass adds
`Δ = request − achieved/c` of the previous pass to the request (one Newton
step with a unit Jacobian), after the relaxation of the intended current, so
the delivered current approaches the intended composition. `Δ` carries no
net current in the linear Ip measure and is part of the state a draw
inherits. Its default is to be decided after the solver fix, by re-running
the distance-to-input table (`tests/probes/measure_engine.py`, part
`recon_dc`).

## Draws

`bouquet/engine_draws.py`; `Bouquet.generate()` builds a
`GenerateEngineDraws` from the live reconstruction and hands it to
`generate_bouquet(engine_draw=...)`, whose per-draw loop then calls it in
place of the legacy `perturb_kinetic_equilibrium` (everything else --
the warm start, the coil regularisation, the homotopy, the archive, the
until-N ledger -- is the same code). The parallel launchers need nothing
new: every worker runs `prepare_baseline()` + `generate()`.

```
 reconstruction state: G*, x*, lambda_BS*, (Delta*), request R*
            |
            v
 sample (legacy stream): kinetics (pressure-matched), aux, ONE inductive
 candidate (toroidal sigma_jphi, j_ls); the run's bootstrap scale
            |
            v
 anchor (0 solves): lambda_0 = scale [lambda_BS* + Redl(draw kin) - Redl(recon kin)]
            |
            v
 ONE loop (run_jbs_loop; ceiling jbs_max_passes_draw; current gate standing)
   compose on G_k with x* HELD, p' from the draw's own pressure
   close the Ip row: J + d_ind * (inductive term), exact measure
   [+ q0 row: the two-scalar increments, AxisRowPin moves the axis row]
   relax (beta) -> ONE solve -> measure (Redl with the draw's kinetics)
            |
            v
 coil homotopy (engine_draw_homotopy, default on) -> post-homotopy check
 (the existing _post_homotopy_jbs, the engine draw's own passes, the
 saturation guard)
            |
            v
 post-hoc filters on the archived draw: l_i band (l_i_tolerance), 
 constrain_sawteeth; coil + boundary as always  ->  in_spec / selected
```

**Inputs.** Every perturbed quantity is formed as `base + (drawn - base)`:
the kinetics on the kinetic grid (then PCHIP'd as the adapter does), the
solve pressure from the adapter's own assembly (thermal + impurity + fast),
the auxiliary channels, the parallel inductive. The random stream is the
legacy one through the first inductive candidate (`engine_draws.RNG_STREAM`):
the kinetic channels `ne, Te, (Zeff -> ni | ni), Ti` redrawn together until
the flux-integrated thermal pressure matches within `p_thresh`, the
auxiliary channels in their order, then one inductive candidate drawn IN
TOROIDAL UNITS with the legacy call on `s_ind(x*) kappa* lambda_ind` (so its
toroidal perturbation is the legacy draw's for the same normals: today's
`sigma_jphi` and `j_ls`) and mapped back to `lambda_ind`; it is redrawn only
while negative where its mean exceeds its sigma, and clipped to zero in the
floor zone (the standard route's rule). The legacy routes then draw further
candidates (Fix C band resampling, the standard route's l_i pre-screen); the
engine draw does not, so a seeded engine run and a seeded legacy run share
their streams up to the first legacy draw that resampled. The bootstrap
scale is the run's `jBS_scale_range` sample, used as is (1.0 is the
reconstruction's value -- `s_bs(x*)` already carries the reconstruction's
scaling, so the legacy `bs_scale` centring is not applied).

**The Ip row.** The one row a draw closes: an increment on the inductive
term, `J = J0 + d_ind s_ind kappa lambda_ind'`, with
`d_ind = [(Ip_lin* - Ip_lin(J0; G_k)) + (c* - c(G_k))] / Ip_lin(ind; G_k)`
in the exact (`jphi-linterp`) measure, the target being the Ip the
reconstruction's delivered composition carries in that measure on `G*`
(`Ip` itself on the hard g-file row, the posterior on the soft IDS row).
Zero extra solves; `a_ind = 1 + d_ind` is recorded per pass. With
`engine_draw_q0_row=True` (needs the reconstruction's active q0 row) the
sawtooth two-scalar system is solved in the same increment form on the
inductive and bootstrap terms, and `AxisRowPin` moves the axis row once per
pass from the measured q0 (the q0 criteria are then loop criteria).

**Zero-perturbation identity by construction.** With every perturbation
zero and the scale 1.0: the inputs are the base exactly, the first pass
composes on `G*` (its `p'` shifted by the draw's pressure change, zero), the
anchor increment is zero, and the Ip increment is formed from differences
that vanish exactly -- so the first request IS the stored request, bit for
bit (recorded per draw as `identity.pass1_request_bit_identical`). The solve
of it from the warm state reproduces the reconstruction, the loop's pass-1
residuals are the reconstruction's delivered ones, and the draw delivers the
reconstruction to the loop tolerances. With the current gate standing (it is
measured one pass late) the loop takes `JBS_REQUIRED_CONSECUTIVE + 1 = 3`
passes. `verify_sigma0_consistency()` under the engine runs exactly this
draw and gates `passed` on the draw-route rule at the unchanged tolerances
(request identical, loop converged, `r_j`, `r_I` against `lambda_BS*`,
`|dl_i| <= jbs_tol_li`), reporting `dq0` at its labelled radius and `dq95`.

**Post-hoc filters, not matching.** A draw matches no l_i, q0 or MSE row;
l_i and beta_N drift and are recorded. The l_i band
(`|l_i - l_i*| <= l_i_tolerance l_i*`, default 0.05, around the
reconstruction's l_i) and `constrain_sawteeth` (`q0 >= 1` at the q-row
radius, psi_N = psi_pad, the legacy gate's radius) are applied to the
ARCHIVED draw: a draw outside a band is archived with `in_spec=False` and
`passes_draw_band=False`, never dropped; the until-N ledger and
`.filter()`'s `selected` both AND the band into the coil + boundary verdict.
Rejected (never archived, never counted), with their
`DRAW_REJECTION_REASONS` code: a loop that does not converge
(`jbs_not_converged`), a non-finite bootstrap (`jbs_non_finite`, at once), a
failed first solve (`anchor_solve_failed`, the anchor's analog: the stored
state composed with the draw's components), a refused amplitude closure
(`engine_closure_refused`), a coil saturation (`coil_saturation_jbs_loop` /
`_post_homotopy`), the homotopy and post-homotopy codes as before, and --
only with `draw_solve_maxits` set -- `homotopy_maxits` /
`post_homotopy_maxits` (below).

**Bootstrap refresh after the first solve (`engine_draw_bootstrap_refresh`,
default off).** The loop's start is computed on the reconstruction's
geometry `G*`; the first solve moves the geometry (the flux range by
several per cent for a typical inductive sample) and with it the Redl
bootstrap, so a relaxed blend toward the stale start costs passes. With the
setting on, after the loop's FIRST solve the anchor's form is re-evaluated
with the draw-kinetics Redl taken on that solved geometry,
`scale [lambda_BS* + Redl(draw kin; G_1) - Redl(recon kin; G*)]`
(`engine_draws.REFRESH_SOURCE`), and pass 2 restarts from it instead of the
blend `(1 - omega) lambda_0 + omega J_1`; every later pass blends as usual
(`jbs_loop.run_jbs_loop(start_refresh=...)`). Zero extra solves and zero
extra Redl evaluations (pass 1's Redl is the loop's own). It changes the
PATH only: the first request (still bit-identical at zero perturbation),
every criterion, tolerance and the ceiling are untouched, and every pass is
judged against the bootstrap it was solved with. At zero perturbation the
refreshed bootstrap is `lambda_BS*` plus the change of Redl between the
stored and the re-solved equilibrium: rounding on the toy stand-in, and on
the live solver the re-solve's own reproduction of `G*` (measured on the
g-file example: a refresh step of r_j = 1.9e-5, against `jbs_rtol_j` =
1e-3; the pass count at zero perturbation is unchanged, 3). The loop
record carries `bootstrap_refresh` (pass 1's residuals before it, pass 2's
after it, the refresh step, `I_BS` start / evaluated / refreshed).

### l_i controllability

How much the draws' l_i scatters, and why, is recorded per draw so it is
predictable and known rather than tuned away. Expected sizes (design note
§2.8): the inductive-shape sampling dominates (the golden in-spec ensemble's
sigma(l_i) = 0.021, about 3 %, band-truncated); kinetics through Redl and the
pressure term and the Ip amplitude about 0.5-1.5 % (not yet measured); the
loop itself at most `jbs_tol_li` (1e-3); the solver's delivery defect is a
common shift, not a spread.

**The attribution record** (`engine.attribution` per draw). The closure's
own l_i gradient (`utils.structured_li_model` / `structured_li_gradient`,
li_3) on the RECONSTRUCTION geometry `G*`, applied along the toroidal
directions of the draw's change: `inductive` (`s_ind kappa* (lambda_ind' -
lambda_ind)`), `bootstrap` (`s_bs kappa* (lambda_BS' - lambda_BS*)`),
`pressure` (`P(p'_draw) - P(p'*)`), `amplitude` (`d_ind s_ind kappa*
lambda_ind'`) and, with the q0 row, `q0_row`. `remainder = delta_l_i -
linear_total` is split into `nonlinear_frozen_geometry` (the full model on
`G*` minus the linear sum) and `geometry_and_delivery` (the delivered l_i
minus the model on `G*`: the geometry's response and the solver). With Ip
pinned by the amplitude the model is linear in the current, so on a fixed
geometry the attribution is exact (tested on the toy: remainder at
rounding); the remainder is the geometry and delivery part.

**Cost.** Every draw records solves, passes and wall time by stage
(`anchor` -- no solve --, `loop`, `homotopy`, `post_homotopy`, `filters`,
`archive`), counted by the draw loop's `DrawSolveGuard` (every GS solve of
the draw). On the toy a draw takes 3-5 loop passes (3 at zero perturbation)
and one solve per pass; the homotopy adds one solve per stage it runs.
`python tests/probes/measure_engine.py OUTDIR --draws 6 --seed 12345` writes
the live numbers for the g-file example (the legacy batch measured 405-1407 s
per draw, 4 archived / 2 rejected / 1 in spec at that seed).

**The solve cap.** `draw_solve_maxits` (the ported #57 cap, default `None`
= the solver's own cap) is applied by the engine's solve wrapper
(`TokaMakerBackend.solve`) to EVERY engine solve -- reconstruction, the
draw's loop and its post-homotopy passes -- and restored after each; under
`Bouquet.generate` the draw loop's `DrawSolveGuard` also sets it on the
solver for the whole of `generate()`, so the homotopy's own solves run
under it; an engine draw additionally installs it for its homotopy stage
when the solver does not already carry it (`generate_bouquet` driven
directly) and puts the previous value back afterwards. A capped solve that
does not converge (the solver's own `Exceeded "maxits"`) REJECTS the draw,
loudly and with its own code: `homotopy_maxits` for a homotopy pass or a
rollback re-solve -- never rolled back to a looser pass and archived --
and `post_homotopy_maxits` for a post-homotopy pass. A loop solve that hits
it keeps the loop's codes (`anchor_solve_failed` on pass 1,
`perturb_failed` after). With the default `None` nothing is re-classified.
The cap's value is a decision for the owner (no default is set).

## Cost

Measured on the toy: Ip + l_i converges in 5–7 passes (the toy's flux range
responds to l_i with the measured log-gain of 2); anchor 2 + passes + delivery
2 solves. The MSE stage adds 1 + 8 finite-difference solves and 4–8 passes.
A draw: 3-5 loop passes of one solve each (see [Draws](#draws)). The live
numbers are written by the solver probe.

## Deviations from the design note

See the Stage 2 report; in short: the l_i discrepancy is measured against the
model on the NEW geometry (the note's `model(x_k; G_k)` form is unstable on
the gain-2 geometry mode); the MSE linearisation is centred on the solved
(β-blended) coefficients; `"fd_chord"` is offered beside `"fd_broyden"`;
the q0 row starts from the anchor's achieved axis current; the IDS q0 target
is the source's own q at the row radius. The draws: the inductive is drawn
in toroidal units and its negative excursions follow the standard route's
floor-zone rule (not Fix C's all-positive retry); the loop starts from
`lambda_BS*` plus the Redl kinetic increment on the starting equilibrium
(zero at identity); the first pass shifts `G*`'s `p'` by the draw's pressure
change; the Ip-row target is the Ip the delivered composition carries in the
exact measure (the posterior on a soft row); the optional q0 row is the
two-scalar closure in increment form; the delivery correction, when on,
keeps being updated per pass in the draw as in the reconstruction.
