# The unified reconstruction engine (`reconstruction_engine="unified"`)

One reconstruction loop for both input types -- a g-file (+ p-file / IDA
profiles) and a modelling source (IMAS / OMAS IDS) -- behind
`GenerationConfig.reconstruction_engine`. **Default `"legacy"`**: the existing
reconstruction and IMAS baseline paths, bit for bit; nothing in this page
runs unless `"unified"` is set.

**Status (Stage 2).** The engine builds the baseline (`Bouquet.prepare_baseline()`
returns the same `Baseline` the rest of the package consumes, plus
`Baseline.engine`, the full record). `generate()` and
`verify_sigma0_consistency()` **refuse** an engine baseline until the draws
run on the engine (Stage 3): the legacy draw routes compose the bootstrap with
a different conversion and keep the pressure-driven term frozen in the
inductive, so a legacy draw would not reproduce the engine's reconstruction at
zero perturbation.

Code: `bouquet/engine.py` (the engine, the TokaMaker backend, the wiring),
`bouquet/adapters.py` (the source adapters), the kernel
`bouquet.jbs_loop.run_jbs_loop` (unchanged except an opt-in hook for added
criteria). Tests: `tests/test_engine.py` (a toy Grad-Shafranov stand-in),
`tests/test_engine_adapters.py`, `tests/test_engine_wiring.py`,
`tests/test_engine_solver.py` (`-m solver`), probe
`tests/probes/measure_engine.py`.

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

**What a draw inherits (Stage 3).** Composing on the stored geometry
snapshot with the stored components, `x*` and `lambda_BS*` (+ the delivery
correction) reproduces the stored request bit for bit (tested); the state
also carries the q0 row and target, so a draw can keep the q0 row as an
option for sawtoothing discharges.

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

## Cost

Measured on the toy: Ip + l_i converges in 5–7 passes (the toy's flux range
responds to l_i with the measured log-gain of 2); anchor 2 + passes + delivery
2 solves. The MSE stage adds 1 + 8 finite-difference solves and 4–8 passes.
The live numbers are written by the solver probe.

## Deviations from the design note

See the Stage 2 report; in short: the l_i discrepancy is measured against the
model on the NEW geometry (the note's `model(x_k; G_k)` form is unstable on
the gain-2 geometry mode); the MSE linearisation is centred on the solved
(β-blended) coefficients; `"fd_chord"` is offered beside `"fd_broyden"`;
the q0 row starts from the anchor's achieved axis current; the IDS q0 target
is the source's own q at the row radius; generate() refuses engine baselines
until Stage 3.
