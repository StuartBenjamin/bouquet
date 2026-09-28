# Golden bouquet test fixtures

Git-tracked regression fixtures for `tests/test_golden_bouquet.py`.

| file | what it is |
|------|------------|
| `D3Dlike_Hmode_golden_slim.h5` | a slimmed real bouquet run (~11.8 MB): `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles). |
| `golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points) with tolerances. |
| `rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the slim fixture's baseline profiles + sigma envelopes. |
| `make_golden_fixture.py` | regenerates the three files above from a full run. |

## The draw-stream golden

`rng_stream_manifest.json` is the only draw-*level* golden here, and it became
possible only when `GenerationConfig.seed` started reaching the GPR: before
that, every draw site re-seeded from OS entropy and no drawn value was
reproducible. It replays what `perturb_kinetic_equilibrium` does for one draw
— ne, Te, ni, Ti through `_draw_monotonic_perturbation`, then the `j_phi` GPR
candidate — off one `make_rng(seed)` Generator. Pure NumPy: no solver, no
mesh, so it is bitwise identical on any machine.

Re-pin it on its own (no full run needed) with

```bash
python tests/golden/make_golden_fixture.py --rng-stream-only
```

The manifest carries sampled values and per-channel min/max alongside each
hash, so the git diff shows roughly *where* a stream moved, not just that it
did. A changed hash with unchanged samples means the change is elsewhere in
the profile.

The geqdsks are deliberately retained: geqdsk is a coarse-at-the-separatrix
format and exercising its read/parse path on real files (see the
`test_geqdsk_*` tests) is worthwhile. They are stored as gzipped `uint8`
under their original `.eqdsk` dataset names, so every reader
(`bytes(grp[k][()])`) is unaffected. `make_golden_fixture.py --eqdsk` chooses
retention: `all` (default, ~11.8 MB), `subset` (baseline + representative
draws, ~5 MB), or `none` (~3.7 MB, no geqdsk-handling coverage). Eventually,
when the default interchange migrates to IMAS/OMAS, the fixture can store
those instead.

The full-fidelity 30 MB run (with p-file bytes and uncompressed eqdsks) stays
as the shareable example artifact under
`bouquet/examples/D3D-like/D3Dlike_Hmode_golden.h5` (not tracked here).

## Updating the golden set (on purpose)

1. Re-run the example notebook to produce a fresh full `.h5`.
2. Regenerate the fixture + manifest:
   ```bash
   python tests/golden/make_golden_fixture.py \
       --source /path/to/D3Dlike_Hmode_golden.h5
   ```
   (defaults to the D3D-like example artifact path if `--source` is omitted;
   `rng_stream_manifest.json` is re-pinned from the new slim fixture in the
   same command).
3. Review the `golden_manifest.json` git diff — it shows exactly which physics
   values moved — then commit the new fixture + manifests together.

The builder refuses to write a fixture that names an absolute filesystem path
anywhere (string attrs, `config_json`, geqdsk headers) and reduces the paths
inside the self-consistent bootstrap records (`jbs_loop_json`, which carry the
OFT package path) to basenames: this repository is public, and the OFT build
is identified by the content digests in the provenance stamp instead.

## The self-consistent-bootstrap refresh

The fixture is a loop-on run: `GenerationConfig.jbs_self_consistent=True`
(the default), regenerated from the previous fixture's own stored config with
only the code and the bootstrap model changed, on the OFT line that carries
the bootstrap stencil change (build identity in the provenance stamp). Pass
ceilings 12 per draw loop and 4 post-homotopy; no tolerance moved. Of 20
requested draws, 17 are archived (10 in spec): one draw was skipped when an
l_i-match candidate's solve exhausted `maxits`, and two were rejected at the
post-homotopy stage (see the known limitation below). Every archived draw's
loops converged, the longest in 9 passes. `test_the_fixture_is_a_self_consistent_bootstrap_run`
asserts the stored config, a converged `jbs_loop` block on the baseline and on
every draw, and the path guard.

`rng_stream_manifest.json` was re-pinned on the same machine class as before:
the ne / Te / ni / Ti stream hashes are unchanged, and only the `jphi` hash
moved, because that channel is drawn from the fixture's baseline j_phi, which
now carries the self-consistent bootstrap (baseline j_phi moved by ~1 % of its
peak, sigma_jphi by ~1.5 %).

## Known build-dependent failure: `test_systematics::test_mode3_production_reproduces_golden`

On OFT builds that carry the 2026-08 bootstrap stencil change (the current
line, including the macOS development build and the Linux production build)
this replay misses the recorded l_i(1) of draw 3 by 3.74 % against its 3 %
bar (`l_i(1) replay 0.8240 vs golden 0.8560`), with l_i(3) (2.06 %) and the
boundary RMS inside their bars. The miss is the fixture's, not bouquet's: the
commit that produced the fixture reproduces it to four decimals on the newer
OFT, and passes on the older one; the signature (an l_i(1)-only, edge-localised
change) is the end-stencil difference of the bootstrap-gradient formula. The
bar is not widened; the failure is expected until the fixture is regenerated
on the current OFT line (see the pending refresh above).

## Known limitation: a standard draw's post-homotopy re-solve can diverge slowly

Seen in the loop-on regeneration of this example (config seed 12345, 20
requested draws, pass ceilings 12 per draw loop / 4 post-homotopy): archive
counts **1** and **17** were rejected this way, both before and after the
ceilings were raised. It is a rejected draw, loudly, never an accepted one;
nothing here caps, retries or falls back.

**What happens.** The draw's own loops converge (count 1: anchor 5 passes,
l_i-match candidate 6; count 17: anchor 3, then four l_i-match candidates of
7-8 passes each). The coil homotopy then delivers the draw (count 1 stops at
pass 2 of 3, F +/-2 %, VSC +/-5 %, because its natural VSC drift of 4.17 %
already exceeds the next pass's +/-1 %; count 17 reaches pass 3, F/VSC
+/-1 %, with 0.54 % / 0.29 % drift). Redl on the delivered equilibrium misses
the bootstrap the draw carries (count 1: `r_j = 2.0e-3`, `r_I = 2.4e-4`;
count 17: `r_j = 9.1e-4` inside, `r_I = 1.8e-4` outside), so the post-homotopy
stage takes its first pass: the delivered inductive held, the relaxed
bootstrap swapped in, the target renormalised to I_p (x1.0035 and x0.9957)
and handed to `_corrective_jphi_iteration`. The corrective iteration's
`jphi-linterp` solve then does not converge. A `jphi-linterp` solve
flux-surface-averages every nonlinear iterate, so once the iterate loses its
nested surfaces every iteration's surface trace fails (the log fills with
`gs_get_qprof: Trace did not complete` and `DLSODE ... R1 = NaN`) and the
solver spends its whole iteration budget: roughly 25-40 min for count 1 and
33-43 min for count 17 at one thread, inside `TokaMaker.solve` in every
periodic stack dump. For count 1 the log shows `[jphi_corr] WARNING: the
FIRST corrective solve failed (Error in solve: Exceeded "maxits")`; for
count 17 no such line appears (the post-homotopy call runs with
`verbose=False`, so a later-iteration failure, or a solve that returned a
degenerate state, is not distinguishable in the log). The call runs without
`protect_state`, so the solver is left in that state; the loop then measures
it, `fsa_current_geometry`'s guard refuses it (`get_q returned a CONSTANT <R>
across all 257 surfaces ... the surface tracer collapsed onto the magnetic
axis`), and the draw is rejected as OUT_OF_SPEC with a NaN VSC drift. This is
the broken state's collapse, not the build-dependent near-axis collapse of the
section above; it occurs on the Linux build.

**What the standard-draw post-homotopy fix changed, and why it did not remove
this.** Before that fix the pass solved `j_ind_used + j_BS` as ONE
`jphi-linterp` request, where `j_ind_used` is derived from the draw's stored
j_phi -- the ACHIEVED current of its corrective iteration, not the solver
input that achieved it; on count 1 that solve exhausted `maxits` (a scratch
60-iteration cap made it fail in 34 s; uncapped it ran for tens of minutes).
The fix routes the pass through the draw's own I_p renormalisation +
corrective iteration. But the corrective iteration's first input IS the
target (`j_phi_input = target_jphi.copy()`), so its first solve is handed the
same achieved-derived profile, rescaled by <0.5 %: not the input that
produced the delivered state (the draw's corrective iteration had moved its
input off the target by its Newton edge corrections, edge RMS
~0.014 -> ~0.001 MA/m^2). The failure moved from "the single solve exhausts
maxits" to "the corrective iteration's first solve exhausts maxits"; wall
time and outcome are the same.

**Why these draws.** Not established. Count 1 sits at a loose coil stage with
the VSC near its bound, which suggested a basin-escape under tight bounds;
count 17 falsifies that as the whole story (tightest stage, 0.29 % VSC
drift, a smaller first residual). What the two share is a new
`jphi-linterp` request solved from a converged free-boundary state under
homotopy-tightened coil bounds with I_p and p_axis pinned. 17 of 19 draws that
reached this stage passed it (3 accepted without passes, the rest in 2-4
passes).

**Options (none implemented; each needs a decision):**

1. *Fail fast on a diverging solve* (robustness guard, not a tolerance
   change). Stop a corrective solve whose nonlinear residual grows over
   consecutive iterations, or whose surface trace fails, and fail the pass at
   once: the same rejection in seconds instead of ~30-45 min. Needs a
   per-iteration residual / trace-status hook from the solver, or a
   per-solve iteration budget for this stage (the latter changes a solver
   limit and needs its own approval).
2. *Start the post-homotopy corrective iteration from the draw's last
   corrective INPUT* (plus the bootstrap change) instead of from the target,
   so the solver is asked for a near neighbour of a state it just reached.
   Path-only (the fixed point is unchanged); may rescue these draws rather
   than reject them faster. Requires carrying that input in the draw context.
3. *`protect_state=True` for the post-homotopy corrective call*, so a failed
   solve restores the pre-solve state and the pass fails as a non-converged
   loop with the solver's own message rather than through the axis-collapse
   guard. Clearer reason; wall time unchanged.
4. *`verbose=True` (or a recorded failure field) for that call*, so the log
   and the draw record say which corrective iteration failed and how.
   Diagnostic only.
5. *Status quo*: a rejected draw, ~30-60 min of wall time per occurrence
   (2 of 20 draws, ~1.6 h of a 6.5 h single-thread run here).

The `*.h5` glob in `.gitignore` is negated for `tests/golden/*.h5` so the slim
fixture is tracked while ad-hoc run outputs elsewhere stay ignored.
