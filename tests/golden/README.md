# Golden bouquet test fixtures

Git-tracked regression fixtures. Every golden is a real bouquet run, made by
`regenerate_golden_run.py` and slimmed by `make_golden_fixture.py`. The recipe
runs the synthetic test fixtures in `tests/data/` at the library's defaults.
**Status (2026-10-15):** the IDS golden is such a run; the g-file h5 golden and
the legacy JSON golden are still the earlier runs of the older D3D-like
example g-file + p-file (2026-10-09); the synthetic g-file pair they are to be
regenerated on was corrected 2026-10-15 (the g-file is now thermal-only; see
"The current goldens" below).

| golden | source | engine | files | read by |
|--------|--------|--------|-------|---------|
| g-file (h5) | recipe: `g_synthetic.geqdsk` + `diiid_profs_synthetic.cdf` (the synthetic D3D-like g-file and its IDA-layout profile file); committed record: the D3D-like example `D3Dlike_Hmode_baseline.geqdsk` + `.peqdsk` | unified (default) | `D3Dlike_Hmode_golden_slim.h5`, `golden_manifest.json`, `rng_stream_manifest.json` | `tests/test_golden_bouquet.py`, `tests/test_rng_reproducibility.py`, and the archive/filter/stats tests that read a real archive |
| IDS (h5) | `dd_synthetic.json.gz` at 1.2 s, IDA-hybrid with `diiid_profs_synthetic.cdf` (the production IDS route) | unified (default) | `D3Dlike_ids_golden_slim.h5`, `ids_golden_manifest.json`, `ids_rng_stream_manifest.json` | `tests/test_golden_bouquet.py` (every test runs on both h5 goldens), `tests/test_golden_ids.py`, `tests/test_rng_reproducibility.py` |
| legacy (JSON) | the g-file golden's source pair (committed record: the D3D-like example pair) | legacy | `D3Dlike_Hmode_legacy_golden.json` | `tests/test_systematics.py` (the legacy replay, `-m solver`), `tests/test_legacy_golden.py` |

| file | what it is |
|------|------------|
| `*_golden_slim.h5` | a slimmed real bouquet run: `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x; all of them on the g-file golden, 12.0 MB; a representative subset of 4 on the IDS golden, 7.5 MB), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles, the stored config). |
| `golden_manifest.json`, `ids_golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points, `in_spec`) with tolerances, and the provenance (bouquet commit, OFT build, numerics). |
| `rng_stream_manifest.json`, `ids_rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the golden's baseline profiles + sigma envelopes. |
| `D3Dlike_Hmode_legacy_golden.json` | the slim LEGACY golden (~0.2 MB): the g-file golden's recipe run with `reconstruction_engine="legacy"` -- stored config, reconstruction scalars/profiles/coils, a 1-in-8 subsample of its LCFS trace, every draw's scalars, and the profiles + coils + boundary RMS of the first two in-spec draws (what `tests/test_systematics.py` replays). Built by `make_golden_fixture.py --legacy-json`. |
| `regenerate_golden_run.py` | produces a full run: the recipe (the synthetic fixtures at the defaults, class API, INPUT-current archival), and the sigma=0 probe of the same reconstruction. |
| `make_golden_fixture.py` | regenerates the slim fixture + manifests of one golden from a full run (`--golden gfile|ids`, `--legacy-json`). |
| `structured_closure_pre_mse.json` | a separate record of the structured closure (not a run golden). |

## The draw-stream goldens

`rng_stream_manifest.json` and `ids_rng_stream_manifest.json` are the
draw-*level* goldens, and they became possible only when
`GenerationConfig.seed` started reaching the GPR: before that, every draw site
re-seeded from OS entropy and no drawn value was reproducible. Each replays
what `perturb_kinetic_equilibrium` does for one draw -- ne, Te, ni, Ti through
`_draw_monotonic_perturbation`, then the `j_phi` GPR candidate -- off one
`make_rng(seed)` Generator, from its golden's baseline profiles. Pure NumPy: no
solver, no mesh, so it is bitwise identical on the machine that pinned it and
equal to `1e-7` elsewhere.

Re-pin one on its own (no full run needed) with

```bash
python tests/golden/make_golden_fixture.py --rng-stream-only [--golden ids]
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
retention: `all` (default), `subset` (baseline + representative draws: the
first in spec, the first out of spec, the min and max boundary deviation), or
`none` (no geqdsk-handling coverage).

## Updating the golden set (on purpose)

**The recipe** (`regenerate_golden_run.py`):

* the **config is built fresh** by the class constructor
  (`Bouquet.from_geqdsk(g_synthetic.geqdsk, profiles=diiid_profs_synthetic.cdf,
  ...)`; `Bouquet.from_imas(dd_synthetic.json, time=1.2,
  ida_path=diiid_profs_synthetic.cdf, ...)` for the IDS golden), **every
  setting at its default** except the run's identity: 20 attempts (no
  until-N), seed 12345, `nthreads=1`, the engine. So a golden embodies the
  defaults of the code that made it; when a default moves, the goldens are
  regenerated, not edited. (`--config-from FIXTURE` replays a golden's stored
  config verbatim instead.)
* the **inputs are linked** into the run directory under their basenames (the
  dd unpacked from its `.gz`), so the stored config names no path;
* the **class API**: `Bouquet(cfg) -> setup_solver() -> prepare_baseline() ->
  generate()`;
* **input-current archival** (`store_achieved_jphi=False`): `_baseline/j_phi`
  and each draw's `j_phi` / `j_inductive` hold the current the generator
  handed the solver, not the achieved flux-surface average. The systematics
  replay feeds `_baseline/j_phi` back as the input of its baseline solve, so
  this is its premise ("archived current reproduces archived LCFS").
  **`Bouquet.generate()` hard-wires `store_achieved_jphi=True`**, so a plain
  notebook run does NOT follow the recipe (an earlier refresh built that way
  failed the mode-1 replay of `tests/test_systematics.py`; see "Why
  input-current archival: the mode-1 coil drift" below).
  `store_achieved_jphi` changes only what is written, never what is solved.
  `regenerate_golden_run.py` flips only `store_achieved_jphi` on
  `generate()`'s own `generate_bouquet` call, refuses to run if `generate()`
  stops passing it as `True`, and stamps the archive root with
  `golden_jphi_archival = "input"`; the builder copies that attr into the
  fixture and records it in the manifest's provenance
  (`generator_args.jphi_archival`);
* one thread (`OMP_NUM_THREADS=1`), on the OFT build the fixture pins, stated
  through `BOUQUET_OFT_COMMIT` / `BOUQUET_OFT_BRANCH` / `BOUQUET_OFT_BUILD_ID`
  when slimming;
* the **sigma=0 probe** (`--sigma0-probe OUT.json`, same arguments): the same
  config reconstructed in its own process and
  `Bouquet.verify_sigma0_consistency()` run on it (the zero-perturbation draw
  must reproduce the reconstruction). It makes no draws: the probe leaves the
  solver re-anchored, and the golden run must be the production sequence.

The commands (each run ~45-60 min at one thread on the engine, ~4 h on the
legacy path; run them on a machine with the disk and the time):

```bash
# the g-file h5 golden (+ golden_manifest.json, rng_stream_manifest.json)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_G --golden gfile --reconstruction-engine unified --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --golden gfile --source RUN_G/D3Dlike_Hmode_golden.h5
# the IDS h5 golden (+ ids_golden_manifest.json, ids_rng_stream_manifest.json)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_I --golden ids --reconstruction-engine unified --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --golden ids --eqdsk subset --source RUN_I/D3Dlike_ids_golden.h5
# the legacy JSON golden
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_L --golden gfile --reconstruction-engine legacy --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --legacy-json --source RUN_L/D3Dlike_Hmode_golden.h5
# the sigma=0 probe of any of them (same arguments, its own run directory)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_S --golden ids --reconstruction-engine unified --sigma0-probe sigma0.json
```

Slim each golden from a clean checkout of the commit that ran it (the builder
records `git status` in the provenance: build one golden, move its files
aside, restore the tree, build the next), then review the manifest / JSON git
diff -- it shows exactly which physics values moved -- and commit the new
fixtures and manifests together. The builder refuses a run whose stored
source or engine is not the golden's (an h5 golden is a unified-engine run of
its own source class; the legacy JSON a legacy-path g-file run).

The builder refuses to write a fixture that names a filesystem path anywhere
a reader sees text: string attrs, string datasets (`config_json`), string
ARRAYS (attrs and datasets, element by element), and geqdsk headers. A path is
an absolute path under a well-known root (`/Users`, `/home`, `/usr`,
`/Volumes`, `/opt`, `/mnt`, ...), any absolute path of two or more components
at a token boundary, a `~/` or `~user/` path, a `../` path, or a Windows
drive path; units and ratios such as `A/m^2` or `1/R` are not
(`tests/test_no_paths_in_records.py` tries to defeat the guard). It reduces
the paths inside the self-consistent bootstrap records (`jbs_loop_json`) to
basenames (records written by earlier builds of the loop carried the OFT
package path in `oft_build.path`; current builds record a path-free
`oft_build = {version, git_hash, build_id}`). This repository is public, and
the OFT build is identified by the content digests in the provenance stamp.

## Build-specific goldens

The goldens record the OFT build they were generated with, and are valid
for that build only. The record is `provenance.oft` of
`golden_manifest.json`, `ids_golden_manifest.json` and
`D3Dlike_Hmode_legacy_golden.json`:
`build_id` as stated through `BOUQUET_OFT_BUILD_ID` when slimming,
`library_sha256` and `sources_sha256` as measured. The policy (owner
decision, 2026-10-09):

* Every golden comparison runs at its existing bar, on every build.
  `tests/_harness.golden_build_check` compares the installed library's
  SHA-256 (`bouquet.jbs_loop.oft_build_info()`) with the fixture's
  `library_sha256`; the `build_id` comparison is informational. A fixture
  with no `library_sha256` is "unstamped" and counts as a mismatch.
* Same build: a pass is a pass, a failure is a regression.
* Different build: a pass passes with a `GoldenBuildMismatchWarning` that
  names both builds; a failure FAILS (never xfail, never skip), its message
  starting with `OFT build mismatch: fixture generated with
  <build_id>/<sha8>, installed <build_id>/<sha8>` and the regeneration
  commands, followed by the original failure.
* No band and no tolerance depends on the build.

Known cross-build deviation (measured on the earlier D3D-like example
golden): the mode-1 replay of the legacy golden in
`tests/test_systematics.py` (pinned baseline, bar 0.3 %) gave a maximum
coil drift of 0.027 % on the build the golden was generated with, and
1.45 % on OFT builds that evaluate the edge FF' from an exact per-node
`<R>` at the diverted LCFS node: the lower coils move, the boundary stays
within its bar. Which edge convention is right is an open physics
question; until it is answered, regenerate on the build in use, never widen
the bar.

Regenerating on the installed build: the commands of "Updating the golden
set" above (`tests/_harness.py` prints them in a mismatch message), with
`BOUQUET_OFT_BUILD_ID` naming the installed build; review the git diff of
the manifests / JSON before committing.

## The current goldens (2026-10-15)

**The IDS golden** is the recipe's run at bouquet `cced718` (main `dc03156`
plus this recipe, a clean tree), OFT `fix/jphi-update-ravgs-and-nonfinite-abort`
7da4f18 (build `20260929_7da4f18`, lib `7885fedc1e62`), one thread. The
defaults it embodies: the in-spec rule is the magnetics (no l_i band),
IDA-hybrid takes the IDA file's Z_eff, the radial sigma templates (the profile
file's own envelopes here), the warm baseline re-solve in `generate()`, engine
draws held on the reconstruction's isoflux, `check_bistability` on, the
coil-saturation test against the installed bound, and the Redl epsilon
`(R_max - R_min)/(R_max + R_min)` (`evaluate_jBS/4`). An engine draw's
`in_spec` is its coil verdict AND the opt-in bands, none of which is on: the
coil verdict.

| golden | attempts | archived | in spec | reconstruction | draws | bistability | sigma=0 probe |
|--------|----------|----------|---------|----------------|-------|-------------|---------------|
| IDS, unified | 20 | 19 (1 perturb_failed) | 18 (all archived but 15) | 108 s | 44.0 min | single | PASS (dl_i +3.1e-5, dq0 -5.3e-5, r_j 1.6e-4) |

* The IDS route archives no schema `jbs_loop` block on its `_baseline` (the
  g-file route does); its reconstruction's loop verdict is in the baseline's
  engine record (`engine_json`: `converged`, `loop_converged`), which the
  golden test reads there.

**The g-file h5 golden and the legacy JSON golden are NOT yet regenerated.**
The synthetic g-file pair was internally inconsistent: the g-file's on-axis
pressure (90.4 kPa) was the dd's TOTAL pressure, fast ions included, while
its profile file carries the thermal pressure only (46.1 kPa), and the g-file
route has no fast-pressure source, so a reconstruction on the pair gave
beta_N 0.80 against the g-file's 1.68. Corrected 2026-10-15 (owner decision):
`g_synthetic.geqdsk` is now the thermal-only equilibrium of the same slice
(46.1 kPa on axis, beta_N 0.80, q0 1.41, l_i(3) 0.835), the total-pressure
file is kept as `g_synthetic_pfast.geqdsk` with a p-file
(`tests/data/README.md`). Both goldens are to be regenerated on the corrected
pair with the commands above.
Until then the committed records are the earlier runs of the older D3D-like
example g-file + p-file, regenerated 2026-10-09 at `67aba59` on the same OFT
build under the defaults of that date (the old l_i band among them): unified
20 attempts, 16 archived, 6 in spec (reconstruction 62 s, draws 43.8 min);
legacy 20 archived, 11 in spec (reconstruction 218 s, draws 184.2 min). Their
stored configs replay with the stored-config compatibility rules (the l_i
band loads OFF, with a note). The legacy replay (`tests/test_systematics.py`)
reconstructs from whatever pair the legacy golden's stored config names.

* The legacy record archives the pressure-driven current as its own
  `j_pressure` beside a `j_inductive` that does not carry it, and says so
  (`current_split_convention = "pressure_separate"`); the legacy replay
  composes the legacy path's carried (in-memory) inductive, `j_inductive +
  j_pressure`, before handing it to `generate_bouquet` as `input_jinductive`
  -- a record from before that convention replays as is.

## Why input-current archival: the mode-1 coil drift

Mode 1 of `tests/test_systematics.py` builds its baseline from the fixture's
`_baseline/j_phi`, handed to the unperturbed `jphi-linterp` baseline solve,
and compares the coil currents after that solve with
`_baseline/coil_currents`. The generator handed the same solve the
reconstruction's own delivered current, so the replay is faithful only if
`_baseline/j_phi` holds that INPUT current. A refresh made through plain
`Bouquet.generate()` (which archives the ACHIEVED flux-surface-averaged
current) failed mode 1 at 1.23 % max coil drift on one coil against the
0.3 % bar; the same run rebuilt with `regenerate_golden_run.py`
(`store_achieved_jphi=False`) passed at 0.019 %. No bar was changed.
`test_the_fixture_archives_the_input_current` asserts the archival stamp in
the fixture, its provenance and the manifest.

The refresh history of earlier fixtures is in the messages of the commits
that changed them (`git log -- tests/golden/`).

## Test scope: what the tests around this fixture do not cover

Stated so that a pass is not read as more than it is. None of these is a
loosened bar; each narrows what a passing suite shows.

* **The axis-collapse regression test is a build-dependent skip.**
  `tests/test_fsa_current_integral.py::test_get_q_collapses_silently_on_an_unclipped_grid`
  was a hard assertion (`unclipped <R> span < 1e-9`, absolute, over all
  surfaces) and is now a measured skip: it asserts only on a build whose
  unclipped grid collapses (at least 2 traced surfaces with a relative `<R>`
  span <= 1e-9) and skips, with a reason, on every other build. It can
  therefore skip with the defect present (a PARTIAL collapse that pins only
  some surfaces, or a build that traces fewer than 2 surfaces). What runs on
  every build instead: the unconditional invariant that the CLIPPED geometry
  is real (`<R>` spans more than 10 % of its mean) and two solver-free tests
  of the guard. The production guard in `utils.fsa_current_geometry` uses the
  raw `<R>` span, zero (untraced) rows included, so an untraced row can hide
  a collapse from it; `physics.evaluate_jBS` now refuses a zero (failed-trace)
  row on its own surfaces.
* **Six zero-perturbation tests run on the legacy path only.** The route-R2
  sigma=0 tests of `tests/test_seeded_reproducibility.py`
  (`test_sigma0_r2_reproduces_the_baseline_jbs`,
  `test_sigma0_r2_exact_measure_lands_in_its_own_budget`,
  `test_sigma0_r2_exact_measure_reports_a_plausible_inductive_share`,
  `test_sigma0_r2_exact_measure_still_recovers_the_recon_li`,
  `test_sigma0_r2_exact_measure_is_bit_reproducible`,
  `test_sigma0_r2_exact_measure_leaves_the_bootstrap_alone`) build their
  baseline with `jbs_self_consistent=False`: they test the legacy frozen-SWB
  path, not the shipped default. The default path's counterpart,
  `tests/test_jbs_loop_solver.py::test_f_sigma0_route_r2_draw_converges_near_the_baseline`,
  checks only that the loop converges, the `|s-1|*f_ind <= 3.86e-3` budget
  and the init-source bookkeeping -- not the other five properties.
* **The structured solver tests and the shipped default both use a pass
  ceiling of 12.** `tests/test_jbs_loop_solver.py` sets
  `jbs_max_passes = _STRUCTURED_TEST_PASSES = 12`, which is now the default
  (raised from 8 on 2026-10-07 with the owner's approval, after the synthetic
  structured case needed exactly 8 passes on the development build). The
  tests therefore show convergence within the default on this build only;
  they do not show headroom on another build.

## Known limitation: a standard draw's post-homotopy re-solve can diverge slowly

In the loop-on regeneration of the earlier D3D-like example golden, two of
twenty draws were
rejected after their coil homotopy: Redl on the delivered equilibrium missed
the bootstrap the draw carried by `r_j` of order 1e-3, the post-homotopy
stage handed the renormalised target to the corrective `jphi-linterp`
iteration, and that solve did not converge. Once the iterate loses its nested
surfaces every nonlinear iteration's surface trace fails and the solver spends
its whole iteration budget, roughly 20-60 min at one thread. The call runs
without `protect_state`, so the loop then measures the broken state, the
`fsa_current_geometry` guard refuses it, and the draw is rejected as
OUT_OF_SPEC with a NaN coil drift. The outcome is always a loud rejection,
never an accepted draw; nothing caps, retries or falls back. The cause (why
these draws, and why the corrective iteration's first solve diverges from a
target rescaled by less than 0.5 %) is not established; the full
investigation is kept in the private validation record.
