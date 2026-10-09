# Golden bouquet test fixtures

Git-tracked regression fixtures for `tests/test_golden_bouquet.py`.

| file | what it is |
|------|------------|
| `D3Dlike_Hmode_golden_slim.h5` | a slimmed real bouquet run (~11.3 MB): `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles). |
| `golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points) with tolerances. |
| `rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the slim fixture's baseline profiles + sigma envelopes. |
| `regenerate_golden_run.py` | produces that full run: the recipe (stored config, class API, INPUT-current archival). |
| `make_golden_fixture.py` | regenerates the three files above from a full run. |
| `D3Dlike_Hmode_legacy_golden.json` | the slim LEGACY golden (~0.2 MB): the same recipe run with `reconstruction_engine="legacy"` -- stored config, reconstruction scalars/profiles/coils, a 1-in-8 subsample of its LCFS trace, every draw's scalars, and the profiles + coils + boundary RMS of the first two in-spec draws (what `tests/test_systematics.py` replays). Built by `make_golden_fixture.py --legacy-json`. |

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
retention: `all` (default, ~11.3 MB), `subset` (baseline + representative
draws, ~5 MB), or `none` (~3.7 MB, no geqdsk-handling coverage). Eventually,
when the default interchange migrates to IMAS/OMAS, the fixture can store
those instead.

The full-fidelity 30 MB run (with p-file bytes and uncompressed eqdsks) stays
as the shareable example artifact under
`bouquet/examples/D3D-like/D3Dlike_Hmode_golden.h5` (not tracked here).

## Updating the golden set (on purpose)

**The recipe** (every systematics golden: 0f92d28, bc7a49a, 060bc1f, and the
input-current rebuild of the self-consistent-bootstrap refresh):

* the fixture's **own stored config** (`scan/0/config_json`), verbatim:
  20 draws, seed 12345, the synthetic-IDA sigmas, the `jbs_*` loop settings
  and pass ceilings, `solver.nthreads=1`; only the archive name and the log
  verbosity are set;
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
  input-current archival: the mode-1 coil drift" below). The current fixture
  was built with this recipe.
  `store_achieved_jphi` changes only what is written, never what is solved;
* one thread (`OMP_NUM_THREADS=1`), on the OFT build the fixture pins, stated
  through `BOUQUET_OFT_COMMIT` / `BOUQUET_OFT_BRANCH` / `BOUQUET_OFT_BUILD_ID`
  when slimming.

The earlier goldens switched the archival off the same way: their
regeneration script wrapped `bouquet.TokaMaker_interface.generate_bouquet`
(which `generate()` imports at call time) to inject
`store_achieved_jphi=False`. That script was never committed; the commits
stated the convention, not the mechanism, and the refresh lost it.
`regenerate_golden_run.py` is that recipe, in the tree. It flips only
`store_achieved_jphi` on `generate()`'s own `generate_bouquet` call, refuses
to run if `generate()` stops passing it as `True`, and stamps the archive root
with `golden_jphi_archival = "input"`. The builder copies that attr into the
fixture and records it in the manifest's provenance
(`generator_args.jphi_archival`), so a fixture says how it was archived.

1. Produce the full run (~6.5 h at one thread for this example; run it on a
   machine with the disk and the time for it):
   ```bash
   OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR \
       --reconstruction-engine unified --verbose
   ```
   (`--reconstruction-engine legacy` plus `make_golden_fixture.py
   --legacy-json` in step 2 rebuilds the legacy JSON golden.)
2. Regenerate the fixture + manifest:
   ```bash
   python tests/golden/make_golden_fixture.py \
       --source RUN_DIR/D3Dlike_Hmode_golden.h5
   ```
   (defaults to the D3D-like example artifact path if `--source` is omitted;
   `rng_stream_manifest.json` is re-pinned from the new slim fixture in the
   same command).
3. Review the `golden_manifest.json` git diff — it shows exactly which physics
   values moved — then commit the new fixture + manifests together.

The builder refuses to write a fixture that names a filesystem path anywhere
a reader sees text: string attrs, string datasets (`config_json`), string
ARRAYS (attrs and datasets, element by element), and geqdsk headers. A path is
an absolute path under a well-known root (`/Users`, `/home`, `/usr`,
`/Volumes`, `/opt`, `/mnt`, ...), any absolute path of two or more components
at a token boundary, a `~/` or `~user/` path, a `../` path, or a Windows
drive path; units and ratios such as `A/m^2` or `1/R` are not
(`tests/test_no_paths_in_records.py` tries to defeat the guard). It reduces
the paths inside the self-consistent bootstrap records (`jbs_loop_json`) to
basenames: records written by earlier builds of the loop carried the OFT
package path in `oft_build.path`; current builds record a path-free
`oft_build = {version, git_hash, build_id}`. This repository is public, and
the OFT build is identified by the content digests in the provenance stamp.
The committed fixture passes the extended guard as it stands (it predates the
path-free record, so its loop records carry the scrubbed basename
`oft_build.path = "OpenFUSIONToolkit"`).

## The current fixture (unified-engine default, 2026-10-07)

The h5 fixture is a run of the unified engine, the default reconstruction
engine: the recipe above with the fixture's stored config (seed 12345, 20
draws, one thread), OFT `fix/jphi-update-ravgs-and-nonfinite-abort` 7da4f18
(build `20260929_7da4f18`), run with `regenerate_golden_run.py
--reconstruction-engine unified`. That switch puts the legacy-only fields the
engine never reads back to their defaults (printed and recorded in the run's
`summary.json`; here only `isolate_edge_jBS` False -> True). No bar changed.

* Unified engine: 20 attempts, **17 archived** (3 rejected:
  `homotopy_maxits`), **4 in spec** (draws 0, 9, 13, 17). An engine draw's
  `in_spec` is the coil verdict AND its post-hoc l_i band
  (`passes_draw_band`, one of `filtering._FILTER_FLAGS`): 5 of 17 pass the
  coil rule, 11 of 17 the band. Wall time: reconstruction 66 s, draws
  42.4 min (123.6 s per equilibrium).
* The legacy path keeps its numeric record in
  `D3Dlike_Hmode_legacy_golden.json`, made from a legacy-engine run of the
  same recipe at the same code: 20 archived, 12 in spec. Wall time:
  reconstruction 217 s, draws 153.6 min (457.7 s per equilibrium).
  `tests/test_systematics.py` (legacy replay) reads it;
  `tests/test_legacy_golden.py` checks what it is.
* `rng_stream_manifest.json` is pinned from this fixture's baseline.
* Validation of the fixture and the suites at `1a15685`:
  [docs/validation-provenance.md](../../docs/validation-provenance.md).
* **The bouquet stamp reads dirty.** The fixture and both manifests stamp
  bouquet commit `7bd48fb` with `dirty: true`. The branch was re-ordered
  after generation: `7bd48fb` is the pre-reorder name of the commit whose
  tree is now `8285201` (the trees are identical). The generator edits the
  run used (`--reconstruction-engine`, `--legacy-json`) were not yet
  committed when the run was made, and were committed together with the
  fixture in `1a15685`; the fixture is regenerable from `1a15685`'s tree.
  (`3d974e6` later changes how the generator's engine switch records
  `isolate_edge_jBS` -- that field now defaults to `None` and is resolved
  per engine -- not what runs: it still records and applies False -> True.)

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
