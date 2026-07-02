# UX review + improvement plan — `feat/bouquet-class-api`

> **Temporary working document** (2026-07-01). Review of the class-API branch's user
> experience and the HDF5 output architecture, with a concrete change plan.
> Delete (or fold into `docs/CHANGES_SUMMARY.md`) before merging to `main`.
>
> Evidence base: direct read of `bouquet/{config,run,filtering,utils,paths}.py`;
> walkthrough of the three `examples/D3D-like/bouquet_D3Dlike_*_example.ipynb`
> notebooks; the downstream shot notebooks + debug scripts in the IDA_run repo
> (9 structurally identical workflow notebooks, `shot_debug/*.py`); and a full
> map of every h5 write/read path.

---

## STATUS — all phases implemented 2026-07-02 (187 tests green)

| Phase | Status |
|---|---|
| 1 config serialization + provenance (F8/F9/F10) + F25 | ✅ |
| 2 `BouquetArchive` reader (F13, F12-read) | ✅ |
| 3 de-thread (header, scan_key) (F1, F7) | ✅ (example-notebook rewrite deferred) |
| 4 schema v2 — **clean break, no legacy** (F11/F12/F15/F16a) | ✅ (adversarial review caught 1 silent coil bug, fixed) |
| 5 config/API simplifications (F3/F4) | ✅ (5.1 jphi-default **won't do**; imas-summary formatting deferred) |
| 6 sweeps + plotting (F5/F6) | ✅ (6.3 IDA-run templating **won't do**) |
| 7 parallel hardening (F17–F26) | ✅ (was already done; F25 finished with Phase 1) |

**Won't-do (user decisions):** 5.1 (keep `jphi_scalar_sigma=0.10`), 6.3/F8
(keep Nelson's split IDA_run notebooks). **Deferred (cosmetic):** rewriting the
3 example notebooks to run-object forms (Phase 3.5); reconstruction-style IMAS
baseline summary (5.4). New modules: `bouquet/schema.py`, `bouquet/archive.py`.
This document can be deleted / folded into `docs/CHANGES_SUMMARY.md` at merge.

---

## Verdict

The core design is solid and should not be reworked:

- Typed dataclass config with early `__post_init__` validation.
- Staged orchestrator (`setup_solver → prepare_baseline → generate → filter →
  export`) with an inspection gate at every stage and a `run()` convenience.
- Non-destructive filter flags (`passes_*`, `selected = AND(flags)`) + pruned
  `export_filtered` copy.
- The per-path workflow guard (`run._validate_workflow`) that raises on
  known-bad knob combos.
- `paths.py` resolution (env var → hint → candidates → walk-up, actionable
  errors).
- Byte-opaque eqdsk/pfile storage (bit-perfect round-trip).
- Solver-chatter capture to `reconstruction_log` / `generation_log`.

The UX debt concentrates in four places:

1. The `(header, scan_key)` string pair is manually re-threaded into every
   post-run call — fragile, and mismatches fail **silently**.
2. Notebooks restate ~10 config values that are (or should be) defaults, copied
   verbatim across 12+ notebooks in two repos.
3. No high-level archive reader — downstream scripts hand-roll h5 tree walking
   and byte extraction (schema knowledge duplicated outside the package).
4. No provenance in the h5 file (no schema version, package version, or config
   snapshot).

---

## Part 1 — findings

### 1.1 UX / API surface

**F1. `(header, scan_key)` re-threading (biggest friction + silent-failure source).**
The `Bouquet` object knows both, yet every notebook does:

```python
bq.plot_traces(HEADER + '.h5', scan_key=2100, ...)   # wants the .h5 path
bq.filter_coil_currents(HEADER, scan_key=2100)        # wants the bare header
sel = bq.select_indices(HEADER, scan_key=2100, selection='selected')
```

- A wrong `scan_key` returns empty results with no error.
- Renaming `HEADER` mid-notebook silently reads a stale file (observed in the
  omas notebook: `HEADER_FULL = HEADER + '_full'` mid-run; and in IDA_run).
- Inconsistent path conventions: filtering accepts header-or-path
  (`filtering._resolve`), plotting wants the `.h5` path.
- `Bouquet.plot_bouquet()` and `Bouquet.filter()` already exist and auto-wire
  header/scan_key, but the notebooks call the module-level functions with
  strings instead (they appear to predate the methods).

**F2. The "validated production settings" block restates defaults.**
The same ~10-line block appears verbatim in all 3 example notebooks and all 9
IDA_run shot notebooks:

```python
run.uncertainty.ne_scalar_sigma = 0.05      # == default
run.uncertainty.te_scalar_sigma = 0.05      # == default
run.uncertainty.ti_scalar_sigma = 0.10      # == default
run.uncertainty.jphi_scalar_sigma = 0.05    # DEFAULT IS 0.10 — real delta
run.uncertainty.zeff_scalar_sigma = 0.05    # == default
run.uncertainty.n_ls, ..., = 0.5, 0.4, 0.25 # == defaults
run.generation.seed = 42                    # real delta (default None)
run.generation.l_i_tolerance = 0.05         # == default
run.generation.recalculate_j_BS = True      # == default
run.generation.jBS_scale_range = (0.99, 1.01)  # == default
run.filtering.inspec_F_max = 0.02           # == default
run.filtering.inspec_VSC_max = 0.02         # == default
run.filtering.rms_max_mm = 5.0              # == default
```

Only `jphi_scalar_sigma` and `seed` differ from the dataclass defaults. If
0.05 is the validated production value, the default in `config.py` is wrong;
align it and the whole block collapses to one line.

**F3. Path asymmetry at the baseline gate.**
geqdsk: `run.reconstruct()` — one call, curated PASS/CHECK fidelity summary.
IMAS: `run.setup_solver(); run.prepare_baseline()` — two calls, one-line
`[imas forward-solve]` print. A reader of both notebooks concludes the IMAS
path has no validation step.

**F4. Workflow knobs encode an enum as booleans.**
`_validate_workflow` enforces that only two combinations of
`perturb_jind_in_anchor` / `jBS_baseline_mode` / (`isolate_edge_jBS`) are
valid — `imas → diff+C`, `geqdsk → standard l_i loop`. That's a missing named
preset; the guard + four flags + `allow_unsafe_workflow` is the hard way to
say `workflow = "imas-diff-c" | "geqdsk-standard" | "custom"`.

**F5. Multi-slice sweeps are boilerplate, and there are two overlapping
organizing mechanisms.** The omas timeseries loop (`set_slice → prepare →
generate → filter → export` + hand-collected `li/Ip/counts` dict) is ~15 lines
repeated per notebook; `parallel.py` re-implements its own per-slice loop.
Separately: `scan/<key>/` inside one file AND one-file-per-slice both exist,
and the notebooks use **both at once** (per-slice headers and per-slice
scan_keys). Every downstream call then needs both coordinates.

**F6. Plotting surface proliferation.** `__all__` exports ~20 plot functions
including near-duplicates (`plot_bouquet` / `plot_imas_bouquet` /
`plot_geqdsk_bouquet` / `plot_pfile_bouquet`). `source_kind` is already stored
in the baseline attrs precisely so plotting can dispatch on path.

**F7. Filter summary duplication.** `run.filter()` already prints the curated
generation summary, but the notebooks then re-call `filter_coil_currents` /
`filter_boundaries` / `select_indices` manually to print counts and get the
distribution figures (`run.filter()` hardcodes `plot=False`).

**F8. Per-shot notebooks in IDA_run are full copies differing in 5 strings**
(`GFILE`, `IDA`, `TIME`, `HEADER`, `SCAN_KEY`). That's a config file, not a
notebook copy — blocked on config serialization (see F9/P1).

**F9. `BouquetConfig` has no serialization.** Needed independently by: h5
provenance, per-shot templating, and `parallel.py` shipping configs to
SLURM workers.

### 1.2 HDF5 architecture

Current schema (as written by this branch — `GenerationConfig.scan_key`
defaults to `0`, so the class API always writes the `scan/` layout):

```
{header}.h5
└── scan/{scan_key}/
    ├── _baseline/
    │   ├── psi_N, n_e, T_e, n_i, T_i, pressure[, pressure_thermal]
    │   ├── j_phi, [j_BS, j_inductive]           (all "[unit]"-suffixed names)
    │   ├── sigma_ne/te/ni/ti/jphi, [sigma_aux_*], [aux_*]
    │   ├── baseline.eqdsk, [baseline.pfile], [recon_lcfs_ref], [x_points]
    │   ├── [coil_currents [A] + coil_names dataset]
    │   └── attrs: Ip_target, l_i_target, source_kind, [diverted]
    └── {count}/                                  (integer; gaps allowed)
        ├── {header}_{key}_{count}.eqdsk          (np.void bytes)
        ├── [{...}.pfile]
        ├── psi_N, j_phi, j_BS, j_inductive, n_e, T_e, n_i, T_i, w_ExB
        ├── [pressure, pressure_thermal, psi_N_kinetic, Zeff, j_BS,edge]
        ├── [coil_currents [A]] (+ coil_names JSON attr)
        ├── [perturbed_lcfs_ref, x_points, aux_*]
        └── attrs: l_i(1), l_i(3), count, homotopy_*, max_*_drift_pct,
                   in_spec, inspec_*, l_i_target_used, [diverted],
                   [passes_coil_filter, passes_boundary_filter, selected]
```

**F10. No provenance.** No schema version, no `bouquet.__version__`, no
timestamp, no record of the config that produced the file.

**F11. Eqdsk dataset names embed `{header}_{scan}_{count}`.**
Consequences: `load_equilibrium` must *reconstruct* the exact string;
`merge_archives` must rename every eqdsk/pfile dataset; renaming the file
breaks lookups. Meanwhile `filtering.py` ignores the name and scans for the
`.eqdsk` suffix — two lookup strategies for the same dataset in one codebase.

**F12. Dual layout (flat vs `scan/`).** Flat only exists for pre-class files,
yet every reader branches on it and downstream users have learned to probe for
it (`if 'scan' in hf: ... else:` in IDA_run `shot_debug/169510_debug.py`).

**F13. No archive reader class.** `169510_debug.py` contains ~100 lines of
hand-rolled tree-walking, `.eqdsk`-suffix scanning, byte extraction, and
`PFile.from_bytes` — package schema knowledge duplicated downstream. The
in-package idiom is also awkward:
`bq.read_eqdsk_from_bytes(d['eqdsk_bytes'], bq.read_geqdsk)`.

**F14. `merge_archives` trusts shard baselines.** Copies the baseline from the
first non-empty shard with no check that other shards' baselines match
(a worker config drift would merge silently).

**F15. Coil-current representation is split.** Values dataset + JSON-names
attr per draw; values dataset + names *dataset* in the baseline. Two
zip-at-read conventions.

**F16. Units live in dataset names** (`"j_phi [A m^-2]"`). Forces exact-string
knowledge into every reader and makes ad-hoc h5py access awkward. (Schema-v2
material only — breaking.)

---

## Part 2 — the plan

Ordered so each phase is independently landable; phases 1–3 are the payoff
core. No physics code changes anywhere in this plan.

### Phase 1 — config serialization + h5 provenance  *(unlocks F9, F10, F8)*  ✅ IMPLEMENTED 2026-07-02

1. ✅ `bouquet/config.py`: `BouquetConfig.to_dict()` / `from_dict()` +
   `to_json` / `from_json`. ndarray fields (isoflux, `sigma_profiles`, `aux_*`,
   fixed components, `profile_overrides`) encode as `{"__ndarray__": [...]}` and
   restore to ndarray; source recorded via a `"source_type"` discriminator
   (inferred from keys if absent). Round-trip test exercises every optional
   field on both source types (`tests/test_config.py::TestSerialization`).
2. ✅ `bouquet/utils.py` `write_provenance(h5, config, scan_key)` called from
   `Bouquet.generate`: file-level attrs `schema_version` (`utils.SCHEMA_VERSION`,
   **currently `1`** — the class-API `scan/` layout; Phase 4 bumps it to `2`),
   `bouquet_version`, `created` (set once, ISO), `updated`, and a `config_json`
   dataset at the file root **and** mirrored per `scan/<key>/` group.
   *(Deviation from the plan: version starts at 1 to honestly reflect the
   current on-disk layout; the v2 stamp lands with the Phase-4 writer changes.)*
3. ✅ `bouquet/parallel.py`: SLURM bundle is now `{job}_bundle.json`
   (`config.to_dict()` + scalar params) loaded via `BouquetConfig.from_dict`;
   `pickle` dropped (F25 — was blocked on this phase).
4. ✅ `bq.load_config(h5path_or_header, scan_key=None)` → `BouquetConfig`
   (per-scan `config_json` first, else file root; raises a clear `KeyError`
   on pre-provenance files). Tests in `tests/test_config.py::TestProvenance`.

### Phase 2 — `BouquetArchive` reader class  *(fixes F13, F1-read-side, F12-read-side)*  ✅ IMPLEMENTED 2026-07-02

Done as specified: new `bouquet/archive.py` with `BouquetArchive` / `ScanView` /
`DrawView` (lazy, open-per-access). `ar.scan_keys` (flat → `[None]`),
`ar[key]` / `ar.scan()`, `sc.indices/baseline/all/selected/excluded`, `sc[i]`,
`d.li1/li3/attrs/flags/selected/profiles`, `d.equilibrium()`, `d.pfile()`,
`d.extract(...)`, `ar.provenance` (Phase-1 attrs + `load_config`).
`Bouquet.archive` property added. Eqdsk/pfile by **suffix scan** (v2 fixed name
first, else `.eqdsk`/`.pfile`) — verified it reads the golden fixture where
`load_equilibrium`'s exact-name lookup fails (a live demonstration of F11).
The functional readers stay the public API and are reused internally
(`select_indices`, `list_equilibrium_indices`, `load_baseline_profiles`,
`read_filter_flags`, `read_eqdsk_from_bytes`). Tests: `tests/test_archive.py`
(golden round-trip + legacy/v2 suffix-scan).

**Not done (deferred, non-breaking):** demoting `read_eqdsk_from_bytes` from
`__all__` — kept exported for now; revisit at merge.

Original spec follows.

New module `bouquet/archive.py`; single authoritative home for schema
knowledge. The existing functional readers
(`load_equilibrium`, `select_indices`, …) become thin wrappers (keep them —
they're public API — but implement them on the archive internally).

```python
ar = bq.BouquetArchive("run.h5")          # or bq.BouquetArchive(run) / run.archive
ar.scan_keys                              # ['4400']  (always list; flat legacy → [None])
sc = ar["4400"]                           # ScanView (default scan if only one)
sc.indices, sc.baseline                   # gap-tolerant indices; baseline dict/handle
sc.selected, sc.excluded, sc.all          # lists of DrawView, per filter flags
d = sc[3]                                 # DrawView (lazy)
d.li1, d.li3, d.flags, d.attrs            # scalars + filter/spec metadata
d.profiles                                # dict of named arrays (psi_N, j_phi, ...)
d.equilibrium()                           # parsed GEQDSKEquilibrium from stored bytes
d.pfile()                                 # parsed PFile (None if absent)
d.extract("out/", formats=("geqdsk", "pfile"))   # write files, return paths
for d in sc.selected: ...
```

Implementation notes:
- Eqdsk/pfile lookup by **suffix scan** (the `filtering.py` strategy), so it
  works on legacy names and the phase-4 fixed names alike.
- `ar.provenance` property surfaces the phase-1 attrs/config.
- `Bouquet.archive` property returns `BouquetArchive(self.config.output_header)`.
- Port the readable parts of `filtering.read_filter_flags` /
  `utils.load_equilibrium` here; deprecate `read_eqdsk_from_bytes` from
  `__all__` (keep importable).
- Tests: build a small archive via the existing golden-fixture path; assert
  round-trip against `load_equilibrium` outputs.

### Phase 3 — kill the `(header, scan_key)` re-threading  *(fixes F1, F7)*  ✅ IMPLEMENTED 2026-07-02 (item 5 deferred)

1. ✅ `utils._resolve_h5` now duck-types a `Bouquet` (→ `config.output_header`)
   / `BouquetArchive` (→ `.path`) / header / path, so **every** reader accepts a
   run or archive as the first arg; `filtering._resolve` delegates to it.
   `utils._default_scan_key(ref, scan_key)` fills a missing key from a passed
   `Bouquet`.
2. ✅ `select_indices` raises a listing `KeyError` on an unknown explicit
   `scan_key` (`filtering._require_scan_key`) — the silent-empty failure is gone.
   *(Applied at `select_indices`, the primary offender; the same guard can be
   dropped into the other accessors trivially — left as a fast follow.)*
3. ✅ `Bouquet.plot_traces()` / `plot_coil_currents()` / `plot_spec_summary()` /
   `selected_indices()` added (auto-wire header + `scan_key`), mirroring
   `plot_bouquet()`.
4. ✅ `run.filter(plot=True)` returns the coil + boundary distribution figures
   under `summary["figures"]` (F7).
5. ⬜ **Deferred**: rewriting the 3 example notebooks to the run-object forms
   end-to-end. The API now supports it (`run.plot_traces()`,
   `run.selected_indices()`, pass `run` to any reader); the notebook edits are a
   separate, low-risk sweep left for the notebook-proofing pass.

Tests: `tests/test_archive.py::TestDethread`. Original spec follows.

1. Module-level readers (`plot_*`, `filter_*`, `select_indices`,
   `read_filter_flags`, `export_filtered`, `load_*`): accept a `Bouquet` or
   `BouquetArchive` as the first argument in addition to header/path.
   One shared `_coerce_archive(obj)` helper; when given a `Bouquet`, default
   `scan_key` to `config.generation.scan_key`.
2. `select_indices` / plotting: **raise `KeyError`** (or warn loudly) when an
   explicit `scan_key` is absent from the file, listing available keys.
   Silent-empty is the current failure mode.
3. Add the missing thin methods on `Bouquet`, mirroring `plot_bouquet`:
   `plot_traces()`, `plot_coil_currents()`, `selected_indices()`.
4. `run.filter(plot=False)` → add `plot=True` support (return the two
   distribution figures alongside the summary dict).
5. Update the 3 example notebooks to use the run-object forms end-to-end
   (no bare `HEADER` strings after construction).

### Phase 4 — schema v2 (CLEAN BREAK, no legacy)  *(fixes F11, F12, F15, F16)*  ✅ IMPLEMENTED 2026-07-02

**Decision (user 2026-07-02): no active users yet, so this is a clean break —
no legacy readers / flat fallback / schema-version gating.** New `bouquet/schema.py`
is the single source of truth (`PROFILE_UNITS`, `write_profile`, fixed names).

- ✅ **F16(a) full**: profile datasets renamed to **bare** names (`j_phi`, `n_e`,
  `pressure`, …) with the unit in `ds.attrs["units"]`. Exhaustive global rename
  of the 17 bracketed literals (171 sites) + writer routed through
  `schema.write_profile` (30 creates). Confirmed the literals were only
  dataset-names/keys (plot labels use mathtext), so the rename was mechanical.
- ✅ **F11**: draws + baseline store fixed `eqdsk` / `pfile` names (no
  `{header}_{scan}_{count}` embedding, no `baseline.` prefix); `_eqdsk_dataset_name`
  returns `"eqdsk"`; `merge_archives` rename step deleted.
- ✅ **F12**: writer is scan-only (already true); the `.eqdsk`/`.pfile` suffix-scan
  readers replaced with fixed-name lookups.
- ✅ **F15**: coil currents = `coil_currents` values dataset + `coil_names` string
  **dataset** in BOTH draw and baseline groups (per-draw was a JSON attr);
  readers use `utils._read_coil_names`.
- ✅ `schema_version` now **2** (`utils.SCHEMA_VERSION` imports from `schema`).
- ✅ Golden fixture migrated in place to v2 (bare names + units, fixed byte
  names, coil_names dataset); tests updated.

**Verification (the user-requested second pass):**
- Full suite **187 pass** on v2 (incl. 2 new coil-drift regression tests).
- End-to-end: a real `reconstruct → generate → filter` through the NEW writer
  produces a v2 file that load_equilibrium / BouquetArchive / load_config /
  plot_bouquet / plot_traces / plot_coil_currents all read.
- **Adversarial diff review caught one real (silent) bug**: `plot_coil_currents`
  still read per-draw `coil_names` from the old JSON attr → every per-draw
  coil-drift cell was NaN on a v2 file (no error, just an empty heatmap). Fixed
  to read the dataset (matching the baseline path) + added a
  `test_plot_coil_currents_finite_drift` regression. Also aligned the stale
  `SCHEMA_VERSION=1`, removed dead `_eqdsk_dataset_name` imports, fixed the
  `merge_archives` docstring. No other reader/writer mismatches found.

**Deviation from the original plan:** F14 (merge baseline verification) already
landed in Phase 7; F16 units-as-attrs was done as the full bare-name rename
(a), not skipped. Original v2-with-legacy spec follows.

Gate all of these behind the `schema_version = 2` stamp from phase 1.

1. **Fixed in-group dataset names**: draws store `eqdsk` / `pfile`; baseline
   stores `eqdsk` / `pfile` (drop the `baseline.`-prefix duplication too).
   Readers already suffix-scan after phase 2, so legacy files keep working.
   `merge_archives` deletes its rename step entirely.
2. **Scan layout only**: writers drop the flat branch (`_group_path` always
   emits `scan/<key>/`); readers keep flat fallback for legacy files.
3. **Unify coil-current storage**: one representation both places —
   `coil_currents` values dataset + `coil_names` string dataset (baseline
   convention wins). Reader zips; legacy JSON-attr fallback retained.
4. **`merge_archives` baseline verification**: compare `Ip_target`,
   `l_i_target`, and `psi_N`/`j_phi` array hashes across shards; raise on
   mismatch (message: which shard, which field).
5. *(Optional, decide before stamping v2)* **Units to attrs**: dataset names
   `j_phi`, `n_e`, … with `ds.attrs["units"]`. Touches every reader/writer —
   only worth it bundled with 1–3 since v2 is the one-time break. If skipped,
   keep the bracketed names forever; do not do this later as a v3.

### Phase 5 — config/API simplifications  *(fixes F2, F4, F3)*  ✅ IMPLEMENTED 2026-07-02 (5.1 won't-do; imas-summary formatting deferred)

- ✅ **5.2 `run.describe()`** (below).
- ✅ **5.3 workflow preset** — `GenerationConfig.workflow = "auto" | "geqdsk-standard"
  | "imas-diff-c" | "custom"`. `_validate_workflow` now folds `allow_unsafe_workflow`
  into `"custom"` (deprecated alias, still honoured), asserts named presets match
  the source, and validates the value in `__post_init__`. `"auto"` keeps the
  existing per-source flag resolution.
- ✅ **5.4 symmetric `run.prepare()`** = `setup_solver()+prepare_baseline()` on
  either path (`reconstruct()` stays the g-file alias). ⬜ The IMAS baseline still
  prints its one-line `[imas forward-solve] converged (…Ip%…TokaMaker vs IDS li…)`
  summary (which already carries the key validation metrics) rather than a
  reconstruction-style block — richer formatting deferred as cosmetic.

1. ❌ **WON'T DO (user decision 2026-07-02): keep `jphi_scalar_sigma = 0.10`.**
   The default stays; notebooks continue to set 0.05 explicitly.
2. ✅ **`run.describe()`** implemented — prints/returns the config grouped by
   section, showing source paths (required) + only the knobs that differ from
   their dataclass default (safe ndarray/dict handling). Replaces the ~60-line
   commented knob-reference cell. *(Original spec:)* print the config grouped by
   section, showing only
   non-default values (plus source paths + output header). Replaces the
   ~60-line commented knob-reference cell that is already diverging between
   the geqdsk and omas notebooks.
3. **Workflow preset enum**: `GenerationConfig.workflow: str = "auto"` with
   values `"auto" | "imas-diff-c" | "geqdsk-standard" | "custom"`.
   - `"auto"`: resolved per source type at `generate()` (what
     `from_geqdsk`/`from_imas` hardcode today).
   - Presets set `perturb_jind_in_anchor` / `jBS_baseline_mode` /
     `isolate_edge_jBS`; `"custom"` leaves flags as-is and downgrades the
     guard to a warning (subsumes `allow_unsafe_workflow`, which stays as a
     deprecated alias).
   - `_validate_workflow` reduces to: preset says X, flags say Y.
4. **Symmetric baseline gate**: `run.prepare()` = `setup_solver()` +
   `prepare_baseline()` on either path (`reconstruct()` stays as the
   recon-path alias); add `_print_imas_summary()` in the same visual format
   as the reconstruction block (Ip err, TokaMaker vs IDS li_1/li_3, SWB/FUSE
   peak ratio, nl its, boundary source).

### Phase 6 — sweeps + notebooks  *(fixes F5, F6, F8)*  ✅ IMPLEMENTED 2026-07-02 (6.3 won't-do)

- ✅ **6.1 `run.run_slices(times, scan_keys=None, header=None, export=False)`** —
  wraps the `set_slice → prepare_baseline → generate → filter` loop into one
  archive (one `scan_key`/slice, defaulting to `round(t*1000)` ms), returns
  `{scan_key: {time, n_all, n_sel, l_i, Ip}}`. (A full IMAS multi-slice run isn't
  in the unit suite — exercised via the notebooks.)
- ✅ **6.2 plotting surface** — `plot_bouquet` is already source-agnostic; the
  wrapper `plot_imas_bouquet` demoted from `__all__` (still importable). The
  source-specific *input* viewers (`plot_geqdsk_bouquet`/`plot_pfile_bouquet`)
  stay advertised — they're the §1 input plotters, not `plot_bouquet` duplicates.
- ❌ **6.3 IDA_run JSON templating (F8): WON'T DO** (user 2026-07-02) — the
  per-shot notebooks are Nelson's originals; kept split (plus our E_r/rotation
  adds). The Phase-1 serialization is in place if that ever changes.

Original spec follows.

1. **`run.run_slices(times, scan_keys=None, header=None)`**: wraps the
   `set_slice` loop; one output file, one `scan_key` per slice by default
   (per-slice headers remain possible but stop being the documented pattern);
   returns `{scan_key: {n_all, n_sel, li, Ip, ...}}`. Refactor the
   `parallel.py` per-slice loop onto the same iteration helper.
2. **Plotting consolidation**: `plot_bouquet` dispatches on stored
   `source_kind`; demote `plot_imas_bouquet` / `plot_geqdsk_bouquet` /
   `plot_pfile_bouquet` from `__all__` (keep importable). Trim `__all__`
   plotting exports to the ~8 actually used in notebooks.
3. **IDA_run templating** (downstream repo, after phases 1+5): one template
   notebook that loads a per-shot JSON (5 paths/values) via
   `BouquetConfig.from_json`, replacing the 9 copied notebooks. Optional
   `python -m bouquet run shot.json` CLI entry point in `__main__.py`.

### Suggested sequencing / sizing

| Phase | Depends on | Size | Risk | Status |
|---|---|---|---|---|
| 1 config serialization + provenance | — | S–M | low | ✅ done 2026-07-02 |
| 2 `BouquetArchive` | — (better after 1) | M | low (read-only) | ✅ done 2026-07-02 |
| 3 de-thread header/scan_key | 2 | S | low | ✅ done 2026-07-02 (notebook rewrite deferred) |
| 4 schema v2 | 1, 2 | M | medium (writer change) | ✅ done 2026-07-02 (clean break, F16a full; review caught 1 silent bug, fixed) |
| 5 config simplifications | — | S | low | ✅ done 2026-07-02 (5.1 **won't do**; imas-summary formatting deferred) |
| 6 sweeps + notebooks | 1, 3, 5 | M | low | ✅ done 2026-07-02 (6.3 IDA templating **won't do** — keep Nelson's split notebooks) |

Regression guardrails for every phase: `tests/test_golden_bouquet.py` must
pass unchanged on a **pre-branch legacy h5** (add one as a fixture if not
already covered), and the geqdsk + omas example notebooks must run end-to-end.
For phase 5.1 (default `jphi_scalar_sigma` change), verify draw yield on both
the D3D-like geqdsk and OMAS testbeds before landing.

---

## Part 3 — parallel mode (`bouquet/parallel.py`, laptop + SLURM)

Doctrine this section is checked against: **every TokaMaker instance needs its
own process and its own core** (`OFT_env` per-process singleton; `nthreads=1`
for determinism and to avoid DLSODE hangs on stiff slices).

Already correct: spawn context (fork would break OFT's cached state);
BLAS/OpenMP env pinned before workers spawn (with restore); laptop cross-worker
baseline guard (`li`/`Ip` @ rtol 1e-6); contiguous renumber + eqdsk rename in
merge; `OMP_PROC_BIND=close OMP_PLACES=cores` on SLURM; honest
"parallel ≠ bit-identical to serial" docstring.

### Contamination risks

**F17. Stale-shard contamination (bug — laptop AND slurm).**
`run_shard` writes `{header}_w{i}.h5` but `initialize_equilibrium_database`
opens **append** and `store_equilibrium` only deletes the group it rewrites.
Leftover shards (crashed laptop run never reaches cleanup; cancelled SLURM
array; re-run with fewer draws/worker) keep their old higher-index draws, and
`merge_archives` copies every integer group it finds → merged bouquet silently
contains previous-config draws. *Fix: `run_shard` removes its shard file
before generating (one line).*

**F18. No baseline cross-check on SLURM.** Laptop checks worker `li`/`Ip`
before merging; the `python -m bouquet.parallel merge` path checks nothing.
Heterogeneous nodes (AVX2 vs AVX512 BLAS/libm paths) can break bit-identical
baselines even at `nthreads=1`. Each shard's `_baseline` attrs already store
`Ip_target`/`l_i_target` → *fix: do the check inside `merge_archives` itself
(covers both backends, zero extra plumbing).*

**F19. `threads_per_worker>1` violates the one-core doctrine — and the
committed example does it.** `run_shard` maps `threads_per_worker` →
`cfg.solver.nthreads`; `examples/D3D-like/slurm_jobs/*_array.sbatch` ships
`--cpus-per-task=4` / `OMP_NUM_THREADS=4`. Consequences: ±1.3% li_1 jitter →
workers accept draws against **different** `l_i_target`s (contamination;
silent on SLURM per F18), plus occasional DLSODE hard-hangs → array task burns
its time limit and loses its whole shard (yield). Laptop's baseline guard
would at least raise. *Fix: regenerate the example at 1 cpu/task; warn or
require an explicit override for `threads_per_worker > 1`.*

**F20. Seeding soft spots.** Workers use global `np.random.seed(seed_base +
worker_id)` (MT19937). (a) Adjacent integer seeds aren't provably independent
— rigorous fix is `SeedSequence(seed_base).spawn(n_workers)` + a `Generator`
threaded through `generate_bouquet` (opportunistic). (b) **Real**: the
timeseries pattern reuses the same `SEED` every slice → draw *i* is correlated
across time slices (same perturbation stream). *Fix: fold slice index /
scan_key into `seed_base` per slice; document.*

### Yield / robustness

**F21. Stuck or silently-shrunk merges.** `afterok` + one timed-out array task
= merge pending forever, no message. And the merge CLI existence-filters shard
paths (`if os.path.exists`) — a missing worker quietly shrinks the bouquet.
*Fix: merge validates expected-vs-found shards loudly (then `afterany` is
safe); laptop worker crash currently also skips merge/cleanup and leaves
shards → feeds F17.*

### Operational

**F22. SLURM emitted-script path inconsistency.** Array script references the
bundle as `{out_dir}/bundle.pkl` (relative to parent) while `submit.sh` does
`sbatch {job_name}_array.sbatch` (relative to `out_dir`) — no CWD satisfies
both. *Fix: absolutize bundle + `out_header` in the scripts;
`cd "$(dirname "$0")"` in submit.sh.*

**F23. No environment setup hook in sbatch** (no venv/conda, `module load`,
`OFT_PYTHONPATH`) → shard dies at `import OpenFUSIONToolkit` on most clusters.
*Fix: `setup=[...]` lines param on `emit_slurm_script`.*

**F24. Blind cluster debugging.** `run_shard` fd-suppresses stdout/stderr
unless `verbose=True`; the CLI never passes it → per-task `slurm-*.out` files
are empty. SLURM already isolates per-task logs, so the notebook flooding
rationale doesn't apply. *Fix: CLI runs `verbose=True`.*

**F25. Pickle bundle fragility.** `.pkl` config ties the cluster run to the
emitting package/Python version → replaced by phase-1 config JSON.

**F26. Laptop core budgeting.** `os.cpu_count()` is logical cores; with one
solver per core, default the worker ceiling to physical cores.

### Phase 7 — parallel hardening  *(fixes F17–F26; items 1–6 IMPLEMENTED 2026-07-02 except the JSON-bundle part of 6, which waits on phase 1)*

1. ✅ `run_shard`: delete pre-existing shard file before generating (F17).
2. ✅ `merge_archives`: cross-shard baseline check from `_baseline` attrs
   (`Ip_target`, `l_i_target`; `baseline_match_rtol` arg, default 1e-6,
   loosenable for heterogeneous clusters), raises **before** anything is
   copied; missing shard paths raise. CLI `merge` counts expected-vs-found
   shards (zero-draw workers excluded), aborts on missing with the exact
   `--array=` re-run hint, `--allow-missing` merges a partial set loudly
   (F18, F21). Laptop orchestrator keeps its pre-merge check.
3. ✅ `threads_per_worker>1` now warns (li_1 jitter / DLSODE-hang rationale)
   in both `parallel_generate` and `emit_slurm_script`; the committed
   `examples/D3D-like/slurm_jobs/` set (bundle + sbatch + submit) was
   re-emitted at `threads_per_worker=1` / `--cpus-per-task=1` — it previously
   shipped `threads_per_worker=4` in the bundle, i.e. `nthreads=4` TokaMakers
   (F19). Verified: 8-case synthetic-shard test (renumber/rename, drift
   detection, rtol, missing-shard accounting, partial merge, zero-draw
   workers, warning) + full test suite 161 passed.
4. ✅ Seeding: worker seeds now derive from
   `SeedSequence([seed, worker_id, sha256(scan_key)])` (`_derive_seed`)
   instead of `seed + worker_id` — provably separated worker streams AND
   slice decorrelation for timeseries sweeps run with one seed (F20).
   **Note: this changes the draw set produced by a given `seed` relative to
   runs made before 2026-07-02** (statistically equivalent; baseline
   unchanged). The full `SeedSequence.spawn` + `Generator` threading through
   `generate_bouquet` (replacing global `np.random.seed`) remains future
   work.
5. ✅ `emit_slurm_script`: `setup=[...]` env lines in both sbatch scripts
   (module load / conda / `OFT_PYTHONPATH`); bundle referenced by basename +
   `submit.sh` cd's to its own dir (works from any CWD, no machine-specific
   absolute paths baked into committed examples); CLI shard runs
   `verbose=True` so per-task `slurm-*.out` captures solver output;
   `afterok`→`afterany` (safe now that the merge validates shards and aborts
   loudly) (F22–F24).
6. ✅ Physical-core default: `parallel_generate(n_workers=None)` resolves to
   physical (not logical/SMT) cores via psutil → `sysctl` → fallback (F26).
   ✅ Bundle → config JSON (F25) — done with Phase 1 (2026-07-02):
   `emit_slurm_script` writes `{job}_bundle.json` via `config.to_dict()`, the
   CLI rebuilds it with `BouquetConfig.from_dict`, `pickle` removed.

Verified: 12-case synthetic-shard test + full suite (161 passed); committed
`examples/D3D-like/slurm_jobs/` re-emitted with the new emitter.

---

## Explicitly out of scope / leave alone

- Homotopy & coil-drift knob set and their documented values.
- `_finalize_scan_result` shape-follows-the-call convention in `filtering.py`.
- Byte-opaque eqdsk/pfile storage.
- Solver-chatter capture design (`verbose=False` + logs).
- `paths.py` resolution logic.
- Any physics: SWB split modes, li targeting, Zeff-primary density scheme.

---

## Post-implementation review fixes (2026-07-02)

An 8-angle adversarial review of the implementation diff surfaced 8 verified
findings; all fixed on this branch:

1. **Solver-marked tests were broken** (hidden by default deselection):
   `test_systematics.py` read per-draw `coil_names` as a v1 JSON attr →
   KeyError on the v2 golden. Now uses `utils._read_coil_names` everywhere.
2. **Merged archives had no provenance**: `merge_archives(config=...)` now
   stamps run-level `config_json` (wired from `parallel_generate` + the CLI);
   without it the deliverable cluster file carried no config record.
3. **Root `config_json` was last-writer-wins in multi-slice files**:
   `load_config` is now scan-aware — per-scan copies authoritative, a
   multi-scan file with `scan_key=None` raises listing the keys.
4. **`DrawView` efficiency**: attrs cached per view (snapshot semantics +
   `refresh()`), `flags` built from the draw's own attrs (was a whole-scan
   sweep per access → O(N²) loops), `extract()` reads both blobs in one open.
5. **Schema lookup consolidated**: new `schema.find_bytes_dataset()` (fixed
   name first, legacy suffix-scan fallback) replaces the 8 copy-pasted
   lookups in plotting/filtering and `archive._find_suffixed`.
6. **`_eqdsk_dataset_name` removed** (returned a constant, ignored its args);
   `eqdsk_out_dir` extraction now writes coordinate-carrying FILENAMES again
   (the fixed dataset name made every extracted draw clobber `eqdsk`).
7. **Legacy signal**: `initialize_equilibrium_database` stamps
   `schema_version`/`bouquet_version`/`created` at creation (single
   chokepoint), `BouquetArchive` warns on unstamped files, and
   `load_equilibrium` raises a clear "pre-v2 archive" error naming any
   legacy-suffixed dataset it finds.
8. Refuted at verify (no code change): config tuple→list JSON drift (all
   consumers sequence-generic), provenance write race (workers write
   distinct files), v1 silent-read claim (failures were already loud).

Scope note: the diff also carries an unrelated new physics feature
(`read_ida_cer` + `radial_field_from_cer`, impurity radial force balance
E_r) — committed separately from the refactor.
