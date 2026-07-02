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

### Phase 1 — config serialization + h5 provenance  *(unlocks F9, F10, F8)*

1. `bouquet/config.py`: add `BouquetConfig.to_dict()` / `from_dict()` (and
   `to_json` / `from_json` thin wrappers).
   - ndarray fields (isoflux, sigma_profiles, aux_*, fixed components) →
     lists on dump, back to ndarray on load; source type recorded via a
     `"source_type": "reconstruction" | "imas"` discriminator.
   - Round-trip test: `from_dict(to_dict(cfg))` equality on a config
     exercising every optional field.
2. `bouquet/utils.py` `initialize_equilibrium_database()` (or a new
   `_write_provenance(hf, config)` called from `Bouquet.generate` and
   `run_shard`): write file-level attrs
   `schema_version` (int, start at `2`), `bouquet_version`, `created`
   (ISO timestamp), and dataset `config_json`.
   - Multi-scan files: also mirror `config_json` per `scan/<key>/` group
     (each slice can have a different header/time/scan_key).
3. `bouquet/parallel.py`: replace whatever ad-hoc config shipping `run_shard`
   / `emit_slurm_script` do with `to_json`/`from_json`.
4. New `bq.load_config(h5path_or_header)` → `BouquetConfig` (reads
   `config_json`; raises with a clear message on pre-provenance files).

### Phase 2 — `BouquetArchive` reader class  *(fixes F13, F1-read-side, F12-read-side)*

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

### Phase 3 — kill the `(header, scan_key)` re-threading  *(fixes F1, F7)*

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

### Phase 4 — schema v2 (writer changes, read-compat kept)  *(fixes F11, F12, F15, F14; optionally F16)*

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

### Phase 5 — config/API simplifications  *(fixes F2, F4, F3)*

1. **Align defaults with validated production values**: `jphi_scalar_sigma`
   0.10 → 0.05 in `UncertaintyConfig` (confirm 0.05 is intended for both
   paths). Then strip the settings blocks from the example notebooks down to
   the true deltas (`seed`, `scan_key`).
2. **`run.describe()`**: print the config grouped by section, showing only
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

### Phase 6 — sweeps + notebooks  *(fixes F5, F6, F8)*

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

| Phase | Depends on | Size | Risk |
|---|---|---|---|
| 1 config serialization + provenance | — | S–M | low |
| 2 `BouquetArchive` | — (better after 1) | M | low (read-only) |
| 3 de-thread header/scan_key | 2 | S | low |
| 4 schema v2 | 1, 2 | M | medium (writer change; legacy readers must stay green) |
| 5 config simplifications | — | S | low; default change needs a yield A/B on both testbeds |
| 6 sweeps + notebooks | 1, 3, 5 | M | low |

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

### Phase 7 — parallel hardening  *(fixes F17–F26; items 1–3 IMPLEMENTED 2026-07-02)*

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
4. Seeding: per-slice seed offset in sweep helpers now; `SeedSequence.spawn`
   when `generate_bouquet` grows an `rng` parameter (F20).
5. `emit_slurm_script`: absolute bundle/output paths, `cd` in submit.sh,
   `setup=[...]` env lines, CLI `verbose=True`; swap `afterok`→`afterany` once
   merge validates (F22–F24).
6. Bundle → config JSON (after phase 1) (F25); physical-core default for
   laptop `n_workers` (F26).

Depends on: nothing (items 1–5); item 6 on phase 1. Size S–M, risk low —
items 1 and 2 are the priority (both are silent-contamination holes).

---

## Explicitly out of scope / leave alone

- Homotopy & coil-drift knob set and their documented values.
- `_finalize_scan_result` shape-follows-the-call convention in `filtering.py`.
- Byte-opaque eqdsk/pfile storage.
- Solver-chatter capture design (`verbose=False` + logs).
- `paths.py` resolution logic.
- Any physics: SWB split modes, li targeting, Zeff-primary density scheme.
