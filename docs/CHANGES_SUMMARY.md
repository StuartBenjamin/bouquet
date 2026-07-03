# Bouquet — change summaries

## Class API + HDF5 schema v2 round (2026-07, PR #8)

Decisions and outcomes folded from the (deleted) working document
`docs/ux-review-feat-bouquet-class-api.md`:

1. **Config serialization + provenance**: `BouquetConfig.to_dict/from_dict`
   (JSON); every archive stores `schema_version` / `bouquet_version` /
   `created` (stamped at file creation) + per-scan `config_json`
   (`write_provenance` / `load_config`).
2. **`BouquetArchive`** reader class (ScanView/DrawView, lazy, cached attrs,
   fixed-name + legacy suffix-scan eqdsk/pfile lookup) — replaces downstream
   hand-rolled h5 traversal.
3. **De-threaded readers**: module-level plot/filter/select functions accept a
   `Bouquet` / `BouquetArchive` / header / path uniformly; a missing explicit
   `scan_key` raises listing available keys (was silent-empty).
4. **Schema v2 — clean break, no legacy readers** (decision: no external users
   yet): bare dataset names + `units` attrs, fixed `eqdsk`/`pfile` names,
   coil names as a string dataset everywhere, `scan/<key>/` layout only.
   Legacy files: `BouquetArchive` warns, `load_equilibrium` raises clearly.
   Spec: `docs/archive-schema.md`; source of truth: `bouquet/schema.py`.
5. **Config/API simplifications**: `run.describe()` (non-default knobs),
   `workflow` preset enum (`auto` / `geqdsk-standard` / `imas-diff-c` /
   `custom`), source-agnostic `run.prepare()`.
6. **Sweeps + plotting**: `run.run_slices()` (one archive, one scan_key per
   slice); `plot_bouquet` dispatches on stored `source_kind`
   (`plot_imas_bouquet` demoted from `__all__`).
7. **Parallel hardening**: fresh shards, merge-side baseline guard +
   expected-shard accounting (`--allow-missing`), nthreads=1 doctrine warning,
   `SeedSequence(seed, worker, scan_key)` slice-decorrelated seeding, JSON
   SLURM bundles, CWD-independent submit scripts, physical-core default.
8. **Post-review fixes** (8-angle adversarial review of the final diff):
   solver-marked tests un-broken (coil_names dataset read), merged archives
   get run-level `config_json`, multi-scan `load_config` disambiguation,
   `DrawView` O(N) flags + cached attrs, `schema.find_bytes_dataset`
   consolidation (8 duplicated lookups), unique eqdsk extraction filenames,
   legacy-archive warnings/errors.
9. **Won't-do (user decisions)**: `jphi_scalar_sigma` default stays 0.10;
   IDA_run per-shot notebooks stay split (no templating).
10. **Deferred**: example-notebook rewrite to run-object idioms;
    reconstruction-style IMAS baseline summary block.

Also in this round: scipy≥1.18/numpy≥2.5 compatibility (`.item()` at the
axis-point `.ev()` call), and the CER/E_r feature (`read_ida_cer`,
`radial_field_from_cer`).

---

# Bouquet + OFT — change summary (golden suite, filtering, Ip-secant removal, systematics)

> **Historical document** (2026-05): a snapshot of the coil-bounds/golden-suite
> round of work, kept for context. Branch/repo layout below reflects the
> author's working setup at that time; PR #3 (feat/coil-bounds) has since
> merged to `main`.

| repo | path | branch | remote |
|---|---|---|---|
| OpenFUSIONToolkit (fork) | `OpenFUSIONToolkit/` | `feat/jphi-linterp-Ip-cutcell-fix` | `d-burg/OpenFUSIONToolkit` |
| bouquet backend (package + tests) | `bouquet_coil_bounds/` | `feat/coil-bounds` | `d-burg/bouquet` |
| bouquet examples (D3D-like) | `bouquet/` | `feat/lock-coils-pr1` | `d-burg/bouquet` |

> The two bouquet clones are the same repo on different branches; the backend
> path-insert in the notebooks is temporary until they're merged into one
> shipped `bouquet`.

---

## 1. OpenFUSIONToolkit — native Ip hold + bootstrap cleanup

**Already committed** on `feat/jphi-linterp-Ip-cutcell-fix` (the Ip "hot fixes"):
- `5b06f66` jphi-linterp Ip-correction **outer iteration** in the GS solve.
- `328163f` restore `Itor_target` at outer-loop exit.
- `97436b2` opt-in trace prints (`oft_debug_print`).
- `b032e4f` safety-bail on corrupted `ip_phys`.
- (plus the earlier cut-cell / `jphi_update` structural fix and the
  `#248` `TokaMaker_equilibrium` merge.)

These make the Fortran backend hold `Ip` to target natively (≈0.05 %),
which is what lets us delete the Python Ip workarounds below.

**New, uncommitted** (`src/python/OpenFUSIONToolkit/TokaMaker/bootstrap.py`):
- Removed the **`find_j0=False` (Ip-scale) secant** call from
  `solve_with_bootstrap` → `final_scale_Ip = 1.0` (Ip held natively).
- Stripped the now-dead `find_j0=False` branch + `get_Ip_error` + the
  `find_j0`/`scale_j0` params from `find_optimal_scale`; it is now a clean
  **core-j0-only** scaler. Callers updated.
- (Mirrored into `build_release/` + `install_release/` so the running env
  matches; only `src/` is tracked.)

→ **PR target:** `feat/jphi-linterp-Ip-cutcell-fix` → upstream (or fork main).
See `OpenFUSIONToolkit` PR body draft.

---

## 2. bouquet backend (`bouquet_coil_bounds/`, `feat/coil-bounds`)

### X-point detection (TokaMaker, not geometric)
- Capture `mygs.get_xpoints()` at generation time (baseline + per draw) at the
  same solver state the eqdsk is saved from; store `x_points` dataset +
  `diverted` attr in the H5 (`utils.store_equilibrium` / `store_baseline_profiles`).
- `plot_boundary_point_traces` now uses the stored true B_p=0 saddles
  (`_xpoints_on_lcfs`) instead of the erratic geometric corner finder; clean
  fallback to the axis-line intersection for pre-existing H5s.
- Added `utils.list_equilibrium_indices()`; fixed a `KeyError` in
  `plot_coil_currents` (and silent last-draw drop in other plots) on
  band-rejection index gaps.

### Postprocessing filters (`bouquet/filtering.py`, new)
- `filter_coil_currents()` and `filter_boundaries()` — **non-destructive**:
  write `passes_coil_filter` / `passes_boundary_filter` + derived `selected`
  flags, return `(summary, distribution-figure)`. Coil thresholds default to
  the stored `inspec_F_max/VSC_max` (reproduce in-loop in_spec); boundary filter
  is diagnostic-only until a `rms_max_mm`/`max_max_mm` is given.
- `read_filter_flags()`, `select_indices('all'|'selected'|'excluded')`,
  `export_filtered()` (pruned copy, baseline preserved, source untouched).
- `plot_bouquet(..., selection=...)` honours the flags.

### Units, flags, RNG (no env vars / no percentages in the user API)
- `l_i_tolerance` and `p_thresh` are now **fractions** (e.g. `0.05`), converted
  to percent internally; defaults updated.
- `jphi_baseline=True` flag replaces the `JPHI_BASELINE` env var.
- `pin_jphi=False` flag replaces the `PIN_JPHI` env var.
- `seed=None` parameter seeds the RNG inside `generate_bouquet`.
- (env vars retained only as back-compat overrides.)

### Removed all Python Ip-rescaling secants (kept core-j0)
- Per-draw post-perturb Ip-secant (was a no-op; dead block deleted).
- `reconstruct`/`fit_inductive_profile` **§6 Ip-correction secant** deleted
  (kept `Ip_desired` + `j_ind_li` that §7 consumes).
- OFT `find_j0=False` call (see §1).
- Inert `final_scale_Ip = 1.0` vestige in the l_i loop.
- **Kept:** all `find_j0=True` core-j0 secants + the l_i-match inductive secant.

### Tests + golden suite (`tests/`)
- `tests/golden/`: `D3Dlike_Hmode_golden_slim.h5` (~12 MB, **geqdsks retained
  gzip-compressed** for g-file handling tests, p-files dropped) +
  `golden_manifest.json` + `make_golden_fixture.py` (`--eqdsk all|subset|none`)
  + `README.md`. `.gitignore` negation `!tests/golden/*.h5`.
- `tests/test_golden_bouquet.py` (15 tests): scalars/coils/x-points/boundary
  vs manifest, geqdsk parse + separatrix-coarseness, filtering/selection/export.
- `tests/test_systematics.py` (2 tests, **opt-in** via
  `pytest -m solver`): σ=0 pinned → baseline (RMS<0.8 mm, drift<0.3 %).
- `tests/test_synthetic_sigma.py` (11 tests): the sine-basis IDA σ helper.
- **Protected proprietary data:** added `*.cdf`/`*.nc` to `.gitignore`
  (fixed a latent inline-comment bug) — operational `IDA_*.cdf` files must never
  be committed.

**Fast suite: 123 passed, 17 skipped (the 2 solver tests skip by default).**

---

## 3. bouquet examples (`bouquet/`, `feat/lock-coils-pr1`)

- **Non-proprietary baseline:** `D3Dlike_Hmode_baseline.geqdsk` / `.peqdsk` +
  `D3Dlike_Hmode_baseline_RECIPE.md` + overview PNG (147131-derived, COCOS 1,
  Ip +1.20 MA, Bt −2.0 T, physical bootstrap, smooth edge).
- **Updated example notebook** `bouquet_D3Dlike_example.ipynb`:
  top-level `REGENERATE` toggle (load golden vs rebuild), decimal knobs,
  `jphi_baseline`/`seed` flags, §9 filtering demo, **no shot-number references**.
- **New systematics notebook** `bouquet_D3Dlike_systematics.ipynb`: runs 3 modes
  (pinned σ=0 / pinned IDA-σ / production, n=10) and checks all traces + coil
  currents with signed-drift summaries to expose any bias.
- `legacy/` folder for the superseded example; builder/helper scripts.
- The full 30 MB golden run stays **untracked** (`*.h5`); only the slim fixture
  in the backend repo is tracked.

---

## 4. Validation

- **Recon Ip (native):** baseline **0.0000 %**, per-draw median +0.03 %, max
  0.04 % — confirms the OFT native hold; Python Ip secants were redundant.
- **No-systematic floor (σ=0 pinned, live solve):** boundary RMS **0.525 mm**
  (deterministic across draws), max coil drift **0.054 %**.
- **VSC drift (production golden):** signed F9A/F9B means −0.19 % / +0.24 %
  (vs σ≈3 %), non-VSC F-coils ±0.005 % → symmetric scatter, **no systematic
  bias**. The 10/20 in-spec yield is honest spread straddling the ±2 % spec.
- **New golden:** 20 draws, 10 in-spec, recon `l_i` 0.842; manifest diff vs the
  prior golden is small (l_i_target +0.2 %, slightly wider boundary/VSC spread).

---

## 5. New / changed public API

```python
generate_bouquet(..., l_i_tolerance=0.01, p_thresh=0.05,   # now fractions
                 jphi_baseline=True, seed=None, pin_jphi=False)
from bouquet import (filter_coil_currents, filter_boundaries,
                     select_indices, read_filter_flags, export_filtered)
plot_bouquet(..., selection='all'|'selected'|'excluded')
from bouquet import list_equilibrium_indices, synthetic_ida_sigma
# OFT: find_optimal_scale(...) is now core-j0 only (no find_j0/scale_j0 args)
```

---

## 6. Follow-ups

- **Open issue:** jphi-linterp edge / separatrix `j_φ` handling in reconstruction
  (the ~0.5 mm σ=0 floor) — see `docs/ISSUE_jphi_edge_reconstruction.md`
  (backend task #41).
- Modes 3–4 (free-jphi ±homotopy) broader validation (task #32).
- Eventually merge the two bouquet branches into one shipped package (removes the
  notebook path-insert).
