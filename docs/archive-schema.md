# The bouquet HDF5 archive — schema v3

Authoritative description of the on-disk layout written by bouquet ≥ 1.0.0
(schema v2 first shipped in the 1.0.0 release; v3 adds the self-consistent
bootstrap record — see [v2 → v3](#v2--v3-the-self-consistent-bootstrap-record)).
The single source of truth in code is [`bouquet/schema.py`](../bouquet/schema.py)
(`SCHEMA_VERSION`, `PROFILE_UNITS`, fixed dataset names, `write_profile` /
`find_bytes_dataset`); this document mirrors it for human readers. Prefer
reading archives through [`bouquet.BouquetArchive`](../bouquet/archive.py) or
the functional readers (`load_equilibrium`, `load_baseline_profiles`,
`select_indices`, `load_config`) rather than raw `h5py` — see
[workflows.md](workflows.md#reading-an-archive-back) for worked examples.

## Layout

```
{header}.h5                            file attrs: schema_version (=3),
│                                      bouquet_version, created, updated
├── config_json                        JSON dump of the run BouquetConfig
│                                      (root copy = most recent write; the
│                                      per-scan copies below are authoritative)
└── scan/<scan_key>/                   one group per scan point / time slice
    ├── config_json                    this slice's exact config
    │   attrs: [coil_filter]           'chi2' | 'legacy' -- which coil filter wrote
    │          [coil_sigma_model]      passes_coil_filter last (chi2: the sigma model JSON)
    ├── _baseline/                     written once per scan point
    │   ├── eqdsk, [pfile]             raw byte-perfect g-file / p-file
    │   ├── psi_N, psi_N_kinetic
    │   ├── n_e, T_e, n_i, T_i         kinetic profiles
    │   ├── pressure[, pressure_thermal]
    │   ├── j_phi[, j_BS, j_inductive] separated toroidal currents
    │   ├── sigma_ne/te/ni/ti/jphi     the uncertainty envelope used
    │   ├── [aux_<name>, sigma_aux_<name>]   switchboard channels
    │   ├── [recon_lcfs_ref]           10k-pt LCFS reference (boundary metric)
    │   ├── [x_points], [coil_currents, coil_names]
    │   ├── [structured_mse/]          structured closure + mse_data only:
    │   │                              per-chord arrays as DATASETS (chord_*,
    │   │                              tgamma_meas, sigma_eff, residual_sigma_*,
    │   │                              tgamma_pred_*, jacobian, excluded_*;
    │   │                              unified engine: engine_mse_*)
    │   ├── [engine_json]              the engine record as a string DATASET
    │   │                              when too large for an attribute
    │   └── attrs: Ip_target, l_i_target, l_i_scale, source_kind, [diverted],
    │              [source_current_sign, source_b0_sign,
    │               source_current_sign_origin, current_frame]
    │              [li_metrics_json, closure_limited]   baseline provenance (1.4+):
    │              Baseline.li_metrics as JSON, incl. the ip_closure health
    │              record on hybrid baselines; load_baseline_profiles() decodes
    │              it to li_metrics / ip_closure / closure_limited
    │              [jbs_converged, jbs_n_passes, jbs_loop_json]
    │                                  ← the baseline's jbs_loop block (v3)
    │              [delivered_state_json]
    │                                  ← the ONE reconstruction state (loop)
    │              [engine_json]      ← the unified engine's record (added;
    │                                    reconstruction_engine="unified"),
    │                                    with its "draws" settings block
    │              [edge_pressure_json] ← the edge-pressure record (added):
    │                                    edge_pprime_pin, separatrix_pressure,
    │                                    p_sep, p_sep_applied, pax_target
    └── <count>/                       one group per accepted draw
        │                              (integer; gaps = rejected draws)
        ├── eqdsk, [pfile]             raw bytes, fixed names
        ├── psi_N[, psi_N_kinetic]
        ├── j_phi, j_BS, j_inductive[, j_BS,edge]
        ├── n_e, T_e, n_i, T_i, w_ExB[, Zeff]
        ├── [pressure, pressure_thermal]
        ├── [aux_<name>]               perturbed switchboard channels
        ├── [coil_currents, coil_names]
        ├── [perturbed_lcfs_ref], [x_points]
        ├── [eq_fsa/]                  live-equilibrium flux-surface averages
        │   ├── psi_N                  (subgroup; see below)
        │   ├── F, avg_inv_R, avg_inv_R2, avg_B2
        │   └── q, dV_dpsi, f_trap, B_avg
        ├── [engine_json]              engine draw record as a string DATASET
        │                              when too large for an attribute
        └── attrs: l_i(1), l_i(3), count, homotopy_*, max_F_drift_pct,
                   max_VSC_drift_pct, in_spec, inspec_*, l_i_target_used,
                   [diverted], [passes_coil_filter, passes_boundary_filter,
                   selected]           ← filter flags, written post-hoc
                   [jbs_converged, jbs_n_passes, jbs_loop_json]
                                       ← the draw's jbs_loop block (v3)
                   [engine_json, passes_draw_band]
                                       ← engine draws only (added; see below)
                   [edge_pressure_json] ← the draw's edge-pressure record
                                       (added): its own p_sep, the offset
                                       applied, and beta / W_MHD in both
                                       pressure frames ("frames")
```

`edge_pressure_json` is read with `bouquet.edge_pressure.load_record(header,
count=None, scan_key=None)` (`count=None`: the baseline's). See
[physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin).
Under `separatrix_pressure="offset"` (the default since 2026-10-02; archives
written before then were made with `"legacy"`)
the stored `eqdsk` bytes carry the full pressure (`PRES` = the solver's
pressure + that equilibrium's `p_sep`).

## Conventions

- **Bare dataset names, units in attrs.** Profile datasets carry plain names
  (`j_phi`, `n_e`, …) with the unit string in `ds.attrs["units"]`
  (`PROFILE_UNITS` in `schema.py`). v1 archives embedded units in the name
  (`"j_phi [A m^-2]"`).
- **Fixed byte-blob names.** The g-file / p-file bytes are stored as `eqdsk` /
  `pfile` inside each group — the group path carries the coordinates. Bytes
  are stored opaque (`np.void`) and round-trip bit-perfect.
- **Always `scan/<key>/`.** The scan key is a user-chosen label
  (`GenerationConfig.scan_key`, default `0`) — a time in ms, a beta value, …
  Several bouquets can share one file under different keys.
- **Gap-tolerant indices.** Rejected draws leave gaps; iterate with
  `list_equilibrium_indices` / `BouquetArchive`, never `range(n)`.
- **Filtering is non-destructive.** Filters write boolean attrs
  (`passes_*`, `selected` = AND of applied flags); `export_filtered` produces
  a pruned copy, the source is never modified.
- **Provenance.** `schema_version` / `bouquet_version` / `created` are stamped
  at file creation; `config_json` is added by `write_provenance` (called from
  `Bouquet.generate`, `run_shard`, and `merge_archives`). Recover the exact
  run configuration with `bq.load_config(path, scan_key=...)`.
- **Current orientation.** Every archived current and eqdsk is in bouquet's
  positive-current frame (TokaMaker native: `Ip > 0`, `F0 > 0`). IMAS-path
  archives record the source's own orientation on `_baseline`:
  `source_current_sign` (the factor the reader multiplied every source current
  by; `-1.0` for a reversed-current source), `source_current_sign_origin`
  (whether that factor was `sign(ip)` or set by
  `ImasSource.current_orientation`), `source_b0_sign` (the source's
  vacuum-field sign; absent when `b0` is zero or unreadable) and
  `current_frame` (a plain statement of the frame).
  Absent on g-file-path archives and on IMAS archives written before the
  reader's normalisation. See
  [physics-notes](physics-notes.md#current-and-field-orientation).
- **Structured-closure MSE record (`_baseline/structured_mse/`).** Written
  only when `closure_channel="structured"` ran with `mse_data`
  (`Baseline.mse_record`). The per-chord arrays and the `n_chords × 2K`
  Jacobian grow with the chord count, so they are datasets here and never
  part of the closure record's single JSON attribute (HDF5 caps an attribute
  at 64 kB); the closure record carries only chord-count-independent
  summaries and points here. `load_baseline_profiles` returns the subgroup as
  a dict under `"structured_mse"`.
- **Live-equilibrium FSA (`eq_fsa/`).** Optional per-draw subgroup of
  flux-surface averages captured directly from the live TokaMaker object at
  generate time (`GenerationConfig.capture_live_eq`, on by default), on the
  `psi_N` grid of `capture_npsi` points. Keys and units are `EQ_FSA_GROUP` /
  `EQ_FSA_UNITS` in `schema.py`: `F` (T m), `avg_inv_R` (⟨1/R⟩, m⁻¹),
  `avg_inv_R2` (⟨1/R²⟩, m⁻²), `avg_B2` (⟨B²⟩, T²), `q`, `dV_dpsi`
  (m³ Wb⁻¹), `f_trap`, `B_avg` (⟨B⟩, T). `⟨1/R²⟩` is computed by exact
  FSA quadrature (`capture_exact_inv_R2`, default) with a fast path for
  `sauter_fc`'s native value when present. This is what enables the exact
  parallel↔toroidal current split in the IMAS/OMAS exporter
  (`write_imas_draw(..., fidelity="exact")`); read it back with
  `bq.load_eq_fsa`.

- **Self-consistent bootstrap record (`jbs_loop` block, schema v3).** When
  the bootstrap came from the self-consistent loop
  (`GenerationConfig.jbs_self_consistent`, **the default**) every draw group
  carries `jbs_converged` (bool), `jbs_n_passes` (int, all loops of the draw)
  and the full loop record as JSON in `jbs_loop_json` (residual histories,
  relaxation factors, the solved-vs-closure gap and the record-only
  unrelaxed closure residual `current_residual_unrelaxed`, tolerances, the
  post-homotopy check, the evaluator version, the OFT build as a path-free
  identifier `oft_build = {version, git_hash, build_id}` (archives written by
  earlier builds of this branch carry `oft_build.path`, the install
  location, instead), and `init_source` -- what each loop started from,
  per loop under `loops` and for the draw's first loop at the top level); the `_baseline` group carries the same three
  attrs for the baseline's own loop (`jbs_n_passes` = its pass count). Names
  in `schema.JBS_LOOP_ATTRS`; write/read with `schema.write_jbs_loop` /
  `schema.read_jbs_loop`, or read with
  `bouquet.utils.load_jbs_loop(header, count, scan_key)` (`count="_baseline"`
  for the baseline), `DrawView.jbs_loop` / `DrawView.jbs_converged`,
  `ScanView.baseline_jbs_loop` and `ScanView.bootstrap_model`. A group
  **without** the block carries a frozen (`solve_with_bootstrap`) bootstrap.

- **The one reconstruction state (`_baseline@delivered_state_json`, loop
  only).** The design rule: the input (g-file or modelling-source IDS), the
  bouquet reconstruction (as close to the input as it can be while physically
  valid and carrying a Redl bootstrap -- allowed to differ from the input),
  and the draws (perturbations of the reconstruction; at zero perturbation
  they reproduce it). With the loop on, the reconstruction is ONE
  equilibrium, and `_baseline` records it and says whether the run's
  baseline re-solve -- the saved `eqdsk` above and every draw's warm start --
  is it. JSON keys (`utils.DELIVERED_STATE_ATTR`, written by
  `utils.store_baseline_state`): `convention` (what the in-memory split is,
  below), `path`, `l_i` (= `l_i_target`, l_i(3)/'iter'), `q0`, `q95`
  (`get_stats` on the delivered state), `Ip_target`,
  `request_normalisation` / `achieved_normalisation` (the uniform factors
  that put the stored request / the achieved current at `Ip_target` in the
  'exact' FSA current measure), `n_floored_inductive`,
  `n_floored_target_inductive` (points where a zero-perturbation draw cannot
  reproduce the state), on the g-file path `li_corrective_state`,
  `li_step6_matched` and `li_input`, `how`, then `l_i_target`,
  `baseline_resolve` (`l_i`, `q0`, `q95`, `Ip` of the run's baseline re-solve
  and their differences from the recorded values) and
  `archived_j_phi_rel_l2_vs_delivered` (the archived `j_phi` against the
  delivered state's achieved current). Absent with
  `jbs_self_consistent=False` (legacy archives are unchanged bit for bit).
  With the loop on, `_baseline/j_phi` is the delivered state's ACHIEVED FSA
  current (as before: `store_achieved_jphi`), `j_BS` the draws' σ=0
  bootstrap composition on it (+ `jBS_diff`) and `j_inductive` their
  residual; the in-memory `Baseline` split the draws consume is the
  Ip-normalised jphi-linterp REQUEST of the same state (one solve of it is
  the state), with `Baseline.jphi_request_offset` = request − achieved
  (not archived).

- **The unified engine's records (added fields; schema stays v3).** Written
  only with `reconstruction_engine="unified"`; a legacy archive carries none
  of them and reads exactly as before.
  - `engine_json` (on `_baseline` and on every draw group): the JSON record,
    as the attribute when it fits in 60 000 bytes
    (`engine.ENGINE_ATTR_MAX_BYTES`), else a string DATASET of the same name
    with the attribute holding the pointer `{"stored_as": "dataset"}` (HDF5
    caps a group's object header at 64 KiB). Read with
    `bouquet.engine.read_engine_json(group)`,
    `engine.load_baseline_engine(header, scan_key)` or
    `engine_draws.read_draw_engine(header, count, scan_key)`.
  - The baseline record is the reconstruction (contract, settings,
    convergence constants and their origins, per-pass log, delivery checks,
    the state, solves, and `coil_solve_mode`: `"bounded"` -- the solver's
    coil least-squares mode, entered once at `Bouquet.setup_solver`, see
    `bouquet.solver_state`; absent in records written before 2026-10-06) plus, once `generate()` ran, a `draws` block (the
    draws' loop settings, `rng_stream`, `q0_row`, `homotopy`,
    `l_i_tolerance`, the Ip-row target `Ip_target_A` in the exact measure,
    and the `reference` values every draw is compared with). Its per-chord
    MSE arrays and Jacobians are NOT in the JSON: they are datasets
    `engine_mse_*` under `_baseline/structured_mse/`, and the JSON holds
    `"mse_record[<key>]"` in their place.
  - The per-draw record (`engine_draws`, version `unified-engine-draw/1`):
    `identity` (whether the first request was the stored one, bit for bit),
    `inputs` (bootstrap scale, pressure-match iterations, inductive tries),
    `rng_stream`, `amplitude` (per pass: `a_ind`, `a_bs` with the q0 row,
    the Ip increment and the Ip in the exact measure against its target),
    `loop` (the kernel record), `passes`, `q0_row`, `delivered` (the loop's
    delivered draw: `l_i_3`, `l_i_1`, `beta_n`, `q0` at `q0_psi_N`,
    `q0_stats` at `q0_stats_psi_N`, `q95`, `Ip`, the delivery check,
    request − achieved, `flux_range`), `archived` (the same after the
    homotopy stage, plus `split`: see the next item),
    `reference` (the reconstruction's DELIVERED measurement -- every value,
    its `flux_range` and the q-row radius included), `deltas`
    (against the reconstruction: `l_i_3`, `l_i_1`, `beta_n`, `q0`, `q95`,
    and the poloidal flux range `psi_b - psi_a` -- `flux_range` [Wb/rad]
    and `flux_range_rel`; `archived.deltas` the same for the archived
    state; `delivered.flux_range` / `archived.flux_range` the values),
    `post_hoc`
    (the l_i band and `constrain_sawteeth` verdicts, `coil_in_spec`,
    `in_spec`), `homotopy`, `post_homotopy`, and `cost` (solves, passes and
    wall time for the stages `anchor`, `loop`, `homotopy`, `post_homotopy`,
    `filters`, `archive`, and their `total`).
  - **The current split of an engine draw group** (`j_phi`, `j_BS`,
    `j_inductive`) is evaluated on the draw's own ARCHIVED (final,
    post-homotopy) equilibrium: `j_phi` its achieved FSA current; `j_BS`
    the draw's bootstrap model on that state, `s_bs (1 + d_bs) x scale x
    Redl(final) x F<1/R>/<B^2>(final)`; the fixed beam / RF parts the
    contract's `<j.B>` times the same final-state factor (recorded in
    `archived.split.j_NBI` / `j_RF`); `j_inductive` the residual `j_phi -
    j_BS - j_NBI - j_RF` (it carries the pressure-driven term) and NEVER
    clipped -- a negative value is recorded in `archived.split`
    (`n_negative_inductive`, `min_inductive`, `negative_inductive_psi_N`)
    and printed, never altered or filtered. (Legacy draws keep their split:
    the residual floored at zero with the sliver moved into `j_BS`.)
  - `passes_draw_band` (bool attr, engine draws only): the post-hoc band
    verdict. It is one of the filter flags ANDed into `selected`
    (`filtering._FILTER_FLAGS`), so `.filter()` selects what the until-N
    ledger counted; `in_spec` is the coil verdict AND this band.
  - A reader that predates these fields misreads nothing: every field it
    knows keeps its name, unit and meaning. (An older package that
    RE-FILTERS an engine archive would not AND `passes_draw_band` into
    `selected`; `in_spec` still carries the band.)

## v2 → v3: the self-consistent bootstrap record

Schema v3 (this release) is **additive**: it adds the `jbs_loop` block above
and changes no v2 dataset, attr, name, unit or meaning.

- **No migration.** A v2 archive reads as a v3 archive whose bootstrap is
  frozen everywhere (no `jbs_loop` block); every v3 reader accepts it
  unchanged, and every v2 reader ignores the new attrs. Do not gate readers on
  `schema_version == 2`.
- **Which bootstrap is in an archive** is decided by the block, never by the
  version number: a v3 archive written with `jbs_self_consistent=False`
  carries no block either (it is the frozen path, bit for bit: the one solver
  change the loop needed, the soft closure's noise-floor acceptance, is
  opt-in via `close_ip_structured_soft(accept_noise_floor=True)` and passed by
  the loop's closure calls only), and appending
  to a v2 file with a current bouquet restamps `schema_version` to 3 while its
  old draws keep reading as frozen. `ScanView.bootstrap_model` gives the label
  ("self-consistent Redl bootstrap" / "frozen SWB bootstrap (legacy)").
- **Replaying an old archive's config.** A `config_json` written before the
  loop existed has no `jbs_self_consistent` field; `load_config` /
  `BouquetConfig.from_dict` rebuild it with `jbs_self_consistent=False` (and
  warn), the behaviour it was produced with.
- **Comparisons across the change.** The self-consistent bootstrap moves the
  bootstrap/inductive split (and with it l_i, q0 and the per-draw responses);
  compare a v2/frozen archive with a v3 loop archive as two bootstrap models,
  not as a regression.

## Legacy (pre-v2) archives

Schema v2 was a clean break (2026-07). Files without the `schema_version`
attr are pre-v2: `BouquetArchive` opens them with a warning (byte blobs still
resolve via a suffix scan; profile keys keep their v1 bracketed names), and
`load_equilibrium` raises a clear error. Regenerate old archives with the
current package for full support.
