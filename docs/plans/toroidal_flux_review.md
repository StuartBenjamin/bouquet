# Review: toroidal_flux_mapping (c0c3337..2136767), 2026-09-24

End-to-end phi_n review. Each item carries an ID; **Fix:** lines record the applied change (branches tfrev-{io,ti,plot,cfg,solver}, merged into toroidal_flux_mapping; line numbers at the merge).

## 1. Coordinate mapping at the solver boundary

Verified correct: all profile dicts via `oft_prof`/`pp_prof`; all 6 SWB calls carry grid+coord; paired readbacks via `psi_of`; ψ-integrals via `psi_at`; windows via `window_x`/`_w`; q0 checks; homotopy rollback; `capture_equilibrium_fsa`; per-draw p-file on the draw's own map.

- **S1** Cached jBS-delta/DIFF_BS seed built before the cache's own anchor solve (TokaMaker_interface.py:4781 vs 2533/2327) → σ=0 seeds differ under `seed_coord="psi_n"`.
  **Fix:** cache seed built after the cache anchor solve (bouquet/TokaMaker_interface.py:4836); new `swb_seed_ref` kwarg of `perturb_kinetic_equilibrium` (bouquet/TokaMaker_interface.py:1752, passed at 5283) reused by DIFF_BS (bouquet/TokaMaker_interface.py:2336) and the delta-composition draw path (bouquet/TokaMaker_interface.py:2543). Reset to None if cache setup fails.
- **S2** `psi_at` evaluated unconditionally as `_r2_ip_scale` argument (2776, 2847), incl. after a failed band-resample solve.
  **Fix:** `_r2_ip_scale(..., coord)` maps the grid with `psi_at` only inside the legacy root branch (bouquet/TokaMaker_interface.py:1520); callers pass the run grid + coord.
- **S3** `check_backend` does not require `solve_with_bootstrap` itself to take `coord`.
  **Fix:** `check_backend` also requires `"coord"` in `solve_with_bootstrap`'s signature (bouquet/coords.py:215).
- **S4** `coord`/`window_coord`/`seed_coord` validated only in `Bouquet._check_coord`; setting `run.baseline` directly then `generate()` skips `check_backend` and the `use_python_solve` guard.
  **Fix:** values validated at construction (`coords.check_source_coord`/`check_native`, coords.py:42-54; config.py:204, 254, 1040-1042; phi_n + `use_python_solve` refused at config.py:1358). New `coords.check_run` (coords.py:240) is the prepare guard (run.py:646) and re-runs in `generate()` on the baseline's coord (run.py:3299).
- **S5** (doc) GPR length scales are Φ_N lengths in a phi_n run; `smooth_jbs_transition` window is index-based.
  **Fix:** docs/workflows.md (length scales are Φ_N lengths; window helpers take the grid given); comments at bouquet/TokaMaker_interface.py:1633 (index-based smoothing window), 1803, 2032.

## 2. io read-in

FF′/P′: g-file PPRIME/FFPRIM enter only via `j_tor_averaged_direct` (values on ψ nodes → relabelled by `rhovn²`, tag `phi_n_relabel`); pressure built on Φ_N nodes, P′ sent as dP/dΦ_N (tag `phi_n`). IMAS `dpressure_dpsi`/`f_df_dpsi` never read; p-file derivative columns unused.

- **IO1** IMAS equilibrium `j_tor`/`pressure` placed onto core_profiles nodes by ψ_N (io/imas.py ~1067, ~1112). Real FUSE dd 174956: up to 1.8 % of peak j_tor (0.3 % pressure) vs placement by ρ_tor. Should use `equilibrium.profiles_1d.rho_tor_norm²` in phi_n.
  **Fix:** phi_n places `equilibrium.profiles_1d` pressure/j_tor by the equilibrium's own Φ_N: `rho_tor_norm²` (new `_phi_n_from_rho`, bouquet/io/imas.py:367, same placeholder/missing checks as core_profiles), else by its q (`phi_n_from_q`, imas.py:1080). psi_n unchanged.
- **IO2** `write_imas_draw` (phi archive): exact fidelity samples eq_fsa at template ψ_N, not the draw's ψ_N of the Φ nodes (io/imas.py ~1392); `grid.psi` left as the template's.
  **Fix:** phi_n archives: the draw's ψ_N at the template Φ_N nodes from the draw eqdsk's `rhovn²` (imas.py:1379) addresses `eq_fsa` and is written to `grid.psi`; `rho_tor_norm` kept.
- **IO3** Fixed components / `sigma_profiles` / `aux_*` can only be supplied on the run grid; no ψ_N option.
  **Fix:** `FixedComponentsConfig.coord` (`"run"` default | `"psi_n"`, config.py ~299) mapped through `coords.to_run_grid` (coords.py:56) with the source's own map: g-file path baseline.py:1022, IMAS path imas.py:1038. Docs state `sigma_profiles`/`aux_*` are on the kinetic run grid.
- **IO4** Per-draw p-file SOL (ψ>1) filled flat in phi_n (kinetic grid truncated at LCFS).
  **Fix:** phi_n per-draw p-file keeps the source p-file values outside the draw's ψ range (bouquet/TokaMaker_interface.py:6000).
- **IO5** `UncertaintyConfig.ida_path` ≠ source IDA: sigmas mapped through the other source's `psi_map` (clamped); should use the file's own q.
  **Fix:** a sigma IDA file that is not the source's and has q is placed by its own q (baseline.py:611).
- **IO6** g-file path: IDA without q silently falls back to the g-file map (`baseline._kinetic_phi_n`); IMAS ida_hybrid raises. Inconsistent.
  **Fix:** phi_n g-file path raises for an IDA .cdf without q; p-file keeps the g-file map (baseline.py:969).

## 3. Plots / metadata

- **P1** `plot_jphi` source overlay interpolates FUSE/g-file by ψ_N onto the Φ_N archive grid (plotting.py:3653-3664).
  **Fix:** phi_n archive: IMAS source placed by normalised `rho_tor_norm²`, g-file by `rhovn²` (plotting.py `plot_jphi`, :3623).
- **P2** `_relabel_x` relabels q/FF′ panels (eqdsk ψ_N grid) as Φ_N (plotting.py:1128/1286/1018).
  **Fix:** `_load_flux_functions` (plotting.py:1253) returns q/FF′ on `rhovn²` in phi_n archives, so the Φ_N label is true.
- **P3** `plot_jphi` (3718) / `plot_aux_profiles` (2712) pass unresolved `scan_key=None` → `profile_coord` returns "psi_n" on scan layout.
  **Fix:** resolved scan keys passed; `utils.profile_coord(scan_key=None)` (utils.py:3091) reads the scan points' shared coord (raises if mixed).
- **P4** `plot_input_vs_recon` interpolates run-grid baseline arrays at the solver's ψ grid.
  **Fix:** `plot_input_vs_recon` (plotting.py:1632) maps baseline grids to ψ via `coords.psi_at` before interpolating.
- **P5** Recon result `'pprime'` is dP/dΦ_N/psi_range; `plot_tokamaker_comparison` overlays it on g-file dP/dψ; x labels ψ_N.
  **Fix:** result `pprime` is dP/dψ in phi_n (bouquet/TokaMaker_interface.py:6968); `plot_tokamaker_comparison` relabels Φ_N axes (`_result_is_phi`/`_relabel_comparison`, plotting.py:240-249).
- **P6** Hard ψ_N labels: `run.plot_baseline`, gui.py, `plot_bouquet_timeseries` (no cross-archive coord check).
  **Fix:** `plot_baseline` (run.py:2865) labels from `bl.coord`; gui relabels after redraw (gui.py:268); `plot_bouquet_timeseries` (plotting.py:2774) labels from `profile_coord` and warns on mixed archives.
- **P7** Draw groups / `profiles_doc` carry no `profile_coord`; `eq_fsa/psi_N` (always ψ_N) undocumented.
  **Fix:** `store_equilibrium` writes a draw-group `profile_coord` attr (utils.py:3787); `profiles_doc` carries it (archive.py:220); schema.py + docs/archive-schema.md state `eq_fsa/psi_N` is always ψ_N.
- **P8** `merge_archives` does not check shard `profile_coord`.
  **Fix:** `merge_archives` raises before writing if shard `profile_coord`s differ (parallel.py:287-306).
- **P9** `verify_sigma0_consistency` `psi_worst` / `floor_inductive_split` message say psi_N for Φ_N.
  **Fix:** `verify_sigma0_consistency` adds `x_worst` + `coord` and labels its print (run.py:2970-2992); `floor_inductive_split(..., coord=)` message (baseline.py:807, 1133).
- **P10** gui.py:239 plots kinetics against `psi_N`, not `psi_N_kinetic` (shape mismatch in phi_n recon).
  **Fix:** gui kinetics use `psi_N_kinetic` (gui.py:235).

## 4. Missing tests

- **T1** IMAS psi_n vs phi_n read = relabel (real `rho_tor_norm`); `rho_tor` spelling; placeholder dd refused end-to-end; ρ not 0→1 renormalised.
  **Fix:** tests/test_phi_io_imas.py `TestReadPhi` (relabel, eq by own rho or q, rho_tor, unusable rho refused, renormalisation, fixed comps on psi_n).
- **T2** Stubbed recon baseline: `x_run == rhovn²`, kinetics truncated/PCHIP, `psi_map`, fixed comps on Φ, `seed_coord`.
  **Fix:** tests/test_phi_cfg.py (stubbed recon baseline).
- **T3** `pp_prof(Φ)·dΦ/dψ ≈ pp_prof(ψ)` with non-identity map (replaces weak same-array test).
  **Fix:** tests/test_phi_ti.py (pp Jacobian consistency; S2 laziness). Weak same-array test kept as the ψ-scale check.
- **T4** Analytic IDA placement; IDA ω; 2-D ensemble q.
  **Fix:** tests/test_phi_cfg.py (analytic IDA placement, ensemble q).
- **T5** `write_imas_draw` phi archive all channels + exact; `profile_coord` round trip + `_relabel_x`.
  **Fix:** tests/test_phi_io_imas.py `write_imas_draw` phi/psi archives; tests/test_phi_archive_plots.py (profile_coord round trip, draw attr, `_relabel_x`, merge refusal, overlay placement).
- **T6** Config: reserved `coord`/`x`/`psi_N`; `check_backend` fake toolkit; JSON round trip of coord fields.
  **Fix:** tests/test_phi_cfg.py (reserved kwargs, fake-toolkit `check_backend`, round trip, construction-time validation).
- **T7** (solver) `get_torflux_map` vs ∫q; Φ-tagged vs ψ-tagged pp; plan §7 Φ round trip; `window_coord="psi_n"` classify parity.
  **Fix:** tests/test_phi_solver.py (15 `solver` tests): torflux map vs ∫q (2.2e-4) and inverse∘forward; ψ- vs Φ-tagged solve (p 1.3e-4, q95 4.7e-4); g-file σ=0 round trip psi_n vs phi_n (l_i +0.06 %, q0 +0.29 %, q95 0.03 %); `window_coord="psi_n"` classify parity (H_mode both). IMAS round trip not added (needs a dd with real `rho_tor_norm`).

Real-dd probe: `.bouquet_wt/torflux_jobs/ddprobe/ddprobe.py`.

## Verification (merged tree, SLURM)

- Fast suite, no OFT: 1238 passed, 2 skipped.
- coords/config/phi test files with OFT (oftstage_swbx): 113 passed. This includes the base-branch failure `test_config::test_without_the_toolkit_only_the_reserved_check_runs`, which now stubs `_bootstrap_kwarg_names` so it holds with OFT importable.
- `tests/test_phi_solver.py` with OFT: 15 passed (1:44).

## IMAS Φ round trip: real FUSE dd, 174956 @ 2.0 s (2026-09-24)

Setup: σ=0 `prepare()` using the production config: g-file LCFS, structured `li_soft_onesided`, ohmic split. OFT is oftstage_swbx. Drivers and logs are in `.bouquet_wt/torflux_jobs/imas_rt/` (`imas_rt.py`, `swb_ab.py`).

| run | l_i(3) | l_i(1) | q0 | q95 | SWB/FUSE jBS peak |
|---|---|---|---|---|---|
| IDS (FUSE) | 0.781 | 0.987 | — | — | — |
| base c0c3337, psi_n, fuse | 0.789 | 1.039 | — | — | 1.037 |
| branch, psi_n, fuse | 0.7215 | 0.949 | 1.047 | 4.231 | 1.230 |
| branch, phi_n, fuse | 0.7677 | 1.008 | 1.047 | 4.144 | 1.000 |
| branch, psi_n, ida_hybrid | 0.7301 | 0.960 | 1.049 | 4.231 | 1.182 |
| branch, phi_n, ida_hybrid | 0.7691 | 1.010 | 1.047 | 4.134 | 0.995 |

- **psi_n vs phi_n differ** by +6.4 % in l_i(3) and −2 % in q95. Ip matches to 1e-6 and q0 to 0.3 %.
  - The phi_n run is the closer of the two to the IDS l_i: −1.7 % against −7.6 % for psi_n.
  - The phi_n run's SWB pedestal bootstrap matches FUSE's; the psi_n run's is 23 % above it.
  - Both runs reproduce their own input j_phi (achieved against input: rms 0.01 MA/m²).
- **Maps:** the solver's ψ_N at the dd's Φ_N nodes differs from the dd's own ψ_N by up to 0.007 in the core (ψ 0.3–0.7). At the pedestal the difference is 0.001.
- **OFT consistency A/B** (one equilibrium state, same kinetics, seed shape in ψ_N):
  - The non-uniform IMAS ψ_N grid and a uniform grid give the same SWB result (jBS peak 0.863 against 0.860).
  - A Φ-tagged SWB, with Φ labels taken from the ψ-tagged run's end state, reproduces that run exactly: peak 0.8633 against 0.8632, l_i 0.7930 against 0.7933, and every node lands back at its ψ.
  - Conclusion: the toroidal-flux path in OFT is correct. The psi/phi gap is the physical effect of which coordinate the profiles are held in.
  - Sensitivity: when the Φ labels come from a different equilibrium, a Φ-held SWB moves l_i a lot (0.79 → 0.88 with labels from the pre-SWB state).
- **ψ-mode change from the base branch:** the psi_n IMAS numbers moved (l_i(3) 0.789 → 0.7215, jBS peak ratio 1.04 → 1.23). This is the phase-1 grid-registration fix. The old code placed the non-uniform core_profiles arrays at ψ_N = i/(n−1), which is ≈ ρ on this grid. That was wrong in the core but roughly right at the pedestal. **Resolved (the flux-surface shift of OFT `tests/physics/tokamaker_torflux_motivation.py`):**
  - The final equilibria of the psi_n and phi_n runs have the same pedestal:
    - peak |P′| 0.998e6 vs 0.990e6, at ψ_N 0.971/0.972;
    - q there 4.55 vs 4.58;
    - ψ_N = 0.97 and 0.90 outboard R within 0.5 mm;
    - psi_range differs by 2.7 %.
  - The difference arises *inside* SWB. There the current profile is the generic seed plus bootstrap, not FUSE's, so the equilibrium relaxes far from the dd's. ψ_N-pinned kinetics then land at a different real-space position (and gradient) than in FUSE, while Φ_N-pinned kinetics stay close.
  - A/B from the same starting state: ψ-pinned gives a pedestal jBS peak of 0.863 MA/m²; Φ-pinned with labels consistent with that state gives 0.714. FUSE's peak is ≈ 0.70 (0.863/1.23).
  - The closure then carries SWB's j_BS into the final solve. The edge current differs by the same factor (achieved j at ψ_N 0.95: 0.584 vs 0.473 MA/m², ratio 1.23), which moves l_i by 6 %.
  - So the phi_n result is the faithful one. The seed-shape dependence of the SWB iteration makes the ψ_N-pinning error larger; see `torflux_jobs/ISSUE_swb_seed_shape.md`.
