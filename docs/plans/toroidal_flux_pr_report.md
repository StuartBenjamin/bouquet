# PR: profiles on normalised toroidal flux (`coord="phi_n"`)

**Branch:** `toroidal_flux_mapping` → base `kwargs_for_bootstrap_fortran_backend_v2` (c0c3337)
**Companion OFT branch:** `profiles_on_toroidal_normalised_flux` (7239b2f); needed only for `coord="phi_n"`
**Attachment:** `torflux_imas_results.zip`, the output of `examples/torflux_imas/torflux_imas_effect.py` on DIII-D 174956 @ 2.0 s

## Summary

A bouquet run can now hold every profile and envelope on **one** radial coordinate chosen at io: `psi_n` (normalised poloidal flux, the default and the previous behaviour) or `phi_n` (normalised toroidal flux Φ_N; `rho_tor` is accepted as input and converted as Φ_N = ρ²). Kinetic profiles are never resampled onto ψ_N. In a Φ_N run they cross into TokaMaker tagged with their coordinate, and OFT remaps them onto ψ_N at every nonlinear step. ψ_N appears in bouquet only as the address at which solver readbacks are sampled, and in a Φ_N run that address comes from the solver's own map.

The motivation is that FUSE, IMAS, IDA and transport codes hold profiles fixed in ρ_tor. A profile pinned to ψ_N moves in real space whenever the solve changes the current profile. The attached example shows the effect on a real FUSE dd: the Φ_N run reproduces FUSE's pedestal bootstrap (ratio 1.03–1.05, against 1.23 held on ψ_N) and comes closer to the dd's l_i (−2.6 % against −8.1 %).

## What changes

**`bouquet/coords.py` (new)**
- The single place for coordinate-sensitive operations:
  - profile dicts (`oft_prof`, `pp_prof`: jphi tagged `phi_n_relabel`, P′ as dP/dΦ_N tagged `phi_n`);
  - SWB grid and seed (`swb_grid_kwargs`, `swb_seed`, `seed_psi`);
  - readback addresses (`psi_at`, `psi_of`, from the solver's `get_torflux_map`);
  - hard-coded windows (`window_x`);
  - io helpers (`phi_n_from_q`, `to_run_grid`);
  - guards (`check_backend`, `check_run`).
- A `psi_n` run never sends a `coord` key or argument, so its calls match those made to an OFT without toroidal-flux support.

**io**
- **IMAS:** core_profiles nodes are relabelled with `grid.rho_tor_norm²`. A √ψ_N placeholder or a missing ρ is refused. Equilibrium `j_tor` and pressure are placed by the equilibrium's own Φ_N (its ρ, else its q). IDA-hybrid fits and IDA ω are placed by the IDA file's own q.
- **g-file:** the run grid is the g-file's `rhovn²`. IDA kinetics are placed by their own q, p-file kinetics by the g-file map. IDA files are refused if they carry no q.
- **Write-back:** `write_imas_draw` puts Φ_N archives back onto the template's Φ_N nodes, samples the draw's flux-surface geometry at the draw's own ψ_N, and writes that ψ_N to `grid.psi`. Per-draw p-files are sampled on each draw's own map and keep the source p-file's SOL.
- **Fixed components:** `FixedComponentsConfig.coord` (`"run"` default, or `"psi_n"`) maps user arrays through the source's own map.

**Solver boundary** (`TokaMaker_interface.py`, `run.py`, `sampling.py`, `utils.py`)
- All 17 `set_profiles` sites and all 6 `solve_with_bootstrap` calls go through `coords`. SWB receives its grid explicitly (`x=`), falling back to `psi_N=` on older OFT.
- Readbacks paired with run-grid arrays are sampled at the nodes' ψ_N.
- ψ-integrals use ψ_N as their abscissa.
- Windows follow `GenerationConfig.window_coord` (`"psi_n"` default, or `"native"`).
- The SWB seed coordinate follows `GenerationConfig.seed_coord`.
- σ=0 exactness is preserved: the jBS-delta/DIFF_BS cache seed is built after its anchor solve and reused by the draws.

**Config and guards**
- Coordinates are validated when the config is built.
- A Φ_N run requires an OFT with toroidal-flux support (introspected) and the internal bootstrap solve. The guard runs at prepare time and again in `generate()`.
- `x`, `psi_N` and `coord` are reserved `bootstrap_kwargs`.

**Archive and plots**
- `profile_coord` is recorded on the baseline group, each draw group and `profiles_doc`. `merge_archives` refuses shards with mixed coordinates.
- Plots label and place their axes by the archive's coordinate: overlays, flux functions, `plot_input_vs_recon`, `plot_tokamaker_comparison`, the GUI and the time series.

**Docs and examples**
- `docs/workflows.md` covers the run coordinate, the `window_coord`/`seed_coord` options, and the fact that length scales are in Φ_N units.
- `docs/archive-schema.md` covers `profile_coord`.
- Plans: `docs/plans/toroidal_flux_mapping.md` and the review record `docs/plans/toroidal_flux_review.md`.
- New `examples/torflux_imas/`.

## Compatibility

- **`psi_n` runs:** the calls sent to OFT are unchanged in form, and existing configs and archives load unchanged (`profile_coord` defaults to `"psi_n"`).
- **One intended numerical change in ψ_N runs (IMAS path only):** SWB now receives the real non-uniform core_profiles grid. Readbacks are sampled at the nodes instead of on OFT's uniform grid. Before, the arrays were placed at ψ_N = i/(n−1), which is ≈ ρ on a FUSE grid.
  - On 174956 (g-file boundary) this moves the σ=0 l_i(3) from 0.789 to 0.722.
  - g-file runs, whose grid is uniform, move only by the end-point padding.
- **OFT versions:** OFT `main` works for `psi_n`. `phi_n` needs the companion OFT branch. On an older toolkit, `phi_n` is refused before any solve rather than silently solved on ψ_N.

## Validation

All runs are SLURM jobs against the staged companion OFT.

- **Fast suite (no OFT):** 1238 passed, 2 skipped. The coordinate/config/Φ_N test files with OFT importable: 113 passed.
- **New pure tests:**
  - `test_coords.py`: helpers, tagging, guards, IDA placement;
  - `test_phi_io_imas.py`: IMAS read is a relabel, equilibrium placement, refusals, renormalisation, fixed-component coord, write-back;
  - `test_phi_cfg.py`: validation, fake-toolkit backend check, stubbed g-file baseline, analytic IDA placement;
  - `test_phi_ti.py`: P′ Jacobian consistency, lazy map lookup;
  - `test_phi_archive_plots.py`: `profile_coord` round trips, relabelling, merge refusal.
- **Solver tests (`test_phi_solver.py`, 15 passed):**
  - OFT's Φ_N map matches ∫q dψ_N to within 2.2e-4, and inverse∘forward is the identity;
  - ψ-tagged and Φ-tagged solves of the same profiles agree to ~1e-4;
  - g-file σ=0 round trip, psi_n vs phi_n: l_i +0.06 %, q0 +0.29 %, q95 0.03 %, Ip 4e-6;
  - `window_coord="psi_n"` in a Φ_N run keeps the ψ_N classifier mode.
- **IMAS, real FUSE dd** (example, attached zip):

  | | l_i(3) | q95 | SWB/FUSE pedestal j_BS |
  |---|---|---|---|
  | IDS (dd) | 0.781 | | |
  | held on ψ_N | 0.717 | 4.236 | 1.233 |
  | held on Φ_N | 0.761 | 4.153 | 1.045 |

  - **Mechanism:** inside SWB the generic seed relaxes the equilibrium (ψ_b − ψ_a −18 %). Held on ψ_N, dp/dψ = (dp/dψ_N)/(ψ_b − ψ_a) rises 21 %, and j_BS with it. Held on Φ_N, j_BS moves 2 %.
  - **Realistic seed:** seeded with the run's ohmic shape, the two coordinates agree to 2 %.
  - **Control:** Φ_N labels taken from the ψ_N end state reproduce that run exactly, which checks OFT's Φ_N path.

## Known limitations and follow-ups

- **SWB seed shape:** the fixed inductive seed `(1−ψ_N^1.5)^1.5` sets the bootstrap in the `ohmic`/`rescale` IMAS modes and the g-file split. That issue is separate from this PR. Φ_N runs are less sensitive to it (see the example).
- **GPR length scales** (`n_ls`, `t_ls`, `j_ls`, aux) keep their values but are Φ_N lengths in a Φ_N run. There is no automatic conversion.
- **Time-dependent solves:** OFT's time-dependent solvers refuse toroidal-flux profiles, and a Φ_N solve costs about 1.4× a ψ_N solve.
- **Selecting the coordinate:** it is chosen on the source. A `GenerationConfig`-level choice was deferred.
- **IMAS Φ_N round trip:** it needs the 500 MB dd, so it lives in `examples/`, not in the test suite.
