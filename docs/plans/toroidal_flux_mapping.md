# Plan: bouquet profiles on normalised toroidal flux (Φ_N) or poloidal flux (ψ_N)

- **bouquet branch:** `toroidal_flux_mapping`, created off `kwargs_for_bootstrap_fortran_backend_v2` (c0c3337). The worktree goes at `/fusion/projects/tmdb/src/.bouquet_wt/torflux`.
- **Companion OFT plan:** `/home/benjamins/.claude/plans/foamy-mixing-candle.md` (OFT branch `profiles_on_toroidal_normalised_flux`). §6 below lists what bouquet needs from OFT, with its status.
  - As of 2026-09-23, items 1–4 are committed on the OFT branch at `a206c21`, which bouquet was checked against (§6). Since then a failed toroidal-flux map update fails the solve (`gs_solve` error -9, a Python `ValueError`) instead of aborting the process (OFT `7778cb5`).
- **Findability:** step 0 copies this file to `docs/plans/toroidal_flux_mapping.md` on the new branch. Both plans should link to each other.
- **Status:** implemented on `toroidal_flux_mapping` (phases 0–5, commits `8c67437`..`bc58a50`), checked against OFT `a206c21`. Additions beyond the plan: `GenerationConfig.seed_coord` (the SWB inductive seed's coordinate), snapshot-based homotopy rollback, and ida_hybrid IDA fits placed by their own Φ_N (from the IDA file's q).

## Context
FUSE and IMAS hold kinetic profiles fixed in ρ_tor while they solve. IDA and transport codes also tabulate in toroidal flux. bouquet currently converts everything to ψ_N at io and then works in ψ_N. As a result, a profile tied to toroidal flux lands at a shifted radius once the equilibrium (q) changes, and the drift check `io/imas.py:_psi_rho_drift` exists because of that. The OFT plan lets a TokaMaker `flux_func` be declared on Φ_N and remaps it internally at every nonlinear step. bouquet should then choose **one** coordinate per run at io, `psi_n` (today's behaviour) or `phi_n`, and keep every profile and every error envelope on it all the way to the OFT call.

**User intent (governs this plan):** kinetic profiles never leave Φ_N outside OFT's internal solve. bouquet never resamples ne, Te, ni, Ti, Zeff (or the pressure built from them) onto ψ_N. ψ_N positions appear in bouquet only as sampling addresses for OFT readbacks, and they come from OFT's own map (`psi_of`, §2).

---

## 1. Inventory: every place bouquet gives OFT a profile

In the tables, TI = `bouquet/TokaMaker_interface.py`. Line numbers are from c0c3337.

### 1a. `set_profiles` (pp `linterp` + ffp `jphi-linterp`, both with `x=psi_N`), 17 sites
| Site | Function | ffp y comes from |
|---|---|---|
| TI:322-324 | `_corrective_jphi_iteration` | Newton update: target + (target − readback) |
| TI:2259-2268 | `perturb_kinetic_equilibrium`, PIN_JPHI | input j_phi |
| TI:2355-2361 | same, DIFF_BS anchor | input + Δspike (difference of two SWB readbacks) |
| TI:2478-2487 | same, SWB state anchor | input j_phi |
| TI:2769-2778 | same, recon-anchor | GPR draw + SWB spike + fixed |
| TI:2802-2803 | same, anchor fallback | `results["total_j_phi"]`, **SWB output reused as input** |
| TI:2838-2840 | same, band resample | GPR draw + spike + fixed |
| TI:3167-3183 | same, l_i loop | dicts handed to OFT `find_optimal_scale(mygs, psi_N, …)` |
| TI:3230-3246 | same, corrector | reaches TI:322 |
| TI:3966-4042 | `generate_bouquet`, jphi baseline | input (+ jphi_diff) |
| TI:4795-4805 | `generate_bouquet`, delta cache | input |
| TI:6492-6499, 6560-6565, 6786-6794 | `reconstruct_equilibrium` / `_solve_and_get_li` | fit + SWB readback |
| run.py:2203-2206 | `_forward_solve_imas_baseline.solve_jphi` | bl.j_phi / rebuilt total / closure-rescaled |
| run.py:2755-2760 | same, opt-in corrector | reaches TI:322 |
| run.py:3048-3049, 3092-3093 | `verify_sigma0_consistency` | bl.j_phi |

At every one of these sites, pp is `pchip_derivative(psi_N, p)/psi_range`. That is a **poloidal Jacobian**. The same sites call `set_targets(pax=p[0])`.

### 1b. `solve_with_bootstrap` (ne, Te, ni, Ti, Zeff arrays + inductive seed), 6 calls
TI:2311 (DIFF_BS), TI:2570 (main draw), TI:4829 (σ=0 cache, via `_swb`), TI:6375 (reconstruction), run.py:2335 (IMAS forward solve), run.py:3059 (σ=0 verify).
- **No call passes `psi_N=`.** OFT therefore assumes `linspace(0,1,n)` (bootstrap.py:841/1010).
- The seed is `create_power_flux_fun(n)["y"]`, which is built on a uniform grid.

### 1c. Other profile handoffs
- `mygs.flux_integral(psi_N, …)`: TI:1931, 2056, 3128; utils.py:3010 (`Ip_flux_integral_vs_target`); run.py:2472.
- `find_optimal_scale(mygs, psi_N, pressure, …)`: TI:3183.
- `set_targets(pax=p[0])`: this is an axis value, so it is valid on either coordinate.

### 1d. Readbacks that are fed back into bouquet arithmetic (they matter in Φ mode)
- **Uniform-grid readbacks paired index-by-index with a bouquet array.** These call `get_profiles`, `get_q` or `sauter_fc` with `npsi=len(arr), psi_pad`, which returns values on `linspace(pad,1-pad,n)`, and never interpolate onto bouquet's grid:
  - TI:343-347 (corrective loop)
  - TI:1051-1056 (`_swb_jbs_to_toroidal`, applied to every SWB output)
  - sampling.py:641/700 (l_i proxies)
  - TI:2992/3018-3020 (mixes two grids in one dict)
- **Readbacks that do interpolate or are sampled explicitly:** TI:976-986 `_achieved_jphi_fsa`, utils.py:533-614 `fsa_current_geometry` (`psi_q = clip(psi_N)`), utils.py:2969 `eq_jphi_profile`, physics.py:386 `capture_equilibrium_fsa`, and the `get_q(psi=psi_q)` q0 reads in run.py.
- **SWB outputs** (`j_BS`, `isolated_j_BS`, `total_j_phi`) come back on OFT's uniform grid and are reused as if they were on `psi_N`: TI:2328, 2687-2689, 2801, 4842, 6390; run.py:2344, 3066.

### 1e. Bugs that already exist in ψ mode (found during the inventory)
The IMAS `psi_N` is **non-uniform** (`io/imas.py:794`, from `cp.grid.psi`). As a result:
- every run.py SWB call hands OFT arrays that OFT places on the wrong ψ_N;
- every uniform-readback pairing in 1d mis-registers by up to the grid non-uniformity.

On the g-file path the error is only the `psi_pad` end offset. The fix in §3 corrects these as a side effect. The fix goes in its **own commit** (phase 1) so that golden-test movement can be attributed to it.

### 1f. Elongation, and the rest
No bouquet site sends OFT anything else that depends on the coordinate: no `set_kinetic_profiles`, `create_prof_file`, `set_resistivity` or `set_boot_ops` calls.

---

## 2. Design decision: one run-wide coordinate tag plus a small helper module (not per-array dicts)

**Recommendation: a hybrid.** Keep the arrays as plain numpy on one grid. Carry a single run-wide `coord` tag. Route every coordinate-sensitive operation through a new module `bouquet/coords.py`. Then edit each site to call it.

Why not `{'x','y','coord'}` for every array?
- bouquet does heavy array arithmetic on these profiles: sums of splits, GPR draws, Newton updates, masks.
- Wrapping every array would touch about 200 lines and every archive field, and would add no information. With one coordinate per run, the tag on each array is always identical.
- Dicts do appear, but only at the OFT boundary, where OFT wants them anyway.

Why not pure case-by-case edits?
- The same four operations (build profile dict, pressure derivative, SWB grid kwargs, readback sampling points) recur at about 40 sites.
- Inlining them would scatter `if coord == 'phi_n'` branches through TI and run.py.

### Carrying the tag
- `ImasSource.coord` and `ReconstructionSource.coord`: `"psi_n"` (default) or `"phi_n"`. This is the io choice.
  - `"rho_tor"` is also accepted as an input spelling and converted exactly to Φ_N = ρ² at read. The rest of the run is then `phi_n`.
- `Baseline.coord`: set by the reader. The fields keep their names (`psi_N`, `psi_N_kinetic`) and are documented as "the run grid in `Baseline.coord`". A rename would cascade through schema, archive and plotting for no gain.
- **Archive:** add a root attribute `profile_coord`. Datasets keep their names. Readers and plotting read the attribute; when it is missing, it defaults to `psi_n`.
- `FixedComponentsConfig.psi_N`, `UncertaintyConfig.sigma_profiles`, and `aux_*` are interpreted on the run coordinate.

### `bouquet/coords.py` (new, about 120 lines)
```python
PSI, PHI = "psi_n", "phi_n"

def oft_prof(kind, x, y, coord):          # {'type': kind, 'x': x, 'y': y[, 'coord': ...]}
def pp_prof(mygs, x, p, coord):           # psi: dp/dψ_N / psi_range, coord omitted
                                          # phi: dp/dΦ_N, coord='phi_n' (OFT applies J=q/Q)
def swb_grid_kwargs(x, coord):            # {'psi_N': x} (+ 'coord': coord in phi mode)
def psi_of(mygs, x, coord, psi_pad):      # ψ_N sample points for grid x on the *current* equilibrium
                                          # psi: clip(x); phi: clip(mygs.get_torflux_map(x, inverse=True)[0])
def window_x(mygs, x, coord, window_coord):  # abscissa for hard-coded windows (§3.6):
                                          # 'psi_n' (default) -> psi_of(x); 'native' -> x
def check_backend(coord):                 # phi mode: require 'coord' in signature(TokaMaker_equilibrium.solve_bootstrap)
                                          # and hasattr(TokaMaker, 'get_torflux_map');
                                          # older OFT silently IGNORES a 'coord' key in profile dicts, so never rely on that
```
- **Key rule:** when `coord == "psi_n"`, **no `coord` key or kwarg is ever sent**. The call is byte-identical to today, so OFT main keeps working. This follows the same approach as the `bootstrap_kwargs` validation.
- **Choice of per-profile mode in Φ runs**, using the exact OFT tag strings:
  - `jphi-linterp` ffp → `'coord': 'phi_n_relabel'`. OFT stops with an error if a jphi profile uses `'phi_n'`.
  - pp → `'coord': 'phi_n'`, with y = dp/dΦ_N. The overall scale does not matter, because `pax` sets it, so no `psi_range` factor is needed.
  - Kinetics (ne, Te, ni, Ti, Zeff) are **not tagged by bouquet**. Pass `coord='phi_n'` to `solve_with_bootstrap`, and OFT tags them `phi_n_relabel` internally. If a kinetic dict is ever sent directly, it must be `phi_n_relabel`; OFT stops with an error on `phi_n`.
- **Where `psi_of` comes from:** always OFT's own map, `mygs.get_torflux_map(x, inverse=True)` (§6.3). It is the map the last solve used, so readback addresses are consistent with the solve by construction. No `get_q`-integral fallback is kept: `check_backend` refuses Φ mode on an OFT without the map.
  - After an SWB call, the result's `'psi_n'` holds the same positions (index-aligned with `x`); either source may be used.
  - The map exists only after a solve with Φ-tagged profiles. Every Φ-mode readback in bouquet follows such a solve; `psi_of` raises otherwise.
  - Readback APIs (`get_q`, `get_profiles`, `sauter_fc`, `flux_integral`) keep taking ψ_N; no `coord=` on them is needed.
- **Python-solver guard:** OFT raises if `solve_with_bootstrap(use_python_solve=True)` gets `coord≠'psi_n'`. bouquet must not select the Python solver in Φ mode (validate at config time).

---

## 3. Site-by-site changes (pattern; representative sites)
1. **Profile dicts (§1a):** `{"type":"linterp","x":psi_N,"y":pp}` becomes `coords.pp_prof(...)`, and the ffp dict becomes `coords.oft_prof("jphi-linterp", ...)`. This mechanical change covers all 17 sites.
2. **SWB calls (§1b):** add `**coords.swb_grid_kwargs(psi_N, coord)` to all 6 calls. Add `psi_N` and `coord` to `GenerationConfig._RESERVED` (config.py:970). The seed `create_power_flux_fun(n)["y"]` becomes a seed evaluated on `psi_N` (`(1-x**1.5)**1.5`), so it is no longer implicitly uniform.
3. **Readbacks (§1d):** replace `get_profiles/get_q/sauter_fc(npsi=len(arr), psi_pad)` with `(psi=coords.psi_of(mygs, x, coord, pad))`, which gives values that line up with bouquet's grid by construction. This applies to TI:343, TI:1051 (`_swb_jbs_to_toroidal`, which now takes `x`), sampling.py:641/700, and TI:2992/3018.
4. **SWB outputs:** OFT already returns them on the input `ffp` nodes, flipped to axis-first. So once `psi_N=x` is passed (§3.2), TI:2801/2328/4842/6390 and run.py:2344/3066 are on `x`. In Φ mode, `results['psi_n']` holds the matching ψ_N positions (returned by `solve_with_bootstrap` on the OFT branch).
5. **Integrals** (`flux_integral`, `find_optimal_scale`, `fsa_current_geometry`, `Ip_fsa_weights`): evaluate on `psi_of(x)`. The integration variable in `trapezoid(…, psi_N)` (utils.py:648) and the Jacobian `|dψ/dψ_N|` (utils.py:586) become the *ψ positions*, not `x`. Values do not change under relabelling, so only the abscissa moves.
6. **Hard-coded locations** (`psi_N > 0.9` TI:291, classify 0.85/0.5 TI:404/530, shelf/bridge/core windows TI:861-900, `_edge_mask` TI:6854, structured basis centres utils.py:1132, q95 at 0.95):
   - New option `window_coord ∈ {'psi_n' (default), 'native'}` (config, alongside the run `coord`). The thresholds keep their numbers; `coords.window_x` gives the abscissa they are compared against: `psi_of(x)` for `'psi_n'`, or `x` itself for `'native'` (thresholds then mean Φ_N in Φ runs).
   - In ψ runs both options are identical, so ψ runs stay bitwise unchanged.
   - q95 is a definition in ψ_N and always uses `psi_of(x)`, whatever `window_coord` is.
   - io-time gates in `io/imas.py` and `io/ida.py` run on the source's own ψ grid before the coordinate switch, and are left unchanged.
7. **io readers:**
   - IMAS (`io/imas.py:740`): in `phi_n` mode, build x = `grid.rho_tor_norm**2` directly from the dd. This is the natural, native path, with no q needed. Equilibrium-side profiles (`p_equilibrium`, `j_tor`) are placed through the dd's own ψ_N↔ρ pair.
   - g-file + IDA/p-file (`baseline.py:887`): the inputs are on ψ_N, so map them with the g-file's `rhovn`, once, at io. That map is the source's own equilibrium. IDA sigmas are moved by the same map.
   - `_psi_rho_drift` stays as a diagnostic.
8. **Envelopes:**
   - `resolve_uncertainty` is unchanged apart from grid provenance, because everything is already on `psi_N_kinetic`, which is now the run grid.
   - GPR `cdist` distances become Φ_N distances. **`n_ls`/`t_ls`/`j_ls`/aux length scales then mean Φ_N units.** Defaults stay the same, and the docs say so.
   - `synthetic_ida_sigma` and `new_uncertainty_profiles` are unit-agnostic helpers, so only their docstrings change.
9. **Plotting:** the x label comes from the `profile_coord` attribute. The existing `_resolve_x_coord` (plotting.py:1539) gains an identity branch for the case where the run is already on Φ.

## 4. Phases (each its own commit)
0. Create the branch and worktree. Copy this plan into `docs/plans/`.
1. **ψ-mode grid-registration fix (§1e):** `coords.py` with only the ψ path, readbacks via `psi_of`, and `psi_N=` on SWB calls. Rerun the golden tests. Record and explain any movement (expected on the IMAS path).
2. `coord` plumbing: source → Baseline → archive attribute. Add `check_backend`. ψ runs must stay bitwise unchanged against phase 1.
3. Φ branches in `coords.py` (pp, profile dicts, `psi_of` via `get_torflux_map`), plus the §3.5 changes and the §3.6 `window_coord` option.
4. io: IMAS `rho_tor_norm` path, then the g-file `rhovn` path.
5. Docs: `workflows.md` (config rows, length-scale units), `docs/flowchart/graph.json` nodes.

Phases 3 and 4 need OFT (§6) for end-to-end runs. The pure-numpy parts of `coords.py` can be unit-tested before then.

## 5. Open points (my defaults; say if you want otherwise)
- **Fix §1e in ψ mode now:** I propose yes, as a separate commit. It changes IMAS-path numerics.
- **Coord on the source rather than on `GenerationConfig`:** chosen because you said "set during the io step".
  ^USER SAYS: we should be able to set this in generation config. This will require case-by-case work, so don't focus on this problem for now.
- **Length scales:** same numbers, Φ_N units. There are no automatic conversions.
- **Window coordinate (decided):** `window_coord` option, default `'psi_n'` (§3.6).

## 6. What bouquet needs from OFT (mirrored in `foamy-mixing-candle.md` §8, with status)
1. **Available.** `set_profiles` / `create_prof_file` accept `profile_dict['coord'] ∈ {'psi_n','phi_n','phi_n_relabel'}`.
   - `psi_n` is the default and writes an unchanged file.
   - `jphi-linterp` supports `phi_n_relabel`.
   - pp uses `phi_n`, where y is the derivative with respect to Φ_N and OFT applies J = q/Q internally.
2. **Available.** `solve_with_bootstrap(..., psi_N=x, coord='phi_n')` already takes `psi_N=`, and passes `coord` on to `solve_bootstrap(coord=...)`. (2026-09-24: the grid keyword is now `x`, with `coord` an explicit argument, so no argument named ψ_N carries Φ_N. bouquet passes whichever of `x` / `psi_N` the toolkit takes, via `coords._swb_grid_arg`.) There:
   - P′ is formed on `x` as dP/dΦ_N;
   - kinetics and jphi are relabelled (stored on their Φ_N nodes);
   - outputs are index-aligned with the input `x`, and the result includes `'psi_n'` (node ψ_N positions).
   - `use_python_solve=True` with `coord≠'psi_n'` raises.
3. **Available.** `mygs.get_torflux_map(x, inverse=False)` → (Φ_N(x), dΦ_N/dψ_N); with `inverse=True` → (ψ_N(x), dΦ_N/dψ_N). User convention (0 = axis). It exposes the last solve's map and raises if there is none. This is the `psi_of` backend. A `coord=` argument on `get_profiles`/`get_q`/`sauter_fc`/`compute_flux_integral` is **not planned**, because readbacks are coordinate-free values at ψ positions.
4. **No OFT change needed.** Call `find_optimal_scale(mygs, psi_of(x), …)`. Its `psi_N` argument is used only to sample readbacks. It overwrites only `ffp_prof['type']` and `['y']`, so a `'coord'` key on the caller's dicts survives. Note that the dict is modified in place.
5. **Constraints to respect in bouquet:**
   - OFT's time-dependent solvers stop with an error on toroidal-flux profiles.
   - `tokamaker_set_psi` leaves the map stale until the next solve.
   - A Φ-mode solve costs about 1.4× a ψ-mode solve (0.47 s against 0.34 s for ITER with 4 threads), using OFT's cut-cell q backend (tracing cost about 3×).

## 7. Verification
- **Unit tests** (`tests/test_coords.py`, no OFT needed):
  - `psi_n` produces dicts with no coord key and identical pp;
  - `window_x` returns `x` unchanged in ψ runs for both options;
  - `rho_tor → phi_n` conversion;
  - `_RESERVED` refuses `psi_N`/`coord`.
- **ψ regression:** run the full non-solver suite (currently 1146 passed / 2 skipped) and the golden tests (`test_golden_bouquet`, `test_seeded_reproducibility`).
  - Phase 2 must be bitwise against phase 1.
  - Phase 1 movement is documented.
- **Backwards compatibility:** with OFT `origin/main` on the path, ψ runs work unchanged and `coord="phi_n"` raises at config time (the same introspection pattern as `bootstrap_kwargs`).
- **`psi_of`** (needs the OFT branch): `get_torflux_map` against `GEQDSKEquilibrium.rhovn`-style `get_q` integration on a solved Φ case (≲1e-4), and inverse∘forward = identity.
- **Φ round trip** (needs the OFT branch):
  1. Take a ψ-mode σ=0 run on the D3D-like example.
  2. Convert its inputs to Φ_N with the solved equilibrium's map and rerun in `phi_n`.
  3. Ip, l_i, q0, q95 and j_phi (mapped back) should match to within solver tolerance.
  4. Do this for both the IMAS example and the g-file example.
- Python: `/fusion/projects/tmdb/scripts/eq_stab/venvs/genvenv1/.venv/bin/python -m pytest tests/ -m "not solver"`, then the solver tests with the OFT branch installed.
