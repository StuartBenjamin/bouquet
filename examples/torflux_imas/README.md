# Profiles held on Φ_N vs ψ_N: an IMAS forward solve

`torflux_imas_effect.py` compares two ways of holding a FUSE dd's profiles during bouquet's σ=0 IMAS forward solve: on normalised poloidal flux (`coord="psi_n"`) or on normalised toroidal flux (`coord="phi_n"`). It then isolates the cause of the difference with a bootstrap (SWB) A/B run from one equilibrium state.

This is the IMAS counterpart of OFT's `src/tests/physics/tokamaker_torflux_motivation.py`. That script shows that ψ_N surfaces move when the current profile changes, while Φ_N stays tied to the geometry.

## Running it

```
cp /path/to/dd_sim.json data/            # the FUSE dd (not in the repo)
python torflux_imas_effect.py --zip      # -> out/ and out.zip
```

Requirements:
- OpenFUSIONToolkit with toroidal-flux profile support, found through `OFT_PYTHONPATH`.
- About 3 GB of memory per worker (two workers run in parallel) and about 5 minutes.

Options:
- `--dd`: path to the dd, default `data/dd_sim.json`.
- `--time`: slice time [s], default 2.0.
- `--gfile`: an LCFS g-file to use in place of the dd boundary.
- `--mesh`: default `examples/D3D-like/DIIID_mesh.h5`.
- `--saddle`: JSON list of X-point targets.
- `--out`: output directory.
- `--replot`: redraw the figures from an existing `--out` without solving again.

The defaults match the operational DIII-D 174956 @ 2.0 s setup: FUSE ohmic current plus the SWB bootstrap (`jBS_baseline_mode="ohmic"`), with the structured `li_soft_onesided` closure.

Outputs in `<out>/`:
- `fig1_forward_solve.png`: the σ=0 solve held on ψ_N and on Φ_N — achieved j_φ, q, and the bootstrap it used, against the dd.
- `fig2_swb_ab.png`: the SWB A/B from one state. The top row uses the generic inductive seed, the bottom row seeds with the run's own ohmic shape.
- `summary.md` / `summary.json`: the numbers.
- `psi_n/`, `phi_n/`: each worker's `result.json` and solver log.

## What it shows (174956 @ 2.0 s, dd boundary; g-file boundary in brackets)

| run | l_i(3) | q95 | SWB/FUSE pedestal j_BS |
|---|---|---|---|
| IDS (dd) | 0.781 | | |
| held on ψ_N | 0.717 [0.722] | 4.236 [4.231] | 1.233 [1.227] |
| held on Φ_N | 0.761 [0.768] | 4.153 [4.144] | 1.045 [1.029] |

The Φ_N run reproduces FUSE's bootstrap and comes closer to the dd's l_i.

The gap arises inside `solve_with_bootstrap`. There the current is SWB's inductive seed plus bootstrap, so the equilibrium relaxes away from the dd's. With the generic seed `(1-ψ_N^1.5)^1.5`, l_i falls from ≈0.77 to 0.60 and core q rises from 1.0 to 1.4. The bootstrap follows the absolute pressure gradient dp/dψ, and the two coordinates respond to that change differently:

- **Held on ψ_N:** dp/dψ = (dp/dψ_N)/(ψ_b − ψ_a). The poloidal-flux range is a global quantity, set by the whole current profile. The generic seed shrinks it by 18 % (0.316 → 0.259 Wb/rad), so the pedestal dp/dψ, and with it j_BS, rise by 21 % (0.714 → 0.863 MA/m²). In real space the ψ_N = 0.97 surface moves 1.3 mm closer to the LCFS, about 15 % of its 8–9 mm stand-off. The same pedestal in ψ_N is steeper in R.
- **Held on Φ_N:** dp/dψ = (dp/dΦ_N)(dΦ_N/dψ), with dΦ_N/dψ = q/Φ̄. That depends on local q and on the toroidal flux, which is fixed by B_T times the area. The same change of seed moves the pedestal j_BS by only 2 % (0.729 → 0.714 MA/m²).
- **Control:** Φ_N labels taken from the ψ_N run's own end state reproduce it exactly (0.8633 vs 0.8632 MA/m², same l_i). So OFT's toroidal-flux path is consistent, and the difference comes from the coordinate the profiles are held on.
- **Realistic seed:** seeded with the run's ohmic shape, the two coordinates agree to 2 % (0.714 vs 0.729) and both sit near FUSE (0.665).

In practice, a Φ_N-held run is insensitive to the intermediate current profile inside SWB. A ψ_N-held run inherits the seed's error through ψ_b − ψ_a. A better seed shrinks the gap in either coordinate (see the SWB seed-shape issue).
