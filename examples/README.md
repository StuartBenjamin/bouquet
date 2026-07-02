# Bouquet examples

`bouquet` generates a *bouquet* — a family of perturbed Grad–Shafranov
equilibria — from a single baseline, so you can propagate kinetic/current
uncertainty into the equilibrium reconstruction. These examples get you running
on fully synthetic, non-proprietary DIII-D-like data.

## Prerequisites

The package needs the **OpenFUSION Toolkit** (TokaMaker) on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/OpenFUSIONToolkit/build_release/python:$PYTHONPATH
```

`import bouquet` itself works without OFT (all TokaMaker imports are lazy); a
missing path surfaces later, at `setup_solver()` / `reconstruct()`, as
`ModuleNotFoundError: OpenFUSIONToolkit`. Set the path above before running
the solver sections.

## Where to start

Both notebooks are **layered**: a minimal "set inputs, press run" example at the
top, then sections that each add one axis of complexity (inputs → filtering →
switchboard extras → … → tuning). Run section 1, then stop wherever you have
what you need. The two share a skeleton — the only real difference is the
**baseline source**, so learning one teaches the other.

| Notebook | Baseline source | What it shows | Full run |
|---|---|---|---|
| [`D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb`](D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb) | EFIT **g-file** + Osborne **p-file** (reconstructed) | reconstruction fidelity report, perturbed LCFS/profile overlays, manual switchboard extras | ~7–10 min |
| [`D3D-like/bouquet_D3Dlike_omas_example.ipynb`](D3D-like/bouquet_D3Dlike_omas_example.ipynb) | FUSE **IMAS/OMAS** IDS (pre-separated) | pre-separated currents, switchboard extras from the IDS, 3-slice L→H time evolution | ~10 min/slice |

**New to bouquet?** Start with the **geqdsk** notebook — it's fully
self-contained (reconstructs from the committed fixtures) and the reconstruction
fidelity summary is a good first confidence check.

The MVP cell is nearly identical across the two:

```python
# geqdsk
run = bq.Bouquet.from_geqdsk('…baseline.geqdsk', profiles='…baseline.peqdsk',
                             mesh='DIIID_mesh.h5', n_draws=5, header='…')
run.reconstruct()                 # reconstruct + fidelity summary
run.generate(); run.filter(); run.export()

# IMAS
run = bq.Bouquet.from_imas('…omas.json', mesh='DIIID_mesh.h5', time=2.3043,
                           n_draws=5, header='…')
run.run()                         # setup → baseline → generate → filter → export
```

## Regenerating vs. loading

Each notebook has a `REGENERATE` flag. The HDF5 outputs (`*.h5`) are gitignored,
so a fresh checkout regenerates them:

- **geqdsk** is self-contained — `REGENERATE=True` reconstructs and solves from
  the committed g-file/p-file.
- **IMAS** single slices regenerate from the committed OMAS json; the 3-slice
  **time series** is produced by `D3D-like/_run_omas_timeseries.py` (one
  subprocess per slice, because `OFT_env` is a per-process singleton).

Rendered outputs are saved in the committed notebooks, so you can preview results
on GitHub without running anything.

## Advanced

- [`D3D-like/bouquet_D3Dlike_systematics.ipynb`](D3D-like/bouquet_D3Dlike_systematics.ipynb)
  — a bias/systematics check that runs the bouquet three ways (j_phi pinned with
  σ=0, pinned IDA-σ, full production) to expose any inherent offset.

## Fixtures (all synthetic, non-proprietary)

- `D3Dlike_Hmode_baseline.geqdsk` / `.peqdsk` — H-mode equilibrium + kinetic profiles
- `D3Dlike_baseline_omas.json` — synthetic FUSE OMAS data dictionary (3-slice L→H)
- `DIIID_mesh.h5` — TokaMaker mesh

Real machine data (`*.cdf`, proprietary g-files, FUSE `dd_sim.json`) is
gitignored and never committed.
