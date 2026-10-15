# Bouquet examples

`bouquet` generates a *bouquet* — a family of perturbed Grad–Shafranov
equilibria — from a single baseline, so you can propagate kinetic/current
uncertainty into the equilibrium reconstruction. The notebooks run on the
fully synthetic fixtures the test suite is built on
([`tests/data`](../tests/data/README.md), specified in
[`FIXTURE_SPEC.md`](../tests/data/FIXTURE_SPEC.md)), so nothing external is
needed.

## Prerequisites

The package needs the **OpenFUSION Toolkit** (TokaMaker) importable — either
call `bq.add_oft_to_path()` (resolves `OFT_PYTHONPATH`, known install
locations, or a sibling checkout; every notebook does) or set the path
yourself:

```bash
export OFT_PYTHONPATH=/path/to/OpenFUSIONToolkit/build_release/python
```

`import bouquet` itself works without OFT (all TokaMaker imports are lazy); a
missing path surfaces later, at `setup_solver()` / `reconstruct()`, as
`ModuleNotFoundError: OpenFUSIONToolkit`. The mesh is found by
`bq.find_mesh()` (`BOUQUET_MESH`, else the one shipped in `examples/`).

## The notebooks

| Notebook | Inputs (`tests/data`) | What it shows |
|---|---|---|
| [`D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb`](D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb) | `g_synthetic.geqdsk` + `diiid_profs_synthetic.cdf` (H-mode); second case `g_synthetic_lmode.geqdsk` + `ida_synthetic_lmode.cdf` (L-mode) | g-file + IDA-layout fit reconstruction, the bistability check, the σ=0 identity, traces, the in-spec filter, a second case on the same solver (`set_case`), reading the archive |
| [`D3D-like/bouquet_D3Dlike_omas_example.ipynb`](D3D-like/bouquet_D3Dlike_omas_example.ipynb) | `dd_synthetic.json.gz` at 1.2 s + `diiid_profs_synthetic.cdf` | IMAS/OMAS source on the IDA-hybrid route, perturbed IDS output per draw |
| [`D3D-like/bouquet_D3Dlike_parallel_IMAS_example.ipynb`](D3D-like/bouquet_D3Dlike_parallel_IMAS_example.ipynb) | `dd_synthetic.json.gz`, all five slices (0.7, 1.1, 1.2, 1.4, 3.0 s) | `parallel_cases`: many bouquets on one node, merged into one archive; the SLURM route |
| [`D3D-like/bouquet_D3Dlike_systematics.ipynb`](D3D-like/bouquet_D3Dlike_systematics.ipynb) | the 1.2 s pair `g_synthetic.geqdsk` + `diiid_profs_synthetic.cdf` | σ=0 identity, pressure-only and production draws; where the baseline sits among the draws |
| [`COCOS_Bt_Ip/`](COCOS_Bt_Ip/) | their own g-file | COCOS conventions, the B_t / I_p signs, saving |

**New to bouquet?** Start with the **geqdsk** notebook.

Every notebook:

- draws **until 24 draws are in spec** (`generation.n_inspec_target = 24`, the
  default for examples; use 30 for 5/95 bands, 50 for widths quoted in a
  paper);
- ends with an **Error bars** section: `bq.bands_summary(archive, key)` prints
  the median, the 16–84 % band, the full range and n over the in-spec draws for
  β_N, l_i(3), Λ = β_p + l_i(3)/2 and the gradients dp/dψ_N and dj_φ/dψ_N at
  each draw's own q = 2 surface, plus one band figure with n stated
  ([workflows.md](../docs/workflows.md#error-bars-from-an-archive)).

The minimal pattern:

```python
b = bq.Bouquet.from_geqdsk("g_synthetic.geqdsk", profiles="diiid_profs_synthetic.cdf",
                           mesh=bq.find_mesh(), n_draws=24, header="my_run")
b.generation.n_inspec_target = 24
b.reconstruct(); b.generate(); b.filter(); b.export()
print(bq.bands_summary("my_run.h5", 0).table())
```

A p-file (`profiles="....peqdsk"`) takes the same route as the IDA-layout fit;
the synthetic set has no p-file, so the notebooks show the IDA route only.

## In-spec counts

A draw is **in spec** when it passes the magnetics: the coil χ² test and the
boundary-deviation cut (`run.filter()` prints the counts and names the rule;
see [the in-spec rule](../docs/workflows.md#the-in-spec-rule)). The l_i band
and a β_p band are opt-in for a source that states their uncertainty. β_p and
Λ are reported per draw, never filtered on.

Regenerating a notebook is **not bit-reproducible** run to run (the solver's
own path, not the seeded draws), so the in-spec yield and the band edges move
slightly between regenerations; the notebooks print the counts rather than
quoting them in the text.

## Outputs

The HDF5 archives (`*.h5`) are written next to the notebooks and gitignored.
Rendered outputs are saved in the committed notebooks, so you can preview the
results on GitHub without running anything.

Rendered outputs are saved in the committed notebooks, so you can preview results
on GitHub without running anything.

## Fixtures (all synthetic, non-proprietary)

- `D3Dlike_Hmode_baseline.geqdsk` / `.peqdsk` — H-mode equilibrium + kinetic profiles
- `D3Dlike_baseline_omas.json` — synthetic FUSE OMAS data dictionary (3-slice L→H)
- `DIIID_mesh.h5` — TokaMaker mesh
- `legacy/TkMkr_D3Dlike_Hmode_BSamp=*.geqdsk` / `.peqdsk` — the earlier
  TokaMaker-generated equilibria the legacy notebook and the
  `COCOS_Bt_Ip/` notebooks read

These are the older D3D-like example set, kept because a number of solver
tests, the `COCOS_Bt_Ip/` notebooks and (until their regeneration) the g-file
regression goldens still read them (the other notebooks run on `tests/data/`); their generator scripts and recipe were retired
(they are in the git history). The golden recipe (`tests/golden/`) runs the
newer synthetic fixtures in `tests/data/` (`g_synthetic.geqdsk` +
`diiid_profs_synthetic.cdf`, and `dd_synthetic.json.gz` with the same profile
file; see [`tests/data/README.md`](../tests/data/README.md)).

Real machine data (`*.cdf`, proprietary g-files, FUSE `dd_sim.json`) is
gitignored and never committed.

`reconstruction_engine="legacy"` / `solve_method="legacy"` are deprecated in
1.4.0 and removed in 1.5 (#90); the notebooks run the unified engine (the
default).
