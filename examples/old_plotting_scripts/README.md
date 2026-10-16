# Old plotting scripts (from tmdb_local_scripts)

Copied as-is from `StuartBenjamin/tmdb_local_scripts` (branch `laszlo_v0_branch`, commit `6dcb7f6`) for
adoption into bouquet. They target the API on `parallel_omega` (`plot_geqdsk_bouquet(..., ranges=,
return_data_only=)`, `bouquet.parallel`).

- `gifplot.py`: animates a bouquet run over time. Pass 1 computes shared axis ranges across all
  `worker_*/{HEADER}_idx*.h5` files via `plot_geqdsk_bouquet(return_data_only=True)`; pass 2 renders one SVG
  per time slice with those ranges; `_gif()` stamps `time = ...ms` on each frame and stitches a GIF
  (`cairosvg` + `imageio`). Output: `<run>/bouquet_plots_gif/`.
  Usage: `uv run gifplot.py 174956 250 bigT250` or `uv run gifplot.py --dirs 174956bigT250 174990bigT250`.
- `runner.py`, `runner_new.py`: parallel bouquet runners for the Laszlo_v0 DIII-D shots; after a run they call
  `plot_bouquet(..., mode='all')` and `plot_geqdsk_bouquet` to write static figures to `<run>/bouquet_plots/`.

Paths (`TMDB_v0_dir`, OFT install, shot lists) are hardcoded to the omega/cscratch setup. The
`/cscratch/benjamins/TMDB_v0` data now lives at
`/fusion/projects/disruption_warning/data/tmdb/tars_output/`. `gifplot.py` also needs `imageio` and `cairosvg`.
