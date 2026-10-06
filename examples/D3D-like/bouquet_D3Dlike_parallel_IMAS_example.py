#!/usr/bin/env python3
"""
Parallel bouquet over IMAS time slices
=======================================

Script version of ``bouquet_D3Dlike_parallel_IMAS_example.ipynb``, transcribed
line for line (matplotlib figure saved to disk instead of shown inline).

    python bouquet_D3Dlike_parallel_IMAS_example.py
"""

import os, copy, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bouquet as bq

OMAS = 'D3Dlike_baseline_omas.json'   # FUSE-style IMAS/OMAS data dictionary
MESH = 'DIIID_mesh.h5'                # TokaMaker finite-element mesh

# the 3 L->H slices, with the fake step labels [ms] used as the scan key / time axis
TIMES = [2.10, 2.20, 2.3043]
T2MS  = {2.10: 1600, 2.20: 1800, 2.3043: 2000}
SEED  = 12345
print('bouquet', bq.__version__, '| cores', os.cpu_count())

PARALLEL = dict(
    backend            = "slrum",               # "laptop"/"pc" -> run now ; "slurm" -> emit job array
    n_workers          = min(4, os.cpu_count()), # 4 is a RAM-safe default for personal machines
    threads_per_worker = 1,                      # 1 thread/worker -> bit-reproducible + no oversubscription
)

# oversubscription guard: workers are thread-pinned, so this is a hard core budget
assert PARALLEL["n_workers"] * PARALLEL["threads_per_worker"] <= os.cpu_count(), \
    "n_workers x threads_per_worker exceeds the physical cores"
print(PARALLEL, "| cores", os.cpu_count())

N_EQUILS = 8   # total draws per slice (split across workers)

template = bq.Bouquet.from_imas(OMAS, mesh=MESH, time=TIMES[-1],
                                n_draws=N_EQUILS, header='_tmpl')
# uncertainty envelope (per-channel fractional sigma)
template.uncertainty.ne_scalar_sigma = 0.05
template.uncertainty.te_scalar_sigma = 0.05
template.uncertainty.ti_scalar_sigma = 0.10
template.uncertainty.jphi_scalar_sigma = 0.05
template.uncertainty.zeff_scalar_sigma = 0.05   # enable the Z_eff channel -> n_i derived per draw
template.solver.nthreads = PARALLEL["threads_per_worker"]
print(f"per-slice ensemble: {template.generation.n_equils} draws  "
      f"-> {min(PARALLEL['n_workers'], N_EQUILS)} workers")

t_ref = TIMES[-1]; ms_ref = T2MS[t_ref]
ser_cfg = copy.deepcopy(template.config)              # same pattern as the sweep below
ser_cfg.source.time         = t_ref
ser_cfg.generation.scan_key = ms_ref
ser_cfg.generation.n_equils = 2                       # a couple draws is enough to time
ser_cfg.output_header       = 'D3Dlike_par_serial_ref'
ser = bq.Bouquet(ser_cfg)

t0 = time.perf_counter()
ser.setup_solver(); ser.prepare_baseline(); ser.generate()
dt_serial = time.perf_counter() - t0
n_ser = len(bq.list_equilibrium_indices('D3Dlike_par_serial_ref', scan_key=ms_ref))
per_draw = dt_serial / max(n_ser, 1)
print(f"serial: {n_ser} draws in {dt_serial:.0f}s  (~{per_draw:.0f}s/draw, incl. one baseline solve)")
print(f"=> a {N_EQUILS}-draw slice serially ~ {per_draw*N_EQUILS/60:.0f} min; "
      f"with {min(4, os.cpu_count())} workers, ~{per_draw*N_EQUILS/min(4,os.cpu_count())/60:.0f} min/slice (minus setup)")

TS = {}            # {ms: header} for the time-series plot
t0 = time.perf_counter()
for t in TIMES:
    ms = T2MS[t]
    cfg = copy.deepcopy(template.config)
    cfg.source.time        = t
    cfg.generation.scan_key = ms
    cfg.output_header       = f'D3Dlike_par_ts_{ms}ms'

    s = bq.parallel_generate(cfg,
                             n_workers          = PARALLEL["n_workers"],
                             threads_per_worker = PARALLEL["threads_per_worker"],
                             seed               = SEED,
                             backend            = "laptop")
    TS[ms] = cfg.output_header
    print(f"  slice {ms}ms: {s['n_draws']} draws / {s['n_workers']} workers  "
          f"baseline li={s['li_target']:.4f}  Ip={s['Ip_target']/1e6:.3f} MA")

print(f"\ntotal wall time: {time.perf_counter() - t0:.1f}s  "
      f"({PARALLEL['n_workers']} workers x {PARALLEL['threads_per_worker']} thread)")

for ms, h in TS.items():
    cs, _ = bq.filter_coil_currents(h, scan_key=ms, plot=False)
    sel   = bq.select_indices(h, scan_key=ms, selection='selected')
    idx   = bq.list_equilibrium_indices(h, scan_key=ms)
    print(f"{ms}ms: draws {idx}  | in-spec {cs['n_pass']}/{cs['n_total']}  "
          f"| selected {len(sel)}")

bq.plot_bouquet_timeseries(TS, time_label='step time [ms]')
plt.savefig('bouquet_D3Dlike_parallel_IMAS_example_timeseries.png', dpi=150)

cfg = copy.deepcopy(template.config)
cfg.source.time         = 2.3043
cfg.generation.scan_key = 2000
cfg.generation.n_equils = 256           # a big ensemble worth a cluster
cfg.output_header       = 'D3Dlike_par_ts_2000ms'

paths = bq.parallel_generate(cfg, n_workers=32, threads_per_worker=4, seed=SEED,
                             backend="slurm",
                             slurm=dict(out_dir='slurm_jobs', job_name='bouquet_2000ms',
                                        partition=None, time_limit='02:00:00',
                                        mem_per_task='16G', python='python'))
print("wrote:", *paths.values(), sep="\n  ")
print("\n--- submit.sh ---")
print(open(paths['submit']).read())
print("--- array.sbatch ---")
print(open(paths['array']).read())
