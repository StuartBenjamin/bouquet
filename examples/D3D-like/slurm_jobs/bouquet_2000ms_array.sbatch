#!/bin/bash
#SBATCH --job-name=bouquet_2000ms
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
# compute-node environment (adjust for your cluster), e.g.:
# export OFT_PYTHONPATH=/path/to/OpenFUSIONToolkit/build_release/python
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m bouquet.parallel shard bouquet_2000ms_bundle.pkl $SLURM_ARRAY_TASK_ID
