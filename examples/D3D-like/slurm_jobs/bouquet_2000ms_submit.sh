#!/bin/bash
# chain the array + merge with an afterok dependency
aid=$(sbatch --parsable bouquet_2000ms_array.sbatch)
sbatch --dependency=afterok:$aid bouquet_2000ms_merge.sbatch
