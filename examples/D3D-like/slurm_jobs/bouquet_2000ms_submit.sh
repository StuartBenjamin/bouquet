#!/bin/bash
# run from anywhere: everything below is relative to this script
cd "$(dirname "$0")"
# chain array + merge; afterany (not afterok) so the merge still
# runs -- and reports exactly which shards are missing -- after a
# partial array, instead of pending forever.
aid=$(sbatch --parsable bouquet_2000ms_array.sbatch)
sbatch --dependency=afterany:$aid bouquet_2000ms_merge.sbatch
