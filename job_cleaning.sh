#!/bin/bash
# Step 2 cleaning-optimality study: one array task per line of
# data/dbmarvel/cleaning/jobs_X.txt (a method chain or a MILP). CPU only.
#   hpc-data data/dbmarvel/cleaning
#   hpc-submit job_cleaning.sh --array=1-<n>%30
# Outputs land in data/dbmarvel/cleaning/ (= ~/data/MARVEL_GNN/... on Hypatia).

#SBATCH -p GPU
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=cleaning

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/dbmarvel/cleaning/jobs_X.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${line}"
ctr uv run python scripts/cleaning_optimality.py run ${line}
