#!/bin/bash
# Hypatia job template. Copy next to the project's pyproject.toml, edit the run lines, then from the repo:
#   hpc-submit [--watch] path/to/job.sh [sbatch overrides, e.g. --gres=gpu:a100:1]
#
# hpc-submit sets: cwd = snapshot of this script's dir, $RUN_DIR = runs/<ts>/ (provenance + slurm log),
# UV_PROJECT_ENVIRONMENT = shared venv in the mirror. Write outputs anywhere under cwd or $RUN_DIR.

#SBATCH -p GPU
#SBATCH --gres=gpu:1                # any type; pin with --gres=gpu:a100:1 or v100, L40S via --nodelist=compute-gpu-0-5
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=bench_lev

set -euo pipefail

# Opt-in: stage heavy inputs on node-local NVMe (much faster than NFS home), cleaned up on exit.
# SCRATCH=/state/partition1/$USER/$SLURM_JOB_ID
# mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
# cp -rL data/big_input "$SCRATCH/"

ctr nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
ctr uv run python scripts/bench_leverage.py --gpu
