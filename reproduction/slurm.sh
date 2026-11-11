#!/usr/bin/env bash
#SBATCH --job-name=robot-retention
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=robot-%j.log
set -euo pipefail
# Submit from the repository root. Supply site-specific partition/account flags to sbatch.
# sbatch reproduction/slurm.sh --root /path/robot-work --run retrain train
REPO="${SLURM_SUBMIT_DIR:?Submit this script from the cloned repository}"
GPU="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$GPU" || "$GPU" == *,* ]]; then
    echo 'Expected one allocated GPU in SLURM_JOB_GPUS or CUDA_VISIBLE_DEVICES.' >&2
    exit 2
fi
exec python3 "$REPO/reproduce.py" --gpu "$GPU" "$@"
