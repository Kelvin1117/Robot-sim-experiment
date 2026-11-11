#!/bin/bash
#SBATCH --job-name=dygro-v2-smoke
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -euo pipefail
RUN="$1"
export DYGRO_CODE="$RUN/code"
source "$RUN/environment.sh"
nvidia-smi > "$RUN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/allocation.txt"
trap 'status=$?; echo "exit_status=$status job=$SLURM_JOB_ID time=$(date -u +%FT%TZ)" > "$RUN/shell_exit.txt"' EXIT
"$DYGRO_PYTHON" -u -B "$RUN/code/scripts/check_live.py" --config "$RUN/config.json" --run "$RUN"
for STAGE in stage1 validate stage2 evaluate; do
  "$DYGRO_PYTHON" -u -B "$RUN/code/run.py" --stage "$STAGE" --config "$RUN/config.json" --run "$RUN"
done
for STAGE in stage1 stage2; do
  "$DYGRO_PYTHON" -u -B "$RUN/code/run.py" --stage "$STAGE" --config "$RUN/config.json" --run "$RUN" --resume
done
echo PASS > "$RUN/SMOKE_PASS"
