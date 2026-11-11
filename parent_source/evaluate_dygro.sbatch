#!/bin/bash
#SBATCH --job-name=dygro-v2-evaluate
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
RUN="$1"
export DYGRO_CODE="$RUN/code"
source "$RUN/environment.sh"
STAGE=evaluate
exec 9>"$RUN/$STAGE.lock"
flock -n 9 || { echo "Another $STAGE job owns this run" >&2; exit 1; }
echo "job=$SLURM_JOB_ID stage=$STAGE started=$(date -u +%FT%TZ)"
scontrol show job "$SLURM_JOB_ID" > "$RUN/logs/$STAGE-$SLURM_JOB_ID-allocation.txt"
nvidia-smi > "$RUN/logs/$STAGE-$SLURM_JOB_ID-gpu.txt"
nvidia-smi --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu --format=csv -l 30 > "$RUN/logs/$STAGE-$SLURM_JOB_ID-gpu.csv" &
MONITOR=$!
cleanup() {
  STATUS=$?
  kill "$MONITOR" 2>/dev/null || true
  echo "exit_status=$STATUS job=$SLURM_JOB_ID stage=$STAGE ended=$(date -u +%FT%TZ)" > "$RUN/logs/$STAGE-$SLURM_JOB_ID-exit.txt"
}
trap cleanup EXIT
RESUME=()
if [[ "${2:-}" == --resume ]]; then RESUME=(--resume); fi
"$DYGRO_PYTHON" -u -B "$RUN/code/run.py" --stage "$STAGE" --config "$RUN/config.json" --run "$RUN" "${RESUME[@]}"
