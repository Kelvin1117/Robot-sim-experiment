#!/bin/bash
#SBATCH --job-name=oneshot-dygro-resume-smoke
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
PORT=/common/users/cs2202/dygro_pi05_phase012_20261106/oneshot_openpi
source "$PORT/environment.sh"
RUN="$1"
test "$(cat "$RUN/SMOKE_PASS")" = PASS
"$DYGRO_PYTHON" -u -B "$PORT/run.py" --stage stage1 --config "$RUN/config.json" --run "$RUN" --resume
"$DYGRO_PYTHON" -u -B "$PORT/run.py" --stage validate --config "$RUN/config.json" --run "$RUN"
"$DYGRO_PYTHON" -u -B "$PORT/run.py" --stage stage2 --config "$RUN/config.json" --run "$RUN" --resume
echo PASS > "$RUN/RESUME_PASS"
