#!/bin/bash
#SBATCH --job-name=oneshot-original-flow-smoke
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
PORT=/common/users/cs2202/dygro_pi05_phase012_20261106/oneshot_openpi
source "$PORT/environment.sh"
RUN="$1"
"$DYGRO_PYTHON" -u -B "$PORT/scripts/check_live.py" --config "$RUN/config.json" --run "$RUN"
echo PASS > "$RUN/COMPAT_PASS"
