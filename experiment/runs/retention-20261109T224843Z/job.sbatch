#!/bin/bash
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --no-requeue
set -euo pipefail
RUN="$1"
STAGE="$2"
export DYGRO_CODE="$RUN/code"
source "$RUN/environment.sh"
exec 9>"$RUN/$STAGE.lock"
flock -n 9 || { echo "Another $STAGE job owns this run" >&2; exit 1; }
scontrol show job "$SLURM_JOB_ID" > "$RUN/logs/$STAGE-$SLURM_JOB_ID-allocation.txt"
nvidia-smi > "$RUN/logs/$STAGE-$SLURM_JOB_ID-gpu.txt"
"$DYGRO_PYTHON" - "$RUN" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name, expected in json.loads((root/'source_manifest.json').read_text()).items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Run artifact changed: {name}')
print('Run artifact hashes verified', flush=True)
PY
STARTED="$(date -u +%FT%TZ)"
cleanup() {
    STATUS=$?
    echo "exit_status=$STATUS job=$SLURM_JOB_ID stage=$STAGE started=$STARTED ended=$(date -u +%FT%TZ)" > "$RUN/logs/$STAGE-$SLURM_JOB_ID-exit.txt"
}
trap cleanup EXIT
"$DYGRO_PYTHON" -u -B "$RUN/code/recovery_run.py" --run "$RUN" --stage "$STAGE"
