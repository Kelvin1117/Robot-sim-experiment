"""Snapshot runtime code and launchers before any Slurm process starts."""
import hashlib
import json
import os
from pathlib import Path
import shutil

PORT=Path(__file__).resolve().parents[1]


def runtime_manifest(root):
    paths=list(Path(root).glob('*.py'))+list((Path(root)/'dygro').glob('*.py'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def snapshot(run,config):
    run=Path(run);run.mkdir()
    for name in ['logs','stage1','stage2','evaluation']:(run/name).mkdir()
    code=run/'code';code.mkdir()
    for path in PORT.glob('*.py'):shutil.copy2(path,code/path.name)
    for name in ['dygro','scripts']:
        shutil.copytree(PORT/name,code/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PORT/'environment.sh',run/'environment.sh')
    for path in PORT.glob('*.sbatch'):shutil.copy2(path,run/path.name)
    shutil.copy2(config,run/'config.json')
    (run/'source_manifest.json').write_text(json.dumps(runtime_manifest(code),indent=2)+'\n')
    for path in run.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    return run
