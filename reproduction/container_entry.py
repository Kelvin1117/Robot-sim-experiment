"""Run the frozen experiment inside its original internal filesystem layout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path('/common/users/cs2202/dygro_pi05_phase012_20261106')
PARENT = ROOT / 'oneshot_paper_v2/runs/paper-v2-20261108T174348Z'
FROZEN = ROOT / 'oneshot_retention/runs/retention-20261109T224843Z'
VLA = Path('/common/home/cs2202/VLARLKit')
POLICY_PYTHON = VLA / 'model_backends/openpi/.venv/bin/python'
SIM_PYTHON = Path('/common/home/cs2202/miniconda3/envs/libero/bin/python')
REPO = Path('/reproduction')
OUTPUTS = Path('/outputs')
CACHE = Path('/cache')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def environment(cpu_only=False):
    env = dict(os.environ)
    for name in ('home', 'tmp', 'torch', 'triton', 'cuda', 'numba', 'matplotlib', 'hf', 'libero', 'openpi'):
        (CACHE / name).mkdir(parents=True, exist_ok=True)
    code = FROZEN / 'code'
    env.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME=str(CACHE / 'home'),
               PYTHONPATH=str(code), DYGRO_CODE=str(code), DYGRO_ROOT=str(ROOT),
               DYGRO_PORT=str(ROOT / 'oneshot_paper_v2'), DYGRO_PYTHON=str(POLICY_PYTHON),
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
               HF_HOME=str(CACHE / 'hf'), XDG_CACHE_HOME=str(CACHE), TMPDIR=str(CACHE / 'tmp'),
               TORCH_HOME=str(CACHE / 'torch'), TRITON_CACHE_DIR=str(CACHE / 'triton'),
               CUDA_CACHE_PATH=str(CACHE / 'cuda'), NUMBA_CACHE_DIR=str(CACHE / 'numba'),
               MPLCONFIGDIR=str(CACHE / 'matplotlib'), OPENPI_DATA_HOME=str(CACHE / 'openpi'),
               JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', USE_TF='0', LIBERO_CONFIG_PATH=str(CACHE / 'libero'),
               WANDB_MODE='disabled', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    # LIBERO accepts JSON as YAML; this writable config keeps all assets in the mounted backup.
    libero = VLA / 'third_party/LIBERO/libero/libero'
    config = dict(benchmark_root=str(libero), assets=str(libero / 'assets'),
                  bddl_files=str(libero / 'bddl_files'), init_states=str(libero / 'init_files'),
                  datasets=str(ROOT / 'oneshot_openpi/data'))
    (CACHE / 'libero/config.yaml').write_text(json.dumps(config, indent=2) + '\n')
    if cpu_only:
        env.update(CUDA_VISIBLE_DEVICES='', ROBOT_CPU_ONLY='1')
    else:
        env.pop('ROBOT_CPU_ONLY', None)
    return env


def verify_inputs():
    for path in (POLICY_PYTHON, SIM_PYTHON, PARENT / 'config.json', FROZEN / 'best.pt'):
        if not path.is_file():
            raise RuntimeError('Missing restored input: ' + str(path))
    for relative, expected in json.loads((FROZEN / 'source_manifest.json').read_text()).items():
        if digest(FROZEN / relative) != expected:
            raise RuntimeError('Frozen source/configuration changed: ' + relative)
    settings = json.loads((FROZEN / 'config.json').read_text())
    for relative, expected in settings['source_hashes'].items():
        if digest(PARENT / relative) != expected:
            raise RuntimeError('Frozen parent changed: ' + relative)
    model = json.loads((PARENT / 'config.json').read_text())['model']
    for key in ('weights', 'normalization', 'tokenizer'):
        if digest(model[key]) != model[key + '_sha256']:
            raise RuntimeError('Model asset changed: ' + key)
    complete = json.loads((FROZEN / 'training_completed.json').read_text())
    if digest(FROZEN / 'best.pt') != complete['best_sha256']:
        raise RuntimeError('Published checkpoint changed')
    return dict(frozen_sources=True, parent_hashes=True, model_assets=True, selected_checkpoint=True)


def doctor(env, cpu_only):
    report = dict(inputs=verify_inputs(), cpu_only=cpu_only)
    for kind, python in (('policy', POLICY_PYTHON), ('simulator', SIM_PYTHON)):
        process = subprocess.run([str(python), '-B', str(REPO / 'reproduction/probe.py'), kind],
                                 env=env, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(kind + ' environment check failed:\n' + process.stdout + process.stderr)
        marker = 'ROBOT_PROBE_JSON='
        line = next((s for s in reversed(process.stdout.splitlines()) if s.startswith(marker)), None)
        if not line:
            raise RuntimeError('Missing probe result: ' + kind)
        report[kind] = json.loads(line[len(marker):])
    report['passed'] = True
    report['gpu_execution_validated'] = False
    report['next_step'] = 'Run smoke to validate actual GPU inference, rendering and learner updates.'
    return report


def run_name(name):
    if name in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name):
        raise ValueError('Expected a simple run name')
    return name


def prepare_run(action, name, resume=False, from_run=None):
    destination = OUTPUTS / run_name(name)
    if action == 'evaluate':
        source = OUTPUTS / run_name(from_run) if from_run else FROZEN
        complete = json.loads((source / 'training_completed.json').read_text())
        settings = json.loads((source / 'config.json').read_text())
        if (complete['episodes'] != settings['episodes'] or
                digest(source / 'config.json') != complete['config_sha256'] or
                digest(source / 'best.pt') != complete['best_sha256']):
            raise RuntimeError('Evaluation source is incomplete or has changed')
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ('config.json', 'training_completed.json', 'best.pt'):
            shutil.copyfile(source / filename, destination / filename)
    else:
        if from_run:
            raise ValueError('--from-run is only supported for evaluate')
        if not destination.exists():
            if resume:
                raise ValueError('Cannot resume a run that does not exist')
            destination.mkdir(parents=True)
            shutil.copyfile(FROZEN / 'config.json', destination / 'config.json')
        if digest(destination / 'config.json') != digest(FROZEN / 'config.json'):
            raise RuntimeError('Run settings differ from the frozen experiment')
        if resume and not (destination / 'latest.pt').is_file():
            raise RuntimeError('Resume requires latest.pt')
        if not resume and any((destination / name).exists() for name in ('latest.pt', 'training_completed.json')):
            raise RuntimeError('Training output exists; choose another --run or use --resume')
        if action == 'smoke' and (destination / 'gpu_smoke.json').exists():
            raise RuntimeError('Smoke result exists; choose another --run')
    provenance = dict(action=action, from_run=from_run, resume=resume,
                      frozen_config_sha256=digest(FROZEN / 'config.json'),
                      started_utc=datetime.now(timezone.utc).isoformat())
    (destination / 'launch.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return destination


def launch_args(stage, run, resume=False):
    args = [str(POLICY_PYTHON), '-u', '-B', str(FROZEN / 'code/recovery_run.py'), '--run', str(run), '--stage', stage]
    if resume:
        args.append('--resume')
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('doctor', 'smoke', 'train', 'evaluate'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--from-run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_only and args.action != 'doctor':
        parser.error('--cpu-only applies to doctor only')
    if args.resume and args.action != 'train':
        parser.error('--resume applies to train only')
    env = environment(args.cpu_only)
    try:
        if args.action == 'doctor':
            report = doctor(env, args.cpu_only)
            destination = OUTPUTS / run_name(args.run)
            destination.mkdir(parents=True, exist_ok=False)
            (destination / 'doctor.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)
            return
        # recovery_run.prepare also verifies all model/parent hashes before GPU work.
        for relative, expected in json.loads((FROZEN / 'source_manifest.json').read_text()).items():
            if digest(FROZEN / relative) != expected:
                raise RuntimeError('Frozen source/configuration changed: ' + relative)
        run = prepare_run(args.action, args.run, args.resume, args.from_run)
        if args.action == 'train':
            smoke = run / 'gpu_smoke.json'
            passed = json.loads(smoke.read_text()) if smoke.exists() else {}
            if not passed.get('passed') or passed.get('config_sha256') != digest(run / 'config.json'):
                subprocess.run(launch_args('smoke', run), env=env, check=True)
            if not json.loads(smoke.read_text()).get('passed'):
                raise RuntimeError('GPU smoke did not pass')
        os.execve(str(POLICY_PYTHON), launch_args(args.action, run, args.resume), env)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
