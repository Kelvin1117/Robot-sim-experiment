"""Separate simulator, rendering and policy repeatability without relaxing checks."""
import gc
import shutil
import subprocess
import traceback
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import recovery_run as recovery
import run as original
from experiment_checks import tensor_hash
from simulator import Environment


ROOT = Path(sys.argv[1])
SOURCE = Path(sys.argv[2])
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['DYGRO_CAPTURE_STATE'] = '1'


def differences(first, second):
    result = {}
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if a.shape != b.shape or a.dtype != b.dtype:
            result[key] = dict(first_shape=a.shape, second_shape=b.shape,
                               first_dtype=str(a.dtype), second_dtype=str(b.dtype))
        elif np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            result[key] = dict(bytewise_different=True, changed=int(np.count_nonzero(a != b)),
                               max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
    return result


class CaptureEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        self.phase = 'initial'
        self.episodes = []
        self.episode = None
        super().__init__(*args, **kwargs)

    def reset(self, task, initial):
        self.episode = dict(phase=self.phase, task=task, initial=initial,
                            seed=self.seed, snapshots=[], actions=[])
        self.episodes.append(self.episode)
        return super().reset(task, initial)

    def request(self, message):
        row = super().request(message)
        if 'obs' in row:
            snapshot = {**row['obs'], **row['diagnostic_state']}
            self.episode['snapshots'].append(snapshot)
            name = '%s-episode%02d-step%03d.npz' % (
                self.phase, len(self.episodes)-1, len(self.episode['snapshots'])-1)
            np.savez_compressed(ROOT/name, **snapshot)
            if message['command'] == 'step':
                self.episode['actions'].append(np.asarray(message['action']))
        return row


def compare_episodes(first, second):
    result = []
    for i, (a, b) in enumerate(zip(first['snapshots'], second['snapshots'])):
        diff = differences(a, b)
        if diff:
            result.append(dict(step=i, differences=diff))
    return dict(first_phase=first['phase'], second_phase=second['phase'],
                task=first['task'], differing_steps=result,
                first_steps=len(first['snapshots']), second_steps=len(second['snapshots']))


torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
report = dict(job_id=os.environ.get('SLURM_JOB_ID'), source=str(SOURCE),
              gpu=str(torch.cuda.get_device_properties(0)), fixed_actions=[],
              frozen_input_policy=[], closed_loop=[], action_replay=[], renderer_variants={})


def save_report():
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


env = CaptureEnvironment(ROOT/'simulator')
data = None
try:
    for repetition in range(3):
        for task in [0, 3]:
            env.phase = 'fixed%d' % repetition
            env.seed = 910100024 if task == 0 else 910100324
            env.reset(task, 24)
            for step in range(20):
                env.step(np.random.RandomState(1000*task+step).uniform(-.1, .1, 7))
    fixed = list(env.episodes)
    for index in range(2, len(fixed)):
        report['fixed_actions'].append(compare_episodes(fixed[index % 2], fixed[index]))
    save_report()
    for mode in ['no_dither', 'no_msaa']:
        os.environ['DYGRO_RENDER_MODE'] = mode
        variant = CaptureEnvironment(ROOT/mode)
        trial = []
        try:
            for repetition in range(4):
                for task in [0, 3]:
                    variant.phase = '%s-repeat%d' % (mode, repetition)
                    variant.seed = 910100024 if task == 0 else 910100324
                    variant.reset(task, 24)
                    for step in range(20):
                        variant.step(np.random.RandomState(1000*task+step).uniform(-.1, .1, 7))
            for index in range(2, len(variant.episodes)):
                trial.append(compare_episodes(variant.episodes[index % 2], variant.episodes[index]))
        finally:
            variant.close()
        report['renderer_variants'][mode] = trial
        save_report()
    os.environ.pop('DYGRO_RENDER_MODE', None)

    settings = json.loads((SOURCE/'config.json').read_text())
    recovery.seed_all(settings['seed'])
    config, adapter, pre, data, learner, _, _ = recovery.prepare(settings, ROOT, False)
    actor = learner.actor
    actor.enabled = False
    actor.eval()
    for task in [0, 3]:
        raw = fixed[task == 3]['snapshots'][0]
        # Use the exact original instruction without changing the frozen observation.
        env.phase = 'instruction%d' % task
        env.seed = 910100024 if task == 0 else 910100324
        _, info = env.reset(task, 24)
        batch = pre.live(raw, info['instruction'])
        hashes = []
        with torch.no_grad():
            for repetition in range(5):
                generator = torch.Generator(device='cuda').manual_seed(env.seed)
                noise = torch.randn((1, config['model']['horizon'], 32),
                                    device='cuda', generator=generator)
                with original.amp():
                    latent = adapter.encode(batch, False)
                    base = adapter.base_chunk(batch, latent, noise=noise)
                action = actor.sample(latent.float(), base, deterministic=True,
                                      task_ids=torch.tensor([task], device='cuda'))['action']
                tensors = dict(latent=latent, base=base, action=action)
                hashes.append({key: tensor_hash(value) for key, value in tensors.items()})
                np.savez_compressed(ROOT/('frozen-task%d-repeat%d.npz' % (task, repetition)),
                                    **{k: v.float().cpu().numpy() for k, v in tensors.items()})
        report['frozen_input_policy'].append(dict(task=task, hashes=hashes,
                                                  exact_repeat=all(h == hashes[-1] for h in hashes)))
    save_report()
    closed = []
    for repetition in range(2):
        env.phase = 'policy%d' % repetition
        start = len(env.episodes)
        recovery.evaluate(config, {**settings, 'max_steps':20}, adapter, pre, actor, env,
                          original.Log(ROOT/'policy.jsonl'), [9101], [24], tasks=[0, 3])
        closed.append(env.episodes[start:])
    for first, second in zip(*closed):
        result = compare_episodes(first, second)
        result['action_differences'] = differences(
            {'commands': np.stack(first['actions'])}, {'commands': np.stack(second['actions'])})
        report['closed_loop'].append(result)
        env.phase = 'replay%d' % first['task']
        env.seed = first['seed']
        env.reset(first['task'], first['initial'])
        for command in first['actions']:
            env.step(command)
        report['action_replay'].append(compare_episodes(first, env.episode))
    save_report()
finally:
    env.close()
    if data is not None:
        data.close()

# A candidate is accepted only by the original full GPU smoke, including exact
# observation/action/trajectory equality and a real learner update.
env.close()
if 'A100' not in torch.cuda.get_device_name(0):
    report['control_only'] = True
    save_report()
    sys.exit(0)
del env, learner, actor, adapter, pre, data, latent, base, action, tensors, batch
gc.collect()
torch.cuda.empty_cache()
os.environ.pop('DYGRO_CAPTURE_STATE', None)
report['candidate_smokes'] = []
for mode in ['no_dither', 'no_msaa']:
    trials = report['renderer_variants'][mode]
    if not trials or any(row['differing_steps'] for row in trials):
        report['candidate_smokes'].append(dict(mode=mode, skipped='fixed-action repeats differ'))
        save_report()
        continue
    candidate = SOURCE.parent / ('render-fix-' + str(report['job_id'])) / mode
    candidate.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, candidate/'code',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['environment.sh', 'job.sbatch']:
        shutil.copy2(SOURCE/name, candidate/name)
    (candidate/'logs').mkdir()
    candidate_settings = dict(settings)
    candidate_settings['rendering_fix'] = dict(mode=mode, source_run=str(SOURCE),
        diagnostic_report=str(ROOT/'report.json'),
        change='Disable OpenGL dithering' if mode == 'no_dither' else 'Disable offscreen multisample antialiasing',
        original_repeatability_criterion_preserved=True,
        comparison='Parent and recovered policies must both use this renderer; historical renderer scores are separate')
    (candidate/'config.json').write_text(json.dumps(candidate_settings, indent=2)+'\n')
    with (candidate/'environment.sh').open('a') as stream:
        stream.write('\nexport DYGRO_RENDER_MODE=' + mode + '\n')
    from storage import digest
    manifest = {str(path.relative_to(candidate)):digest(path)
                for path in (candidate/'code').rglob('*') if path.is_file()}
    for name in ['config.json', 'environment.sh', 'job.sbatch']:
        manifest[name] = digest(candidate/name)
    (candidate/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    os.environ['DYGRO_RENDER_MODE'] = mode
    try:
        for smoke_pass in [1, 2]:
            with (candidate/'logs'/('acceptance-%d.out' % smoke_pass)).open('w') as out, \
                 (candidate/'logs'/('acceptance-%d.err' % smoke_pass)).open('w') as err:
                subprocess.run(['bash', str(candidate/'job.sbatch'), str(candidate), 'smoke'],
                               check=True, stdout=out, stderr=err)
            shutil.copy2(candidate/'gpu_smoke.json', candidate/('gpu_smoke-pass%d.json' % smoke_pass))
        result = json.loads((candidate/'gpu_smoke.json').read_text())
        if not result['passed']:
            raise RuntimeError('GPU smoke did not pass')
        report['candidate_smokes'].append(dict(mode=mode, passed=True, fresh_process_smoke_passes=2, run=str(candidate)))
        (ROOT/'chosen_run.txt').write_text(str(candidate)+'\n')
        (SOURCE.parents[1]/'latest_run.txt').write_text(str(candidate)+'\n')
        report['fixed_and_verified'] = True
        save_report()
        break
    except Exception:
        report['candidate_smokes'].append(dict(mode=mode, passed=False, run=str(candidate),
                                               error=traceback.format_exc()))
        save_report()
        gc.collect()
        torch.cuda.empty_cache()
else:
    report['fixed_and_verified'] = False
    save_report()
    raise RuntimeError('No renderer candidate passed the original full GPU smoke; see report.json')
