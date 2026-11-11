import datetime
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'oneshot_paper_v2/runs/paper-v2-20261108T174348Z'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'runs'/f'retention-{stamp}'
    run.mkdir(parents=True)
    (run/'logs').mkdir()
    shutil.copytree(SOURCE/'code', run/'code', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['retention.py', 'recovery_run.py', 'test_retention.py']:
        shutil.copy2(ROOT/name, run/'code'/name)
    for name in ['environment.sh']:
        shutil.copy2(SOURCE/name, run/name)
    shutil.copy2(ROOT/'job.sbatch', run/'job.sbatch')
    original_manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    for name, expected in original_manifest.items():
        if sha(run/'code'/name) != expected:
            raise ValueError(f'Original source snapshot no longer matches: {name}')
    probabilities=[.4/7]*10
    for task, probability in [(0,.15),(6,.15),(9,.30)]:
        probabilities[task]=probability
    settings=dict(source_run=str(SOURCE), seed=1042, target_tasks=[0,6,9],
        method='Frozen Stage I and parent residual; new instruction/task-gated bounded correction; BC + residual RL + retention',
        provenance='Experimental anti-forgetting repair; modified DyGRO port, not exact paper reproduction',
        source_hashes={name:sha(SOURCE/name) for name in ['config.json','stage1/validated.json','stage1/validated.pt','stage2/best.pt','stage2/latest.pt']},
        task_probabilities=probabilities, correction_scale=.2, actor_lr=3e-5, critic_lr=1e-4,
        entropy_temperature=.01, bc_weight=10., retention_weight=20., trust_weight=.1,
        offline_ratio=.5, offline_capacity=20000, online_capacity=30000, batch_size=128,
        bc_warmup=2000, critic_warmup=500, episodes=1200, updates_per_episode=40,
        validation_every=200, checkpoint_every=100, max_steps=400,
        train_initial_states=list(range(25,50)), validation_initial_states=list(range(20,25)),
        validation_seeds=[8123,8124,8125,8126], protection_tolerance=0.,
        evaluation_seeds=[0,1,2], evaluation_initial_states=list(range(20)),
        selection='Require all protected and target validation rates >= parent; maximize weakest target then target mean; identity fallback',
        evaluation='Parent and recovered each 10 tasks x 3 seeds x 20 states = 600; paired RNG per episode',
        limitation='Protects the current DyGRO parent; original SFT restoration is not guaranteed. Validation repeats five held-out states across four seeds, not twenty distinct states.')
    (run/'config.json').write_text(json.dumps(settings, indent=2)+'\n')
    readme='''# DyGRO task-gated retention experiment

Frozen parent: paper-v2-20261108T174348Z Stage I validated checkpoint and Stage II best (episode 600).
Original runs, checkpoints and scripts remain untouched. This run has an isolated source snapshot.

Only tasks 0, 6, 9 enable a new bounded correction. The other seven use the exact parent inference path.
No shared backbone, fusion, parent router or parent experts are updated. The correction starts at zero
deterministic output. Task gating uses the known LIBERO task/instruction; this is an explicit method change.

Training: 1200 online episodes, 40 updates/episode; 60% target / 40% protected sampling in both rollout
and per-task replay. Reuse all 5884 training-demo replay chunks from the original run. No final evaluation
trajectories enter replay. Unfiltered target-demo BC (weight 10), old-policy retention (20), small target
trust penalty (0.1), reduced entropy (0.01 averaged per valid action dimension), new actor LR 3e-5.
Critics are initialized from the original latest checkpoint and recalibrated; parent actor comes from best.
The correction scale 0.2 is an experimental normalized-space bound; final commands still clamp to [-1,1].

Validation: states 20-24 x four seeds = 20 episodes/task. These are five distinct states; training resets
remain 25-49. Select only checkpoints with no per-task regression versus the paired parent validation;
maximize minimum target success, then target mean. If none improve, select the original zero correction
and explicitly report no recovery. Do not use states 0-19 to choose checkpoints.

Final evaluation: parent 600 + recovered 600 episodes; each model performs 60 per task (three seeds x
20 initial states), 400-step limit. Episode RNGs are paired independently, so target duration changes do
not shift protected-task random draws. This RNG protocol differs from historical job 1096493; use the
new paired parent as the comparison. Output: parent_summary.json, recovered_summary.json,
comparison.json, comparison.md. Protected-task identity is also checked by success and step outcomes.

Slurm: GPU smoke -> afterok training -> afterok paired evaluation, rlab2, 1 A100, 96 GiB each.
The smoke executes real replay actor/critic updates and simulator episodes, and fails on parent mutation.
CPU tests cover no-op/RNG identity, trained target gating, frozen teacher gradients, real learner updates,
selection rejecting regressions, replay balancing, and sampler/replay resume.

This is an experimental repair to the DyGRO OpenPI port. Preserving the current parent is different
from restoring all original SFT ability; improvements and statistical uncertainty must be read from
the final report. No success claim is made at submission time.
'''
    (run/'README.md').write_text(readme)
    manifest={str(p.relative_to(run)):sha(p) for p in sorted((run/'code').rglob('*.py'))}
    for name in ['config.json','environment.sh','job.sbatch']:
        manifest[name]=sha(run/name)
    (run/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'latest_run.txt').write_text(str(run)+'\n')
    print(run)


if __name__ == '__main__':
    main()
