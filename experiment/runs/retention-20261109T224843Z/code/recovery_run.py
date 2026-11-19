import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

import run as original
from storage import atomic_json, digest, load, restore_rng, rng_state, save
from dygro.critic import CriticEnsemble, update_target
from dygro.pipeline import OnlinePipeline
from dygro.replay import ReplayBuffer, mixed_batch
from dygro.trainer import RLSettings
from retention import FocusSampler, RetentionLearner, TaskReplay, candidate_score, make_policy


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GatedEnvironment:
    def __init__(self, environment, actor):
        self._environment, self._actor = environment, actor

    def reset(self, task, initial):
        self._actor.set_task(task)
        return self._environment.reset(task, initial)

    def step(self, action):
        return self._environment.step(action)


def prepare(settings, run, with_replay):
    source = Path(settings['source_run'])
    config = json.loads((source/'config.json').read_text())
    for relative, expected in settings['source_hashes'].items():
        if digest(source/relative) != expected:
            raise ValueError(f'Source changed: {relative}')
    for key in ['weights', 'normalization', 'tokenizer']:
        if digest(config['model'][key]) != config['model'][key+'_sha256']:
            raise ValueError(f'Original model asset changed: {key}')
    adapter, pre, data, marker = original.validated_adapter(config, source, digest(source/'config.json'))
    adapter.eval()
    if any(p.requires_grad for p in adapter.parameters()):
        raise ValueError('The full Stage I adapter must be frozen')
    parent = original.make_learner(config).actor
    parent.load_state_dict(load(source/'stage2/best.pt')['actor'])
    policy = make_policy(parent, config, settings).to('cuda')
    s, f = config['stage2'], config['fusion']
    critics = CriticEnsemble(f['width'], config['model']['horizon'], 7, s['hidden'], s['critic_layers'], s['critics']).to('cuda')
    rl = RLSettings(**{k:s[k] for k in RLSettings.__dataclass_fields__})
    rl.entropy_temperature = settings['entropy_temperature']
    learner = RetentionLearner(policy, critics,
        torch.optim.Adam(policy.correction.parameters(), lr=settings['actor_lr']),
        torch.optim.Adam(critics.parameters(), lr=settings['critic_lr']), rl, recovery=settings)
    offline = TaskReplay(ReplayBuffer, settings['offline_capacity'], True, settings['seed'], settings['task_probabilities'])
    online = TaskReplay(ReplayBuffer, settings['online_capacity'], False, settings['seed']+1, settings['task_probabilities'])
    if with_replay:
        state = load(source/'stage2/latest.pt')
        learner.critics.load_state_dict(state['learner']['critics'])
        learner.target.load_state_dict(state['learner']['target'])
        counts = [0]*10
        for row in state['offline']['rows']:
            offline.add(row)
            counts[int(row['task_id'])] += 1
        if any(n == 0 for n in counts):
            raise ValueError('Source demonstration replay must cover all ten tasks')
        atomic_json(run/'offline_provenance.json', dict(source_sha256=settings['source_hashes']['stage2/latest.pt'],
                    source_task_counts=counts, retained_transitions=len(offline),
                    source='existing Stage I latents/base draws; training demos only; no evaluation trajectories'))
        del state
        gc.collect()
    return config, adapter, pre, data, learner, offline, online


@torch.no_grad()
def evaluate(config, settings, adapter, pre, actor, env, log, seeds, initials, tasks=range(10)):
    counts = [[0, 0] for _ in range(10)]
    results = []
    actor.eval()
    for seed in seeds:
        for task in tasks:
            for episode, initial in enumerate(initials):
                # Independent episode RNGs make protected-task comparisons paired,
                # even when earlier target trajectories change length.
                episode_seed = seed*100000+task*100+initial
                seed_all(episode_seed)
                env.seed = seed
                raw, info = env.reset(task, initial)
                steps, success = 0, False
                while steps < settings['max_steps'] and not success:
                    batch = pre.live(raw, info['instruction'])
                    with original.amp():
                        latent = adapter.encode(batch, False)
                        base = adapter.base_chunk(batch, latent)
                    ids = torch.tensor([task], device=base.device)
                    action = actor.sample(latent.float(), base, deterministic=True, task_ids=ids)['action']
                    commands = pre.denormalize(action).clamp(-1, 1)[0].cpu().numpy()
                    for command in commands:
                        raw, _, _, _, step_info = env.step(command)
                        steps += 1
                        success = bool(step_info['is_success'])
                        if success or steps >= settings['max_steps']:
                            break
                row = dict(seed=seed, task=task, episode=episode, initial_state=initial,
                           episode_seed=episode_seed, success=success, steps=steps)
                log(event='evaluation_episode', **row)
                results.append(row)
                counts[task][0] += int(success)
                counts[task][1] += 1
    return dict(counts=counts, episodes=results,
                per_task_success_rate=[a/b if b else None for a,b in counts],
                overall_success_rate=sum(a for a,b in counts)/sum(b for a,b in counts),
                protocol=dict(seeds=seeds, initial_states=initials, max_steps=settings['max_steps'],
                              independent_episode_rng=True, success_signal='env.check_success()',
                              camera_resolution=config['environment']['camera_resolution']))


def inference_state(actor):
    return {k:v.detach().cpu().clone() for k,v in actor.correction.state_dict().items()}


def train(settings, run, smoke=False, resume=False):
    seed_all(settings['seed'])
    config, adapter, pre, data, learner, offline, online = prepare(settings, run, True)
    actor = learner.actor
    sampler = FocusSampler(settings['task_probabilities'], settings['seed'])
    env = original.Environment(run/('smoke_simulator' if smoke else 'train_simulator'), settings['seed'], config['environment'])
    log = original.Log(run/('smoke.jsonl' if smoke else 'training.jsonl'))
    parent_before = {k:v.detach().cpu().clone() for k,v in actor.parent.state_dict().items()}
    correction_before = inference_state(actor)
    def context(raw, instruction):
        batch = pre.live(raw, instruction)
        with torch.no_grad(), original.amp():
            latent = adapter.encode(batch, False)
            base = adapter.base_chunk(batch, latent)
        return latent.float(), base.float()
    def command(action):
        return pre.denormalize(action).clamp(-1, 1).cpu().numpy()
    pipeline = OnlinePipeline(GatedEnvironment(env, actor), context, command, learner, offline, online, sampler,
        batch_size=settings['batch_size'], offline_ratio=settings['offline_ratio'],
        updates_per_episode=settings['updates_per_episode'], max_steps=20 if smoke else settings['max_steps'],
        train_initial_states=settings['train_initial_states'], bootstrap_time_limits=False,
        seed=settings['seed'], device='cuda')
    def batch():
        return mixed_batch(offline, online, settings['batch_size'], settings['offline_ratio'], 'cuda')
    def save_best(rates, episode, score):
        save(run/'best.pt', dict(correction=inference_state(actor), episode=episode, rates=rates,
                                score=score, config_sha256=digest(run/'config.json'), source_hashes=settings['source_hashes']))
    def checkpoint(best_score, baseline):
        save(run/'latest.pt', dict(learner=learner.state_dict(), online=online.state_dict(),
             offline_random=[b['random_state'] for b in offline.state_dict()['buffers']],
             offline_generator=offline.state_dict()['generator'], sampler=sampler.state_dict(), pipeline=pipeline.state_dict(),
             rng=rng_state(), best_score=best_score, baseline=baseline, config_sha256=digest(run/'config.json')))
    try:
        if smoke:
            learner.actor_step(batch())
            learner.critic_step(batch())
            result = pipeline.rollout()
            metrics = learner.update(batch())
            if all(torch.equal(correction_before[k], v) for k,v in inference_state(actor).items()):
                raise RuntimeError('Correction did not update')
            if any(not torch.equal(parent_before[k], v.cpu()) for k,v in actor.parent.state_dict().items()):
                raise RuntimeError('Frozen parent changed')
            # End-to-end simulator/inference check at an evaluation-independent state.
            result_eval = evaluate(config, {**settings, 'max_steps':20}, adapter, pre, actor, env, log,
                                   [9001], [24], tasks=[0, 1, 6, 9])
            atomic_json(run/'gpu_smoke.json', dict(passed=True, rollout=result, metrics=metrics,
                        parent_unchanged=True, correction_updated=True, short_evaluation=result_eval,
                        config_sha256=digest(run/'config.json'),
                        note='20-step smoke checks execution, not policy success'))
            return
        if resume:
            state = load(run/'latest.pt')
            if state['config_sha256'] != digest(run/'config.json'):
                raise ValueError('Recovery config changed')
            learner.load_state_dict(state['learner'])
            online.load_state_dict(state['online'])
            for b,saved in zip(offline._buffers, state['offline_random'], strict=True):
                b._random.setstate(saved)
            offline._generator.set_state(state['offline_generator'])
            sampler.load_state_dict(state['sampler'])
            pipeline.load_state_dict(state['pipeline'])
            restore_rng(state['rng'])
            best_score, baseline = tuple(state['best_score']), state['baseline']
            del state
        else:
            if (run/'latest.pt').exists():
                raise FileExistsError('Use --resume to continue recovery')
            actor.enabled = False
            baseline = evaluate(config, settings, adapter, pre, actor, env, log,
                                settings['validation_seeds'], settings['validation_initial_states'])
            actor.enabled = True
            atomic_json(run/'validation_parent.json', baseline)
            best_score = candidate_score(baseline['per_task_success_rate'], baseline['per_task_success_rate'],
                                         settings['target_tasks'], settings['protection_tolerance'])
            save_best(baseline['per_task_success_rate'], 0, best_score)
            seed_all(settings['seed'])
            # Supervised warm-up anchors all target demonstration chunks, rather
            # than allowing a Q filter to suppress the recovery signal.
            for i in range(settings['bc_warmup']):
                b = batch()
                old = actor.parent_action(b['latent'], b['base'])
                row = actor.correction.sample(b['latent'], old, deterministic=True, valid_mask=b['valid_mask'])
                from regularization import chunk_bc
                mask = actor.target_mask(b['task_id'])
                bc = chunk_bc(row['action'], b['action'], b['valid_mask'], mask)
                retain = chunk_bc(row['action'], old, b['valid_mask'], ~mask)
                loss = settings['bc_weight']*bc+settings['retention_weight']*retain
                learner.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.correction.parameters(), 1., error_if_nonfinite=True)
                learner.actor_optimizer.step()
                if i%100 == 0:
                    log(event='bc_warmup', step=i+1, bc=float(bc.detach()), retention=float(retain.detach()))
            for i in range(settings['critic_warmup']):
                metrics = learner.critic_step(batch())
                update_target(learner.critics, learner.target, learner.settings.target_rate)
                if i%100 == 0:
                    log(event='critic_warmup', step=i+1, **metrics)
            checkpoint(best_score, baseline)
        for episode in range(pipeline.episodes+1, settings['episodes']+1):
            actor.train()
            env.seed = settings['seed']
            result = pipeline.rollout()
            for _ in range(settings['updates_per_episode']):
                metrics = learner.update(batch())
            log(event='training_episode', **result, updates=learner.updates, **metrics)
            if episode%settings['validation_every'] == 0 or episode == settings['episodes']:
                saved_rng = rng_state()
                candidate = evaluate(config, settings, adapter, pre, actor, env, log,
                                     settings['validation_seeds'], settings['validation_initial_states'])
                restore_rng(saved_rng)
                score = candidate_score(candidate['per_task_success_rate'], baseline['per_task_success_rate'],
                                        settings['target_tasks'], settings['protection_tolerance'])
                improved = score is not None and score > best_score
                candidate.update(episode=episode, admissible=score is not None, selected=improved)
                atomic_json(run/f'validation-{episode:05d}.json', candidate)
                if improved:
                    best_score = score
                    save_best(candidate['per_task_success_rate'], episode, score)
                log(event='validation_summary', episode=episode, score=score, best_score=best_score, selected=improved,
                    rates=candidate['per_task_success_rate'])
            if episode%settings['checkpoint_every'] == 0 or episode == settings['episodes']:
                checkpoint(best_score, baseline)
        parent_unchanged = all(torch.equal(parent_before[k], v.cpu()) for k,v in actor.parent.state_dict().items())
        if not parent_unchanged or any(p.requires_grad for p in adapter.parameters()):
            raise RuntimeError('Frozen model integrity failed')
        best = load(run/'best.pt')
        atomic_json(run/'training_completed.json', dict(episodes=pipeline.episodes, best_episode=best['episode'],
                    selected_validation_rates=best['rates'], parent_unchanged=parent_unchanged,
                    stage1_frozen=True, best_sha256=digest(run/'best.pt'), config_sha256=digest(run/'config.json'),
                    validation_improved=best['episode']>0, source_hashes=settings['source_hashes']))
    finally:
        env.close()
        data.close()


def final_evaluation(settings, run):
    complete = json.loads((run/'training_completed.json').read_text())
    if complete['episodes'] != settings['episodes'] or complete['config_sha256'] != digest(run/'config.json'):
        raise ValueError('Recovery training is incomplete or configuration changed')
    if digest(run/'best.pt') != complete['best_sha256']:
        raise ValueError('Selected correction changed')
    config, adapter, pre, data, learner, _, _ = prepare(settings, run, False)
    learner.actor.correction.load_state_dict(load(run/'best.pt')['correction'])
    env = original.Environment(run/'evaluation_simulator', settings=config['environment'])
    try:
        summaries = {}
        for name, enabled in [('parent', False), ('recovered', True)]:
            learner.actor.enabled = enabled
            result = evaluate(config, settings, adapter, pre, learner.actor, env, original.Log(run/f'{name}_episodes.jsonl'),
                              settings['evaluation_seeds'], settings['evaluation_initial_states'])
            atomic_json(run/f'{name}_summary.json', result)
            summaries[name] = result
        old, new = summaries['parent'], summaries['recovered']
        targets = settings['target_tasks']
        protected = [t for t in range(10) if t not in targets]
        def rows(summary, task):
            return [(e['seed'], e['initial_state'], e['success'], e['steps']) for e in summary['episodes'] if e['task'] == task]
        exact_outcomes = all(rows(old,t) == rows(new,t) for t in protected)
        delta = [b-a for a,b in zip(old['per_task_success_rate'], new['per_task_success_rate'], strict=True)]
        report = dict(parent_rates=old['per_task_success_rate'], recovered_rates=new['per_task_success_rate'],
            per_task_delta=delta, parent_overall=old['overall_success_rate'], recovered_overall=new['overall_success_rate'],
            episodes_per_task_per_model=60, total_evaluation_episodes=1200,
            protected_tasks=protected, protected_episode_outcomes_identical=exact_outcomes,
            targets_all_non_regressing=all(delta[t]>=0 for t in targets),
            targets_all_improved=all(delta[t]>0 for t in targets),
            target_mean_delta=sum(delta[t] for t in targets)/len(targets),
            selected_training_episode=complete['best_episode'], source_run=settings['source_run'],
            limitation='Task-gated recovery relative to the current DyGRO parent; does not prove restoration to original SFT or exact DyGRO reproduction.',
            protocol='Paired independent episode RNG; seeds 0/1/2 x initial states 0-19; 400 steps; final states never used for training/selection')
        atomic_json(run/'comparison.json', report)
        lines = ['# Task-gated DyGRO recovery', '', '| Task | Parent | Recovered | Delta (pp) |', '|---|---:|---:|---:|']
        for t in range(10):
            lines.append(f'| {t} | {old["per_task_success_rate"][t]:.1%} | {new["per_task_success_rate"][t]:.1%} | {100*delta[t]:+.1f} |')
        lines += ['', f'Protected episode outcomes identical: {exact_outcomes}',
                  f'All three targets improved: {report["targets_all_improved"]}', '', report['limitation']]
        (run/'comparison.md').write_text('\n'.join(lines)+'\n')
    finally:
        env.close()
        data.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'train', 'evaluate'], required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    settings = json.loads((run/'config.json').read_text())
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    if args.stage == 'evaluate':
        final_evaluation(settings, run)
    else:
        if args.stage == 'train':
            smoke = json.loads((run/'gpu_smoke.json').read_text())
            if not smoke['passed'] or smoke['config_sha256'] != digest(run/'config.json'):
                raise ValueError('Matching GPU smoke is required before training')
        train(settings, run, args.stage == 'smoke', args.resume)


if __name__ == '__main__':
    main()
