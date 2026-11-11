import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from dataset import GoalData,validate_data
from native import OpenPIAdapter,Preprocessor,load_base
from storage import atomic_json,digest,load,promote,restore_rng,rng_state,save
from dygro.representation import DVCritic,frozen_parameters
from dygro.policy import DifficultySampler,ResidualPolicy
from dygro.critic import CriticEnsemble
from dygro.trainer import RLSettings,ResidualLearner
from dygro.replay import ReplayBuffer,mixed_batch
from dygro.pipeline import OnlinePipeline,offline_warmup
from simulator import Environment
from regularization import actor_objective


def amp():
    return torch.autocast('cuda',dtype=torch.bfloat16)


class Log:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def __call__(self,**values):
        row=dict(time=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),**values)
        text=json.dumps(row,allow_nan=False)
        with self.path.open('a') as stream:stream.write(text+'\n')
        if 'step' not in values or values.get('step',0)%100==0 or values.get('step')==1 or 'validation_flow_loss' in values:
            print(text,flush=True)


def build(config):
    core=load_base(config['model'],'cuda')
    adapter=OpenPIAdapter(core,config['fusion'],config['model']['denoise_steps']).to('cuda')
    adapter.set_stage1_trainable(config['stage1']['train_base'])
    pre=Preprocessor(config['model'],'cuda')
    data=GoalData(config['data']['directory'],pre,config['model']['horizon'],config['data']['validation_episodes'])
    return adapter,pre,data


def check_provenance(state,config_hash,config):
    if state['config_hash']!=config_hash or state['base_sha256']!=config['model']['weights_sha256']:
        raise ValueError('Checkpoint configuration/base provenance mismatch')


@torch.no_grad()
def validation_loss(adapter,data,config):
    settings=config['stage1'];losses=[]
    mode=adapter.training;adapter.eval()
    generator=np.random.default_rng(settings['validation_seed'])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(settings['validation_seed'])
        for _ in range(settings['validation_batches']):
            batch,mask=data.sample(settings['batch_size'],generator,True)
            with amp():loss=adapter.flow_loss(adapter.encode(batch,False),batch['action'],mask)
            losses.append(float(loss))
    adapter.train(mode)
    value=float(np.mean(losses))
    if not math.isfinite(value):raise ValueError('Nonfinite validation loss')
    return value


def train_stage1(config,run,config_hash,resume,stop_at=None):
    out=run/'stage1';out.mkdir(exist_ok=True)
    log=Log(out/'metrics.jsonl');settings=config['stage1']
    adapter,pre,data=build(config)
    atomic_json(out/'pretrained_load_audit.json',adapter.core.pretrained_load_audit)
    training_frames=data.training_frames()
    target_steps=stop_at if stop_at is not None else settings['initial_steps']
    if target_steps>settings['steps']:raise ValueError('Stage I budget exceeded')
    parameters=[p for p in adapter.parameters() if p.requires_grad]
    names=[n for n,p in adapter.named_parameters() if p.requires_grad]
    atomic_json(out/'parameter_rules.json',dict(updated=names,frozen=[n for n,p in adapter.named_parameters() if not p.requires_grad]))
    generator=np.random.default_rng(config['seed'])
    example,_=data.sample(settings['batch_size'],generator)
    dv=DVCritic(adapter.raw_observation_vector(example).shape[-1],config['fusion']['width']*config['fusion']['queries'],settings['dv_width']).to('cuda')
    optimizer_class={'AdamW':torch.optim.AdamW,'Adam':torch.optim.Adam}
    opt=optimizer_class[settings['optimizer']](parameters,lr=settings['lr'],weight_decay=settings['weight_decay'],betas=tuple(settings['betas']),eps=settings['eps'])
    dvopt=optimizer_class[settings['dv_optimizer']](dv.parameters(),lr=settings['dv_lr'],betas=tuple(settings['betas']),eps=settings['eps'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,settings['steps'],eta_min=settings['min_lr'])
    start=0;best=float('inf')
    if resume:
        state=load(out/'latest.pt');check_provenance(state,config_hash,config)
        adapter.load_delta(state['delta']);dv.load_state_dict(state['dv'])
        opt.load_state_dict(state['optimizer']);dvopt.load_state_dict(state['dv_optimizer'])
        scheduler.load_state_dict(state['scheduler']);start=state['step'];best=state['best_loss']
        generator.bit_generator.state=state['data_rng'];restore_rng(state['rng'])
    elif (out/'latest.pt').exists():raise FileExistsError('Use --resume for existing Stage I')
    adapter.train()
    for step in range(start+1,target_steps+1):
        batch,mask=data.sample(settings['batch_size'],generator)
        permutation=torch.randperm(settings['batch_size'],device='cuda')
        with amp():
            z=adapter.encode(batch,True)
            observation=adapter.raw_observation_vector(batch)
        for _ in range(settings['critic_updates']):
            dvopt.zero_grad(set_to_none=True)
            estimate=dv.estimate(observation.detach(),z.detach().float().flatten(1),permutation)
            (-estimate).backward()
            dvnorm=torch.nn.utils.clip_grad_norm_(dv.parameters(),settings['max_grad_norm'],error_if_nonfinite=True)
            dvopt.step()
        dvopt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
        with frozen_parameters(dv):
            with amp():flow=adapter.flow_loss(z,batch['action'],mask)
            ib=dv.estimate(observation,z.float().flatten(1),permutation)
            objective=flow+settings['ib_weight']*ib
            objective.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,settings['max_grad_norm'],error_if_nonfinite=True)
        if step==1:
            gradients={n:float(p.grad.norm()) for n,p in adapter.named_parameters() if p.requires_grad and p.grad is not None}
            for key in ['latent_to_prefix.weight','fusion.queries','fusion.action.weight','core.action_in_proj.weight']:
                if gradients.get(key,0)<=0:raise ValueError(f'Missing representation gradient: {key}')
            atomic_json(out/'first_step_gradients.json',gradients)
        opt.step();scheduler.step()
        metrics=dict(step=step,epochs=step*settings['batch_size']/training_frames,
                     flow_loss=float(flow.detach()),ib_loss=float(ib.detach()),dv_ascent=float(estimate.detach()),
                     loss=float(objective.detach()),gradient_norm=float(norm),dv_gradient_norm=float(dvnorm),lr=scheduler.get_last_lr()[0])
        improved=False
        if step%settings['validation_every']==0 or step==target_steps:
            score=validation_loss(adapter,data,config);metrics['validation_flow_loss']=score
            improved=step>=settings['min_completed_steps'] and score<best
            if improved:best=score
        log(**metrics)
        if improved or step%settings['checkpoint_every']==0 or step==target_steps:
            save(out/'latest.pt',dict(format_version=2,stage='stage1',step=step,config_hash=config_hash,
                base_sha256=config['model']['weights_sha256'],delta=adapter.trainable_state(),dv=dv.state_dict(),
                optimizer=opt.state_dict(),dv_optimizer=dvopt.state_dict(),scheduler=scheduler.state_dict(),
                best_loss=best,validation_loss=metrics.get('validation_flow_loss'),data_rng=generator.bit_generator.state,rng=rng_state()))
            if improved:
                save(out/'best.pt',dict(format_version=3,stage='stage1',step=step,config_hash=config_hash,
                    base_sha256=config['model']['weights_sha256'],delta=adapter.trainable_state(),
                    best_loss=best,validation_loss=score,checkpoint_kind='inference; optimizer retained in latest.pt'))
    atomic_json(out/'completed.json',dict(config_hash=config_hash,steps=target_steps,epoch_equivalents=target_steps*settings['batch_size']/training_frames,latest_sha256=digest(out/'latest.pt'),best_sha256=digest(out/'best.pt'),job_id=os.environ.get('SLURM_JOB_ID')))
    data.close()


def stage1_gate_passes(scores,gate):
    rates=scores['per_task_success_rate']
    return (len(rates)==10 and all(math.isfinite(v) for v in rates) and
            math.isfinite(scores['overall_success_rate']) and
            scores['overall_success_rate']>=gate['minimum_overall_success_rate'] and
            min(rates)>=gate['minimum_per_task_success_rate'])


def validate_stage1(config,run,config_hash):
    """Real base-policy gate. At most one prespecified continuation, then fail closed."""
    out=run/'stage1';gate=config['stage1_gate']
    (out/'validated.json').unlink(missing_ok=True)
    for attempt in range(2):
        completed=json.loads((out/'completed.json').read_text())
        if completed['config_hash']!=config_hash or completed['steps']<config['stage1']['min_completed_steps']:
            raise ValueError('Incomplete Stage I or configuration mismatch')
        if completed['epoch_equivalents']<config['stage1']['minimum_epoch_equivalents']:
            raise ValueError('Insufficient offline training exposure')
        for name in ['best','latest']:
            if digest(out/(name+'.pt'))!=completed[name+'_sha256']:raise ValueError('Checkpoint hash mismatch')
        adapter,pre,data=build(config)
        env=Environment(out/f'validation-simulator-{attempt}',settings=config['environment'])
        checked=set()
        try:
            for name in ['best','latest']:
                sha=completed[name+'_sha256']
                if sha in checked:continue
                checked.add(sha)
                state=load(out/(name+'.pt'));check_provenance(state,config_hash,config)
                if name=='latest' and (not state['optimizer']['state'] or not state['dv_optimizer']['state']):
                    raise ValueError('Missing trained optimizer state')
                if state['step']<config['stage1']['min_completed_steps']:raise ValueError('Checkpoint predates minimum training budget')
                selected_step=state['step']
                if name=='latest' and state['scheduler']['last_epoch']!=state['step']:raise ValueError('Scheduler step mismatch')
                for key,tensor in state['delta'].items():
                    if not torch.isfinite(tensor).all():raise ValueError(f'Nonfinite checkpoint: {key}')
                adapter.load_delta(state['delta']);adapter.eval()
                expected=state['best_loss'] if name=='best' else state['validation_loss']
                del state
                score=validation_loss(adapter,data,config)
                if abs(score-expected)>max(1e-4,abs(expected)*1e-4):raise ValueError('Validation loss reload mismatch')
                batch,_=data.sample(1,np.random.default_rng(914),True)
                with amp(),torch.no_grad():action=adapter.base_chunk(batch)
                if action.shape!=(1,config['model']['horizon'],7) or not torch.isfinite(action).all():
                    raise ValueError('Invalid trained action chunk')
                scores=evaluate(adapter,pre,None,env,config,Log(out/f'gate-{attempt}-{name}-episodes.jsonl'),
                    seeds=gate['seeds'],episodes=len(gate['initial_states']),max_steps=gate['max_steps'],initial_states=gate['initial_states'])
                passed=stage1_gate_passes(scores,gate)
                scores.update(environment_gate_passed=passed,source_checkpoint=name,checkpoint_sha256=sha,
                              training_completed_steps=completed['steps'],validation_flow_loss=score)
                atomic_json(out/f'gate-{attempt}-{name}.json',scores)
                if passed:
                    # Dedicated immutable hard link; continuation cannot invalidate a passing source.
                    promote(out/(name+'.pt'),out/'validated.pt')
                    atomic_json(out/'validated.json',dict(config_hash=config_hash,checkpoint=str((out/'validated.pt').resolve()),
                        sha256=sha,training_completed_steps=completed['steps'],selected_step=selected_step,validation_flow_loss=score,
                        environment_gate_passed=True,overall_success_rate=scores['overall_success_rate'],
                        per_task_success_rate=scores['per_task_success_rate'],protocol=scores['actual_protocol'],
                        gate_thresholds=gate,job_id=os.environ.get('SLURM_JOB_ID')))
                    return
        finally:
            env.close();data.close()
            del adapter,pre,data
            import gc
            gc.collect();torch.cuda.empty_cache()
        if attempt==0 and completed['steps']<config['stage1']['steps'] and gate['allow_budget_extension']:
            Log(out/'gate-continuation.jsonl')(event='continue_paper_stage1',from_step=completed['steps'],to_step=config['stage1']['steps'])
            train_stage1(config,run,config_hash,True,stop_at=config['stage1']['steps'])
        else:break
    raise RuntimeError('Stage I LIBERO base-policy gate failed; Stage II is blocked. Inspect gate-*.json; do not bypass.')


def validated_adapter(config,run,config_hash):
    marker=json.loads((run/'stage1/validated.json').read_text())
    if not marker.get('environment_gate_passed',False):raise ValueError('Stage I environment validation did not pass')
    if marker['config_hash']!=config_hash or digest(marker['checkpoint'])!=marker['sha256']:
        raise ValueError('Stale or incompatible validation marker')
    state=load(marker['checkpoint']);check_provenance(state,config_hash,config)
    adapter,pre,data=build(config);adapter.load_delta(state['delta']);adapter.freeze_for_online()
    return adapter,pre,data,marker


def make_learner(config):
    s=config['stage2'];f=config['fusion'];h=config['model']['horizon']
    actor=ResidualPolicy(f['width'],s['embedding_dim'],10,s['experts'],s['top_m'],h,7,s['hidden'],s['temperature'],s['residual_scale'],s['log_std_bounds']).to('cuda')
    critics=CriticEnsemble(f['width'],h,7,s['hidden'],s['critic_layers'],s['critics']).to('cuda')
    keys=RLSettings.__dataclass_fields__
    settings=RLSettings(**{k:s[k] for k in keys})
    class ExperimentalLearner(ResidualLearner):
        def actor_step(self,batch):
            return actor_objective(self,batch,s['advantage_threshold'])
    optimizer_class={'Adam':torch.optim.Adam,'AdamW':torch.optim.AdamW}[s['optimizer']]
    return ExperimentalLearner(actor,critics,optimizer_class(actor.parameters(),lr=s['actor_lr'],weight_decay=s['weight_decay'],betas=tuple(s['betas']),eps=s['eps']),
                          optimizer_class(critics.parameters(),lr=s['critic_lr'],weight_decay=s['weight_decay'],betas=tuple(s['betas']),eps=s['eps']),settings)


@torch.no_grad()
def evaluate(adapter,pre,actor,environment,config,log,*,seeds,episodes,max_steps,initial_states=None):
    results=[];counts=np.zeros((10,2),int)
    for seed in seeds:
        environment.seed=seed
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        for task in range(10):
            for episode in range(episodes):
                initial=initial_states[episode] if initial_states is not None else episode
                raw,info=environment.reset(task,initial);success=False;steps=0
                while steps<max_steps and not success:
                    batch=pre.live(raw,info['instruction'])
                    with amp():
                        latent=adapter.encode(batch,False);base=adapter.base_chunk(batch,latent)
                    action=actor.sample(latent.float(),base,deterministic=True)['action'] if actor is not None else base
                    command=pre.denormalize(action).clamp(-1,1)[0].cpu().numpy()
                    for item in command:
                        raw,reward,terminal,truncation,step_info=environment.step(item);steps+=1
                        success=bool(step_info['is_success'])
                        if success or steps>=max_steps:break
                row=dict(seed=seed,task=task,episode=episode,initial_state=initial,success=success,steps=steps)
                results.append(row);counts[task]+=[int(success),1];log(event='evaluation_episode',**row)
    rates=(counts[:,0]/counts[:,1]).tolist()
    return dict(actual_protocol=dict(seeds=seeds,episodes_per_task=episodes,max_steps=max_steps,initial_states=initial_states),episodes=results,per_task_success_rate=rates,overall_success_rate=float(counts[:,0].sum()/counts[:,1].sum()),
                task3_success_rate=rates[3],counts=counts.tolist(),config=config['evaluation'],
                model='Rhx11111/pi05_libero_one_shot + DyGRO experimental OpenPI port',
                action_horizon=config['model']['horizon'],denoise_steps=config['model']['denoise_steps'],control_hz=20,
                success_signal='env.check_success(); stop immediately on success; horizon treated as terminal',
                history_comparison='90/92/93.5 are historical, not controlled same-checkpoint comparisons')


def train_stage2(config,run,config_hash,resume):
    out=run/'stage2';out.mkdir(exist_ok=True);log=Log(out/'metrics.jsonl');s=config['stage2']
    adapter,pre,data,marker=validated_adapter(config,run,config_hash)
    learner=make_learner(config)
    initial_actor={n:p.detach().clone() for n,p in learner.actor.named_parameters()}
    initial_critics={n:p.detach().clone() for n,p in learner.critics.named_parameters()}
    offline=ReplayBuffer(s['offline_capacity'],True,config['seed'])
    online=ReplayBuffer(s['replay_capacity'],False,config['seed']+1)
    sampler=DifficultySampler(10,s['difficulty_target'],s['difficulty_temperature'],0.)
    env=Environment(out/'simulator',config['seed'],config['environment'])
    def context(raw,instruction):
        batch=pre.live(raw,instruction)
        with torch.no_grad(),amp():
            z=adapter.encode(batch,False);base=adapter.base_chunk(batch,z)
        return z.float(),base.float()
    clip_stats=dict(commands=0,clipped=0,residual_changed=0)
    def command(action):
        physical=pre.denormalize(action)
        clip_stats['commands']+=physical.numel();clip_stats['clipped']+=int((physical.abs()>1).sum())
        return physical.clamp(-1,1).cpu().numpy()
    pipeline=OnlinePipeline(env,context,command,learner,offline,online,sampler,
        batch_size=s['batch_size'],offline_ratio=s['offline_ratio'],updates_per_episode=s['updates_per_episode'],
        max_steps=s['max_steps'],train_initial_states=s['train_initial_states'],bootstrap_time_limits=s['bootstrap_time_limits'],seed=config['seed'],device='cuda')
    best=-1.;episode_counts=np.zeros((10,2),int);expert_counts=np.zeros(s['experts'],int)
    try:
        if resume:
            state=load(out/'latest.pt');check_provenance(state,config_hash,config)
            if state['stage1_sha256']!=marker['sha256']:raise ValueError('Stage I changed across resume')
            learner.load_state_dict(state['learner']);offline.load_state_dict(state['offline']);online.load_state_dict(state['online'])
            sampler.load_state_dict(state['sampler']);pipeline.load_state_dict(state['pipeline']);restore_rng(state['rng'])
            best=state['best_success'];episode_counts=np.array(state['episode_counts']);expert_counts=np.array(state['expert_counts'])
        else:
            if (out/'latest.pt').exists():raise FileExistsError('Use --resume for existing Stage II')
            with amp():
                data.populate(offline,adapter,s['gamma'],s['offline_episodes_per_task'],
                    lambda task,episode,count:log(event='offline_replay',task=task,episode=episode,transitions=count) if count%100==0 else None,
                    max_chunks_per_episode=s['offline_max_chunks_per_episode'])
            log(event='offline_replay_ready',transitions=len(offline))
            offline_warmup(learner,offline,online,critic_steps=s['critic_warmup'],policy_steps=s['actor_warmup'],
                batch_size=s['batch_size'],device='cuda',on_step=lambda kind,step,metrics:log(event=kind,step=step,**metrics) if step%100==0 or step==1 else None)
        learner.actor.eval()
        atomic_json(out/'parameter_rules.json',dict(frozen=[n for n,p in adapter.named_parameters()],updated_actor=[n for n,p in learner.actor.named_parameters()],updated_critics=[n for n,p in learner.critics.named_parameters()]))
        def checkpoint():
            save(out/'latest.pt',dict(format_version=2,stage='stage2',config_hash=config_hash,base_sha256=config['model']['weights_sha256'],stage1_sha256=marker['sha256'],
                learner=learner.state_dict(),offline=offline.state_dict(),online=online.state_dict(),sampler=sampler.state_dict(),
                pipeline=pipeline.state_dict(),rng=rng_state(),best_success=best,episode_counts=episode_counts.tolist(),expert_counts=expert_counts.tolist()))
        checkpoint()
        learner.actor.train()
        for episode in range(pipeline.episodes+1,s['episodes']+1):
            def on_chunk(row):
                change=float(row['residual'].abs().max())
                if change>0:clip_stats['residual_changed']+=1
                executed_delta=float((pre.denormalize(row['action']).clamp(-1,1)-pre.denormalize(row['base']).clamp(-1,1)).abs().max())
                log(event='chunk',task=row['task'],steps=row['steps'],duration=row['duration'],reward=row['reward'],
                    residual_max=change,executed_residual_delta=executed_delta,selected=row['selected'].tolist(),router=row['router'].tolist())
            result=pipeline.rollout(on_chunk)
            episode_counts[result['task']]+=[int(result['success']),1]
            for _ in range(s['updates_per_episode']):
                metrics=learner.update(mixed_batch(offline,online,s['batch_size'],s['offline_ratio'],'cuda'))
                if 'expert_counts' in metrics:expert_counts+=np.array(metrics['expert_counts'])
                if learner.updates%10==0:log(event='update',updates=learner.updates,**metrics)
            log(event='episode',**result,environment_steps=pipeline.primitive_steps,per_task_success_rate=sampler.success_rates.tolist(),
                expert_utilization=expert_counts.tolist(),clip_stats=clip_stats)
            improved=False
            if episode%s['evaluation_every']==0:
                saved_rng=rng_state()
                scores=evaluate(adapter,pre,learner.actor,env,config,log,seeds=config['stage1_gate']['seeds'],episodes=len(s['validation_initial_states']),max_steps=s['max_steps'],initial_states=s['validation_initial_states'])
                restore_rng(saved_rng);env.seed=config['seed']
                scores.update(training_episode=episode,delta_vs_stage1=scores['overall_success_rate']-marker['overall_success_rate'])
                atomic_json(out/'validation_latest.json',scores)
                atomic_json(out/f'validation-{episode:05d}.json',scores)
                log(event='validation_summary',episode=episode,overall_success_rate=scores['overall_success_rate'],per_task_success_rate=scores['per_task_success_rate'])
                improved=scores['overall_success_rate']>best;best=max(best,scores['overall_success_rate'])
            if improved or episode%s['checkpoint_every']==0 or episode==s['episodes']:
                checkpoint()
                if improved:
                    save(out/'best.pt',dict(format_version=3,stage='stage2',episode=episode,config_hash=config_hash,
                        base_sha256=config['model']['weights_sha256'],stage1_sha256=marker['sha256'],
                        actor=learner.actor.state_dict(),validation_success_rate=best,checkpoint_kind='inference; full learner/replays retained in latest.pt'))
        if not (out/'best.pt').exists():
            raise RuntimeError('No validated Stage II policy checkpoint; evaluate at least once within the training budget')
        actor_changed=[n for n,p in learner.actor.named_parameters() if not torch.equal(initial_actor[n],p)]
        critics_changed=[n for n,p in learner.critics.named_parameters() if not torch.equal(initial_critics[n],p)]
        if not actor_changed or not critics_changed or not any(n.startswith('experts.') for n in actor_changed):
            raise ValueError('Actor/residual experts/critics did not actually update')
        atomic_json(out/'completed.json',dict(config_hash=config_hash,episodes=pipeline.episodes,environment_steps=pipeline.primitive_steps,
            per_task_counts=episode_counts.tolist(),expert_utilization=expert_counts.tolist(),stage1_sha256=marker['sha256'],
            changed_actor_parameters=actor_changed,changed_critic_parameters=critics_changed,
            latest_sha256=digest(out/'latest.pt'),best_sha256=digest(out/'best.pt'),job_id=os.environ.get('SLURM_JOB_ID')))
    finally:env.close();data.close()


def final_evaluation(config,run,config_hash):
    completed=json.loads((run/'stage2/completed.json').read_text())
    if completed['config_hash']!=config_hash or completed['episodes']<config['stage2']['episodes']:
        raise ValueError('Stage II did not complete')
    checkpoint=run/'stage2/best.pt'
    if digest(checkpoint)!=completed['best_sha256']:raise ValueError('Stage II checkpoint changed')
    state=load(checkpoint);check_provenance(state,config_hash,config)
    adapter,pre,data,marker=validated_adapter(config,run,config_hash)
    if state['stage1_sha256']!=marker['sha256']:raise ValueError('Wrong Stage I source')
    learner=make_learner(config);learner.actor.load_state_dict(state['actor']);learner.actor.eval()
    out=run/'evaluation';out.mkdir(exist_ok=True);env=Environment(out/'simulator',settings=config['environment'])
    try:
        e=config['evaluation']
        result=evaluate(adapter,pre,learner.actor,env,config,Log(out/'episodes.jsonl'),seeds=e['seeds'],episodes=e['episodes_per_task'],max_steps=e['max_steps'],initial_states=e['initial_states'])
        result.update(config_hash=config_hash,stage1_sha256=marker['sha256'],stage2_sha256=completed['best_sha256'])
        atomic_json(out/'summary.json',result)
    finally:env.close();data.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['stage1','validate','stage2','evaluate'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text());run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if not run.is_relative_to('/common/users/cs2202'):raise ValueError('Outputs must stay on users storage')
    if not torch.cuda.is_available():raise RuntimeError('Run inside Slurm GPU allocation')
    config_hash=digest(args.config)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(2)
    started=time.time();log=Log(run/(args.stage+'_status.jsonl'))
    log(event='start',stage=args.stage,config_hash=config_hash,gpu=torch.cuda.get_device_name(),gpu_memory=torch.cuda.get_device_properties(0).total_memory)
    try:
        if digest(config['model']['weights'])!=config['model']['weights_sha256']:raise ValueError('Source weight hash changed')
        for key in ['normalization','tokenizer']:
            if digest(config['model'][key])!=config['model'][key+'_sha256']:raise ValueError(f'{key} hash changed')
        if args.stage=='stage1':train_stage1(config,run,config_hash,args.resume)
        elif args.stage=='validate':validate_stage1(config,run,config_hash)
        elif args.stage=='stage2':train_stage2(config,run,config_hash,args.resume)
        else:final_evaluation(config,run,config_hash)
        log(event='exit',exit_status=0,elapsed_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        log(event='failure',exit_status=1,traceback=traceback.format_exc());raise


if __name__=='__main__':main()
