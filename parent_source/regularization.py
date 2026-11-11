"""Paper p6 BC/filter: explicit experimental positive-advantage demo BC.

Eq6 SAC remains active on every mixed-replay state. The unspecified actor
filter applies only to demonstration regression, not to the RL state objective.
MSE is from the cited TD3+BC family; this is not claimed to be exact AWAC.
"""
import torch


def demo_filter(offline,demo_q,current_q,threshold):
    return offline.bool() & ((demo_q-current_q).detach()>threshold)


def chunk_bc(mean_action,demo_action,valid_mask,selected):
    errors=(mean_action-demo_action).square()*valid_mask[...,None]
    per_row=errors.sum((1,2))/(valid_mask.sum(-1)*mean_action.shape[-1]).clamp_min(1)
    return per_row[selected].mean() if selected.any() else per_row.sum()*0


def actor_objective(learner,batch,threshold):
    from dygro.representation import frozen_parameters
    s=learner.settings;actor=learner.actor
    learner.actor_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(learner.critics):
        sample=actor.sample(batch['latent'],batch['base'],valid_mask=batch['valid_mask'])
        values=learner.critics(batch['latent'],batch['base'],sample['action'],batch['valid_mask']).mean(-1)
        selected=batch['offline'].bool()
        if s.advantage_filter=='positive_demo_advantage':
            with torch.no_grad():
                demo_q=learner.critics(batch['latent'],batch['base'],batch['action'],batch['valid_mask']).mean(-1)
            selected=demo_filter(batch['offline'],demo_q,values,threshold)
        terms=s.entropy_temperature*sample['log_prob']-values
        rl=terms.mean()
        cl=actor.task_embedding.contrastive_loss(sample['embedding'],batch['task_id'])
        lb=actor.router.load_balance(sample['dense'])
        bc=chunk_bc(batch['base']+actor.residual_scale*sample['mean'].tanh(),batch['action'],batch['valid_mask'],selected)
        loss=rl+s.contrastive_weight*cl+s.load_balance_weight*lb+s.bc_weight*bc
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),s.max_gradient_norm,error_if_nonfinite=True)
    learner.actor_optimizer.step()
    counts=torch.bincount(sample['indices'].detach().flatten(),minlength=len(actor.experts))
    return dict(actor=float(loss.detach()),rl=float(rl.detach()),contrastive=float(cl.detach()),load_balance=float(lb.detach()),bc=float(bc.detach()),
                actor_gradient_norm=float(norm),filtered_offline_samples=int(selected.sum()),expert_counts=counts.cpu().tolist(),router_mean=sample['dense'].detach().mean(0).cpu().tolist())
