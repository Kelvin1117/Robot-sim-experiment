"""Task-gated correction experiment; not an exact DyGRO reproduction."""
import torch
from torch import nn

from dygro.policy import ResidualPolicy
from dygro.representation import frozen_parameters
from dygro.critic import bellman_target, calibration_loss
from dygro.trainer import ResidualLearner
from regularization import chunk_bc


class RetainedPolicy(nn.Module):
    def __init__(self, parent, correction, targets=(0, 6, 9)):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.correction = correction
        self.targets = tuple(targets)
        self._task = None
        self.enabled = True

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def set_task(self, task):
        self._task = int(task)

    def target_mask(self, task_ids):
        return torch.stack([task_ids == t for t in self.targets]).any(0)

    def parent_action(self, latent, base):
        with torch.no_grad():
            return self.parent.sample(latent, base, deterministic=True)['action']

    def sample(self, latent, base, deterministic=False, valid_mask=None, task_ids=None):
        parent_action = self.parent_action(latent, base)
        if task_ids is None:
            if self._task is None:
                raise ValueError('Task gate requires the environment task or replay task IDs')
            task_ids = torch.full((len(base),), self._task, device=base.device)
        selected = self.target_mask(task_ids) & self.enabled
        action = parent_action.clone()
        log_prob = base.new_zeros(len(base))
        if selected.any():
            mask = valid_mask[selected] if valid_mask is not None else None
            noise = torch.zeros_like(parent_action[selected]) if deterministic else None
            row = self.correction.sample(latent[selected], parent_action[selected], deterministic, mask, noise=noise)
            action[selected] = row['action']
            log_prob[selected] = row['log_prob']
        return dict(action=action, residual=action-parent_action, log_prob=log_prob,
                    parent_action=parent_action, selected=selected)


def make_policy(parent, config, recovery):
    s, f = config['stage2'], config['fusion']
    correction = ResidualPolicy(f['width'], s['embedding_dim'], 10, 4, 2,
                                config['model']['horizon'], 7, s['hidden'], s['temperature'],
                                recovery['correction_scale'], [-5., -2.])
    # Zero mean gives exactly the old deterministic policy at initialization.
    nn.init.zeros_(correction.std_head[-1].weight)
    nn.init.constant_(correction.std_head[-1].bias, -3.)
    return RetainedPolicy(parent, correction, recovery['target_tasks'])


class RetentionLearner(ResidualLearner):
    def __init__(self, *args, recovery, **kwargs):
        super().__init__(*args, **kwargs)
        self._recovery = recovery

    def critic_step(self, batch):
        s = self.settings
        with torch.no_grad():
            sample = self.actor.sample(batch['next_latent'], batch['next_base'], task_ids=batch['task_id'])
            q = self.target(batch['next_latent'], batch['next_base'], sample['action'])
            target = bellman_target(batch['reward'], batch['duration'], batch['terminated'], q, s.gamma)
        self.critic_optimizer.zero_grad(set_to_none=True)
        values = self.critics(batch['latent'], batch['base'], batch['action'], batch['valid_mask'])
        td = (values-target[:, None]).square().mean()
        offline = batch['offline'].bool()
        cal = values.new_zeros(())
        if offline.any():
            with torch.no_grad():
                action = self.actor.sample(batch['latent'][offline], batch['base'][offline],
                                           task_ids=batch['task_id'][offline])['action']
            policy_q = self.critics(batch['latent'][offline], batch['base'][offline], action,
                                    batch['valid_mask'][offline])
            cal = calibration_loss(policy_q, values[offline], batch['mc_return'][offline])
        loss = td+s.calibration_weight*cal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), s.max_gradient_norm, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        return dict(critic=float(loss.detach()), td=float(td.detach()), calibration=float(cal.detach()))

    def actor_step(self, batch):
        actor, r = self.actor, self._recovery
        self.actor_optimizer.zero_grad(set_to_none=True)
        mask = actor.target_mask(batch['task_id'])
        old = actor.parent_action(batch['latent'], batch['base'])
        # Train the correction on every replay task, but enable it only for targets.
        sample = actor.correction.sample(batch['latent'], old, valid_mask=batch['valid_mask'])
        mean = old+actor.correction.residual_scale*sample['mean'].tanh()
        with frozen_parameters(self.critics):
            q = self.critics(batch['latent'], batch['base'], sample['action'], batch['valid_mask']).mean(-1)
            dimensions = (batch['valid_mask'].sum(-1)*mean.shape[-1]).clamp_min(1)
            rl_rows = r['entropy_temperature']*sample['log_prob']/dimensions-q
            rl = rl_rows[mask].mean() if mask.any() else mean.sum()*0
            # Do not discard successful demonstrations using an unreliable Q filter.
            bc = chunk_bc(mean, batch['action'], batch['valid_mask'], mask & batch['offline'].bool())
            retain = chunk_bc(mean, old, batch['valid_mask'], ~mask)
            trust = chunk_bc(mean, old, batch['valid_mask'], mask)
            loss = rl+r['bc_weight']*bc+r['retention_weight']*retain+r['trust_weight']*trust
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.correction.parameters(), 1., error_if_nonfinite=True)
        self.actor_optimizer.step()
        return dict(actor=float(loss.detach()), rl=float(rl.detach()), bc=float(bc.detach()),
                    retention=float(retain.detach()), trust=float(trust.detach()), actor_gradient_norm=float(norm))


class FocusSampler:
    def __init__(self, probabilities, seed):
        self._probabilities = torch.tensor(probabilities, dtype=torch.float64)
        if len(probabilities) != 10 or abs(sum(probabilities)-1.) > 1e-6:
            raise ValueError('Expected ten normalized task probabilities')
        self._generator = torch.Generator().manual_seed(seed)
        self._counts = torch.zeros((10, 2), dtype=torch.long)

    def sample(self):
        return int(torch.multinomial(self._probabilities, 1, generator=self._generator))

    def probabilities(self):
        return self._probabilities

    def update(self, task, success):
        self._counts[task] += torch.tensor([int(success), 1])

    def state_dict(self):
        return dict(counts=self._counts, generator=self._generator.get_state())

    def load_state_dict(self, state):
        self._counts.copy_(state['counts'])
        self._generator.set_state(state['generator'])


class TaskReplay:
    """Separate per-task rings prevent long failed episodes dominating replay."""
    def __init__(self, replay_class, capacity, offline, seed, probabilities):
        self._buffers = [replay_class(max(1, capacity//10), offline, seed+t) for t in range(10)]
        self._probabilities = torch.tensor(probabilities, dtype=torch.float64)
        self._generator = torch.Generator().manual_seed(seed+100)

    def __len__(self):
        return sum(map(len, self._buffers))

    def add(self, transition):
        self._buffers[int(transition['task_id'])].add(transition)

    def sample(self, count):
        weights = self._probabilities*torch.tensor([len(b)>0 for b in self._buffers])
        tasks = torch.multinomial(weights, count, replacement=True, generator=self._generator)
        rows = [self._buffers[int(t)].sample(int((tasks == t).sum())) for t in tasks.unique()]
        return {k: torch.cat([row[k] for row in rows]) for k in rows[0]}

    def state_dict(self):
        return dict(buffers=[b.state_dict() for b in self._buffers], generator=self._generator.get_state())

    def load_state_dict(self, state):
        for buffer, saved in zip(self._buffers, state['buffers'], strict=True):
            buffer.load_state_dict(saved)
        self._generator.set_state(state['generator'])


def candidate_score(rates, baseline, targets, tolerance):
    protected = [i for i in range(10) if i not in targets]
    if any(rates[i] < baseline[i]-tolerance for i in protected):
        return None
    # No target may regress on validation; improve the weakest target first.
    if any(rates[i] < baseline[i] for i in targets):
        return None
    return (min(rates[i] for i in targets), sum(rates[i] for i in targets)/len(targets))
