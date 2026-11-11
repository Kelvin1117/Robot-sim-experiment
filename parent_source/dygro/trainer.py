from dataclasses import asdict, dataclass

import torch

from .critic import bellman_target, calibration_loss, update_target
from .representation import frozen_parameters


@dataclass
class RLSettings:
    """Numbers must be supplied; the DyGRO paper does not publish these defaults."""
    gamma: float
    calibration_weight: float
    contrastive_weight: float
    load_balance_weight: float
    bc_weight: float
    entropy_temperature: float
    target_rate: float
    target_frequency: int
    actor_frequency: int
    advantage_filter: str
    soft_backup: bool
    max_gradient_norm: float


class ResidualLearner:
    def __init__(self, actor, critics, actor_optimizer, critic_optimizer, settings):
        if settings.advantage_filter not in ('off', 'positive_demo_advantage'):
            raise ValueError('Unknown advantage filter; document the selected port interpretation')
        if min(settings.target_frequency, settings.actor_frequency) < 1:
            raise ValueError('Update frequencies must be positive')
        self.actor, self.critics = actor, critics
        self.target = critics.make_target()
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.settings, self.updates = settings, 0

    def critic_step(self, batch):
        s = self.settings
        with torch.no_grad():
            sample = self.actor.sample(batch['next_latent'], batch['next_base'])
            next_q = self.target(batch['next_latent'], batch['next_base'], sample['action'])
            target = bellman_target(batch['reward'], batch['duration'], batch['terminated'], next_q, s.gamma,
                                    sample['log_prob'] if s.soft_backup else None, s.entropy_temperature)
        self.critic_optimizer.zero_grad(set_to_none=True)
        values = self.critics(batch['latent'], batch['base'], batch['action'], batch['valid_mask'])
        td = (values-target[:, None]).square().mean()
        offline = batch['offline'].bool()
        cal = values.new_zeros(())
        if offline.any() and s.calibration_weight:
            with torch.no_grad():
                actions = self.actor.sample(batch['latent'][offline], batch['base'][offline])['action']
            policy_values = self.critics(batch['latent'][offline], batch['base'][offline], actions,
                                         batch['valid_mask'][offline])
            cal = calibration_loss(policy_values, values[offline], batch['mc_return'][offline])
        loss = td+s.calibration_weight*cal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), s.max_gradient_norm, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        return dict(critic=float(loss.detach()), td=float(td.detach()), calibration=float(cal.detach()))

    def actor_step(self, batch):
        from regularization import actor_objective
        return actor_objective(self,batch,0.)

    def update(self, batch):
        result = self.critic_step(batch)
        self.updates += 1
        if self.updates % self.settings.actor_frequency == 0:
            result.update(self.actor_step(batch))
        if self.updates % self.settings.target_frequency == 0:
            update_target(self.critics, self.target, self.settings.target_rate)
        return result

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critics=self.critics.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    updates=self.updates, settings=asdict(self.settings))

    def load_state_dict(self, state):
        if state['settings'] != asdict(self.settings):
            raise ValueError('Learner settings changed across checkpoint resume')
        for name in ['actor', 'critics', 'target', 'actor_optimizer', 'critic_optimizer']:
            getattr(self, name).load_state_dict(state[name])
        self.updates = state['updates']
