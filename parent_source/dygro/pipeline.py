"""Algorithm 1 control flow, with the missing textual warm-up/replay/targets.

No training runs at import. Real-data validation and a Stage I checkpoint are
required before the user-facing launcher can enter online training.
"""
import random

import torch

from .critic import chunk_return
from .replay import mixed_batch
from .representation import OfflineIBOptimizer


class OfflineStage:
    def __init__(self, adapter, dv_critic, base_optimizer, dv_optimizer, ib_weight):
        self.adapter, self.dv_critic = adapter, dv_critic
        self.optimizer = OfflineIBOptimizer(adapter, dv_critic, base_optimizer, dv_optimizer, ib_weight)
        self.updates = 0

    def train(self, batches, steps, critic_updates, on_step=None):
        self.adapter.train()
        iterator = iter(batches)
        for _ in range(steps):
            try:
                batch, action_mask = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch, action_mask = next(iterator)
            result = self.optimizer.step(batch, action_mask, critic_updates)
            self.updates += 1
            if on_step is not None:
                on_step(self.updates, result)

    def state_dict(self):
        return dict(adapter=self.adapter.state_dict(), dv=self.dv_critic.state_dict(),
                    base_optimizer=self.optimizer.base_optimizer.state_dict(),
                    dv_optimizer=self.optimizer.critic_optimizer.state_dict(),
                    updates=self.updates, ib_weight=self.optimizer.ib_weight)

    def load_state_dict(self, state, strict=True):
        self.adapter.load_state_dict(state['adapter'], strict=strict)
        self.dv_critic.load_state_dict(state['dv'], strict=strict)
        self.optimizer.base_optimizer.load_state_dict(state['base_optimizer'])
        self.optimizer.critic_optimizer.load_state_dict(state['dv_optimizer'])
        self.updates = state['updates']
        self.optimizer.ib_weight = state['ib_weight']


def offline_warmup(learner, offline, online, *, critic_steps, policy_steps, batch_size, device, on_step=None):
    if len(online):
        raise ValueError('Offline warm-up must precede online interaction')
    if not len(offline) or min(critic_steps, policy_steps) < 1:
        raise ValueError('Both actor and critic require offline warm-up with real data')
    for i in range(critic_steps):
        result = learner.critic_step(mixed_batch(offline, online, batch_size, 1., device))
        from .critic import update_target
        update_target(learner.critics, learner.target, learner.settings.target_rate)
        if on_step is not None:
            on_step('critic_warmup', i+1, result)
    for i in range(policy_steps):
        # Continue updating Q as the warm-up actor moves; targets remain tracked.
        result = learner.update(mixed_batch(offline, online, batch_size, 1., device))
        if 'actor' not in result:
            result.update(learner.actor_step(mixed_batch(offline, online, batch_size, 1., device)))
        if on_step is not None:
            on_step('policy_warmup', i+1, result)


class OnlinePipeline:
    """Shared multi-task system. Environment returns true final observations.

    context(observation, instruction) returns frozen z and a fresh normalized
    base chunk. Carrying next_base forward makes the conditional actor density
    and the augmented-state Bellman transition refer to the same base sample.
    """
    def __init__(self, environment, context, to_environment_action, learner, offline, online,
                 sampler, *, batch_size, offline_ratio, updates_per_episode, max_steps,
                 train_initial_states, bootstrap_time_limits, seed, device):
        if set(train_initial_states) & set(range(20)):
            raise ValueError('Keep baseline evaluation initial states 0–19 out of the configured training reset set')
        if not train_initial_states:
            raise ValueError('Training reset states must be explicitly configured')
        self.environment, self.context, self.to_environment_action = environment, context, to_environment_action
        self.learner, self.offline, self.online, self.sampler = learner, offline, online, sampler
        self.batch_size, self.offline_ratio, self.updates_per_episode = batch_size, offline_ratio, updates_per_episode
        self.max_steps, self.train_initial_states = max_steps, train_initial_states
        self.bootstrap_time_limits, self.device = bootstrap_time_limits, device
        self.random = random.Random(seed)
        self.episodes, self.primitive_steps = 0, 0

    def rollout(self, on_chunk=None):
        task = self.sampler.sample()
        initial = self.random.choice(self.train_initial_states)
        observation, info = self.environment.reset(task, initial)
        instruction = info['instruction']
        with torch.no_grad():
            latent, base = self.context(observation, instruction)
        steps, success = 0, False
        while steps < self.max_steps and not success:
            with torch.no_grad():
                sample = self.learner.actor.sample(latent, base)
                env_actions = self.to_environment_action(sample['action'])
            rewards = []
            terminated, truncated = False, False
            for action in env_actions[0]:
                next_observation, reward, terminal, timeout, info = self.environment.step(action)
                if info.get('autoreset', False):
                    if 'final_observation' not in info:
                        raise ValueError('Autoreset lost the terminal observation')
                    next_observation = info['final_observation']
                steps += 1
                rewards.append(float(reward))
                success = bool(info['is_success'])
                terminated = bool(terminal or success)
                truncated = bool(timeout or steps >= self.max_steps)
                if terminated or truncated:
                    break
            with torch.no_grad():
                next_latent, next_base = self.context(next_observation, instruction)
            duration = len(rewards)
            gamma = self.learner.settings.gamma
            reward = sum(gamma**i*r for i,r in enumerate(rewards))
            self.online.add(dict(latent=latent[0], base=base[0], action=sample['action'][0],
                                 next_latent=next_latent[0], next_base=next_base[0], reward=reward,
                                 duration=duration, terminated=terminated or (truncated and not self.bootstrap_time_limits),
                                 truncated=truncated, task_id=task,
                                 valid_mask=torch.arange(base.shape[1]) < duration, mc_return=float('nan')))
            self.primitive_steps += duration
            if on_chunk is not None:
                on_chunk(dict(task=task, steps=steps, duration=duration, reward=reward, success=success,
                              base=base.detach().cpu(), action=sample['action'].detach().cpu(),
                              residual=sample['residual'].detach().cpu(), router=sample['dense'].detach().cpu(),
                              selected=sample['indices'].detach().cpu()))
            latent, base, observation = next_latent, next_base, next_observation
            if terminated or truncated:
                break
        self.sampler.update(task, success)
        self.episodes += 1
        return dict(task=task, success=success, steps=steps, episodes=self.episodes,
                    task_probabilities=self.sampler.probabilities().tolist())

    def train(self, episodes, on_episode=None, on_update=None, on_chunk=None):
        for _ in range(episodes):
            result = self.rollout(on_chunk)
            for _ in range(self.updates_per_episode):
                losses = self.learner.update(mixed_batch(self.offline, self.online, self.batch_size,
                                                          self.offline_ratio, self.device))
                if on_update is not None:
                    on_update(self.learner.updates, losses)
            if on_episode is not None:
                on_episode(result)

    def state_dict(self):
        return dict(episodes=self.episodes, primitive_steps=self.primitive_steps, random=self.random.getstate())

    def load_state_dict(self, state):
        self.episodes, self.primitive_steps = state['episodes'], state['primitive_steps']
        self.random.setstate(state['random'])
