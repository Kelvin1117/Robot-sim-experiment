import copy

import torch
from torch import nn


class ResMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                     nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, x):
        return x+self.network(x)


class ChunkCritic(nn.Module):
    def __init__(self, latent_dim, horizon, action_dim, width, layers):
        super().__init__()
        self.state_projection = nn.Linear(latent_dim, width)
        self.action_projection = nn.Linear(2*action_dim, width)
        self.position = nn.Parameter(torch.randn(horizon, width)*0.02)
        self.blocks = nn.Sequential(*[ResMLPBlock(width) for _ in range(layers)])
        self.pool_score = nn.Linear(width, 1)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, latent, base, action, valid_mask=None):
        # Frozen latent already contains instruction/task semantics; no true task ID.
        state = self.state_projection(latent.mean(1))[:, None]
        tokens = self.action_projection(torch.cat([base, action], -1))+state+self.position[None]
        tokens = self.blocks(tokens)
        scores = self.pool_score(tokens).squeeze(-1)
        if valid_mask is not None:
            if (~valid_mask.bool().any(-1)).any():
                raise ValueError('Each chunk must contain a valid action')
            scores = scores.masked_fill(~valid_mask.bool(), float('-inf'))
        pooled = (scores.softmax(-1)[..., None]*tokens).sum(1)
        return self.output(pooled).squeeze(-1)


class CriticEnsemble(nn.Module):
    def __init__(self, latent_dim, horizon, action_dim, width, layers, count):
        super().__init__()
        if count < 2:
            raise ValueError('An ensemble needs at least two critics')
        self.critics = nn.ModuleList([ChunkCritic(latent_dim, horizon, action_dim, width, layers) for _ in range(count)])

    def forward(self, latent, base, action, valid_mask=None):
        return torch.stack([q(latent, base, action, valid_mask) for q in self.critics], -1)

    def make_target(self):
        return copy.deepcopy(self).eval().requires_grad_(False)


@torch.no_grad()
def update_target(online, target, rate):
    if not 0 <= rate <= 1:
        raise ValueError('Target update rate must be in [0,1]')
    for p, q in zip(online.parameters(), target.parameters(), strict=True):
        q.lerp_(p, rate)
    for p, q in zip(online.buffers(), target.buffers(), strict=True):
        q.copy_(p)


def chunk_return(rewards, valid_mask, gamma):
    if not 0 < gamma <= 1 or rewards.shape != valid_mask.shape:
        raise ValueError('Invalid discount or reward mask')
    if ((~valid_mask.bool()).cummax(-1).values & valid_mask.bool()).any():
        raise ValueError('Primitive reward mask must be a contiguous valid prefix')
    discounts = gamma**torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    return (rewards*valid_mask*discounts).sum(-1)


def bellman_target(reward, duration, terminated, next_q, gamma, next_log_prob=None, entropy_temperature=None):
    if (duration < 1).any():
        raise ValueError('Executed duration must be positive')
    # Eq12 has no entropy term. The optional soft backup is a labelled extension.
    continuation = next_q.min(-1).values
    if next_log_prob is not None:
        if entropy_temperature is None:
            raise ValueError('A soft backup requires an explicit temperature')
        continuation = continuation-entropy_temperature*next_log_prob
    continuation = torch.where(terminated.bool(), torch.zeros_like(continuation), continuation)
    return reward+gamma**duration*continuation


def calibration_loss(policy_values, data_values, monte_carlo_returns):
    """Eq14, not the full log-sum-exp CQL objective from the upstream paper."""
    return (torch.maximum(policy_values, monte_carlo_returns[:, None])-data_values).mean()
