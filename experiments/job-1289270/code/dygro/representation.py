import math
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def frozen_parameters(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


class FusionLayer(nn.Module):
    """Appendix A.1: self, action/proprio cross, gated conditioning cross."""
    def __init__(self, width, heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.action_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.condition_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.condition_gate = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, query, action_proprio, condition, condition_mask):
        if not condition_mask.bool().any(-1).all():
            raise ValueError('Each observation needs at least one valid conditioning token')
        q = self.norms[0](query)
        query = query + self.self_attention(q, q, q, need_weights=False)[0]
        query = query + self.action_attention(self.norms[1](query), action_proprio, action_proprio, need_weights=False)[0]
        update = self.condition_attention(self.norms[2](query), condition, condition,
                                          key_padding_mask=~condition_mask.bool(), need_weights=False)[0]
        query = query + self.condition_gate.sigmoid()*update
        return query + self.mlp(self.norms[3](query))


class InformationBottleneck(nn.Module):
    """Encoder architecture/noise magnitude are port decisions, not published defaults."""
    def __init__(self, condition_dim, action_dim, state_dim, width, queries, layers, heads, latent_std):
        super().__init__()
        if latent_std <= 0:
            raise ValueError('A positive stochastic latent scale must be explicitly selected')
        self.condition = nn.Linear(condition_dim, width)
        self.action = nn.Linear(action_dim, width)
        self.proprio = nn.Linear(state_dim, width)
        self.queries = nn.Parameter(torch.randn(queries, width)/math.sqrt(width))
        self.layers = nn.ModuleList([FusionLayer(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.register_buffer('latent_std', torch.tensor(float(latent_std)))

    def forward(self, condition, action_tokens, state, condition_mask, stochastic=True, noise=None):
        cond = self.condition(condition.float())
        act = self.action(action_tokens.float())
        proprio = self.proprio(state.float())[:, None]
        query = self.queries[None].expand(state.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, torch.cat([act, proprio], 1), cond, condition_mask)
        mean = self.norm(query)
        if not stochastic:
            return mean
        if noise is None:
            noise = torch.randn_like(mean)
        return mean + self.latent_std*noise


class DVCritic(nn.Module):
    def __init__(self, observation_dim, latent_dim, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim+latent_dim, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, observation, latent):
        return self.network(torch.cat([observation, latent], -1)).squeeze(-1)

    def estimate(self, observation, latent, permutation):
        if observation.shape[0] < 2:
            raise ValueError('DV product-of-marginals estimate needs at least two samples')
        if permutation.shape != (observation.shape[0],):
            raise ValueError('Invalid permutation shape')
        joint = self(observation, latent)
        marginal = self(observation, latent[permutation])
        return joint.mean() - (torch.logsumexp(marginal, 0)-math.log(len(marginal)))


class OfflineIBOptimizer:
    """Appendix B supremum: critic ascent, then encoder/action-head descent."""
    def __init__(self, adapter, critic, base_optimizer, critic_optimizer, ib_weight):
        self.adapter, self.critic = adapter, critic
        self.base_optimizer, self.critic_optimizer = base_optimizer, critic_optimizer
        self.ib_weight = ib_weight

    def step(self, batch, action_mask, critic_updates=1):
        if critic_updates < 1:
            raise ValueError('At least one DV critic ascent step is required')
        count = batch['action'].shape[0]
        permutation = torch.randperm(count, device=batch['action'].device)
        # Keep the same permutation for both optimization directions in this step.
        with torch.no_grad():
            observation, latent = self.adapter.ib_samples(batch, stochastic=True)
        critic_value = None
        for _ in range(critic_updates):
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_value = self.critic.estimate(observation.detach(), latent.detach(), permutation)
            (-critic_value).backward()
            self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.base_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.critic):
            observation, latent, flow_loss = self.adapter.offline_loss(batch, action_mask)
            dv = self.critic.estimate(observation, latent, permutation)
            loss = flow_loss + self.ib_weight*dv
            loss.backward()
        self.base_optimizer.step()
        return dict(loss=float(loss.detach()), flow_loss=float(flow_loss.detach()),
                    dv=float(dv.detach()), dv_ascent=float(critic_value.detach()))
