import math

import torch
from torch import nn
from torch.nn import functional as F


class TaskEmbedding(nn.Module):
    def __init__(self, latent_dim, embedding_dim, tasks, temperature):
        super().__init__()
        if temperature <= 0:
            raise ValueError('InfoNCE temperature must be positive')
        self.projector = nn.Sequential(nn.Linear(latent_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
        self.prototypes = nn.Embedding(tasks, embedding_dim)
        self.temperature = temperature

    def forward(self, latent):
        return self.projector(latent.mean(1))

    def contrastive_loss(self, embedding, task_ids):
        return F.cross_entropy(embedding @ self.prototypes.weight.T/self.temperature, task_ids.long())


class DynamicRouter(nn.Module):
    def __init__(self, embedding_dim, experts, top_m, hidden):
        super().__init__()
        if not 1 <= top_m <= experts:
            raise ValueError('top_m must be in [1, n_experts]')
        self.logits = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, experts))
        self.top_m = top_m

    def forward(self, embedding):
        dense = self.logits(embedding).softmax(-1)
        selected, indices = dense.topk(self.top_m, dim=-1)
        selected = selected/selected.sum(-1, keepdim=True)
        sparse = torch.zeros_like(dense).scatter(-1, indices, selected)
        return dense, sparse, indices

    @staticmethod
    def load_balance(dense, epsilon=1e-8):
        average = dense.mean(0)
        return (average*(average+epsilon).log()).sum()


class ResidualPolicy(nn.Module):
    """A.2 mean-MoE plus one std head; conditional on recorded base sample.

    This is a squashed Gaussian, not a categorical mixture density. Task IDs
    are deliberately absent from sample()/forward(); only the CL loss sees IDs.
    """
    def __init__(self, latent_dim, embedding_dim, tasks, experts, top_m, horizon,
                 action_dim, hidden, temperature, residual_scale, log_std_bounds):
        super().__init__()
        if residual_scale <= 0 or log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError('Invalid residual distribution bounds')
        self.horizon, self.action_dim = horizon, action_dim
        self.task_embedding = TaskEmbedding(latent_dim, embedding_dim, tasks, temperature)
        self.router = DynamicRouter(embedding_dim, experts, top_m, hidden)
        inputs = latent_dim+horizon*action_dim
        def network():
            return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, horizon*action_dim))
        self.experts = nn.ModuleList([network() for _ in range(experts)])
        self.std_head = network()
        self.log_std_bounds = log_std_bounds
        self.register_buffer('residual_scale', torch.tensor(float(residual_scale)))
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, latent, base):
        if base.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError('Base/residual chunk shape mismatch')
        embedding = self.task_embedding(latent)
        dense, sparse, indices = self.router(embedding)
        inputs = torch.cat([latent.mean(1), base.flatten(1)], -1)
        # Skip globally unselected experts entirely: a zero gradient tensor would
        # still let AdamW weight decay/momentum update an inactive expert.
        means = torch.stack([expert(inputs) if sparse[:, i].any() else torch.zeros_like(base.flatten(1))
                             for i, expert in enumerate(self.experts)], 1)
        mean = (means*sparse[:, :, None]).sum(1).reshape_as(base)
        log_std = self.std_head(inputs).reshape_as(base).clamp(*self.log_std_bounds)
        return mean, log_std, dict(embedding=embedding, dense=dense, sparse=sparse, indices=indices)

    def sample(self, latent, base, deterministic=False, valid_mask=None, noise=None, residual_enabled=True):
        if not residual_enabled:
            # Delta=0 is a point mass: never compute a bogus Gaussian log density.
            return dict(action=base, residual=torch.zeros_like(base), log_prob=None, entropy=None)
        mean, log_std, aux = self(latent, base)
        noise = torch.randn_like(mean) if noise is None else noise
        u = mean if deterministic else mean + log_std.exp()*noise
        residual = self.residual_scale*u.tanh()
        base_log_prob = -0.5*((u-mean)*(-log_std).exp()).square()-log_std-0.5*math.log(2*math.pi)
        log_jacobian = 2*(math.log(2)-u-F.softplus(-2*u))
        element_log_prob = base_log_prob-log_jacobian-self.residual_scale.log()
        if valid_mask is not None:
            element_log_prob = element_log_prob*valid_mask[..., None]
        log_prob = element_log_prob.sum((1, 2))
        return dict(action=base+residual, residual=residual, log_prob=log_prob,
                    entropy=-log_prob, mean=mean, log_std=log_std, pre_tanh=u, **aux)


class DifficultySampler:
    """§4.2 formula with empirical cumulative binary success rates."""
    def __init__(self, tasks, target, temperature, ema_decay):
        if temperature <= 0 or not 0 <= target <= 1 or not 0 <= ema_decay < 1:
            raise ValueError('Invalid difficulty sampling settings')
        self.success_rates = torch.zeros(tasks, dtype=torch.float64)
        self.target, self.temperature, self.ema_decay = target, temperature, ema_decay
        self.counts = torch.zeros(tasks, dtype=torch.long)

    def update(self, task, success):
        if success not in (0, 1, False, True):
            raise ValueError('Success must be binary')
        count = int(self.counts[task])
        self.success_rates[task] = (count*self.success_rates[task]+float(success))/(count+1)
        self.counts[task] += 1

    def probabilities(self):
        return ((self.target-self.success_rates).clamp_min(0)/self.temperature).softmax(0)

    def sample(self, generator=None):
        return int(torch.multinomial(self.probabilities(), 1, generator=generator))

    def state_dict(self):
        return dict(success_rates=self.success_rates.clone(), counts=self.counts.clone(),
                    target=self.target, temperature=self.temperature, ema_decay=self.ema_decay)

    def load_state_dict(self, state):
        self.success_rates.copy_(state['success_rates']); self.counts.copy_(state['counts'])
        self.target, self.temperature, self.ema_decay = state['target'], state['temperature'], state['ema_decay']
