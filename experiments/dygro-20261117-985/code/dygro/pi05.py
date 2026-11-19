"""PI05 port: observation-side action slots, fusion z, original flow expert.

No VLM is duplicated. This modifies conditioning and must be trained in Stage I;
an untrained adapter must never be reported as the original SFT baseline.
"""
import math

import torch
from torch import nn

from .representation import InformationBottleneck


class PI05IBAdapter(nn.Module):
    def __init__(self, policy, width, queries, layers, heads, latent_std):
        super().__init__()
        self.policy = policy
        backbone_width = policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.action_slots = nn.Parameter(torch.randn(queries, backbone_width)/math.sqrt(backbone_width))
        self.fusion = InformationBottleneck(backbone_width, backbone_width, 8, width, queries, layers, heads, latent_std)
        self.latent_to_prefix = nn.Linear(width, backbone_width)
        self.online_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self.online_frozen:
            super().train(False)
        return self

    def freeze_for_online(self):
        self.online_frozen = True
        self.eval().requires_grad_(False)

    @staticmethod
    def raw_observation_vector(batch):
        # Injective flattening of supplied numeric O, rather than secretly replacing
        # I(O;Z) with I(trainable pooled backbone features; pooled Z).
        pieces = [batch[key].float().flatten(1) for key in ['observation.images.image',
                   'observation.images.image2', 'observation.state']]
        pieces.append(batch['observation.language.tokens'].float()/257152.)
        pieces.append(batch['observation.language.attention_mask'].float())
        return torch.cat(pieces, -1)

    def encode(self, batch, stochastic=True):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        core = self.policy.model
        images, image_masks = self.policy._preprocess_images(batch)
        prefix, mask, attention = core.embed_prefix(images, image_masks, batch['observation.language.tokens'],
                                                    batch['observation.language.attention_mask'])
        original_length = prefix.shape[1]
        slots = self.action_slots[None].expand(prefix.shape[0], -1, -1)
        dtype = core.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix = torch.cat([prefix, slots.to(prefix.dtype)], 1).to(dtype)
        mask = torch.cat([mask, torch.ones(slots.shape[:2], device=mask.device, dtype=torch.bool)], 1)
        attention = torch.cat([attention, torch.zeros(slots.shape[:2], device=mask.device, dtype=torch.bool)], 1)
        outputs, _ = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(make_att_2d_masks(mask, attention)),
            position_ids=mask.long().cumsum(1)-1, inputs_embeds=[prefix, None], use_cache=False)
        condition, action = outputs[0][:, :original_length], outputs[0][:, original_length:]
        return self.fusion(condition, action, batch['observation.state'], mask[:, :original_length], stochastic)

    def ib_samples(self, batch, stochastic=True):
        latent = self.encode(batch, stochastic)
        return self.raw_observation_vector(batch), latent.flatten(1)

    def flow_loss(self, latent, actions, action_mask, noise=None, time=None):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, pad_vector
        core = self.policy.model
        padded = pad_vector(actions, self.policy.config.max_action_dim)
        noise = core.sample_noise(padded.shape, padded.device) if noise is None else noise
        time = core.sample_time(padded.shape[0], padded.device) if time is None else time
        noisy = time[:, None, None]*noise+(1-time[:, None, None])*padded
        target_velocity = noise-padded
        suffix, suffix_mask, suffix_attention, cond = core.embed_suffix(noisy, time)
        prefix = self.latent_to_prefix(latent)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
        prefix_attention = torch.zeros_like(prefix_mask)
        mask = torch.cat([prefix_mask, suffix_mask], 1)
        attention = torch.cat([prefix_attention, suffix_attention], 1)
        dtype = core.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        (_, prediction), _ = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(make_att_2d_masks(mask, attention)),
            position_ids=mask.long().cumsum(1)-1, inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
            use_cache=False, adarms_cond=[None, cond])
        velocity = core.action_out_proj(prediction[:, -core.config.chunk_size:].float())
        element_loss = (velocity[..., :actions.shape[-1]]-target_velocity[..., :actions.shape[-1]]).square()
        return (element_loss*action_mask[..., None]).sum()/(action_mask.sum()*actions.shape[-1]).clamp_min(1)

    def offline_loss(self, batch, action_mask):
        latent = self.encode(batch, stochastic=True)
        return self.raw_observation_vector(batch), latent.flatten(1), self.flow_loss(latent, batch['action'], action_mask)

    @torch.no_grad()
    def base_chunk(self, batch, latent=None, noise=None, adapter_enabled=True):
        if not adapter_enabled:
            return self.policy.predict_action_chunk(batch, noise=noise)
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        core = self.policy.model
        if latent is None:
            latent = self.encode(batch, stochastic=False)
        prefix = self.latent_to_prefix(latent)
        mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
        dtype = core.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        _, cache = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(make_att_2d_masks(mask, torch.zeros_like(mask))),
            position_ids=mask.long().cumsum(1)-1, inputs_embeds=[prefix.to(dtype), None], use_cache=True)
        if noise is None:
            noise = core.sample_noise((len(latent), core.config.chunk_size, core.config.max_action_dim), latent.device)
        action = noise
        for step in range(core.config.num_inference_steps):
            time = torch.full((len(latent),), 1-step/core.config.num_inference_steps, device=latent.device)
            action = action-core.denoise_step(mask, cache, action, time)/core.config.num_inference_steps
        return action[..., :self.policy.config.output_features['action'].shape[0]]
