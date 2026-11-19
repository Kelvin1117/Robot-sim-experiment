"""OpenPI π0.5 + DyGRO; a single PyTorch autograd graph after Orbax conversion."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import sentencepiece
import torch
from torch import nn
from safetensors.torch import load_model

from dygro.representation import InformationBottleneck


class Preprocessor:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.stats = json.loads(Path(config['normalization']).read_text())['norm_stats']
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=config['tokenizer'])

    def normalize(self, value, name):
        stats = self.stats[name]
        return (np.asarray(value, dtype=np.float32)-np.array(stats['q01'],np.float32)) / (
            np.array(stats['q99'],np.float32)-np.array(stats['q01'],np.float32)+1e-6)*2-1

    def denormalize(self, actions):
        stats = self.stats['actions']
        lo = actions.new_tensor(stats['q01'])
        hi = actions.new_tensor(stats['q99'])
        # Match OpenPI quantile Unnormalize, including its epsilon convention.
        return (actions+1)/2*(hi-lo+1e-6)+lo

    def frame(self, images, state, instruction, *, flip=True):
        pictures = []
        for image in images:
            image = np.asarray(image, dtype=np.uint8)
            if flip:
                image = image[::-1, ::-1]
            pictures.append(np.asarray(Image.fromarray(image).resize((224,224),Image.Resampling.BILINEAR)).copy())
        state = self.normalize(state,'state')
        prompt = instruction.strip().replace('_',' ').replace('\n',' ')
        if self.config['discrete_state_input']:
            values = np.digitize(state, np.linspace(-1,1,257)[:-1])-1
            prompt = f"Task: {prompt}, State: {' '.join(map(str,values))};\nAction: "
            tokens = self.tokenizer.encode(prompt,add_bos=True)
        else:
            tokens = self.tokenizer.encode(prompt,add_bos=True)+self.tokenizer.encode('\n')
        length = self.config['max_token_len']
        tokens = tokens[:length]
        return dict(images=torch.tensor(np.stack(pictures)).permute(0,3,1,2).float()/127.5-1,
                    state=torch.tensor(state), tokens=torch.tensor(tokens+[0]*(length-len(tokens))),
                    token_mask=torch.arange(length)<len(tokens))

    def batch(self, rows):
        return {k:torch.stack([row[k] for row in rows]).to(self.device) for k in rows[0]}

    def live(self, raw, instruction):
        quat = np.asarray(raw['robot0_eef_quat'], dtype=np.float64)
        quat = quat/np.linalg.norm(quat)
        w = np.clip(quat[3],-1,1)
        denom = math.sqrt(max(0,1-w*w))
        axis_angle = np.zeros(3) if denom<1e-8 else quat[:3]*2*math.acos(w)/denom
        state = np.concatenate([raw['robot0_eef_pos'],axis_angle,raw['robot0_gripper_qpos']])
        return self.batch([self.frame([raw['agentview_image'],raw['robot0_eye_in_hand_image']],state,instruction)])


def load_base(config, device):
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    cfg = Pi0Config(pi05=True, action_horizon=config['horizon'], dtype='float32',
                    max_token_len=config['max_token_len'], discrete_state_input=config['discrete_state_input'])
    # Keep real constructor buffers (RoPE inv_freq / vision position_ids).
    # to_empty(meta) would leave nonpersistent buffers uninitialized, while
    # these buffers are intentionally absent from the checkpoint state_dict.
    from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        core = PI0Pytorch(cfg)
    core.sample_actions = PI0Pytorch.sample_actions.__get__(core, PI0Pytorch)
    core.paligemma_with_expert.gemma_expert.lm_head = None
    core.paligemma_with_expert.paligemma.tie_weights()
    missing, unexpected = load_model(core, config['weights'], strict=True)
    if missing or unexpected:
        raise ValueError((missing,unexpected))
    for name,buffer in core.named_buffers():
        if name.endswith('position_ids'):
            expected=torch.arange(buffer.shape[-1],device=buffer.device)[None]
            if not torch.equal(buffer,expected):raise ValueError(f'Invalid constructor position buffer: {name}')
        if name.endswith('inv_freq'):
            if not torch.isfinite(buffer).all() or not (buffer>0).all() or float(buffer[0])!=1.:
                raise ValueError(f'Invalid constructor RoPE buffer: {name}')
    from safetensors import safe_open
    checked = 0
    with safe_open(config['weights'],framework='pt',device='cpu') as weights:
        params = core.state_dict()
        for name in weights.keys():
            if not torch.equal(params[name].cpu(),weights.get_tensor(name).to(params[name].dtype)):
                raise ValueError(f'Pretrained parameter changed during construction: {name}')
            checked += 1
    core.pretrained_load_audit = dict(stored_tensors_checked=checked,strict_load=True,
        exact_parameter_equality=True,constructor_buffers_checked=True)
    return core.to(device).eval().requires_grad_(False)


class OpenPIAdapter(nn.Module):
    def __init__(self, core, fusion, num_steps):
        super().__init__()
        self.core, self.num_steps = core, num_steps
        width = core.paligemma_with_expert.paligemma.config.text_config.hidden_size
        action_width = core.action_in_proj.out_features
        if fusion['queries'] != core.config.action_horizon:
            raise ValueError('Use one fusion query per executed action slot')
        self.fusion = InformationBottleneck(width,action_width,8,**fusion)
        self.latent_to_prefix = nn.Linear(fusion['width'],width)
        self.online_frozen = False

    def train(self, mode=True):
        super().train(False if self.online_frozen else mode)
        # Frozen perceptual model stays deterministic in both stages.
        self.core.paligemma_with_expert.paligemma.eval()
        return self

    def set_stage1_trainable(self, mode):
        self.core.requires_grad_(False)
        if mode == 'action_expert':
            self.core.paligemma_with_expert.gemma_expert.requires_grad_(True)
            for name in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']:
                getattr(self.core,name).requires_grad_(True)
        elif mode != 'adapter_only':
            raise ValueError('Unsupported explicit Stage I freeze recipe')

    def freeze_for_online(self):
        self.online_frozen=True
        self.eval().requires_grad_(False)

    @staticmethod
    def raw_observation_vector(batch):
        return torch.cat([batch['images'].float().flatten(1),batch['state'].float(),
                          batch['tokens'].float()/257152,batch['token_mask'].float()],-1)

    @torch.no_grad()
    def conditioning(self, batch):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        images = list(batch['images'].unbind(1))+[torch.zeros_like(batch['images'][:,0])]
        valid = torch.ones(len(batch['state']),dtype=torch.bool,device=batch['state'].device)
        prefix,mask,attention = self.core.embed_prefix(images,[valid,valid,~valid],batch['tokens'],batch['token_mask'])
        outputs,cache = self.core.paligemma_with_expert(
            attention_mask=self.core._prepare_attention_masks_4d(make_att_2d_masks(mask,attention)),
            position_ids=mask.long().cumsum(1)-1,inputs_embeds=[prefix,None],use_cache=True)
        return outputs[0].float(),mask,cache

    def encode(self,batch,stochastic=True):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        condition,mask,cache = self.conditioning(batch)
        # pi0.5 has a separate action stream, rather than Qwen action tokens.
        # Read its pretrained hidden states with a fixed zero action probe at t=1.
        # The probe never sees demonstration targets; train/live use the same path.
        probe = torch.zeros(len(condition),self.core.config.action_horizon,32,device=condition.device)
        time = torch.ones(len(condition),device=condition.device)
        suffix,sm,sa,cond = self.core.embed_suffix(torch.nn.functional.pad(batch['state'],(0,24)),probe,time)
        attention = torch.cat([mask[:,None,:].expand(-1,sm.shape[1],-1),make_att_2d_masks(sm,sa)],-1)
        positions = mask.sum(-1)[:,None]+sm.long().cumsum(1)-1
        outputs,_ = self.core.paligemma_with_expert(
            attention_mask=self.core._prepare_attention_masks_4d(attention),position_ids=positions,
            past_key_values=cache,inputs_embeds=[None,suffix],use_cache=False,adarms_cond=[None,cond])
        action = outputs[1][:,-self.core.config.action_horizon:].float()
        return self.fusion(condition,action,batch['state'],mask,stochastic)

    def ib_samples(self,batch,stochastic=True):
        return self.raw_observation_vector(batch),self.encode(batch,stochastic).flatten(1)

    def flow_loss(self,latent,actions,action_mask,noise=None,time=None):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        core=self.core
        actions=torch.nn.functional.pad(actions,(0,32-actions.shape[-1]))
        noise=core.sample_noise(actions.shape,actions.device) if noise is None else noise
        time=core.sample_time(len(actions),actions.device) if time is None else time
        noisy=time[:,None,None]*noise+(1-time[:,None,None])*actions
        suffix,sm,sa,cond=core.embed_suffix(torch.zeros_like(actions[:,0]),noisy,time)
        prefix=self.latent_to_prefix(latent)
        pm=torch.ones(prefix.shape[:2],device=prefix.device,dtype=torch.bool)
        mask=torch.cat([pm,sm],1)
        att=torch.cat([torch.zeros_like(pm),sa],1)
        (_,out),_=core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(make_att_2d_masks(mask,att)),
            position_ids=mask.long().cumsum(1)-1,inputs_embeds=[prefix,suffix],use_cache=False,adarms_cond=[None,cond])
        velocity=core.action_out_proj(out[:,-core.config.action_horizon:].float())
        error=(velocity[...,:7]-(noise-actions)[...,:7]).float().square()
        return (error*action_mask[...,None]).sum()/(7*action_mask.sum()).clamp_min(1)

    def offline_loss(self,batch,mask):
        latent=self.encode(batch,True)
        return self.raw_observation_vector(batch),latent.flatten(1),self.flow_loss(latent,batch['action'],mask)

    @torch.no_grad()
    def base_chunk(self,batch,latent=None,noise=None):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        if latent is None:
            latent=self.encode(batch,False)
        prefix=self.latent_to_prefix(latent)
        mask=torch.ones(prefix.shape[:2],dtype=torch.bool,device=prefix.device)
        _,cache=self.core.paligemma_with_expert(
            attention_mask=self.core._prepare_attention_masks_4d(make_att_2d_masks(mask,torch.zeros_like(mask))),
            position_ids=mask.long().cumsum(1)-1,inputs_embeds=[prefix,None],use_cache=True)
        action=self.core.sample_noise((len(latent),self.core.config.action_horizon,32),latent.device) if noise is None else noise
        state=torch.nn.functional.pad(batch['state'],(0,24))
        for step in range(self.num_steps):
            time=torch.full((len(latent),),1-step/self.num_steps,device=latent.device)
            action=action-self.core.denoise_step(state,mask,cache,action,time)/self.num_steps
        return action[...,:7].float()

    def trainable_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        # Preserve trainable parameters only; frozen base is an immutable hashed reference.
        return {n:t.detach().cpu().clone() for n,t in self.state_dict().items() if n in names}

    def load_delta(self,state):
        expected={n for n,p in self.named_parameters() if p.requires_grad}
        if set(state)!=expected:
            raise ValueError(f'Delta keys mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}')
        current=dict(self.named_parameters())
        with torch.no_grad():
            for name,value in state.items():
                current[name].copy_(value)
