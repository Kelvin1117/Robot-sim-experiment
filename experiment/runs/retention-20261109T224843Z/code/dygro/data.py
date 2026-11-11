"""Strict primitive-rate episode input. Stride-10 snapshots are rejected."""
import json
from pathlib import Path

import numpy as np
import torch


def load_episode(path, *, require_rl=True):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('frame_stride') != 1 or metadata.get('control_frequency_hz') != 20:
        raise ValueError('Require verified 20Hz primitive-rate data, not sampled world-model snapshots')
    if metadata.get('action_coordinates') != 'libero_osc_pose_command':
        raise ValueError('Unknown action control units')
    episode = dict(np.load(path, allow_pickle=False))
    needed = {'image', 'image2', 'state', 'action'}
    if require_rl:
        needed |= {'reward', 'terminated', 'truncated'}
    if not needed.issubset(episode):
        raise ValueError(f'Incomplete episode fields: {needed-set(episode)}')
    length = len(episode['action'])
    if length < 1:
        raise ValueError('Empty demonstration')
    allowed_lengths = (length+1,) if require_rl else (length,length+1)
    if any(len(episode[k]) not in allowed_lengths for k in ['image','image2','state']):
        raise ValueError('Need aligned observations; RL additionally requires the true final observation')
    if episode['action'].shape != (length,7) or episode['state'].shape[1:] != (8,):
        raise ValueError('Incompatible PI05 state/action dimensions')
    if require_rl:
        if any(len(episode[k]) != length for k in ['reward','terminated','truncated']):
            raise ValueError('Action/reward/outcome timestamps disagree')
        if np.any(episode['terminated'][:-1]) or np.any(episode['truncated'][:-1]):
            raise ValueError('Episode boundary occurs before end of data')
        if not (episode['terminated'][-1] or episode['truncated'][-1]):
            raise ValueError('Missing final outcome; do not infer success from end-of-file')
        if not np.isfinite(episode['reward']).all():
            raise ValueError('Nonfinite reward')
    if not np.isfinite(episode['action']).all() or not np.isfinite(episode['state']).all():
        raise ValueError('Nonfinite demonstration')
    return episode, metadata


def episode_chunks(episode, horizon, gamma):
    actions = torch.as_tensor(episode['action'])
    rewards = torch.as_tensor(episode['reward'], dtype=torch.float32)
    mc = torch.empty_like(rewards)
    continuation = rewards.new_zeros(())
    for i in reversed(range(len(rewards))):
        continuation = rewards[i]+gamma*(not bool(episode['terminated'][i]))*continuation
        mc[i] = continuation
    for start in range(0, len(actions), horizon):
        end = min(start+horizon, len(actions))
        duration = end-start
        chunk = torch.zeros(horizon, 7, dtype=actions.dtype)
        chunk[:duration] = actions[start:end]
        mask = torch.arange(horizon) < duration
        yield dict(start=start, next_index=end, action=chunk, valid_mask=mask, duration=duration,
                   reward=(rewards[start:end]*gamma**torch.arange(duration)).sum(), mc_return=mc[start],
                   terminated=bool(episode['terminated'][end-1]), truncated=bool(episode['truncated'][end-1]))
