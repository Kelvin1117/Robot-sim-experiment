"""Reward reconstruction and inference reproducibility checks."""
import hashlib

import numpy as np
import torch


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def observation_hash(raw):
    h = hashlib.sha256()
    for key in sorted(raw):
        value = np.ascontiguousarray(raw[key])
        h.update(key.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def shaped_returns(success_labels, dones, gamma):
    reward = np.where(np.asarray(success_labels) > 0, 1., -.01).astype(np.float64)
    mc = np.empty_like(reward)
    value = 0.
    for i in reversed(range(len(reward))):
        value = reward[i] + gamma*(not bool(dones[i]))*value
        mc[i] = value
    return reward, mc


def rebuild_demo_rewards(rows, data, gamma):
    rebuilt = []
    index = 0
    for task in range(10):
        for episode in data.episode_ids(task):
            group = data.files[task]['data'][episode]
            labels = group['rewards'][:]
            dones = group['dones'][:]
            if not (np.all(labels[:-1] == 0) and labels[-1] == 1 and dones[-1]):
                raise ValueError('Expected successful demonstrations with a single terminal success')
            rewards, mc = shaped_returns(labels, dones, gamma)
            for start in range(0, len(labels)-1, data.horizon):
                begin, end = start+1, min(start+1+data.horizon, len(labels))
                if index >= len(rows):
                    raise ValueError('Original replay is incomplete')
                row = dict(rows[index])
                if int(row['task_id']) != task or int(row['duration']) != end-begin:
                    raise ValueError('Replay order or chunk duration does not match HDF5')
                actions = data.pre.normalize(group['actions'][begin:end], 'actions')
                np.testing.assert_allclose(row['action'][:end-begin].numpy(), actions, rtol=1e-5, atol=1e-6)
                old_reward = sum(gamma**i*r for i,r in enumerate(labels[begin:end]))
                if abs(float(row['reward'])-old_reward) > 1e-5:
                    raise ValueError('Stored replay reward disagrees with original HDF5')
                row['reward'] = float(sum(gamma**i*r for i,r in enumerate(rewards[begin:end])))
                row['mc_return'] = float(mc[begin])
                rebuilt.append(row)
                index += 1
    if index != len(rows):
        raise ValueError('Unmatched original replay rows')
    return rebuilt


def verify_repeatability(evaluate, config, settings, adapter, pre, actor, env, log):
    saved_enabled = actor.enabled
    checks = []
    try:
        for enabled in (False, True):
            actor.enabled = enabled
            first = evaluate(config, settings, adapter, pre, actor, env, log, [9101], [24], tasks=[0, 3])
            second = evaluate(config, settings, adapter, pre, actor, env, log, [9101], [24], tasks=[0, 3])
            if first['episodes'] != second['episodes']:
                raise RuntimeError('Repeated identical policy episodes differ; refusing performance evaluation')
            checks.append(dict(correction_enabled=enabled, episodes=2, exact_repeat=True))
        # A zero-output new correction must be an identity even when enabled.
        if all(torch.count_nonzero(expert[-1].weight) == 0 and torch.count_nonzero(expert[-1].bias) == 0
               for expert in actor.correction.experts):
            actor.enabled = False
            first = evaluate(config, settings, adapter, pre, actor, env, log, [9102], [23], tasks=[3])
            actor.enabled = True
            second = evaluate(config, settings, adapter, pre, actor, env, log, [9102], [23], tasks=[3])
            if first['episodes'] != second['episodes']:
                raise RuntimeError('Zero correction does not preserve parent trajectories')
            checks.append(dict(zero_correction_identity=True))
    finally:
        actor.enabled = saved_enabled
    return dict(passed=True, checks=checks, criterion='exact episode metadata, observation/action hashes, action trace and outcome')
