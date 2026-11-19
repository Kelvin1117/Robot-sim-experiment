import random

import torch


REQUIRED = {'latent', 'base', 'action', 'next_latent', 'next_base', 'reward',
            'duration', 'terminated', 'truncated', 'task_id', 'valid_mask', 'mc_return'}


class ReplayBuffer:
    """Chunk transitions with the actual base draw and separate true terminal flag."""
    def __init__(self, capacity, offline, seed):
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive')
        self.capacity, self.offline = capacity, offline
        self._rows, self._cursor = [], 0
        self._random = random.Random(seed)

    def __len__(self):
        return len(self._rows)

    def add(self, transition):
        if not REQUIRED.issubset(transition):
            raise ValueError(f'Missing replay fields: {REQUIRED-set(transition)}')
        if int(transition['duration']) < 1 or int(transition['duration']) > transition['action'].shape[0]:
            raise ValueError('Invalid executed chunk duration')
        if transition['action'].shape != transition['base'].shape or transition['next_base'].shape != transition['base'].shape:
            raise ValueError('Chunk action shapes disagree')
        if self.offline and not torch.isfinite(torch.as_tensor(transition['mc_return'])):
            raise ValueError('Offline Cal-QL data requires an actual Monte Carlo return')
        duration = int(transition['duration'])
        mask = torch.as_tensor(transition['valid_mask']).bool()
        expected = torch.arange(transition['action'].shape[0], device=mask.device) < duration
        if mask.shape != expected.shape or not torch.equal(mask, expected):
            raise ValueError('valid_mask must identify exactly the executed contiguous prefix')
        row = {k: torch.as_tensor(v).detach().cpu().clone() for k, v in transition.items()}
        row['offline'] = torch.tensor(self.offline)
        if len(self._rows) < self.capacity:
            self._rows.append(row)
        else:
            self._rows[self._cursor] = row
        self._cursor = (self._cursor+1) % self.capacity

    def sample(self, count):
        if not self._rows:
            raise ValueError('Cannot sample an empty replay buffer')
        rows = self._random.choices(self._rows, k=count)
        return {k: torch.stack([row[k] for row in rows]) for k in rows[0]}

    def state_dict(self):
        return dict(capacity=self.capacity, offline=self.offline, rows=self._rows,
                    cursor=self._cursor, random_state=self._random.getstate())

    def load_state_dict(self, state):
        if self.capacity != state['capacity'] or self.offline != state['offline']:
            raise ValueError('Replay configuration changed across resume')
        self._rows, self._cursor = state['rows'], state['cursor']
        self._random.setstate(state['random_state'])


def mixed_batch(offline, online, batch_size, offline_ratio, device):
    if not 0 <= offline_ratio <= 1 or batch_size <= 0:
        raise ValueError('Invalid replay mixture')
    count_off = round(batch_size*offline_ratio)
    if not len(online):
        count_off = batch_size
    if count_off and not len(offline):
        raise ValueError('Required offline buffer is empty; do not silently disable RLPD')
    pieces = []
    if count_off:
        pieces.append(offline.sample(count_off))
    if count_off < batch_size:
        pieces.append(online.sample(batch_size-count_off))
    batch = {k: torch.cat([p[k] for p in pieces]).to(device) for k in pieces[0]}
    permutation = torch.randperm(batch_size, device=device)
    return {k: v[permutation] for k, v in batch.items()}
