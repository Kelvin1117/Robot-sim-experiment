from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import episode_chunks, load_episode


class GoalDemonstrations(Dataset):
    def __init__(self, files, horizon, preprocessor):
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError('No verified full-rate demonstration files')
        self.horizon, self.preprocessor = horizon, preprocessor
        self.indices = []
        for file in self.files:
            episode, metadata = load_episode(file, require_rl=False)
            if not 0 <= metadata.get('task_id', -1) < 10 or not metadata.get('instruction'):
                raise ValueError('Need Goal task ID and instruction in every episode manifest')
            if metadata.get('image_orientation') not in ('mujoco_raw', 'huggingface_libero'):
                raise ValueError('Explicit image orientation is required')
            self.indices.extend((file, i) for i in range(len(episode['action'])))

    @lru_cache(maxsize=2)
    def episode(self, path):
        return load_episode(path, require_rl=False)

    def frame(self, episode, metadata, index, action=None):
        images = {}
        for source, target in [('image','image'), ('image2','image2')]:
            image = torch.tensor(episode[source][index].copy()).permute(2,0,1).float()/255
            if metadata['image_orientation'] == 'mujoco_raw':
                image = image.flip((1,2))
            images['observation.images.'+target] = image
        sample = {**images, 'observation.state':torch.as_tensor(episode['state'][index]).float(),
                  'task':metadata['instruction']}
        if action is not None:
            sample['action'] = action.float()
        return self.preprocessor(sample)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        file, start = self.indices[item]
        episode, metadata = self.episode(file)
        end = min(start+self.horizon, len(episode['action']))
        action = torch.zeros(self.horizon,7)
        action[:end-start] = torch.as_tensor(episode['action'][start:end])
        prepared = self.frame(episode,metadata,start,action)
        mask = (torch.arange(self.horizon,device=prepared['action'].device) < end-start)[None]
        return prepared, mask

    @staticmethod
    def collate(samples):
        dictionaries, masks = zip(*samples)
        # Model inputs are numeric tensors; task text has already been tokenized.
        batch = {k:torch.cat([row[k] for row in dictionaries]) for k,v in dictionaries[0].items() if torch.is_tensor(v)}
        return batch,torch.cat(masks)

    @torch.no_grad()
    def populate_replay(self, replay, adapter, gamma, stride):
        if stride < 1:
            raise ValueError('Offline replay stride must be explicitly positive')
        for file in self.files:
            episode, metadata = load_episode(file, require_rl=True)
            # episode_chunks generates non-overlapping chunks; shifted starts
            # support a requested primitive sampling stride without crossing reset.
            for start in range(0,len(episode['action']),stride):
                sliced = {k:v[start:] for k,v in episode.items()}
                row = next(episode_chunks(sliced,self.horizon,gamma))
                current = self.frame(episode,metadata,start,row['action'])
                final_index = start+row['next_index']
                following = self.frame(episode,metadata,final_index)
                latent = adapter.encode(current,stochastic=False)
                next_latent = adapter.encode(following,stochastic=False)
                replay.add(dict(latent=latent[0],base=adapter.base_chunk(current,latent)[0],action=current['action'][0],
                                next_latent=next_latent[0],next_base=adapter.base_chunk(following,next_latent)[0],
                                reward=row['reward'],duration=row['duration'],terminated=row['terminated'],
                                truncated=row['truncated'],task_id=metadata['task_id'],valid_mask=row['valid_mask'],
                                mc_return=row['mc_return']))
