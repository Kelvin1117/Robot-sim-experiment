import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, learner, offline, online, sampler, *, stage_one=None, provenance=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Use a new checkpoint path; never overwrite source checkpoints')
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=1, learner=learner.state_dict(), offline=offline.state_dict(),
                 online=online.state_dict(), sampler=sampler.state_dict(), provenance=provenance,
                 stage_one=stage_one.state_dict() if stage_one is not None else None,
                 random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temp = path.with_suffix(path.suffix+'.partial')
    with temp.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard link is an atomic create-if-absent; no overwrite race at destination.
    os.link(temp, path)
    temp.unlink()


def load_checkpoint(path, learner, offline, online, sampler, *, stage_one=None, expected_provenance=None):
    # Only load trusted local research artifacts: optimizer/RNG state uses pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format_version'] != 1 or (expected_provenance is not None and state['provenance'] != expected_provenance):
        raise ValueError('Checkpoint version/provenance mismatch')
    learner.load_state_dict(state['learner'])
    offline.load_state_dict(state['offline']); online.load_state_dict(state['online']); sampler.load_state_dict(state['sampler'])
    if stage_one is not None:
        stage_one.load_state_dict(state['stage_one'], strict=True)
    random.setstate(state['random']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    return state['provenance']
