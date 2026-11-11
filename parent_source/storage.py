import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.json.partial')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pt.partial')
    with temp.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def promote(source,target):
    target=Path(target);temp=target.with_suffix('.link.partial')
    if temp.exists():temp.unlink()
    os.link(source,temp);os.replace(temp,target)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)
