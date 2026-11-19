import json
from pathlib import Path
import numpy as np
import torch

from dataset import GoalData,TASKS,validate_data
from native import Preprocessor
from regularization import chunk_bc,demo_filter
from storage import load,promote,save

ROOT=Path(__file__).resolve().parents[1]


def real_data():
    config=json.loads((ROOT/'configs/full.json').read_text())
    pre=Preprocessor(config['model'],'cpu')
    return GoalData(config['data']['directory'],pre,config['model']['horizon'],5)


def test_all_real_goal_files_timing_and_outcomes():
    rows=validate_data(ROOT/'data/libero_goal')
    assert len(rows)==10 and all(r['episodes']==50 for r in rows)


def test_real_chunks_shift_post_action_observation_and_stop_at_boundary():
    data=real_data()
    try:
        for task in range(10):
            ep=data.episode_ids(task)[0];g=data.files[task]['data'][ep]
            for start in [0,len(g['actions'])-2]:
                row,mask,begin,end=data.chunk(task,ep,start)
                assert begin==start+1 and end<=len(g['actions'])
                np.testing.assert_allclose(row['action'][0].numpy(),data.pre.normalize(g['actions'][begin],'actions'),atol=1e-6)
                assert int(mask.sum())==end-begin
                if start==len(g['actions'])-2:assert mask.tolist()==[True]+[False]*9
            assert not set(data.episode_ids(task))&set(data.episode_ids(task,True))
    finally:data.close()


def test_real_normalization_gripper_and_action_roundtrip():
    data=real_data()
    try:
        g=data.files[3]['data'][data.episode_ids(3)[0]]
        actions=g['actions'][:].astype(np.float32)
        normalized=torch.tensor(data.pre.normalize(actions,'actions'))
        np.testing.assert_allclose(data.pre.denormalize(normalized).numpy(),actions,atol=1e-6)
        assert actions.shape[1]==7 and np.abs(actions[:,-1]).max()<=1.01
    finally:data.close()


def test_regularization_filters_detached_advantage_and_masks_tail():
    q=torch.tensor([1.,-1.,2.],requires_grad=True)
    selected=demo_filter(torch.tensor([True,True,False]),q,torch.zeros(3),0.)
    assert selected.tolist()==[True,False,False] and not selected.requires_grad
    predicted=torch.ones(3,2,7,requires_grad=True);demo=torch.zeros_like(predicted)
    mask=torch.tensor([[True,False]]*3)
    loss=chunk_bc(predicted,demo,mask,selected);loss.backward()
    assert predicted.grad[0,0].abs().sum()>0
    assert predicted.grad[:,1].abs().sum()==0


def test_atomic_checkpoint_latest_best_remain_independent(tmp_path):
    latest,best=tmp_path/'latest.pt',tmp_path/'best.pt'
    save(latest,{'step':1});promote(latest,best);save(latest,{'step':2})
    assert load(best)['step']==1 and load(latest)['step']==2
