import json
from pathlib import Path
import numpy as np
import pytest
import torch

from dygro.policy import DifficultySampler
from native import OpenPIAdapter
from regularization import actor_objective
from run import stage1_gate_passes,validated_adapter


def test_gate_rejects_old_fifteen_percent_and_single_task_collapse():
    gate=dict(minimum_overall_success_rate=.75,minimum_per_task_success_rate=.2)
    assert not stage1_gate_passes(dict(overall_success_rate=.15,per_task_success_rate=[.15]*10),gate)
    assert not stage1_gate_passes(dict(overall_success_rate=.8,per_task_success_rate=[0.]+[.9]*9),gate)
    assert stage1_gate_passes(dict(overall_success_rate=.8,per_task_success_rate=[.8]*10),gate)
    assert not stage1_gate_passes(dict(overall_success_rate=float('nan'),per_task_success_rate=[.8]*10),gate)


def test_stage2_rejects_unvalidated_stage1_before_loading_checkpoint(tmp_path):
    (tmp_path/'stage1').mkdir()
    (tmp_path/'stage1/validated.json').write_text(json.dumps(dict(environment_gate_passed=False)))
    with pytest.raises(ValueError,match='environment validation'):
        validated_adapter({},tmp_path,'irrelevant')


def test_difficulty_uses_empirical_rates_and_exact_paper_probabilities():
    s=DifficultySampler(3,.9,.2,0.)
    for value in [1,0,1,0]:s.update(0,value)
    s.update(1,1)
    torch.testing.assert_close(s.success_rates,torch.tensor([.5,1.,0.],dtype=torch.float64))
    expected=torch.softmax(torch.tensor([.4,0.,.9],dtype=torch.float64)/.2,0)
    torch.testing.assert_close(s.probabilities(),expected)


def test_negative_advantage_demo_does_not_remove_sac_state_objective():
    # A detached fake critic makes every offline demonstration inferior. SAC must
    # still update all states; only BC is filtered. This catches the previous bug.
    from dygro.critic import CriticEnsemble
    from dygro.policy import ResidualPolicy
    from dygro.trainer import ResidualLearner,RLSettings
    a=ResidualPolicy(6,4,3,4,2,5,2,16,.2,.1,(-5,1))
    q=CriticEnsemble(6,5,2,16,1,2)
    settings=RLSettings(.99,.1,.1,.01,1.,1.,.005,1,1,'positive_demo_advantage',False,1.)
    learner=ResidualLearner(a,q,torch.optim.Adam(a.parameters()),torch.optim.Adam(q.parameters()),settings)
    z=torch.randn(4,3,6);base=torch.zeros(4,5,2)
    batch=dict(latent=z,base=base,action=base,valid_mask=torch.ones(4,5,dtype=torch.bool),offline=torch.ones(4,dtype=torch.bool),task_id=torch.tensor([0,1,2,0]))
    # Huge threshold guarantees zero BC-selected demonstrations.
    result=actor_objective(learner,batch,1e20)
    assert result['filtered_offline_samples']==0 and result['bc']==0
    assert abs(result['rl'])>1e-5 and result['actor_gradient_norm']>0
