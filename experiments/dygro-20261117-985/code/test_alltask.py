import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dygro.policy import ResidualPolicy
from dygro.critic import CriticEnsemble
from dygro.trainer import RLSettings
from dygro.replay import ReplayBuffer
from retention import RetainedPolicy, RetentionLearner, FocusSampler, TaskReplay, candidate_score
from experiment_checks import shaped_returns, rebuild_demo_rewards


class AllTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        def policy(scale):
            return ResidualPolicy(8,4,10,4,2,2,3,12,.1,scale,[-5,-2])
        old = RetainedPolicy(policy(.1),policy(.2),(0,6,9))
        for expert in old.correction.experts:
            expert[-1].bias.data.fill_(.5)
        self.actor = RetainedPolicy(old,policy(.2),tuple(range(10)))
        self.z, self.base = torch.randn(10,2,8),torch.randn(10,2,3)
        self.ids = torch.arange(10)

    def test_nested_teacher_and_zero_correction_identity_without_rng(self):
        rng = torch.get_rng_state().clone()
        expected = self.actor.parent.sample(self.z,self.base,deterministic=True,task_ids=self.ids)['action']
        actual = self.actor.sample(self.z,self.base,deterministic=True,task_ids=self.ids)['action']
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.actor.set_task(3)
        single = self.actor.sample(self.z[3:4],self.base[3:4],deterministic=True)['action']
        self.assertTrue(torch.allclose(single,actual[3:4]))

    def test_every_task_can_change_but_disabled_correction_is_exact_teacher(self):
        expected = self.actor.sample(self.z,self.base,deterministic=True,task_ids=self.ids)['action']
        for expert in self.actor.correction.experts:
            expert[-1].bias.data.fill_(1.)
        actual = self.actor.sample(self.z,self.base,deterministic=True,task_ids=self.ids)['action']
        self.assertTrue(((actual-expected).abs().flatten(1).max(1).values>0).all())
        self.assertLessEqual(float((actual-expected).abs().max()),.2)
        self.actor.enabled=False
        self.assertTrue(torch.equal(expected,self.actor.sample(self.z,self.base,deterministic=True,task_ids=self.ids)['action']))

    def test_real_learning_cl_gradients_and_frozen_teacher(self):
        settings=RLSettings(.99,.1,.1,.01,1.,.01,.005,1,1,'off',False,1.)
        r=dict(entropy_temperature=.01,bc_weight=10.,retention_weight=20.,trust_weight=1.,contrastive_weight=.1,load_balance_weight=.01)
        learner=RetentionLearner(self.actor,CriticEnsemble(8,2,3,12,1,2),
            torch.optim.Adam(self.actor.correction.parameters(),lr=.001),
            None,settings,recovery=r)
        learner.critic_optimizer=torch.optim.Adam(learner.critics.parameters(),lr=.001)
        before=copy.deepcopy(self.actor.parent.state_dict())
        proto=self.actor.correction.task_embedding.prototypes.weight.detach().clone()
        batch=dict(latent=self.z,base=self.base,task_id=self.ids,next_latent=self.z+.1,next_base=self.base+.1,
            action=self.base+.1,valid_mask=torch.ones(10,2,dtype=torch.bool),reward=torch.ones(10),duration=torch.full((10,),2),
            terminated=torch.zeros(10,dtype=torch.bool),mc_return=torch.ones(10),offline=torch.ones(10,dtype=torch.bool))
        metrics=learner.update(batch)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertIn('contrastive',metrics)
        self.assertIn('load_balance',metrics)
        self.assertFalse(torch.equal(proto,self.actor.correction.task_embedding.prototypes.weight))
        self.assertTrue(all(torch.equal(v,self.actor.parent.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.actor.parent.parameters()))

    def test_difficulty_sampling_and_resume(self):
        sampler=FocusSampler([.1]*10,7)
        for _ in range(10):
            for task in range(10):
                sampler.update(task,task!=3)
        self.assertGreater(float(sampler.probabilities()[3]),float(sampler.probabilities()[0])*10)
        saved=sampler.state_dict()
        expected=[sampler.sample() for _ in range(30)]
        sampler.update(3,True)
        sampler.load_state_dict(saved)
        self.assertEqual(expected,[sampler.sample() for _ in range(30)])

    def test_selection_mean_then_worst_and_regression_guard(self):
        base=[.8]*10
        better=base.copy();better[3]=.9
        self.assertGreater(candidate_score(better,base,range(10),1/30),candidate_score(base,base,range(10),1/30))
        better[9]=.7
        self.assertIsNone(candidate_score(better,base,range(10),1/30))

    def test_reward_and_mc_match_primitive_bruteforce(self):
        labels=np.array([0,0,0,1]);dones=np.array([0,0,0,1])
        reward,mc=shaped_returns(labels,dones,.99)
        np.testing.assert_allclose(reward,[-.01,-.01,-.01,1.])
        for start in range(4):
            self.assertAlmostEqual(mc[start],sum(.99**i*r for i,r in enumerate(reward[start:])))

    def test_rebuild_rewards_checks_replay_and_handles_partial_tail(self):
        groups=[];rows=[]
        for task in range(10):
            labels=np.array([0,0,0,1]);done=np.array([0,0,0,1]);actions=np.arange(8).reshape(4,2).astype(np.float32)
            groups.append({'data':{'demo_0':{'rewards':labels,'dones':done,'actions':actions}}})
            for start in (0,2):
                begin,end=start+1,min(start+3,4);duration=end-begin
                a=torch.zeros(2,2);a[:duration]=torch.tensor(actions[begin:end])
                rows.append(dict(task_id=task,duration=duration,action=a,reward=sum(.99**i*r for i,r in enumerate(labels[begin:end])),mc_return=0.))
        data=SimpleNamespace(files=groups,horizon=2,episode_ids=lambda task:['demo_0'],pre=SimpleNamespace(normalize=lambda x,name:x))
        rebuilt=rebuild_demo_rewards(rows,data,.99)
        self.assertAlmostEqual(rebuilt[0]['reward'],-.01-.0099)
        self.assertAlmostEqual(rebuilt[0]['mc_return'],-.01-.0099+.99**2)
        self.assertEqual(rebuilt[1]['reward'],1.)
        broken=copy.deepcopy(rows);broken[0]['task_id']=9
        with self.assertRaises(ValueError):rebuild_demo_rewards(broken,data,.99)


if __name__=='__main__':
    unittest.main()
