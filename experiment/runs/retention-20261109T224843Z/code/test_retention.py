import copy
import unittest

import torch

from dygro.policy import ResidualPolicy
from dygro.critic import CriticEnsemble
from dygro.replay import ReplayBuffer
from dygro.trainer import RLSettings
from retention import FocusSampler, RetainedPolicy, RetentionLearner, TaskReplay, candidate_score


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        parent = ResidualPolicy(8, 4, 10, 2, 1, 2, 3, 12, .1, .1, [-5, 1])
        child = ResidualPolicy(8, 4, 10, 2, 1, 2, 3, 12, .1, .2, [-5, -2])
        self.actor = RetainedPolicy(parent, child)
        self.latent, self.base = torch.randn(10, 2, 8), torch.randn(10, 2, 3)
        self.ids = torch.arange(10)

    def test_initial_identity_and_rng(self):
        torch.manual_seed(11)
        expected = self.actor.parent.sample(self.latent, self.base, deterministic=True)['action']
        expected_rng = torch.get_rng_state()
        torch.manual_seed(11)
        actual = self.actor.sample(self.latent, self.base, deterministic=True, task_ids=self.ids)['action']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_trained_correction_cannot_change_other_tasks(self):
        for expert in self.actor.correction.experts:
            expert[-1].bias.data.fill_(2)
        torch.manual_seed(11)
        expected = self.actor.parent.sample(self.latent, self.base, deterministic=True)['action']
        torch.manual_seed(11)
        actual = self.actor.sample(self.latent, self.base, deterministic=True, task_ids=self.ids)['action']
        selected = self.actor.target_mask(self.ids)
        self.assertTrue(torch.equal(actual[~selected], expected[~selected]))
        self.assertTrue((actual[selected]-expected[selected]).abs().max() > .1)
        self.assertLessEqual(float((actual-expected).abs().max()), .2)

    def test_real_update_freezes_parent_and_changes_correction(self):
        critics = CriticEnsemble(8, 2, 3, 12, 1, 2)
        settings = RLSettings(.99, .1, .1, .01, 1., .01, .005, 1, 1, 'off', False, 1.)
        learner = RetentionLearner(self.actor, critics,
            torch.optim.Adam(self.actor.correction.parameters(), lr=.001),
            torch.optim.Adam(critics.parameters(), lr=.001), settings,
            recovery=dict(entropy_temperature=.01, bc_weight=10., retention_weight=20., trust_weight=.1))
        before = copy.deepcopy(self.actor.parent.state_dict())
        child_before = copy.deepcopy(self.actor.correction.state_dict())
        batch = dict(latent=self.latent, base=self.base, task_id=self.ids,
            next_latent=self.latent+.1, next_base=self.base+.1, action=self.base+.1,
            valid_mask=torch.ones(10, 2, dtype=torch.bool), reward=torch.ones(10), duration=torch.full((10,), 2),
            terminated=torch.zeros(10, dtype=torch.bool), mc_return=torch.ones(10), offline=torch.ones(10, dtype=torch.bool))
        metrics = learner.update(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(all(torch.equal(v, self.actor.parent.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.actor.parent.parameters()))
        self.assertTrue(any(not torch.equal(v, self.actor.correction.state_dict()[k]) for k,v in child_before.items()))

    def test_unknown_task_rejected(self):
        with self.assertRaises(ValueError):
            self.actor.sample(self.latent[:1], self.base[:1])

    def test_checkpoint_selection_rejects_regression(self):
        baseline = [.8]*10
        better = baseline.copy()
        for t in (0, 6, 9):
            better[t] = .9
        self.assertGreater(candidate_score(better, baseline, [0,6,9], 0), (.8,.8))
        better[2] = .75
        self.assertIsNone(candidate_score(better, baseline, [0,6,9], 0))
        better[2] = .8
        better[9] = .7
        self.assertIsNone(candidate_score(better, baseline, [0,6,9], 0))

    def test_sampler_resume(self):
        sampler = FocusSampler([.1]*10, 7)
        state = sampler.state_dict()
        expected = [sampler.sample() for _ in range(10)]
        sampler.load_state_dict(state)
        self.assertEqual(expected, [sampler.sample() for _ in range(10)])

    def test_replay_task_balance_and_resume(self):
        replay = TaskReplay(ReplayBuffer, 100, True, 7, [.1]*10)
        for task in range(10):
            for _ in range(1 if task else 10):
                replay.add(dict(latent=torch.ones(2,8), base=torch.zeros(2,3), action=torch.ones(2,3),
                    next_latent=torch.ones(2,8), next_base=torch.zeros(2,3), reward=1., duration=2,
                    terminated=False, truncated=False, task_id=task, valid_mask=torch.ones(2,dtype=torch.bool), mc_return=1.))
        saved = replay.state_dict()
        ids = replay.sample(1000)['task_id']
        self.assertTrue(all(60 < int((ids == t).sum()) < 140 for t in range(10)))
        replay.load_state_dict(saved)
        self.assertTrue(torch.equal(ids, replay.sample(1000)['task_id']))


if __name__ == '__main__':
    unittest.main()
