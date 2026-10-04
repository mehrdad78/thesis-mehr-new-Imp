import copy
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
from aero.environment import (TraceEnvironment, observation, prefix_of, read_epochs,
                              reward, reward_components, statistics)
from aero.model import Policy
from aero.train import advantages, update
from aero.worker import plan, Session


class AEROTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)
        self.shards, self.prefixes = 2, 16
        self.assign = [p % 2 for p in range(16)]
        self.edges = [(0, 1), (0, 1), (1, 2), (3, 3)]
        self.stats = statistics(self.edges, self.assign, 2, 16)
        self.zero = statistics([], self.assign, 2, 16)
        self.model = Policy(2, 16, hidden=24, heads=6)

    def test_prefix_normalization_and_validation(self):
        self.assertEqual(prefix_of('0xAb00', 2), 171)
        with self.assertRaises(ValueError):
            prefix_of('xyz', 2)

    def test_relay_counts_reward_and_initial_history(self):
        self.assertEqual(np.array(self.stats['cross']).sum(), 6)
        self.assertEqual(np.array(self.stats['intra']).sum(), 1)
        self.assertLess(reward(self.stats), reward(self.zero))
        obs = observation(self.stats, self.zero, self.assign, [])
        self.assertTrue(all(t[5:7] == [0., 0.] for t in obs['tokens']))

    def test_reward_breakdown_matches_training_formula(self):
        stats = {'cross': [[4, 0, 0, 0], [0, 2, 1, 1]],
                 'intra': [[0, 0, 0, 2], [0, 0, 0, 0]]}
        components = reward_components(stats, 2., 3.)
        self.assertAlmostEqual(components['reward'], -2.32)
        self.assertAlmostEqual(components['cross_ratio'], .8)
        self.assertAlmostEqual(components['cross_variance'], 1.5)
        self.assertAlmostEqual(components['normalized_cross_variance'], .24)
        self.assertEqual(components['reward'], reward(stats, 2., 3.))

    def test_group_migration_future_reward_no_future_state(self):
        e = TraceEnvironment([self.edges, [(0, 1)]], shards=2, digits=1, max_moves=2)
        first = copy.deepcopy(e.observe())
        _, r, done, _ = e.step((0, 1))
        self.assertEqual(r, 0.)
        self.assertFalse(done)
        self.assertFalse(e.observe()['eligible'][0])
        _, r, done, _ = e.step((16, -1))
        self.assertEqual(r, 0.)  # next epoch's pair is now local
        self.assertFalse(done)
        self.assertNotEqual(first, e.observe())

    def test_autoregressive_plan_determinism_and_masks(self):
        request = dict(shards=2, prefix_digits=1, seed=4, max_moves=8,
                       assignment=self.assign, history=[], current=self.stats, previous=self.zero)
        a, b = plan(self.model, request), plan(self.model, request)
        self.assertEqual(a, b)
        prefixes = [m['prefix'] for m in a['moves']]
        self.assertEqual(len(prefixes), len(set(prefixes)))
        for m in a['moves']:
            self.assertEqual(m['sender'], self.assign[m['prefix']])
            self.assertNotEqual(m['sender'], m['receiver'])
        request['current'] = self.zero
        self.assertEqual(plan(self.model, request)['moves'], [])

    def test_ppo_updates_attention_and_has_finite_loss(self):
        env = TraceEnvironment([self.edges] * 4, shards=2, digits=1, max_moves=2)
        obs = env.reset()
        rollout = []
        for _ in range(8):
            with torch.no_grad():
                action, lp, _, v = self.model.act(obs)
            nxt, r, done, info = env.step(action)
            rollout.append((obs, action, float(lp), float(v), r, done, info['epoch_end']))
            obs = env.reset() if done else nxt
        before = self.model.state_encoder[0].weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        args = Namespace(gamma=.99, gae_lambda=.95, update_epochs=2, minibatch=4,
                         clip=.2, value_coef=.5, entropy_coef=.01)
        loss = update(self.model, optimizer, rollout, 0., args)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, self.model.state_encoder[0].weight))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))

    def test_gae_terminal_does_not_bootstrap(self):
        a = advantages([1., 2.], [0., 0.], [False, True], 100., gamma=.5, lam=1.)
        np.testing.assert_allclose(a, [2., 2.])

    def test_decoder_tokens_do_not_advance_epoch_discount(self):
        a = advantages([0., 2.], [0., 0.], [False, True], 100.,
                       gamma=.5, lam=.95, boundaries=[False, True])
        np.testing.assert_allclose(a, [2., 2.])

    def test_csv_flushes_short_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'trace.csv'
            p.write_text('sender,recipient\n0xa000,0xb000\n0xc000,0xd000\n0xe000,0xf000\n')
            epochs = list(read_epochs(p, digits=1, epoch_size=2))
            self.assertEqual([len(e) for e in epochs], [2, 1])

    def test_online_replicas_update_identically(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy.pt'
            config = dict(shards=2, prefixes=16, hidden=24, heads=6, history_window=32)
            torch.save({'model': self.model.state_dict(), 'config': config,
                        'training': {'batch_size': 8, 'update_epochs': 1}}, path)
            a, b = Session(path, True), Session(path, True)
            initial = a.fingerprint()
            assignment = list(self.assign)
            history = []
            for epoch in range(1, 18):
                stats = statistics(self.edges, assignment, 2, 16)
                request = dict(epoch=epoch, shards=2, prefix_digits=1, seed=epoch,
                               max_moves=3, assignment=assignment, history=history,
                               current=stats, previous=self.zero)
                pa, pb = a.run(request), b.run(request)
                self.assertEqual(pa, pb)
                for move in pa['moves']:
                    assignment[move['prefix']] = move['receiver']
                history = (history + pa['moves'])[-32:]
            self.assertGreater(a.updates, 0)
            self.assertNotEqual(initial, a.fingerprint())

    def test_measurement_uses_checkpoint_weights_without_training(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy.pt'
            config = dict(shards=2, prefixes=16, hidden=24, heads=6, history_window=32)
            torch.save({'model': self.model.state_dict(), 'config': config,
                        'training': {'cross_weight': 2., 'variance_weight': 3.}}, path)
            session = Session(path, True)
            before = session.fingerprint()
            result = session.run({'operation': 'reward', 'current': self.stats})
            self.assertEqual(result['reward'], reward_components(self.stats, 2., 3.))
            self.assertEqual(session.updates, 0)
            self.assertEqual(session.fingerprint(), before)
            self.assertEqual(session.pending, [])


if __name__ == '__main__':
    unittest.main()
