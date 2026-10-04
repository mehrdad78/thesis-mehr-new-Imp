"""JSON-lines CPU inference service used by every logical-shard replica."""
import json
import hashlib
import sys
import torch
import numpy as np
from argparse import Namespace
from aero.environment import observation
from aero.environment import reward_components
from aero.model import Policy
from aero.train import update


def load_policy(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = Policy(**{k: config[k] for k in
                     ('shards', 'prefixes', 'hidden', 'heads', 'history_window')})
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, config


def plan(model, request, transitions=None):
    if request['shards'] != model.shards or 16 ** request['prefix_digits'] != model.prefixes:
        raise ValueError('checkpoint does not match shard count / prefix digits')
    if request.get('history_window', model.window) != model.window:
        raise ValueError('checkpoint does not match history window')
    assignment = list(request['assignment'])
    if len(assignment) != model.prefixes or any(s < 0 or s >= model.shards for s in assignment):
        raise ValueError('invalid prefix assignment')
    history, used, moves = list(request.get('history') or []), set(), []
    generator = torch.Generator().manual_seed(request['seed'])
    # Seed is epoch-agreed; sampling a trained distribution enables variable plans.
    with torch.no_grad():
        for _ in range(request['max_moves']):
            obs = observation(request['current'], request['previous'], assignment, history, used)
            pd, _, _ = model.distributions(obs)
            prefix = int(torch.multinomial(pd.probs, 1, generator=generator))
            if prefix == model.prefixes:
                if transitions is not None:
                    lp, _, value = model.evaluate(obs, (prefix, -1))
                    transitions.append((obs, (prefix, -1), float(lp), float(value), 0., False, True))
                break
            _, td, _ = model.distributions(obs, prefix)
            target = int(torch.multinomial(td.probs, 1, generator=generator))
            if transitions is not None:
                lp, _, value = model.evaluate(obs, (prefix, target))
                transitions.append((obs, (prefix, target), float(lp), float(value), 0., False, False))
            move = {'sender': assignment[prefix], 'receiver': target, 'prefix': prefix}
            moves.append(move)
            history.append(move)
            history = history[-model.window:]
            assignment[prefix] = target
            used.add(prefix)
    return {'moves': moves}


class Session:
    def __init__(self, path, online=False):
        self.model, _ = load_policy(path)
        self.online = online
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        defaults = dict(gamma=.99, gae_lambda=.95, minibatch=4, update_epochs=4,
                    clip=.2, value_coef=.5, entropy_coef=.01, learning_rate=1e-5,
                    cross_weight=1., variance_weight=1., batch_size=128, seed=1)
        defaults.update(checkpoint.get('training', {}))
        self.args = Namespace(**defaults)
        self.rng = np.random.RandomState(self.args.seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-5)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.pending, self.rollout, self.updates = [], [], 0

    def fingerprint(self):
        h = hashlib.sha256()
        for value in self.model.state_dict().values():
            h.update(value.detach().numpy().tobytes())
        return h.hexdigest()

    def run(self, request):
        feedback = reward_components(request['current'], self.args.cross_weight,
                                     self.args.variance_weight)
        if request.get('operation') == 'reward':
            # Measurement only: do not consume a pending action or train.
            return {'reward': feedback}
        if self.online and self.pending:
            last = self.pending[-1]
            self.pending[-1] = (*last[:4], feedback['reward'], False, True)
            self.rollout.extend(self.pending)
            self.pending = []
            while len(self.rollout) >= self.args.batch_size:
                with torch.no_grad():
                    obs = (self.rollout[self.args.batch_size][0]
                           if len(self.rollout) > self.args.batch_size
                           else observation(request['current'], request['previous'],
                                            request['assignment'], request.get('history') or []))
                    _, _, bootstrap = self.model.distributions(obs)
                update(self.model, self.optimizer, self.rollout[:self.args.batch_size],
                       float(bootstrap), self.args, rng=self.rng)
                self.rollout = self.rollout[self.args.batch_size:]
                self.updates += 1
        result = plan(self.model, request, self.pending if self.online else None)
        result['model_digest'], result['updates'] = self.fingerprint(), self.updates
        result['reward'] = feedback
        return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    session = Session(sys.argv[1], '--online' in sys.argv[2:])
    for line in sys.stdin:
        try:
            result = session.run(json.loads(line))
        except Exception as exc:
            result = {'error': str(exc)}
        print(json.dumps(result, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
