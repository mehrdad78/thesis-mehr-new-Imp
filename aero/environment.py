"""Prefix-group trace replay. No future trace is exposed in observations."""
import copy
import csv
import math
import numpy as np


def prefix_of(address, digits):
    address = address.lower().removeprefix('0x')
    if len(address) < digits or any(c not in '0123456789abcdef' for c in address):
        raise ValueError('invalid hexadecimal account address')
    return int(address[:digits], 16)


def read_epochs(path, digits=2, epoch_size=100000, limit=None):
    """Accept BlockEmulator CSV or a sender,recipient CSV with a header."""
    epoch, count = [], 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = csv.reader(f)
        first = next(rows, None)
        if first is None:
            return
        lower = [s.lower() for s in first]
        simple = 'sender' in lower and ('recipient' in lower or 'receiver' in lower)
        if simple:
            si = lower.index('sender')
            ri = lower.index('recipient' if 'recipient' in lower else 'receiver')
        else:
            si, ri = 3, 4
        import itertools
        for row in rows if simple else itertools.chain([first], rows):
            if len(row) <= max(si, ri):
                continue
            if not simple and (len(row) < 9 or row[6:8] != ['0', '0']):
                continue
            try:
                s, r = prefix_of(row[si], digits), prefix_of(row[ri], digits)
            except ValueError:
                continue
            if row[si].lower() == row[ri].lower():
                continue
            epoch.append((s, r))
            count += 1
            if len(epoch) == epoch_size:
                yield epoch
                epoch = []
            if limit is not None and count >= limit:
                break
    if epoch:
        yield epoch


def statistics(edges, assignment, shards, prefixes, seconds=1.):
    cross = np.zeros((prefixes, shards))
    intra = np.zeros_like(cross)
    for sender, receiver in edges:
        s, r = assignment[sender], assignment[receiver]
        if s == r:
            intra[sender, s] += 1
        else:
            cross[sender, s] += 1
            cross[receiver, r] += 1  # relay costs one execution in each shard
    c, b = cross.sum(0), intra.sum(0)
    load = c + b
    ratio = np.divide(c, load, out=np.zeros_like(c), where=load > 0)
    return {'cross': cross.tolist(), 'intra': intra.tolist(),
            'throughput': (load / max(seconds, 1e-6)).tolist(),
            'ratio': ratio.tolist(), 'variance': ((c - c.mean()) ** 2).tolist()}


def reward_components(stats, cross_weight=1., variance_weight=1.):
    c = np.array(stats['cross']).sum(0)
    b = np.array(stats['intra']).sum(0)
    # Eq. 9 cross ratio and Eq. 12 cross-execution variance, scaled for PPO.
    ratio = float(c.sum() / max(1., c.sum() + b.sum()))
    raw_variance = float(c.var())
    variance = float(raw_variance / max(1., (c + b).mean() ** 2))
    return {'reward': -cross_weight * ratio - variance_weight * variance,
            'cross_ratio': ratio, 'cross_variance': raw_variance,
            'normalized_cross_variance': variance,
            'cross_weight': float(cross_weight), 'variance_weight': float(variance_weight)}


def reward(stats, cross_weight=1., variance_weight=1.):
    return reward_components(stats, cross_weight, variance_weight)['reward']


def observation(stats, previous, assignment, history, used=()):
    prefixes, shards = len(stats['cross']), len(stats['throughput'])
    cross, intra = np.array(stats['cross']), np.array(stats['intra'])
    pc, pi = np.array(previous['cross']), np.array(previous['intra'])
    scale = max(1., float((cross + intra).sum()))
    t = np.array(stats['throughput'])
    t = t / max(1., float(t.max()))
    tokens = []
    for p in range(prefixes):
        for s in range(shards):
            tokens.append([t[s], stats['ratio'][s], stats['variance'][s] / scale ** 2,
                           cross[p, s] / scale, intra[p, s] / scale,
                           pc[p, s] / scale, pi[p, s] / scale,
                           p / max(1, prefixes - 1), s / max(1, shards - 1)])
    active = (cross + intra + pc + pi).sum(1) > 0
    eligible = [bool(active[p]) and p not in used and shards > 1 for p in range(prefixes)]
    return {'tokens': tokens, 'history': copy.deepcopy(history),
            'assignment': list(assignment), 'eligible': eligible}


class TraceEnvironment:
    def __init__(self, epochs, shards=16, digits=2, max_moves=32, history_window=32,
                 cross_weight=1., variance_weight=1.):
        if not epochs or shards < 1 or not 1 <= digits <= 3 or max_moves < 1:
            raise ValueError('nonempty trace, positive shards/moves, prefix digits 1..3 required')
        self.epochs, self.shards, self.prefixes = epochs, shards, 16 ** digits
        self.max_moves, self.window = max_moves, history_window
        self.weights = cross_weight, variance_weight
        self.reset()

    def reset(self):
        self.assignment = [p % self.shards for p in range(self.prefixes)]
        self.index, self.history, self.used, self.moves = 0, [], set(), 0
        self.previous = statistics([], self.assignment, self.shards, self.prefixes)
        self.stats = statistics(self.epochs[0], self.assignment, self.shards, self.prefixes)
        return self.observe()

    def observe(self):
        return observation(self.stats, self.previous, self.assignment, self.history, self.used)

    def step(self, action):
        p, target = action
        if p != self.prefixes:
            if (not self.observe()['eligible'][p] or not 0 <= target < self.shards
                    or target == self.assignment[p]):
                raise ValueError('invalid or repeated migration')
            self.history.append({'sender': self.assignment[p], 'receiver': target, 'prefix': p})
            self.history = self.history[-self.window:]
            self.assignment[p] = target
            self.used.add(p)
            self.moves += 1
        end_epoch = p == self.prefixes or self.moves >= self.max_moves
        if not end_epoch:
            # Keep the measured epoch state fixed while decoding, matching
            # online inference. Ownership/history change; future data is absent.
            return self.observe(), 0., False, {'epoch_end': False}
        self.previous = self.stats
        self.index += 1
        done = self.index >= len(self.epochs)
        self.used, self.moves = set(), 0
        if not done:
            self.stats = statistics(self.epochs[self.index], self.assignment, self.shards, self.prefixes)
        else:
            self.stats = statistics(self.epochs[-1], self.assignment, self.shards, self.prefixes)
        r = reward(self.stats, *self.weights)
        return self.observe(), r, done, {'epoch': self.index, 'reward': r, 'epoch_end': True}
