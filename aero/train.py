"""PPO with clipped objective, GAE, value loss and entropy regularization."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from aero.environment import TraceEnvironment, read_epochs, reward_components
from aero.model import Policy


def advantages(rewards, values, dones, bootstrap, gamma=.99, lam=.95, boundaries=None):
    result = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.
    for i in reversed(range(len(rewards))):
        nv = bootstrap if i == len(rewards) - 1 else values[i + 1]
        live = 1. - float(dones[i])
        # Time advances at epoch boundaries, not between decoder tokens.
        end_epoch = boundaries is None or boundaries[i]
        discount, trace = (gamma, lam) if end_epoch else (1., 1.)
        delta = rewards[i] + discount * nv * live - values[i]
        gae = delta + discount * trace * live * gae
        result[i] = gae
    return result


def update(model, optimizer, rollout, bootstrap, args, rng=None):
    obs, acts, old_lp, values, rewards, dones = zip(*(row[:6] for row in rollout))
    boundaries = [row[6] if len(row) > 6 else True for row in rollout]
    adv = advantages(rewards, values, dones, bootstrap, args.gamma, args.gae_lambda, boundaries)
    returns = torch.tensor(adv + np.array(values), dtype=torch.float32)
    adv = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
    old_lp = torch.tensor(old_lp)
    loss_value = 0.
    for _ in range(args.update_epochs):
        for ids in (rng or np.random).permutation(len(rollout)).reshape(-1, args.minibatch):
            evaluations = [model.evaluate(obs[i], acts[i]) for i in ids]
            lp, entropy, value = map(torch.stack, zip(*evaluations))
            ratio = (lp - old_lp[ids]).exp()
            pg = torch.maximum(-adv[ids] * ratio,
                               -adv[ids] * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
            vl = .5 * (value - returns[ids]).square().mean()
            loss = pg + args.value_coef * vl - args.entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            loss_value = float(loss.detach())
    return loss_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True)
    p.add_argument('--output', default='aero/checkpoints/policy.pt')
    p.add_argument('--shards', type=int, default=16)
    p.add_argument('--prefix-digits', type=int, default=2)
    p.add_argument('--epoch-size', type=int, default=100000)
    p.add_argument('--limit', type=int)
    p.add_argument('--steps', type=int, default=12800)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--minibatch', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--history-window', type=int, default=32)
    p.add_argument('--max-moves', type=int, default=32)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--value-coef', type=float, default=.5)
    p.add_argument('--entropy-coef', type=float, default=.01)
    p.add_argument('--cross-weight', type=float, default=1.)
    p.add_argument('--variance-weight', type=float, default=1.)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--resume')
    a = p.parse_args()
    if (a.steps < 1 or a.batch_size < 2 or a.minibatch < 1 or
            a.batch_size % a.minibatch or a.steps % a.batch_size):
        p.error('steps must be positive and divisible by batch-size; batch-size by minibatch')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    epochs = list(read_epochs(a.trace, a.prefix_digits, a.epoch_size, a.limit))
    env = TraceEnvironment(epochs, a.shards, a.prefix_digits, a.max_moves,
                           a.history_window, a.cross_weight, a.variance_weight)
    config = dict(shards=a.shards, prefixes=16 ** a.prefix_digits, hidden=a.hidden,
                  heads=a.heads, history_window=a.history_window)
    model = Policy(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate, eps=1e-5)
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=True)
        if ck['config'] != config:
            p.error('resume checkpoint architecture does not match')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
    obs, rollout, total_reward = env.reset(), [], 0.
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode = 1
    with output.with_suffix('.csv').open('w', newline='') as log, \
            output.with_suffix('.rewards.csv').open('w', newline='') as reward_log:
        writer = csv.writer(log)
        writer.writerow(['steps', 'mean_reward', 'loss'])
        reward_writer = csv.writer(reward_log)
        component_names = ['reward', 'cross_ratio', 'cross_variance',
                           'normalized_cross_variance', 'cross_weight', 'variance_weight']
        reward_writer.writerow(['episode', 'epoch', 'steps', *component_names])
        for step in range(a.steps):
            with torch.no_grad():
                action, lp, _, value = model.act(obs)
            nxt, r, done, info = env.step(action)
            rollout.append((obs, action, float(lp), float(value), r, done, info['epoch_end']))
            if info['epoch_end']:
                components = reward_components(env.stats, a.cross_weight, a.variance_weight)
                reward_writer.writerow([episode, info['epoch'], step + 1,
                                        *(components[name] for name in component_names)])
                reward_log.flush()
            if done:
                episode += 1
            obs = env.reset() if done else nxt
            total_reward += r
            if len(rollout) == a.batch_size:
                with torch.no_grad():
                    _, _, bootstrap = model.distributions(obs)
                loss = update(model, optimizer, rollout, float(bootstrap), a)
                writer.writerow([step + 1, total_reward / a.batch_size, loss])
                log.flush()
                print(json.dumps({'steps': step + 1, 'loss': loss}), flush=True)
                rollout, total_reward = [], 0.
    torch.save({'config': config, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'training': vars(a)}, output)


if __name__ == '__main__':
    main()
