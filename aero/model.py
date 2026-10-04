"""Appendix B: state queries, six-head attention over migration history.

The paper's prose and Eq. 14 disagree on Q/K/V. We follow the prose.
Heads use independent 43-dimensional projections so a 256-wide model can
have six heads without the divisibility restriction of MultiheadAttention.
"""
import math
import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, shards, prefixes, hidden=256, heads=6, history_window=32):
        super().__init__()
        self.shards, self.prefixes = shards, prefixes
        self.hidden, self.heads, self.window = hidden, heads, history_window
        self.head_dim = math.ceil(hidden / heads)
        self.state_encoder = nn.Sequential(nn.Linear(9, hidden), nn.Tanh(),
                                           nn.Linear(hidden, hidden), nn.Tanh())
        self.history_encoder = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        width = heads * self.head_dim
        self.query = nn.Linear(hidden, width)
        self.key = nn.Linear(hidden, width)
        self.value = nn.Linear(hidden, width)
        self.output = nn.Linear(width, hidden)
        self.context = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
        self.prefix_score = nn.Linear(hidden, 1)
        self.stop_score = nn.Linear(hidden, 1)
        self.target_score = nn.Linear(hidden, shards)
        self.critic = nn.Linear(hidden, 1)

    def encode(self, obs):
        tokens = torch.as_tensor(obs['tokens'], dtype=torch.float32).reshape(
            self.prefixes, self.shards, 9)
        encoded = self.state_encoder(tokens)
        pooled = encoded.mean((0, 1))
        history = obs.get('history', [])[-self.window:]
        hist = [[a['sender'] / max(1, self.shards - 1),
                 a['receiver'] / max(1, self.shards - 1),
                 a['prefix'] / max(1, self.prefixes - 1),
                 (i + 1) / max(1, len(history))] for i, a in enumerate(history)]
        # A zero sentinel gives a well-defined initial attention context.
        h = self.history_encoder(torch.tensor(hist or [[0., 0., 0., 0.]]))
        q = self.query(pooled).reshape(self.heads, self.head_dim)
        k = self.key(h).reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        v = self.value(h).reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        scores = torch.einsum('hd,hld->hl', q, k) / math.sqrt(self.head_dim)
        attention = torch.einsum('hl,hld->hd', scores.softmax(-1), v).reshape(-1)
        context = self.context(torch.cat([pooled, self.output(attention)]))
        groups = encoded.mean(1) + context
        return groups, context

    def distributions(self, obs, prefix=None):
        groups, context = self.encode(obs)
        logits = torch.cat([self.prefix_score(groups).flatten(),
                            self.stop_score(context).flatten()])
        mask = torch.tensor(obs['eligible'], dtype=torch.bool)
        logits[:self.prefixes] = logits[:self.prefixes].masked_fill(~mask, -1e9)
        pd = Categorical(logits=logits)
        td = None
        if prefix is not None and prefix < self.prefixes:
            targets = self.target_score(groups[prefix])
            source = obs['assignment'][prefix]
            allowed = torch.arange(self.shards) != source
            td = Categorical(logits=targets.masked_fill(~allowed, -1e9))
        return pd, td, self.critic(context).squeeze()

    def act(self, obs, deterministic=False):
        pd, _, value = self.distributions(obs)
        prefix = int(pd.logits.argmax()) if deterministic else int(pd.sample())
        logprob, entropy = pd.log_prob(torch.tensor(prefix)), pd.entropy()
        target = -1
        if prefix < self.prefixes:
            _, td, _ = self.distributions(obs, prefix)
            target = int(td.logits.argmax()) if deterministic else int(td.sample())
            logprob = logprob + td.log_prob(torch.tensor(target))
            entropy = entropy + td.entropy()
        return (prefix, target), logprob, entropy, value

    def evaluate(self, obs, action):
        prefix, target = action
        pd, td, value = self.distributions(obs, prefix)
        lp, entropy = pd.log_prob(torch.tensor(prefix)), pd.entropy()
        if td is not None:
            lp, entropy = lp + td.log_prob(torch.tensor(target)), entropy + td.entropy()
        return lp, entropy, value
