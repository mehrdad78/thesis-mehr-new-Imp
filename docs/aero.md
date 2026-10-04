# AERO on BlockEmulator

This is a research implementation of **AERO: Enhancing Sharding Blockchain via
Deep Reinforcement Learning for Account Migration**, Song et al., WWW 2025,
DOI [10.1145/3696410.3714926](https://doi.org/10.1145/3696410.3714926).
It implements the policy, training, logical migration consensus and account
transfer on this repository's PBFT/Monoxide relay implementation. It does not
claim to reproduce the published performance figures.

## Implemented path

1. Accounts share a group when their hexadecimal addresses have the same prefix.
   Prefixes are configurable (1-3 hexadecimal digits, two by default). Initially
   prefix `p` belongs to physical shard `p % shardCount`. This establishes one
   source shard per prefix, including addresses first observed after migration.
2. The supervisor collects **completed original transfers** from committed blocks:
   intra-shard transfers and the second relay leg, deduplicated by transaction hash.
   It builds the current throughput, cross-execution ratios and variances, current
   prefix/shard cross and intra volumes, previous epoch volumes, and a sliding
   migration history (Section 4, Eq. 5). A relay costs one execution in each shard.
3. Every node has a persistent Python policy worker. Six independent attention
   heads attend from encoded state to encoded migration history. An autoregressive
   categorical decoder selects prefix/STOP and a destination shard; the current
   prefix owner supplies the source field. Invalid destinations, inactive groups,
   and repeated prefixes within a plan are masked. Each migration extends the
   history used by the next decoding step.
4. CPU inference and shared epoch seeds reproduce the same policy output on all
   replicas. A signed logical-shard proposal is agreed using prepare/commit votes
   across **all nodes**, independently checked against each replica's plan and
   model-parameter digest. The quorum is `N - floor((N-1)/3)`; for `N = 3f+1` it is
   `2f+1`. The larger quorum for other memberships preserves quorum intersection
   (the paper specifies 128 nodes). Ed25519 public keys are pinned before startup.
   Votes are deduplicated by identity and buffered across stage orderings. A missing
   logical leader triggers a signed view-change vote after five seconds; a quorum
   rotates to the next leader. The independently derived canonical plan remains
   fixed across views.
5. Once agreed, the existing CLPA **transfer protocol** executes the AERO plan;
   the CLPA partitioning algorithm is not run. It drains in-flight relay rounds,
   transfers account states and pending transactions, and commits the transfer
   through each physical shard's PBFT. Balances and nonces are preserved. Outgoing
   states are deleted from the source's new trie root; historical roots remain.
   Prefix ownership persists so future addresses use the new destination.
6. Injection pauses during reconfiguration. The supervisor resumes after a quorum
   of applied acknowledgements in **each** physical shard, including its node 0,
   which is the upstream emulator's routing destination. Final partial CSV batches
   are flushed. Epochs require the configured block count in every shard.
7. PPO uses GAE, the clipped policy objective, value loss, entropy regularization,
   Adam and gradient clipping. Optional deterministic online learning incorporates
   the next observed epoch's reward for the preceding migration sequence. An update
   runs at an epoch boundary after a full rollout batch has accumulated. Independent
   online replicas are tested for identical updated weights and proposals.

## Resolve ambiguities explicitly

The paper is not an executable specification; its trained weights and complete
training dataset are not supplied with the PDF.

| Paper detail | Choice here |
| --- | --- |
| Eq. 8 gives the positive cross ratio, while the prose says to maximize a reward that reduces it | Reward is `-w1 * crossRatio - w2 * normalizedCrossVariance` |
| Eq. 12 uses variance of cross transaction counts, despite describing workload variance | Use cross-execution count variance, divided by squared mean total shard load for stable reward scale; retain raw variances in the state |
| Appendix B prose and Eq. 14 disagree about query/key/value inputs | State supplies queries; migration history supplies keys and values, following the prose |
| Six attention heads with 256 neurons | Each head has `ceil(256/6)=43` projected dimensions, then a learned output projection back to 256 |
| Eq. 17's three continuous outputs must become valid discrete migrations | Factorized categorical prefix/STOP and target heads; source comes from the prefix table |
| Prefix length, action/history limits, initial placement, PPO clipping and GAE settings are unspecified | Defaults: two digits, 32 moves, 32 history entries, prefix modulo placement, clip 0.2, GAE lambda 0.95 |
| Online update described once per reconfiguration, but batch size is 128 | Accumulate on-policy migration decisions; update at reconfiguration boundaries when 128 decisions are available |

Training defaults use gamma 0.99, six heads, 256 hidden neurons, batch size 128,
minibatch size 4 and learning rate 1e-5 (Appendix C). Each epoch terminates decoding
on STOP or the maximum migration length. Intermediate decoding steps have zero
reward; the next trace epoch supplies the feedback, avoiding future-data leakage.
Epoch measurements remain fixed while decoding the plan; only ownership and
action history change. Discounting advances at epoch boundaries, so longer
migration lists do not inadvertently shorten the learning horizon.
The last training epoch uses its own trace to terminate the episode. Online model
updates are held in memory; the initial checkpoint is never overwritten.

## Setup and train

On Windows, the root `run_aero.bat` accepts the shard count, nodes per shard,
and dataset path. Double-click it for interactive prompts, or run:

```bat
run_aero.bat 4 4 "C:\Projects\block-emulator\selectedTxs_300K.csv"
```

The BAT uses every eligible BlockEmulator CSV row until EOF. It invokes
`aero/run.ps1 -AutoPrepare`, creates fresh identity keys and a localhost port
table for the requested topology, and saves configuration, logs, databases,
reward and standard CSTX/TPS/latency metrics under `expTestAERO/run-.../`.
Relative dataset paths are resolved from the repository root. Go must be on PATH.
Python is selected from `.venv`, the locally available bundled runtime, or PATH;
PyTorch and NumPy must be installed. The current local bundled runtime uses
`tmp/aero-python` automatically when available.

The existing four-shard development checkpoint is reused for four shards.
Other shard counts use `aero/checkpoints/shards-N/policy.pt`. If missing, a small
checkpoint is trained automatically for 128 decisions using at most 2,048
transfers from the chosen dataset, then reused on subsequent runs. This initial
training is for running the implementation, not a reproduction of the paper's
training or results. Online learning remains enabled. The epoch length is 10
blocks with this launcher.

Optional environment variables `AERO_PYTHON`, `AERO_PYTHON_PACKAGES`, and
`AERO_CHECKPOINT` select a custom interpreter, dependencies directory, or trained
checkpoint. A custom checkpoint must match the shard count, two prefix digits,
and history window 32. `AERO_PREPARE_ONLY=1` checks and prepares everything without
starting processes. Ctrl+C stops an active run and its child processes.

From the repository root, install the Python dependencies in your own environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r aero/requirements.txt
$env:GOCACHE = "$PWD\tmp\go-build"
go build -buildvcs=false -o tmp/blockEmulator.exe .
```

Use the same shard count, prefix length and history window for training and every
node. A small development checkpoint is already present locally at
`aero/checkpoints/policy.pt` for **4 shards, two prefix digits**, with the default
256-neuron/six-head architecture. It was trained for only 128 decisions on 2,048
bundled transfers and is a wiring check, not a scientifically trained model.
Checkpoint files and keys are ignored by Git. Generate your own when cloning:

```powershell
.\.venv\Scripts\python.exe -m aero.train `
  --trace selectedTxs_300K.csv --output aero/checkpoints/policy.pt `
  --shards 4 --prefix-digits 2 --epoch-size 100000 --steps 12800
go run -buildvcs=false ./cmd/aero-keygen -shards 4 -nodes 4
```

Key generation refuses to overwrite an existing directory. Local development
keys for the existing 4-shard/4-node IP table have already been generated in this
workspace. On a distributed deployment, provision each replica with only its own
private key and the common public manifest; ID is `shardID * nodesPerShard + nodeID`.
The supervisor identity is `shards * nodesPerShard`.

To reproduce the paper's training scale, supply a separate 10-million-transaction
training trace, a disjoint 1-million-transaction evaluation trace, train with
`--shards 16 --epoch-size 100000`, provision 16x8 nodes in `ipTable.json`, and set
`BlockSize=1000`, `AERO.EpochBlocks=100`. The included dataset is insufficient for
that experiment. Multiple optimization passes and seed runs are needed; there is
no justified claim of convergence from a short wiring run.

## Enable and run

On Windows, `aero/run.ps1` prepares a fresh isolated run, builds the executable,
checks the Python policy, and starts all nodes plus the supervisor with hidden
windows and saved logs. It uses the bundled CSV and up to 300,000 eligible
transfers by default. For a short development run:

```powershell
powershell -ExecutionPolicy Bypass -File aero/run.ps1 `
  -Python .venv/Scripts/python.exe -TotalDataSize 10000 -EpochBlocks 10
```

Remove `-TotalDataSize 10000` to read the full sample. `-PythonPackages` can supply
an additional dependency directory. `-PrepareOnly` checks/builds/prepares without
starting processes. Results and logs are saved under `expTestAERO/run-.../`;
the root configuration and existing databases are preserved. This launcher uses
4 shards/4 nodes by default and requires a matching checkpoint and key directory.
Its 10-block default is for the small sample; `-EpochBlocks 100` selects the paper's
epoch length. The following manual procedure remains available.

Copy the values from `aero/paramsConfig.example.json` into `paramsConfig.json` and
set `AERO.Python` to your Python interpreter's absolute path, such as
`C:/Projects/block-emulator/.venv/Scripts/python.exe`. `ConsensusMethod=4` selects
AERO. Existing methods 0-3 keep their behavior. The original `paramsConfig.json`
has been retained. Configure a fresh `ExpDataRootDir` for every experiment; the
existing emulator does not restore prefix routing or online learner state when
restarting from an old experiment database.

Launch all configured nodes and the supervisor using the usual BlockEmulator
commands (one process for each shard/node pair):

```powershell
.\tmp\blockEmulator.exe -S 4 -N 4 -s 0 -n 0
# Repeat for every s=0..3 and n=0..3, then launch the supervisor:
.\tmp\blockEmulator.exe -S 4 -N 4 -c
```

Alternatively, after setting `paramsConfig.json`, use the existing batch generator
`go run -buildvcs=false . -g -S 4 -N 4` and run the generated launcher. Training,
keys, checkpoint architecture and the IP table must match the launch counts.

The experiment writes normal relay TPS/latency/cross-rate/count metrics plus
`<ExpDataRootDir>/result/aero_epochs.jsonl`, containing agreed input snapshots,
plans and digests. Timed-out logical consensus or transfer fails the experiment
with a specific error, rather than silently injecting into partially updated shards.

### Reward logging

Every run also writes `<ExpDataRootDir>/result/aero_rewards.csv`. Rows contain
`epoch`, `action_epoch`, `reward`, `cross_ratio`, `cross_variance`,
`normalized_cross_variance`, the checkpoint's `cross_weight`/`variance_weight`,
`completed_transfers`, `used_for_online_feedback`, and `final_epoch`.
The reward is `-cross_weight * cross_ratio - variance_weight * normalized_cross_variance`.
Larger values (closer to zero with positive weights) are better.

Migration input `snapshot.epoch=k` contains measurements from execution epoch
`k-1`. The reward row therefore has `epoch=k-1`: it evaluates the preceding
migration, not the new proposal. Epoch 0 is the initial baseline and has an empty
`action_epoch`. Subsequent rows refer to their preceding migration through
`action_epoch`. At normal shutdown, the final execution epoch is recorded even
without a following migration; `final_epoch=true` marks a measurement that was
not passed back into the online learner. Runs without any migration still produce
a final reward row with the actual checkpoint weights. The epoch JSONL records
also include `observed_epoch` and a `reward` object containing these components.
The supervisor log prints the reward at each measurement boundary.
Offline `python -m aero.train` also writes a sibling `policy.rewards.csv` (for an
output checkpoint named `policy.pt`), containing the episode, trace epoch, step
count, exact reward and components at each completed training epoch. The existing
`policy.csv` continues to contain per-batch mean reward and PPO loss.

## Verification

```powershell
go test ./aero ./chain ./consensus_shard/pbft_all ./supervisor/... ./build ./params ./utils
.\.venv\Scripts\python.exe -m unittest aero.test_aero -v
```

The isolated smoke runner creates its own configuration, keys, trace, IP table and
databases. It requires a **2-shard/one-digit** checkpoint and an unused output
directory; it does not edit your root configuration:

```powershell
.\.venv\Scripts\python.exe -m aero.train --trace selectedTxs_300K.csv `
  --output tmp/smoke-policy.pt --shards 2 --prefix-digits 1 `
  --epoch-size 128 --limit 2048 --steps 128 --batch-size 32 `
  --hidden 24 --max-moves 3 --update-epochs 2
go build -buildvcs=false -o tmp/keygen.exe ./cmd/aero-keygen
.\.venv\Scripts\python.exe -m aero.smoke --exe tmp/blockEmulator.exe `
  --keygen tmp/keygen.exe --checkpoint tmp/smoke-policy.pt --output tmp/smoke-run
```

The smoke test requires successful prefix migrations and verifies that all 48
original transfers finish (intra-shard completions plus second relay legs).
Go tests cover prefix ownership, transfer expansion, signatures, quorum ordering,
duplicate votes, equivocation, logical leader rotation, source state deletion and
state preservation through a round-trip migration. Python tests cover statistics,
masked autoregressive decoding, deterministic inference, synchronized online PPO,
finite updates, GAE terminal handling and partial trace epochs.

The upstream `go test ./...` also runs `query.TestFinalResult`, which expects an
already completed experiment database and fails with `No filepath` on a clean
checkout. This was observed before implementation and is not changed here.

## Scope limits

- **Physical node membership is static.** This implements AERO's account migration,
  not live randomized reassignment of validator nodes and their full ledger/state
  bootstrap in Section 3.3. Epoch seeds are reproducible experiment inputs, not a
  distributed unpredictable randomness beacon.
- **The supervisor supplies the global epoch record.** Every replica independently
  validates the policy output and votes on it, but the collection/authentication of
  physical execution records still uses BlockEmulator's supervisor. This does not
  establish the paper's full decentralization/security claims.
- Physical transaction PBFT, relay transport, fixed node-0 routing and view change
  retain upstream emulator assumptions. The added logical signatures do not make
  those upstream messages Byzantine-secure or recover a failed physical routing
  leader. The logical leader failure case is covered independently.
- Identical CPU/PyTorch versions and checkpoint/configuration are required across
  replicas. GPU inference and heterogeneous-library determinism are not claimed.
- The paper's 31.77% throughput improvement, 90 KB model size and timing claims
  are not reproduced. This architecture's saved checkpoint is larger. Full-scale
  training/evaluation, ablations and SPRING/LB-Chain/AERO-S baselines remain separate
  experiments; the existing Relay, Broker and CLPA modes remain available.
