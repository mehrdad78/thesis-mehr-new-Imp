"""Run a small, isolated 2-shard / 8-node AERO experiment and check completion.

Pass a compiled emulator and a trained 2-shard, 1-digit checkpoint. This script
creates experiment files in a new directory and never edits paramsConfig.json.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import math
from aero.environment import reward_components


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exe', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--keygen', required=True)
    p.add_argument('--output', default='tmp/aero-smoke/run')
    args = p.parse_args()
    root, exe, checkpoint = Path.cwd(), Path(args.exe).resolve(), Path(args.checkpoint).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(Path(args.keygen).resolve()), '-dir', str(out / 'keys'),
                    '-shards', '2', '-nodes', '4'], check=True)
    reservations, ports = [], []
    for _ in range(9):
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        reservations.append(s)
        ports.append(s.getsockname()[1])
    ips = {str(s): {str(n): f'127.0.0.1:{ports[s * 4 + n]}' for n in range(4)}
           for s in range(2)}
    ips['2147483647'] = {'0': f'127.0.0.1:{ports[8]}'}
    (out / 'ipTable.json').write_text(json.dumps(ips))
    with (out / 'trace.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['block', 'hash', 'nonce', 'from', 'to', 'gas', 'isError', 'status', 'value'])
        # Prefix activity changes midway through the run; later accounts also use
        # prefixes that may already have migrated.
        for i in range(48):
            a, b = ('a', 'b') if i < 24 else ('c', 'd')
            sender = '0x' + a + f'{i % 4:039x}'
            recipient = '0x' + b + f'{i % 4:039x}'
            w.writerow([0, str(i), i, sender, recipient, 0, 0, 0, 1])
    config = dict(ConsensusMethod=4, PbftViewChangeTimeOut=120000,
                  ExpDataRootDir='results', Block_Interval=1000, BlockSize=8,
                  BlocksizeInBytes=200000, UseBlocksizeInBytes=0, InjectSpeed=4,
                  TotalDataSize=48, TxBatchSize=8, BrokerNum=0, RelayWithMerkleProof=0,
                  DatasetFile='trace.csv', ReconfigTimeGap=10, Delay=0, JitterRange=0,
                  Bandwidth=10000000,
                  AERO=dict(Python=sys.executable, Checkpoint=str(checkpoint),
                            KeyDir=str(out / 'keys'), PrefixDigits=1, EpochBlocks=2,
                            MaxMoves=3, HistoryWindow=32, Seed=1, TimeoutSeconds=90,
                            OnlineTraining=True))
    (out / 'paramsConfig.json').write_text(json.dumps(config))
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    for s in reservations:
        s.close()
    processes, logs = [], []
    try:
        for s in range(2):
            for n in range(4):
                log = (out / f's{s}n{n}.log').open('w')
                logs.append(log)
                processes.append(subprocess.Popen([str(exe), '-S', '2', '-N', '4',
                                                    '-s', str(s), '-n', str(n)],
                                                   cwd=out, env=env, stdout=log, stderr=log,
                                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
        log = (out / 'supervisor.log').open('w')
        logs.append(log)
        supervisor = subprocess.Popen([str(exe), '-S', '2', '-N', '4', '-c'], cwd=out,
                                      env=env, stdout=log, stderr=log,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        processes.append(supervisor)
        deadline = time.monotonic() + 150
        while supervisor.poll() is None and time.monotonic() < deadline:
            for i, process in enumerate(processes[:-1]):
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'node {i} exited with {process.returncode}; inspect {out}')
            time.sleep(.5)
        if supervisor.poll() is None:
            raise RuntimeError(f'experiment timed out; inspect {out}')
        if supervisor.returncode:
            raise RuntimeError(f'supervisor failed; inspect {out}')
        records = [json.loads(line) for line in
                   (out / 'results/result/aero_epochs.jsonl').read_text().splitlines()]
        if not records or not any(r['plan']['moves'] for r in records):
            raise AssertionError('experiment did not perform a prefix migration')
        text = (out / 'supervisor.log').read_text()
        if 'AERO injected 48 transactions' not in text:
            raise AssertionError('not all trace transactions were injected')
        # BlockEmulator's measurement must account for every completed transfer.
        with (out / 'results/result/supervisor_measureOutput/Tx_number.csv').open() as f:
            counts = list(csv.DictReader(f))
        completed = sum(int(r['Normal tx # in this epoch']) + int(r['Relay2 tx # in this epoch'])
                        for r in counts)
        if completed != 48:
            raise AssertionError(f'{completed} completed transfers; expected 48')
        with (out / 'results/result/aero_rewards.csv').open() as f:
            rewards = list(csv.DictReader(f))
        if [int(r['epoch']) for r in rewards] != list(range(len(records) + 1)):
            raise AssertionError('reward epochs are missing or duplicated')
        if rewards[-1]['final_epoch'] != 'true' or rewards[0]['action_epoch'] != '':
            raise AssertionError('reward baseline/final epoch attribution is incorrect')
        for record, row in zip(records, rewards):
            expected = reward_components(record['snapshot']['current'],
                                         float(row['cross_weight']), float(row['variance_weight']))
            if not math.isclose(float(row['reward']), expected['reward'], abs_tol=1e-12):
                raise AssertionError('logged reward differs from the Python PPO reward')
        print(json.dumps({'epochs': len(records),
                          'prefix_moves': sum(len(r['plan']['moves']) for r in records),
                          'completed_transfers': completed,
                          'reward_rows': len(rewards),
                          'output': str(out)}, indent=2))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
