"""Validate a checkpoint before the PowerShell launcher starts any nodes."""
import sys
from aero.worker import load_policy


def main():
    model, config = load_policy(sys.argv[1])
    if model.shards != int(sys.argv[2]) or model.prefixes != 256 or model.window != 32:
        raise ValueError('checkpoint must match shard count, two prefix digits and history window 32')
    print('Policy loaded:', config)


if __name__ == '__main__':
    main()
