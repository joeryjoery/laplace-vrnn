#!/usr/bin/env python

import sys
import os
import yaml

# Specifies where client-states are stored in an experiment-dir.
SAVE_PATH: str = 'client/checkpoint'

# Specifies which config key to override during a sweep
SWEEP_KEY: str = 'parameters'
CONFIG_KEY: str = 'experiment.restore_point.value'
FINETUNE_KEY: str = 'experiment.finetune.value'


if __name__ == "__main__":

    configs = sys.stdin.read().strip().splitlines(keepends=False)

    for group in configs:
        directories = [os.path.dirname(path) for path in group.split()]

        checkpoints = [
            os.path.join(path, SAVE_PATH, file) for path in directories
            for file in os.listdir(os.path.join(path, SAVE_PATH))
            if os.path.exists(os.path.join(path, SAVE_PATH))
        ]

        sweep_config = {
            SWEEP_KEY: {
                CONFIG_KEY: dict(values=checkpoints),
                FINETUNE_KEY: dict(values=[True])
            }
        }

        print(yaml.dump(sweep_config))
