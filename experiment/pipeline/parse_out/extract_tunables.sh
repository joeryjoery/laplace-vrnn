#!/usr/bin/env bash

# parse_output.sh - Parse the output folder of project experiments to find
# configurations trained with a Deterministic or MeanField architecture.
# The checkpoints of these runs can be used to finetune a new agent.

# Usage:    ./parse_output.sh path/to/experiment_output
# Example: ./parse_output.sh experiments/run_123/output

# Arguments:
#   - path/to/experiment_output: Path to the directory containing experiment output.

# This script looks for configuration files in the specified output directory,
# filters out runs that can be fine-tuned, and groups runs by their config data.

# Options:
PATTERN=".yaml"  # Pattern to match configuration files
EXCLUDE="save|wandb|checkpoint|client"  # Patterns to exclude from the results

# Extract all confi files in the given output directory.
configs=$(find "$1" -type f -name "*$PATTERN" | grep -vE "/($EXCLUDE)/")

# Filter out runs that can be fine-tuned
tunable=$(echo "$configs" | python pipeline/parse_out/filter_tunable.py)

# Group runs by their config data (e.g., model, task). Drops duplicates.
grouped=$(echo "$tunable" | python pipeline/parse_out/group_configs.py)

echo "$grouped"
