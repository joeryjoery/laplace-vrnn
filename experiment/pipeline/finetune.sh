#!/usr/bin/env bash

# Default values
CHUNK_SIZE=4
NUM_AGENTS=1

# Ensure that temporary files are always cleaned
cleanup_tmps() {
  rm -f "$tmpfile"
}
trap cleanup_tmps EXIT ERR

# Print script output to console but also capture in a tempfile
tmpfile=$(mktemp)
pipeline/parse_out/generate_sweeps.sh "$@" | tee $tmpfile

# Extract last lines of the tmpfile as these contain all sweep-commands
commands=$(grep "\./run\.sh" "$tmpfile")

# Run the W&B Agent for each sweep.
while read -r cmd; do
  pipeline/sweep/run_agent.sh "$cmd" "$NUM_AGENTS" "$CHUNK_SIZE"
done <<< "$commands"
