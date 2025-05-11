#!/usr/bin/env bash
# Helper script to generate the sweep scripts for all finetuning experiments
# This script should only be run when e.g., the results from `ablations.sh` have completed.

ENTITY=$1
SEARCH_DIR=$2
OUT_DIR=$3

sweeps=(
  "configs/sweeps/finetune/laplace.yaml"
  "configs/sweeps/finetune/variational.yaml"
)

for ((i=0; i<${#sweeps[@]}; i++)); do
  ./run.sh pipeline/parse_out/generate_sweeps.sh "$SEARCH_DIR" \
      -E "$ENTITY" -N FineTune \
      -O "$OUT_DIR" \
      -S "${sweeps[$i]}"
done
