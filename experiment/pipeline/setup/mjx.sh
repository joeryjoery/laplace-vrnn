#!/usr/bin/env bash
# Helper script to generate the scripts that run all the ablation experiments

ENTITY=$1
OUT_DIR=$2

names=(
  "Humanoid"
  "Walker"
)
problems=(
  "configs/combined/mjx_humanoid.txt"
  "configs/combined/mjx_walker.txt"
)
models=(
  "configs/methods/models/deterministic.yaml"
  "configs/methods/models/laplace.yaml"
  "configs/methods/models/variational.yaml"
)
sweeps=(
  "configs/sweeps/mjx/deterministic.yaml"
  "configs/sweeps/mjx/laplace.yaml"
  "configs/sweeps/mjx/variational.yaml"
)

for ((i=0; i<${#names[@]}; i++)); do
  for ((j=0; j<${#sweeps[@]}; j++)); do

    ./run.sh pipeline/sweep/compile_sweep.sh \
      -P "${problems[$i]}" \
      -E "$ENTITY" -N "${names[$i]}" \
      -O "$OUT_DIR" \
      -C "${models[$j]}" \
      -S "configs/sweeps/mjx/seeds.yaml" "${sweeps[$j]}"

  done
done
