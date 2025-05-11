#!/usr/bin/env bash
# Helper script to generate the scripts that run all the ablation experiments

ENTITY=$1
OUT_DIR=$2

names=(
  "Regression"
  "Multinoulli"
#  "BayesOpt"
  "GridWorld"
)
problems=(
  "configs/combined/regression.txt"
  "configs/combined/multinoulli.txt"
#  "configs/combined/bayesopt.txt"
  "configs/combined/gridworld.txt"
)
models=(
  "configs/methods/models/deterministic.yaml"
  "configs/methods/models/laplace.yaml"
  "configs/methods/models/variational.yaml"
)
sweeps=(
  "configs/sweeps/ablations/deterministic.yaml"
  "configs/sweeps/ablations/laplace.yaml"
  "configs/sweeps/ablations/variational.yaml"
)

for ((i=0; i<${#names[@]}; i++)); do
  for ((j=0; j<${#sweeps[@]}; j++)); do

    # Skip Compiling a sweep for BayesOpt + Deterministic
    if [[ "${models[$j]}" == *deterministic* ]] && \
        [[ "${problems[$i]}" == *bayesopt* ]]; then
      continue
    fi

    ./run.sh pipeline/sweep/compile_sweep.sh \
      -P "${problems[$i]}" \
      -E "$ENTITY" -N "${names[$i]}" \
      -O "$OUT_DIR" \
      -C "${models[$j]}" \
      -S "configs/sweeps/ablations/seeds.yaml" "${sweeps[$j]}"

  done
done
