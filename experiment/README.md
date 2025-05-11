## Experiment Module for LVRNN

### Generic Examples

Run one instance of a Regression Task with a Laplace architecture and Console logging.
```bash
./run.sh pipeline/task.sh \
  -P configs/combined/regression.txt \
  -C configs/methods/models/laplace.yaml
  
# Starts an experiment ...
```

Run one instance of a Multinoulli Bandit Task with a Deterministic RNN architecture and Weights & Biases logging to the `TEST` project.
```bash
./run.sh pipeline/task.sh \
  -P configs/combined/multinoulli.txt \
  -C configs/methods/models/deterministic.yaml \
  --wandb TEST
 
 # Starts an experiment ...
```

#### Weights & Biases Sweeps
Perform a sweep on a Gridworld Task: first compile the sweep for the given configuration, then run the compiled script to run the Weights & Biases Agent.
```bash
./run.sh pipeline/sweep/compile_sweep.sh \
  -P configs/combined/regression.txt \
  -C configs/methods/models/deterministic.yaml \
  -E ENTITY \
  -N test \
  -S configs/sweeps/debug.yaml 

# W&B output:
# > wandb: Creating sweep from: /tmp/tmp.rFEoweXb18
# > wandb: Creating sweep with ID: whyyjlfg
# > wandb: View sweep at: https://wandb.ai/ENTITY/test/sweeps/whyyjlfg
# > wandb: Run sweep agent with: wandb agent ENTITY/test/whyyjlfg

# Script output:
# > Generated Agent Script -- Sweep ID: whyyjlfg
# > Run the wandb Agent with:
# > ./run.sh pipeline/sweep/run/whyyjlfg.sh $NUM_AGENTS $CHUNK_SIZE

./run.sh pipeline/sweep/run/whyyjlfg.sh 4 2

# Starts a parameter sweep Experiment ...
```
This can also be done in one joint command with,
```bash
# Compiles the sweep and immediately launches the W&B Agent until completion.
./run.sh pipeline/sweep.sh \
  -P configs/combined/multinoulli.txt \
  -C configs/methods/models/deterministic.yaml \
  -E ENTITY \
  -N test \
  -S configs/sweeps/debug.yaml
  

# Starts a parameter sweep Experiment ...
```
Note that we haven't exhaustively tested all edge-cases for playing around with arguments.
If the arguments aren't properly passed through the script may crash or produce wrong results.

#### FineTuning Previous Runs

For the fine-tuning experiments, we first need to parse the output directory structure and filter for configs that represent the architecture that we wish to fine-tune, this can be done in the following command
```bash
./run.sh pipeline/parse_out/extract_tunables.sh ~/out
# Example output:
> ~/out/AXME/20231206_085232-c5a410574caa66e7/config.yaml ~/out/AXME/20231206_085301-27c4d3a5327541ea/config.yaml
> ~/out/AXME/20231206_085145-55eee9dafbf162cb/config.yaml
> ~/out/AXME/20231206_084945-18345da585df5345/config.yaml ~/out/AXME/20231206_084856-3056025678ac527a/config.yaml
```
The example output shows that the configs are both grouped and separated. 
Each newline indicates a different group (e.g., different model-size) and within each group there are only config-variations for specific fields (e.g., evaluation-configs which do not influence the training progression, or random-key seeds).

To see exactly how we filter out the configs, see the `pipeline/parse_out/extract_tunables.sh`

```bash
 ./run.sh pipeline/parse_out/generate_sweeps.sh ~/out \
  -E ENTITY \
  -N test \
  -S configs/sweeps/debug.yaml

# Generates `n` sweeps for all groups found in ~/out
```

```bash
 ./run.sh pipeline/finetune.sh ~/out \
  -E ENTITY \
  -N test \
  -S configs/sweeps/debug.yaml

# Generates `n` sweeps for all groups found in ~/out
# Then immediately runs all agents until completion.
```


## Reproducing our Experiments

All commands that were run to generate the data in our paper are listed partially below. So, this only includes the commands to generate the **scripts**, not the commands to actually run the resulting scripts. 

Since we generated W&B sweeps for a large number of config options, we deployed the generated scripts on a SLURM based compute-cluster. So the actual used commands are wrapped in their own `srun` files. We do not list the SLURM scripts here as they simply wrap the singular command `./run.sh path/to/my/runnable.sh NUM_AGENTS CHUNK_SIZE`, as explained earlier, while being platform dependent. Of course, one can call all the generated scripts locally, but this will take an immense amount of time to run.

In essence the commands below are calls to scripts that loop over combinations of config-files for `sweep/compile_sweep.sh` and `parse_out/generate_sweep.sh` as explained in the previous sections.

### 1) Generating the Ablations

```bash
# Arguments to ./run.sh: 1) Script 2) W&B Entity 3) output directory
./run.sh pipeline/setup/ablations.sh ENTITY ~/out/ablations
```

Deploy the generated scripts...

### 1.5) Finetuning the Ablations

```bash
# Arguments to ./run.sh: 1) Script 2) W&B Entity 2) input directory 3) output directory
./run.sh pipeline/setup/finetune.sh ENTITY ~/out/ablations ~/out/finetune
```

Deploy the generated scripts...

### 2) Generating the MJx Ablations
Based on inspecting the Ablation results, we tuned the hyperparameters to get decent values for the more expensive/ difficult experiments.

```bash
# Arguments to ./run.sh: 1) Script 2) W&B Entity 3) output directory
./run.sh pipeline/setup/mjx.sh ENTITY ~/out/mjx
```

Deploy the generated scripts...

### 2.5) Finetuning the MJx Ablations

```bash
# Arguments to ./run.sh: 1) Script 2) W&B Entity 2) input directory 3) output directory
./run.sh pipeline/setup/finetune.sh ENTITY ~/out/mjx ~/out/finetune
```

Deploy the generated scripts...


# Debugging

DEBUGGING: RUN THIS CMD
```bash
# 1) Run Deterministic sweep:
./run.sh pipeline/sweep.sh -P configs/combined/debug.txt -C configs/methods/models/deterministic.yam -E lvrnn -N DetTestV2 -S configs/sweeps/debug.yaml -O ~/out/DetTestV2 

# TODO: Run fully Integrated sweep

# 2) Finetune the sweep with Laplace for 1 run
./run.sh pipeline/task.sh -P configs/combined/debug.txt -C configs/methods/models/laplace.yaml -S experiment.restore_point.value=/home/out/test/Test/20231215_162112-5f35045f422cda2c/client/checkpoint/10.cp experiment.finetune.value=True

# 3) Finetune the sweep with VRNN for 1 run
./run.sh pipeline/task.sh -P configs/combined/debug.txt -C configs/methods/models/variational.yaml -S experiment.restore_point.value=/home/out/test/Test/20231215_162112-5f35045f422cda2c/client/checkpoint/10.cp experiment.finetune.value=True
```
