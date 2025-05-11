# FineTune Sweep Configurations

This folder contains the config-files for defining W&B Sweeps for the Variational and Laplace Methods.

The idea is that these config files contain as little ablation-values as possible so that we can take the Cartesian product of the defined values here with checkpoint data from a previous ablation run.
This defines an experiment that resumes at intermediate point of a previous experiment; the resulting evaluation statistics measure how effective it is to train the given algorithm with a different methodology first before finetuning the parameters with another method.
