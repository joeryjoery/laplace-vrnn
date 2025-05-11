# Config Hierarchy

This folder hierarchy splits up all the dependent parameters into config-categories for: experiment parameters, methodology parameters, and problem-dependent IO parameters, etc.

This is done for both readability, reproducibility, and modularity. To define new experiments with similar model architectures, all that is needed is to swap a couple config files (e.g., the problem config and the modality+loss config) instead of having to redefine a completely different monolithic configuration file. The latter is much more prone to mistakes and bloating of the entire configuration space.

## Structure Explained

```markdown
configs/
│
├── combined/
│ ├── combined_config_file.txt
│ └── ...
|
├── experiment/
│ ├── experiment_params.yaml
│ └── ...
│
├── methods/
│ ├── actor/
│ │ ├── control_config.yaml
│ │ └── ...
│ ├── learning/
│ │ ├── algorithm_config.yaml
│ │ └── ...
│ ├── modality/
│ │ ├── model_output_structure.yaml
│ │ └── ...
│ └── models/
│   ├── architecture_config.yaml
│   └── ...
│
├── problems/
│ ├── type/
│ │ ├── sub_problem_config.yaml
│ │ └── ...
│ └── ...
│
├── sweeps/
│ ├── ablations/
│ │ ├── wandb_sweep_config.yaml
│ │ └── ...
│ └── finetune/
│ │ ├── simplified_sweep_config.yaml
│ │ ├── ...
│ │ └── README.md
│
└── README.md  # This file
```
