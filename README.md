# LVRNN: Laplace Variational Recurrent Neural Networks

This repository was used to create the results in our RLC 2025 paper "[Bayesian Meta-Reinforcement Learning with Laplace Variational Recurrent Networks](https://openreview.net/forum?id=YYw6MlEPaU)".

The main idea of this project was to transform a recurrent neural network (RNN) into a variational RNN through the Laplace approximation.
The variational RNN then allows us to use or analyze distributional statistics of the learned hidden state representation.

Please cite us as,
```text
@inproceedings{
    vries2025bayesian,
    title={Bayesian Meta-Reinforcement Learning with Laplace Variational Recurrent Networks},
    author={Joery A. de Vries and Jinke He and Mathijs M. de Weerdt and Matthijs T. J. Spaan},
    booktitle={Reinforcement Learning Conference},
    year={2025},
    url={https://openreview.net/forum?id=YYw6MlEPaU}
}
```

**Notes:**

- The main implementation of learning algorithms and neural network architectures can be found in the `lvrnn` package.
- Experiment code, factories, outer loop logic, hyperparameters and configs can all be found in the `experiment` package. 
