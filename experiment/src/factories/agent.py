from typing import Any, Type, Literal, TYPE_CHECKING
from dataclasses import fields

import flax.linen as nn

import jax

from jit_env import TimeStep, specs

from axme import data
from axme.factory import Factory, default

from lvrnn import agents, vrnn
from lvrnn.agent_model import (
    builder, DefaultEmbedder,
    DefaultAfterState, AgentModel
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


class RecurrentNetworks:
    """Namespace for compatible RNN implementations. Do not instantiate.

    TODO:
     - Implement nn.RNN variant that actually implements a VanillaRNN
     - Extend a RNN variant with a Transformer (FUTURE)
    """
    RNN: Type[nn.RNNCellBase] = nn.RNN
    LSTM: Type[nn.RNNCellBase] = nn.OptimizedLSTMCell
    GRU: Type[nn.RNNCellBase] = nn.GRUCell


@dataclass
class AgentModelFactory(Factory[tuple[builder.ModelBuilder, bool]]):
    embedding_sizes: tuple[int, ...]
    prediction_sizes: tuple[int, ...]
    simulate_sizes: tuple[int, ...]

    embedding_activation: str
    prediction_activation: str
    simulate_activation: str

    recurrent_core: str
    recurrent_size: int

    variational_method: str
    variational_kwargs: dict[str, Any]

    # Choose between :
    #  1. Stochastic State-Space Model: Output = RVs
    #  2. Recurrent SSM: Output = RVs and Deterministic State
    #  3. Bayes-Adaptive Model: Output = RVs and Last Observation
    posterior_type: Literal['ssm', 'rssm', 'bamdp']

    def make(  # type: ignore
            self,
            modalities: dict[str, vrnn.ModelModality]
    ) -> tuple[builder.ModelBuilder, bool]:

        posterior_modalities = [
            v for k, v in modalities.items() if
            (agents.Scope.Policy in k) or (agents.Scope.Value in k)
        ]
        prior_modalities = [
            v for k, v in modalities.items() if
            (agents.Scope.Reward in k) or (agents.Scope.Signal in k)
        ]

        if (not prior_modalities) and (not posterior_modalities):
            raise RuntimeError("No modalities specified!")

        obs_embedder = DefaultEmbedder(
            self.embedding_sizes,
            getattr(jax.nn, self.embedding_activation)
        )
        act_embedder = DefaultEmbedder(
            self.embedding_sizes,
            getattr(jax.nn, self.embedding_activation)
        )

        # Instantiate RNN Architecture
        rnn_base = getattr(RecurrentNetworks, self.recurrent_core)
        core = rnn_base(self.recurrent_size)

        deterministic = True
        if self.variational_method is not None:
            # Wrap RNN core with a Variational Posterior output
            core_type = getattr(vrnn, self.variational_method)
            core = core_type(core=core, **self.variational_kwargs)

            deterministic = (core_type == vrnn.DeterministicRNN)

        # Connect the components into a joint model
        transition = vrnn.RLVMTransition(
            core=core, core_adapter=vrnn.RLVMAdapter,
            posterior_type=self.posterior_type
        )
        model_builder = builder.DefaultModelBuilder(
            obs_embedder, act_embedder, transition, DefaultAfterState()
        )

        if prior_modalities:

            simulater = vrnn.VariationalSimulater(
                layer_sizes=self.simulate_sizes,
                activation=getattr(jax.nn, self.simulate_activation),
                modalities=prior_modalities
            )
            model_builder = model_builder.set_simulate(simulater)

        if posterior_modalities:

            predicter = vrnn.VariationalPredicter(
                layer_sizes=self.prediction_sizes,
                activation=getattr(jax.nn, self.prediction_activation),
                modalities=posterior_modalities
            )
            model_builder = model_builder.set_posterior_predictive(predicter)

        return model_builder, deterministic


@dataclass
class ModalityFactory(
    default.ModalityFactory[
        vrnn.ModelModality | None,
        data.Regression | data.Environment
    ]
):
    name: dict[int, str]

    likelihood: dict[int, str]
    likelihood_kwargs: dict[int, dict[str, Any] | None]

    ensemble_method: dict[int, str | None]
    uniform_ensemble: dict[int, bool]

    def make(
            self,
            task: data.Regression | data.Environment
    ) -> dict[str, vrnn.ModelModality]:

        if isinstance(task, data.Regression):
            predict_spec = task.input_spec() if task.inverse \
                else task.output_spec()

        elif isinstance(task, data.Environment):

            env_spec: TimeStep = task.output_spec()
            predict_spec = env_spec.reward

        else:
            raise NotImplementedError(
                f"Task Type: {type(task)} is not supported!"
            )

        values = fields(self)
        transposed = [
            {field.name: getattr(self, field.name)[key] for field in values}
            for key in self.name.keys()
        ]

        modalities = dict()
        for config in transposed:
            name = config.get('name')

            if (agents.Scope.Value in name) or \
                    (agents.Scope.Reward in name) or \
                    (agents.Scope.Signal in name):

                # TODO: Int modality not yet supported
                modality = vrnn.ModelModality(
                    **config,
                    spec=jax.ShapeDtypeStruct(
                        predict_spec.shape, jax.numpy.float32
                    )
                )

            elif agents.Scope.Policy in name:

                # TODO: Tree action spaces not yet supported
                action_spec = task.env.action_spec()
                if isinstance(action_spec, specs.DiscreteArray):

                    # TODO: Rework this construction of inferring num_values
                    config['likelihood_kwargs'] |= {
                        'num_classes': action_spec.num_values
                    }
                    modality = vrnn.ModelModality(
                        **config,
                        spec=jax.ShapeDtypeStruct(
                            predict_spec.shape, jax.numpy.int32
                        )
                    )
                else:
                    modality = vrnn.ModelModality(
                        **config,
                        spec=jax.ShapeDtypeStruct(
                            predict_spec.shape, jax.numpy.float32
                        )
                    )

            else:
                valid = [
                    k for k in agents.Scope.__dict__ if not k.startswith('_')
                ]
                raise NotImplementedError(
                    f"Name: {name} is not supported. "
                    f"The name must *contain* a value from: {valid}."
                )

            modalities[name] = modality

        return modalities


@dataclass
class PolicyFactory(
    default.PolicyFactory[agents.AgentModelPolicy | None, AgentModel]
):
    agent: str | None = None
    agent_kwargs: dict[str, Any] | None = None

    def make(
            self,
            model: AgentModel,
            test: bool
    ) -> agents.AgentModelPolicy | None:
        if model.posterior_predictive is None:
            return None

        policy = getattr(agents, self.agent)
        return policy(model, test, **(self.agent_kwargs or {}))
