from __future__ import annotations
from typing_extensions import Self

import abc

from .interface import (
    PriorPredictive, PosteriorPredictive, PriorTransition, PosteriorTransition,
    Embedder, ApplyAction
)
from .model import (
    AgentModel, ModelComponents, InitialInference, RecurrentInference,
    DefaultModelState
)


class ModelBuilder(abc.ABC):
    """Builder Class to flexibly instantiate AgentModel.

    This class is a wrapper around the utilities defined in AgentFactory.
    """

    def __init__(
            self,
            observation_embedder: Embedder,
            action_embedder: Embedder,
            core: PosteriorTransition,
            apply_action: ApplyAction,
            name: str = None
    ):
        self.observation_embedder = observation_embedder
        self.action_embedder = action_embedder
        self.core = core
        self.apply_action = apply_action

        self.name = name

        self.simulate = None
        self.posterior_predictive = self.prior_predictive = None

    def set_prior_predictive(
            self,
            prior_predictive: PriorPredictive
    ) -> Self:
        self.prior_predictive = prior_predictive
        return self

    def set_posterior_predictive(
            self,
            posterior_predictive: PosteriorPredictive
    ) -> Self:
        self.posterior_predictive = posterior_predictive
        return self

    def set_simulate(
            self, simulate: PriorTransition
    ) -> Self:
        self.simulate = simulate
        return self

    @abc.abstractmethod
    def components(self, *, option: str | None = None) -> ModelComponents:
        pass

    @abc.abstractmethod
    def build(self, *, option: str | None = None) -> AgentModel:
        pass


class DefaultModelBuilder(ModelBuilder):

    def components(self, *, option: str | None = None) -> ModelComponents:
        init = DefaultModelState(
            observation_embedder=self.observation_embedder,
            action_embedder=self.action_embedder,
            transition=self.core
        )

        return ModelComponents(
            observation_embedder=self.observation_embedder,
            action_embedder=self.action_embedder,

            transition=self.core,
            apply_action=self.apply_action,
            simulate=self.simulate,

            posterior_predictive=self.posterior_predictive,
            prior_predictive=self.prior_predictive,

            initial_state=init,

            name=self.name or ModelComponents.__name__
        )

    def build(self, *, option: str | None = None) -> AgentModel:

        init = DefaultModelState(
            observation_embedder=self.observation_embedder,
            action_embedder=self.action_embedder,
            transition=self.core
        )

        joint_core = InitialInference(
            observation_embedder=self.observation_embedder,
            action_embedder=self.action_embedder,

            transition=self.core,
            apply_action=self.apply_action,

            posterior_predictive=self.posterior_predictive,
            prior_predictive=self.prior_predictive,

            initial_state=init
        )

        joint_simulate = None
        if self.simulate is not None:
            joint_simulate = RecurrentInference(
                action_embedder=self.action_embedder,

                simulate=self.simulate,
                apply_action=self.apply_action,

                posterior_predictive=self.posterior_predictive,
                prior_predictive=self.prior_predictive
            )

        return AgentModel(
            observation_embedder=self.observation_embedder,
            action_embedder=self.action_embedder,

            transition=self.core,
            apply_action=self.apply_action,
            simulate=self.simulate,

            posterior_predictive=self.posterior_predictive,
            prior_predictive=self.prior_predictive,

            joint_transition=joint_core,
            joint_simulate=joint_simulate,

            initial_state=init,

            name=self.name or AgentModel.__name__
        )
