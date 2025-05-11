from typing import Sequence, Any, Union, TYPE_CHECKING

from jaxtyping import PRNGKeyArray

import jax

from axme import data
from axme.actor import Policy
from axme.factory import default
from axme.implement.experiment import EvaluationService

from lvrnn.agent_model import AgentModel

from .. import problems

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class ProducerFactory(
    default.ProducerFactory[
        data.UniformDataGenerator | data.EnvironmentDataGenerator,
        data.Regression | data.Environment,
        Policy | None
    ]
):
    data_batch_dimensions: dict[str, int]

    def make(
            self,
            task: data.Regression | data.Environment,
            policy: Policy | None,
            seed: PRNGKeyArray
    ) -> Union[
         data.UniformDataGenerator,
         data.EnvironmentDataGenerator
    ]:
        if isinstance(task, data.Regression):
            return data.UniformDataGenerator(
                task,
                seed=seed
            ).resize(**self.data_batch_dimensions)
        else:
            return data.EnvironmentDataGenerator(
                task,
                policy,
                seed=seed
            ).resize(**self.data_batch_dimensions)


@dataclass
class EvaluatorFactory(
    default.EvaluatorFactory[
        EvaluationService,
        data.Regression | data.Environment,
        AgentModel,
        Policy | None
    ]
):
    problem_seed: int
    artifact_kwargs: dict[str, Any]

    def make(
            self,
            tasks: Sequence[data.Regression | data.Environment],
            model: AgentModel,
            policy: Policy | None,
            seed: PRNGKeyArray
    ) -> EvaluationService:
        problem_key = jax.random.key(self.problem_seed)

        routines = dict()
        for i, task in enumerate(tasks):
            if isinstance(task, data.Regression):
                evaluator = problems.functions.SupervisedEvaluator(
                    model=model,
                    task=task,
                    fixed_seed=problem_key,
                    **self.artifact_kwargs
                )
                functions = [
                    evaluator.test_functional_metrics, evaluator.test_plots,
                    evaluator.train_functional_metrics, evaluator.train_plots
                ]
                routines |= {
                    f'{f.__name__}/{task.name}/{i}': f
                    for f in functions
                }

            elif isinstance(task.env, problems.bandits.Multinoulli):
                evaluator = problems.bandits.MultinoulliEvaluator(
                    model=model,
                    policy=policy,
                    task=task,
                    fixed_seed=problem_key,
                    **self.artifact_kwargs
                )
                functions = [
                    evaluator.agent_metrics,
                    evaluator.sample_plots
                ]
                routines |= {
                    f'{f.__name__}/{task.name}/{i}': f
                    for f in functions
                }

            elif isinstance(task.env, problems.bandits.GaussianProcessPrior):
                raise NotImplementedError(
                    "GP Prior has no Evaluation implementation!"
                )

            elif isinstance(task.env, problems.gridworld.SquareGrid):
                evaluator = problems.gridworld.GridworldEvaluator(
                    model=model,
                    policy=policy,
                    task=task,
                    fixed_seed=problem_key,
                    **self.artifact_kwargs
                )
                functions = [
                    evaluator.agent_metrics,
                    evaluator.sample_plots
                ]
                routines |= {
                    f'{f.__name__}/{task.name}/{i}': f
                    for f in functions
                }

            elif isinstance(task.env, problems.mjx.Barkour) or \
                    isinstance(task.env, problems.mjx.Humanoid):
                raise NotImplementedError(  # TODO
                    "MJX Environments have no Evaluation implementation yet!"
                )

            else:
                raise NotImplementedError(f"Unknown Task option: {task}")

        return EvaluationService(
            routines=routines,
            seed=seed
        )
