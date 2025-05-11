from typing import TYPE_CHECKING, Sequence
from dataclasses import fields

from axme.factory import default, Factory
from axme.data import UsageFlags, TrainTestTask, supervised, environment

from .. import problems

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class SampleTaskFactory(
    default.TaskFactory[environment.Environment | supervised.Regression]
):
    option: str  # Module choice
    option_kwargs: dict  # e.g, smoothness, number of enemies, difficulty

    task_kwargs: dict  # e.g., input shape, IO-options
    task_flags: Sequence[str]

    def make(self) -> environment.Environment | supervised.Regression:

        module_tree = problems
        for branch in self.option.split('.'):
            module_tree = getattr(module_tree, branch)
        problem_type = module_tree

        # Accumulate flags
        flag = None
        for f in self.task_flags:
            flag = UsageFlags.__getitem__(f.upper()) if flag is None else \
                flag | UsageFlags.__getitem__(f.upper())

        if hasattr(problems.functions, problem_type.__name__):
            return supervised.Regression(
                function=problem_type(**self.option_kwargs),
                name=self.option,
                flag=flag,
                **self.task_kwargs,
            )
        else:
            return environment.Environment(
                env=problem_type(**self.option_kwargs),
                name=self.option,
                flag=flag,
                # **self.task_kwargs
            )


@dataclass
class TrainTestTaskFactory(
    Factory[TrainTestTask[
        supervised.Regression | environment.Environment,
        supervised.Regression | environment.Environment
    ]]
):
    option: dict[int, str]  # Module choice
    option_kwargs: dict[int, dict]  # e.g, smoothness, difficulty

    task_kwargs: dict[int, dict]  # e.g., input shape, IO-options
    task_flags: dict[int, Sequence[str]]  # e.g., producer, evaluator

    def make(self) -> TrainTestTask[
        supervised.Regression | environment.Environment,
        supervised.Regression | environment.Environment
    ]:

        # Transpose multi-element dictionary of attributes to list of
        # single-element dictionaries for each attribute
        values = fields(self)
        transposed = [
            {f.name: getattr(self, f.name)[k] for f in values}
            for k in self.option.keys()
        ]

        tasks = [SampleTaskFactory(**config).make() for config in transposed]

        train = [t for t in tasks if UsageFlags.PRODUCER in t.flag]
        test = [t for t in tasks if UsageFlags.EVALUATION in t.flag]

        if len(train) == 1:
            train, = train
        else:
            # TODO: Future (use-cases; e.g., MAML?)
            raise NotImplementedError('Must specify strictly 1 train task')

        return TrainTestTask(train=train, test=test)
