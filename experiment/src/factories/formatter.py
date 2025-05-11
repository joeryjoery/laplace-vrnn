from typing import Generic, TYPE_CHECKING

from axme.core import Formatter, T
from axme.data import supervised, environment, DataGenerator
from axme.implement.learner import SGDLearner
from axme.implement import formatters
from axme.factory import default


if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class FormatterFactory(
    default.FormatterFactory[
        formatters.NextTokenPrediction | formatters.TimeStepToSGD,
        SGDLearner,
        DataGenerator[T]
    ],
    Generic[T]
):

    def make(
            self,
            receiver: SGDLearner,
            sender: DataGenerator[T]
    ) -> Formatter:

        if isinstance(sender.task, supervised.Regression):
            return formatters.NextTokenPrediction()

        if isinstance(sender.task, environment.Environment):
            return formatters.TimeStepToSGD()

        raise NotImplementedError(sender.task)
