"""
"""
from __future__ import annotations
from typing import Literal, Any, Generator, Hashable, TYPE_CHECKING

from dataclasses import fields, replace

import flax.traverse_util
import jax

from axme.core import Server, Broker
from axme.broker import SynchronousBroker
from axme.factory import default

from axme.experimenter import Experimenter, Evaluation

from axme.implement import profiler
from axme.implement.factory import (
    OptaxFactory, SeedBatchFactory, SGDLearnerFactory
)

from .factories import (
    ELBOFactory,
    AgentModelFactory, EvaluatorFactory,
    TrainTestTaskFactory, PolicyFactory,
    ProducerFactory,
    FormatterFactory, ModalityFactory,
    OfflineTestExperimentFactoryV2
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


def setup_datastream(
        service: Broker,
        consumer: Server,
        producer: Server,
        formatter: Server,
        evaluator: Server
) -> tuple[Generator, set[Hashable], set[Hashable]]:

    service.register(consumer)
    service.register(producer)
    service.register(formatter)
    service.register(evaluator)

    service.subscribe(consumer, streams=(formatter, ))
    service.subscribe(producer, streams=(consumer, ))
    service.subscribe(formatter, streams=(producer, ))
    service.subscribe(evaluator, streams=(consumer, ))

    # Create Profiler to Monitor the DataStream
    stream_profiler = profiler.ResourceProfiler()
    service.register(stream_profiler)
    service.subscribe(stream_profiler, streams=(consumer, producer))

    # Listen to all datastreams
    streams = service.get_streams()

    return (
        service.get_event_loop(streams),
        streams,
        {service.get_key(evaluator)}
    )


@dataclass
class TrainTestEventLoopFactory(default.EventLoopFactory):
    """Factory of Factories."""
    rng: SeedBatchFactory

    task: TrainTestTaskFactory
    modality: ModalityFactory

    model: AgentModelFactory
    policy: PolicyFactory

    optimizer: OptaxFactory
    loss: ELBOFactory

    learner: SGDLearnerFactory
    producer: ProducerFactory
    formatter: FormatterFactory
    evaluator: EvaluatorFactory

    def make(self, broker: Broker) -> Generator:
        # Create initial random-key Tree from seeds.
        consumer_keys, producer_keys, evaluation_keys = self.rng.make()

        # Build the task and its IO.
        train_test = self.task.make()
        modalities = self.modality.make(train_test.train)

        # Build Models for training and data-generation
        model_builder, deterministic = self.model.make(modalities)
        train_model = model_builder.build(option='train')
        inference_model = model_builder.build(option='inference')

        # Override Loss-config depending on the model architecture
        loss_factory = replace(self.loss, deterministic=deterministic)
        loss = loss_factory.make(train_model)

        # Create the Learner/ Consumer objects to compute updates from Loss
        optimizer = self.optimizer.make()
        learner = self.learner.make(loss, optimizer, seed=consumer_keys)

        train_policy = self.policy.make(inference_model, test=False)
        test_policy = self.policy.make(inference_model, test=True)

        # Create Data Producers given the model
        producer = self.producer.make(
            train_test.train, train_policy, seed=producer_keys
        )
        evaluator = self.evaluator.make(
            train_test.test, inference_model, test_policy, seed=evaluation_keys
        )

        # Create Adapter for interoperation between Consumer <-> Producer.
        formatter = self.formatter.make(learner, producer)

        # Given all dependent objects, instantiate the EventLoop by connecting
        # all components as a data-communication graph.
        event_loop, streams, testers = setup_datastream(
            service=broker,
            consumer=learner,
            producer=producer,
            formatter=formatter,
            evaluator=evaluator
        )

        return event_loop


class ConfigSyntax:
    desc: str = 'desc'  # Parameter Description
    value: str = 'value'  # Parameter Value

    # Remaining config-fields not listed in EventLoopFactory.
    experiment: str = 'experiment'


def parse_config(config_dict: dict[str, Any]) -> dict[str, Any]:

    def unpack(k: dict[str, Any] | Any):
        if isinstance(k, dict):
            k.pop(ConfigSyntax.desc, None)

            if ConfigSyntax.value in k:
                return unpack(k.get(ConfigSyntax.value))

            return {a: unpack(b) for a, b in k.items()}

        return k

    return unpack(config_dict)


def logger_formatter(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # TODO: Check for batched data!
    format_names = {k.split('(')[0]: v for k, v in data.items()}
    return flax.traverse_util.flatten_dict(format_names, sep='/')


def stream_callback_formatter(
        metrics: dict[str, jax.Array | dict[str, jax.Array]]
) -> dict[jax.Array]:
    # Filter out only loss-relevant metrics from the logger-format.
    match_names = ['loss', 'grad', 'update']

    flat = logger_formatter(metrics)
    return {
        k: v for k, v in flat.items() for name in match_names
        if name in k.lower()
    }


def from_config(
        config_dict: dict[str, Any],
        service: None
) -> Experimenter:

    # Format configuration file dictionary to a proper config dict
    parsed = parse_config(config_dict)

    # Construct factory of nested factories through nested field-unpacking.
    init_types = {
        field.name: globals().get(field.type)
        for field in fields(TrainTestEventLoopFactory)
    }
    event_loop_factory = TrainTestEventLoopFactory(
        **{name: _typ(**parsed.pop(name)) for name, _typ in init_types.items()}
    )

    # Construct Client Factory
    logger: list[Literal['console']] = ['console']
    if service:
        logger: list[Literal['wandb']] = ['wandb']

    experimenter = OfflineTestExperimentFactoryV2(
        **parsed.pop(ConfigSyntax.experiment),
        logger=logger,
        logger_formatter=logger_formatter,
        service=service,
        stream_callback_formatter=stream_callback_formatter
    )

    if len(parsed) != 0:
        # We expect that TrainTestEventLoopFactory consumes all fields.
        # This helps prevent ambiguities and wrongly constructed objects.
        raise RuntimeError(
            f"Trailing config options remaining after consumption! "
            f"Config: {parsed}"
        )

    # Build the experiment
    broker = SynchronousBroker()
    event_loop = event_loop_factory.make(broker)

    subscriptions = broker.get_streams()
    test_services = {
        str(s) for k, s in broker.streams.items()
        if isinstance(s, Evaluation)
    }

    return experimenter.make(event_loop, subscriptions, test_services)
