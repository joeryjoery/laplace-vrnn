from __future__ import annotations
from typing import TypeVar, Union, Generic, Any

from jaxtyping import PyTree

from flax import linen as nn


TCarry = TypeVar("TCarry")
TV = TypeVar("TV")

T = TypeVar("T")
Batched = Union[T, type('_', (), {})]

# External Input Modality
Action = TypeVar('Action')
Observation = TypeVar('Observation')

# Compatibility Modality between Input and Internal State.
Embedding = TypeVar('Embedding')

# State and AfterState may coincide in different agent_model architectures
State = TypeVar('State')
AfterState = TypeVar('AfterState')

# Internal Output Modality
SignalPrediction = TypeVar('SignalPrediction')
StatePrediction = TypeVar('StatePrediction')
AfterStatePrediction = TypeVar('AfterStatePrediction')


class Embedder(nn.Module, Generic[Observation, Embedding]):

    @nn.compact
    def __call__(
            self,
            o_t: Observation | Action,
            *args, **kwargs
    ) -> Embedding:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class ApplyAction(nn.Module, Generic[State, Action, AfterState]):
    adapter: StateAdapter

    @nn.compact
    def __call__(
            self,
            s_t: State,
            a_embed_t: Embedding,
            *args, **kwargs
    ) -> AfterState:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class PriorTransition(
    nn.RNNCellBase,
    Generic[AfterState, SignalPrediction, State]
):
    adapter: StateAdapter

    @nn.compact
    def __call__(
            self,
            as_t: AfterState,
            *args, **kwargs
    ) -> tuple[SignalPrediction, State]:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class PosteriorTransition(
    nn.RNNCellBase,
    Generic[Embedding, AfterState, State]
):
    adapter: StateAdapter

    @nn.compact
    def __call__(
            self,
            e_t: Embedding,
            as_prev: AfterState,
            *args, **kwargs
    ) -> State:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class PriorPredictive(nn.Module, Generic[AfterState, AfterStatePrediction]):

    @nn.compact
    def __call__(
            self,
            as_t: AfterState,
            *args, **kwargs
    ) -> AfterStatePrediction:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class PosteriorPredictive(nn.Module, Generic[State, StatePrediction]):

    @nn.compact
    def __call__(
            self,
            s_t: State,
            *args, **kwargs
    ) -> StatePrediction:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class StateInitializer(nn.Module, Generic[State, Observation, Action]):

    @nn.compact
    def __call__(
            self,
            observation: Observation,
            action: Action,
            *args, **kwargs
    ) -> State:
        raise NotImplementedError(f"{type(self)} has no implementation!")


class StateAdapter(Generic[TCarry, TV]):
    """Define adapters for connecting State Transformations.

    For example, how to split the output of an LSTM Module into the memory
    variable (substate) and the downstream variable (output).

    Default Intended Design:
     - State + Action       <->     AfterState
     - RNN-Cell + Output    <->     State
     - PyTree + Output      <->     RNN-Cell

    Note that for many architectures, the RNN-cell and Output are the same.
    """
    base_adapter: StateAdapter | None = None

    @staticmethod
    def split_transformable(
            state: Any[TCarry, TV]
    ) -> tuple[TCarry, TV]:
        ...

    @staticmethod
    def combine_transformed(
            sub_state: TCarry,
            transformed: TV
    ) -> Any[TCarry, TV]:
        ...
