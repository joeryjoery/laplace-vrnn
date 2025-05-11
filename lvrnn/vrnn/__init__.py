from .interface import (
    RLVMState, RLVMAdapter, RecurrentLatentVariableModel, ModelModality,
    RLVMTransition
)

from .deterministic import DeterministicRNN, DetRNNState
from .vrnn import VRNNState, VariationalRNN
from .lvrnn import (
    LaplaceVRNN, LVRNNState, ApproxLVRNNState,
    HistoryLaplaceVRNN, LinearizedLaplaceVRNN
)

from . import losses

from .model import (
    VariationalSimulater, VariationalPredicter
)


class Models:
    """Namespace for collecting RNN implementations"""
    delta: DeterministicRNN
    variational: VariationalRNN
    laplace: HistoryLaplaceVRNN
    linearized_laplace: LinearizedLaplaceVRNN
