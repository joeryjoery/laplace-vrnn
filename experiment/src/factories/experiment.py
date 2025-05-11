"""Extend the default axme-client factory to allow for finetune experiments.


"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import warnings

import jax

from axme import broker

from axme.implement.experiment import (
    OfflineTestExperiment, ExperimentState, EvaluationService
)
from axme.implement.factory import OfflineTestExperimentFactory
from axme.implement.learner import SGDLearner

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class OfflineTestExperimentFactoryV2(
    OfflineTestExperimentFactory
):
    """Override the default Experiment Factory for state-overriding.

    This enables restoring a previously saved experiment-state and
    transferring this state to a new Learner Architecture. T
    """

    finetune: bool = False  # Treat restore_point as an initial-state

    def _restore_state(
            self,
            client: OfflineTestExperiment
    ):
        mngr = client.client_callback  # alias

        if (not mngr) or (not self.checkpointer):
            warnings.warn(
                f"Experiment Factory received a Restore Point without the "
                f"means to restore it! Make sure the experiment arguments "
                f"match those of the experiment to be restored.\n"
                f"restore_point: {self.restore_point}\n"
                f"checkpointer: {self.checkpointer}\n"
                f"client_callback: {type(mngr).__name__}."
            )
            # Simply continue with the un-restored client.
            return client

        # Load in Client-State and restore client data
        mngr.checkpointer.restore(self.restore_point)
        state = mngr.checkpointer.load()

        if self.finetune:
            # A more complex hacky alternative to `client.restore`.
            # Manually modifies the data inside `state` to allow the client
            # to treat `state` as an initial-state.
            setup_finetuning(client, state)
        else:
            # Simply loads in state and directly restore it to the client.
            # This will only work for resuming runs or when parameter
            # structures match exactly. Otherwise, this might silently fail
            # or produce different results than originally intended
            client.restore(state)


def left_inner_override(
        a: dict[str, dict | Any],
        b: dict[str, dict | Any],
        parent: bool = False
) -> dict[str, dict | Any]:
    """Recursively override all matching keys in `a` with values from `b`.

    If the dictionaries have a mismatching sub-branch, it traverses `a` until
    a subtree matches with `b`.

    Note: this is hacky code to make a specific use-case work; this will not
        work generically for merging trees.
    """
    result = {}

    for key, value_a in a.items():

        if key in b and isinstance(value_a, dict) and isinstance(b[key], dict):
            # Recursive call for matching sub-dictionaries
            result[key] = left_inner_override(value_a, b[key], parent=True)

        elif isinstance(value_a, dict) and parent:
            result[key] = left_inner_override(value_a, b, parent=True)

        else:
            # Use the value from dictionary b if there's a match otherwise use a
            result[key] = b.get(key, value_a)

    return result


def override_server_states(reference: dict, override: dict) -> dict:
    results = dict()

    for k, v in reference.items():
        if SGDLearner.__name__ not in k:
            results[k] = override[k]
            continue
        # else: override LearnerState

        # Match Params and Merge Learner + Optimizer States
        old_state = override[k]
        new_state = reference[k]

        # Left override the new-state with the old based on matching
        # parameter subtree structure. This is heuristically done for dicts.
        leaves, treedef = jax.tree_util.tree_flatten(
            new_state, is_leaf=lambda x: isinstance(x, dict)
        )
        old_leaves = jax.tree_util.tree_leaves(
            old_state, is_leaf=lambda x: isinstance(x, dict)
        )

        merged = []
        for leaf, old_leaf in zip(leaves, old_leaves):

            if type(leaf) != type(old_leaf):
                old_def = jax.tree_util.tree_structure(
                    old_state, is_leaf=lambda x: isinstance(x, dict)
                )
                raise ValueError(
                    "Optimizer States cannot be merged from old to new! "
                    f"Their structures are different:\n"
                    f"Old: {old_def}\nVs.\nNew: {treedef}"
                )

            if not isinstance(leaf, dict):
                merged.append(old_leaf)
                continue

            merged.append({
                k: left_inner_override(v, old_leaf[k])
                for k, v in leaf.items()
            })

        merged_state = jax.tree_util.tree_unflatten(treedef, merged)

        # Write combined LearnerState to Server
        results[k] = merged_state

    return results


def setup_finetuning(
        client: OfflineTestExperiment,
        restore_state: ExperimentState
):
    """Hacky code to override newly initialized state with an old-state"""

    # 0) Ensure client is in a predictable initial state.
    client.reset()

    # 1) Create New Client State
    backend: broker.BrokerState = client.datastream.send(
        broker.ClientRequest(broker.Control.SAVE)
    )

    # 2) Extract Restored Client State
    client.checkpointer.restore(restore_state.checkpoint)
    restored_backend: broker.BrokerState = client.checkpointer.load()

    # 3) Clear the queue/ schedule in the old evaluation-states in-place.
    for key, state in restored_backend.server_states.items():

        if EvaluationService.__name__ in key:
            queue, key = state
            queue.clear()

    # 4) Partially override new-state with the old-state
    #    Only override matching-keys in the Parameter/ Optimizer dictionary
    merged_backend = override_server_states(
        backend.server_states, restored_backend.server_states
    )

    # 5) Restore updated state to the service
    restored = broker.BrokerState(
        metrics=backend.metrics,
        control_flow=backend.control_flow,
        server_states=merged_backend
    )
    client.datastream.send(
        broker.ClientRequest(broker.Control.RESTORE, restored)
    )

    # 6) Modify Client state to continue the experiment from the restore-point.
    client.start = client.step = restore_state.step
    if client.eval_disabled:
        client.eval_disabled = client._toggle_test_servers(True)
