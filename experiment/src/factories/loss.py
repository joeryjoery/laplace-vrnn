from typing import Any, Generic, TYPE_CHECKING

from axme.core import SafeModule
from axme.consumer import Loss, CompositeLoss
from axme.factory import Factory

from lvrnn import vrnn


if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class ELBOFactory(Factory[Loss], Generic[SafeModule]):
    elbo: str
    elbo_kwargs: dict[str, Any]

    target_loss: list[str]
    target_loss_kwargs: list[dict[str, Any]]
    target_weights: list[float]

    ignore_model_complexity: bool
    simplify_model_complexity: bool

    deterministic: bool = False

    def make(
            self,
            module: SafeModule
    ) -> Loss:
        if not self.target_loss:
            raise ValueError("No Loss specified...")

        target_loss = CompositeLoss(module, [], self.target_weights)

        for option, option_kwargs in zip(
                self.target_loss, self.target_loss_kwargs
        ):

            module_tree = vrnn.losses
            for branch in option.split('.'):
                module_tree = getattr(module_tree, branch)
            loss_fun = module_tree

            target_loss += loss_fun(module, **option_kwargs)

        # Ensure KL-divergence is defined for the given parameterization.
        if self.deterministic or self.ignore_model_complexity:
            divergence = vrnn.losses.elbo.IgnoreDivergence()
        elif not self.deterministic:
            if self.simplify_model_complexity:
                # Prunes the log-determinant and trace terms from the MVN-KL.
                divergence = vrnn.losses.elbo.MahalanobisDivergence()
            else:
                divergence = vrnn.losses.elbo.MAPDivergence()
        else:
            raise RuntimeError()  # Unreachable

        elbo_cls = getattr(vrnn.losses.elbo, self.elbo)
        loss = elbo_cls(
            module,
            target_loss,
            divergence,
            **self.elbo_kwargs
        )

        return loss
