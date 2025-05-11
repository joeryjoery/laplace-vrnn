from __future__ import annotations
from typing import Type, Generic, Any, TypeVar
from typing_extensions import Self

import jax

T = TypeVar("T")


@jax.tree_util.register_pytree_node_class
class SerializeTree(Generic[T]):
    """Creates a Lazy object by deferring class construction.

    This modifies the PyTree serialization to be based on the __init__
    instead of the class __dict__ attribute.

    Warning for class usage with transformations inside the __init__:
    When using `vmap` over this class, be sure to also call vmap when
    using any of class type's methods!. I.e., if you create this
    object within a `vmap`, you may only be able to call `get` within another
    `vmap` scope with compatible batching-dimensions.
    """

    def __init__(
            self,
            cls_type: Type[T],
            *args,
            static_args: tuple[Any] | None = None,
            static_kwargs: dict[str, Any] | None = None,
            **kwargs
    ):
        self.cls_type = cls_type

        self.statics = (static_args or (), static_kwargs or {})
        self.vargs, self.kwargs = args, kwargs

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"cls_type={self.cls_type.__name__}, "
                f"*args={self.vargs}, "
                f"**kwargs={self.kwargs}, "
                f"statics={self.statics})")

    @property
    def get(self) -> T:  # TODO: Also allow lazy access to the class
        sv, sk = self.statics
        try:
            d = self.cls_type(*self.vargs, *sv, **self.kwargs, **sk)
        except Exception as e:
            raise RuntimeError(
                f"Instantiation of {self.cls_type} failed. Did you call "
                f"`get` outside of its creation Scope? This can result in "
                f"mismatching batching dimensions."
            ) from e

        return d

    def tree_flatten(
            self
    ) -> tuple[
        tuple[tuple, dict],
        tuple[Type[T], tuple[tuple | None, dict | None]]
    ]:
        return (self.vargs, self.kwargs), (self.cls_type, self.statics)

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: tuple[Type[T], tuple[tuple | None, dict | None]],
            children: tuple[tuple, dict]
    ) -> Self:
        vargs, kwargs = children
        obj_type, (sv, sk) = aux_data
        return cls(
            obj_type,
            *vargs, **kwargs,
            static_args=sv,
            static_kwargs=sk
        )
