from copy import deepcopy
from typing import Any

import jax.numpy as jnp
import optax
from flax import linen as nn

from jvm.registry import ModelRegistry, Registry


def build_object(cfg: dict, registry: Registry) -> Any:

    _cfg = deepcopy(cfg)
    name = _cfg.pop("name")

    return registry(name)(**_cfg)


def build_model(model_cfg: dict) -> nn.Module:

    _model_cfg = deepcopy(model_cfg)

    model_name = _model_cfg.pop("name")
    model = ModelRegistry(model_name)

    return model()


def build_optax_object(cfg: dict) -> Any:

    _cfg = deepcopy(cfg)
    name = _cfg.pop("name")

    return getattr(optax, name)(**_cfg)


def build_optimizer(
        optimizer_cfg: dict) -> optax.GradientTransformationExtraArgs:

    if 'scheduler' in optimizer_cfg:
        scheduler_cfg = optimizer_cfg.pop('scheduler')
        scheduler = build_optax_object(scheduler_cfg)
        optimizer = getattr(optax, optimizer_cfg.pop('name'))
        optimizer = optax.inject_hyperparams(optimizer)(
            learning_rate=scheduler, **optimizer_cfg)
    else:
        optimizer = build_optax_object(optimizer_cfg)
    return optimizer


def build_loss_function(loss_cfg: dict) -> Any:

    # TODO: build loss function chain
    _loss_cfg = deepcopy(loss_cfg)
    loss_name = _loss_cfg.pop('name')

    return getattr(optax, loss_name)


# class OptaxLossFunction(object):

#     def __init__(self, loss_cfg: dict, reduce: str = "mean") -> None:
#         self._loss_cfg = deepcopy(loss_cfg)
#         self._reduce = reduce
#         self._loss_name = self._loss_cfg.pop('name')
#         self._loss = getattr(optax, self._loss_name)

#     def __call__(self, *args) -> jnp.ndarray:
#         losses = self._loss(*args, **self._loss_cfg)
#         if self._reduce == "mean":
#             return jnp.mean(losses)
#         elif self._reduce == "sum":
#             return jnp.sum(losses)
#         elif self._reduce == "none":
#             return losses

#     def __repr__(self) -> str:
#         return f"OptaxLossFunction(loss_name={self._loss_name})"
