import jax.numpy as jnp
import torch


class ToJAXArray(object):

    def __call__(self, tensor: torch.Tensor) -> jnp.ndarray:

        array = jnp.array(tensor)

        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        elif array.ndim == 3:
            array = array.transpose(0, 2, 1)

        return array

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
