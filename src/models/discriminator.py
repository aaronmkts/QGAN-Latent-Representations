from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class Discriminator(nn.Module):
    channels: Sequence[int]
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


def init_discriminator_params(rng, model: Discriminator) -> dict:
    dummy = jnp.zeros((1, 28, 28, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy)
    return variables["params"]
