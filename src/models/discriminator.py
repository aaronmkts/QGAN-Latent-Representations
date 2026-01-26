from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

class Discriminator(nn.Module):
    channels: int  # Kept for config compatibility, but unused for MLP
    mlp_dim: int   # Hidden layer size (e.g., 100)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input x shape: (batch_size, latent_dim)
        # The paper specifies operating on the latent feature vector [cite: 114]
        
        # 1. First Dense Layer (Paper uses 100 nodes for MNIST) 
        x = nn.Dense(features=100)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 2. Second Dense Layer (Paper uses 50 nodes for MNIST) 
        x = nn.Dense(features=50)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 3. Output Layer (Scalar Score)
        # "returns a scalar value measuring the realness" [cite: 114]
        x = nn.Dense(features=1)(x)
        
        return x

def init_discriminator_params(rng, discriminator: Discriminator, latent_dim: int):
    """
    Initializes the discriminator with a dummy latent vector.
    """
    dummy_input = jnp.ones((1, latent_dim))
    return discriminator.init(rng, dummy_input)["params"]