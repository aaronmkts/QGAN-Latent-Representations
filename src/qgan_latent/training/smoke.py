from __future__ import annotations

import jax.numpy as jnp


def synthetic_mnist_images(num_samples: int) -> jnp.ndarray:
    """Create deterministic MNIST-shaped images for smoke tests."""
    base = jnp.linspace(0.0, 1.0, 28 * 28, dtype=jnp.float32).reshape(1, 28, 28, 1)
    offsets = jnp.arange(num_samples, dtype=jnp.float32).reshape(num_samples, 1, 1, 1)
    return jnp.mod(base + offsets / max(num_samples, 1), 1.0)
