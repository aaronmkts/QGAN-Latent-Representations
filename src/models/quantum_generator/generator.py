from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Callable

from .circuits import CircuitConfig, make_circuit


def init_generator_params(rng, n_qubits: int, depth: int, noise_dim: int, latent_dim: int) -> dict:
    rng, q_rng, noise_rng, linear_rng = jax.random.split(rng, 4)
    q_params = 0.01 * jax.random.normal(q_rng, (depth, n_qubits, 3))
    noise_w = 0.02 * jax.random.normal(noise_rng, (noise_dim, n_qubits))
    noise_b = jnp.zeros((n_qubits,))
    linear_w = 0.02 * jax.random.normal(linear_rng, (n_qubits, latent_dim))
    linear_b = jnp.zeros((latent_dim,))
    return {
        "q_params": q_params,
        "noise_w": noise_w,
        "noise_b": noise_b,
        "linear_w": linear_w,
        "linear_b": linear_b,
    }


def build_generator_apply(
    n_qubits: int,
    depth: int,
    latent_tanh: bool = True,
    architecture: str = "SimpleEntangling",
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:
    circuit = make_circuit(architecture, CircuitConfig(n_qubits=n_qubits, depth=depth))

    def apply(params: dict, noise: jnp.ndarray) -> jnp.ndarray:
        angles = noise @ params["noise_w"] + params["noise_b"]
        measurements = jax.vmap(circuit, in_axes=(None, 0))(params["q_params"], angles)
        if isinstance(measurements, (list, tuple)):
            measurements = jnp.stack(measurements, axis=-1)
        z = measurements @ params["linear_w"] + params["linear_b"]
        if latent_tanh:
            z = jnp.tanh(z)
        return z

    return apply


def sample_noise(rng, batch_size: int, noise_dim: int) -> jnp.ndarray:
    return jax.random.normal(rng, (batch_size, noise_dim))
