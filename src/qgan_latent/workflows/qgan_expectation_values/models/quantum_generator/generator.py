from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable
from .circuits import CircuitConfig, make_style_based_circuit
from .observables import FixedPauliBank

# ... init_generator_params remains the same ...
def init_generator_params(rng, n_qubits: int, depth: int, noise_dim: int) -> dict:
    params_per_gate = 2
    total_circuit_angles = depth * n_qubits * params_per_gate

    rng_w, rng_b = jax.random.split(rng)

    style_w = jax.random.uniform(
        rng_w, shape=(noise_dim, total_circuit_angles), minval=-0.01, maxval=0.01
    )
    style_b = jax.random.uniform(
        rng_b, shape=(total_circuit_angles,), minval=-0.01, maxval=0.01
    )

    return {"style_w": style_w, "style_b": style_b}


def build_generator_apply(
    n_qubits: int,
    depth: int,
    observable_bank: FixedPauliBank | None = None,
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:

    # 1. Create the circuit (single sample version)
    circuit = make_style_based_circuit(
        CircuitConfig(
            n_qubits=n_qubits,
            depth=depth,
            observable_bank=observable_bank or FixedPauliBank(n_qubits),
        )
    )

    # 2. Vmap it immediately to handle batches
    # in_axes=(0,) means "vectorize over the 0-th dimension of the input arguments"
    batched_circuit = jax.vmap(circuit, in_axes=(0,))

    angles_shape = (depth, n_qubits, 2)

    def apply(params: dict, noise: jnp.ndarray) -> jnp.ndarray:
        # Affine Transformation
        flat_angles = noise @ params["style_w"] + params["style_b"]

        # Reshape to (Batch, Depth, Qubits, 2)
        batch_angles = flat_angles.reshape((-1,) + angles_shape)

        # Run the vmapped circuit
        fake_features = batched_circuit(batch_angles)

        # Stack output if PennyLane returns a list/tuple of arrays
        if isinstance(fake_features, (list, tuple)):
            fake_features = jnp.stack(fake_features, axis=-1)

        return fake_features

    return apply

def sample_noise(rng, batch_size: int, noise_dim: int) -> jnp.ndarray:
    return jax.random.normal(rng, (batch_size, noise_dim))
