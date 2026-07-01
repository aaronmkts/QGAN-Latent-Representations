from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FourierRFFSurrogate:
    noise_dim: int
    output_dim: int
    rff_dim: int
    hidden_dim: int
    sigma: float
    seed: int

    def apply(self, params: dict, noise: jnp.ndarray) -> jnp.ndarray:
        return build_fourier_apply(
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            rff_dim=self.rff_dim,
            hidden_dim=self.hidden_dim,
            sigma=self.sigma,
            seed=self.seed,
        )(params, noise)


def matched_rff_dim(n_qubits: int, depth: int) -> int:
    return int(depth) * int(n_qubits) * 2


def fourier_features(noise: jnp.ndarray, *, rff_dim: int, sigma: float, seed: int) -> jnp.ndarray:
    rng_w, rng_b = jax.random.split(jax.random.PRNGKey(int(seed)))
    noise_dim = int(noise.shape[-1])
    weights = jax.random.normal(rng_w, (noise_dim, int(rff_dim))) / float(sigma)
    bias = jax.random.uniform(rng_b, (int(rff_dim),), minval=0.0, maxval=2.0 * jnp.pi)
    projection = noise @ weights + bias
    scale = jnp.sqrt(2.0 / float(rff_dim))
    return scale * jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)


def init_fourier_params(rng, *, noise_dim: int, output_dim: int, rff_dim: int, hidden_dim: int) -> dict:
    del noise_dim
    rng_w1, rng_b1, rng_w2, rng_b2 = jax.random.split(rng, 4)
    feature_dim = int(rff_dim) * 2
    return {
        "w1": jax.random.normal(rng_w1, (feature_dim, int(hidden_dim))) * jnp.sqrt(2.0 / feature_dim),
        "b1": jax.random.normal(rng_b1, (int(hidden_dim),)) * 0.01,
        "w2": jax.random.normal(rng_w2, (int(hidden_dim), int(output_dim))) * jnp.sqrt(2.0 / int(hidden_dim)),
        "b2": jax.random.normal(rng_b2, (int(output_dim),)) * 0.01,
    }


def build_fourier_apply(
    *,
    noise_dim: int,
    output_dim: int,
    rff_dim: int,
    hidden_dim: int,
    sigma: float,
    seed: int,
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:
    del noise_dim, output_dim, hidden_dim

    def apply(params: dict, noise: jnp.ndarray) -> jnp.ndarray:
        features = fourier_features(noise, rff_dim=rff_dim, sigma=sigma, seed=seed)
        hidden = jnp.tanh(features @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]

    return apply


def fourier_parameter_count(noise_dim: int, output_dim: int, rff_dim: int, hidden_dim: int) -> dict[str, int]:
    counts = {
        "rff_weights": int(noise_dim) * int(rff_dim),
        "rff_bias": int(rff_dim),
        "w1": int(rff_dim) * 2 * int(hidden_dim),
        "b1": int(hidden_dim),
        "w2": int(hidden_dim) * int(output_dim),
        "b2": int(output_dim),
    }
    counts["total"] = sum(counts.values())
    return counts


def fourier_spectrum_counts(noise_dim: int, output_dim: int, rff_dim: int, hidden_dim: int) -> dict[str, int]:
    return fourier_parameter_count(noise_dim, output_dim, rff_dim, hidden_dim)
