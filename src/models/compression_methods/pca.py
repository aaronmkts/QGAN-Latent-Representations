from __future__ import annotations

from flax import struct
import jax.numpy as jnp


@struct.dataclass
class PCAConfig:
    n_components: int


@struct.dataclass
class PCAState:
    mean: jnp.ndarray
    components: jnp.ndarray


def _flatten(data: jnp.ndarray) -> jnp.ndarray:
    data = jnp.asarray(data)
    if data.ndim == 1:
        return data[None, :]
    if data.ndim > 2:
        return data.reshape((data.shape[0], -1))
    return data


def fit(data: jnp.ndarray, config: PCAConfig, rng=None) -> PCAState:
    del rng
    x = _flatten(data)
    mean = jnp.mean(x, axis=0)
    x_centered = x - mean
    _, _, vt = jnp.linalg.svd(x_centered, full_matrices=False)
    components = vt[: config.n_components].T
    return PCAState(mean=mean, components=components)


def transform(data: jnp.ndarray, state: PCAState) -> jnp.ndarray:
    x = _flatten(data)
    return (x - state.mean) @ state.components


def fit_transform(data: jnp.ndarray, config: PCAConfig, rng=None) -> tuple[PCAState, jnp.ndarray]:
    state = fit(data, config, rng=rng)
    return state, transform(data, state)
