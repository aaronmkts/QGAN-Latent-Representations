from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class NMFConfig:
    n_components: int
    max_iters: int = 200
    encode_iters: int = 50
    eps: float = 1e-8


@struct.dataclass
class NMFState:
    basis: jnp.ndarray
    config: NMFConfig


def _flatten(data: jnp.ndarray) -> jnp.ndarray:
    data = jnp.asarray(data)
    if data.ndim == 1:
        return data[None, :]
    if data.ndim > 2:
        return data.reshape((data.shape[0], -1))
    return data


def _nmf_update(x: jnp.ndarray, w: jnp.ndarray, h: jnp.ndarray, eps: float) -> tuple:
    h_num = w.T @ x
    h_den = (w.T @ w @ h) + eps
    h = h * (h_num / h_den)
    w_num = x @ h.T
    w_den = (w @ h @ h.T) + eps
    w = w * (w_num / w_den)
    return w, h


def fit(data: jnp.ndarray, config: NMFConfig, rng: jax.random.KeyArray) -> NMFState:
    x = _flatten(data)
    rng, w_rng, h_rng = jax.random.split(rng, 3)
    w = jax.random.uniform(w_rng, (x.shape[0], config.n_components))
    h = jax.random.uniform(h_rng, (config.n_components, x.shape[1]))

    def body(_, state):
        w, h = state
        w, h = _nmf_update(x, w, h, config.eps)
        return w, h

    w, h = jax.lax.fori_loop(0, config.max_iters, body, (w, h))
    return NMFState(basis=h, config=config)


def transform(data: jnp.ndarray, state: NMFState) -> jnp.ndarray:
    x = _flatten(data)
    h = state.basis
    w = jnp.maximum(x @ h.T, state.config.eps)

    def body(_, w_state):
        w_num = x @ h.T
        w_den = (w_state @ h @ h.T) + state.config.eps
        w_state = w_state * (w_num / w_den)
        return w_state

    w = jax.lax.fori_loop(0, state.config.encode_iters, body, w)
    return w


def fit_transform(
    data: jnp.ndarray, config: NMFConfig, rng: jax.random.KeyArray
) -> tuple[NMFState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
