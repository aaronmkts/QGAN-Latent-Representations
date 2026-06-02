from __future__ import annotations

import math
from typing import Sequence

from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

from .autoencoder import Autoencoder, init_autoencoder_variables_with_shape
from qgan_latent.utils.train_state import TrainStateWithBatchStats


@struct.dataclass
class SinkhornAutoencoderConfig:
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 128
    loss: str = "mse"
    sinkhorn_weight: float = 1.0
    lambda_sinkhorn: float | None = None
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    sinkhorn_cost: str = "l2_sq"
    prior: str = "gaussian"


@struct.dataclass
class SinkhornAutoencoderState:
    params: dict
    batch_stats: dict
    config: SinkhornAutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _cost_matrix(x: jnp.ndarray, y: jnp.ndarray, cost_type: str) -> jnp.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    if cost_type == "l2":
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    if cost_type == "l2_sq":
        return jnp.sum(diff**2, axis=-1)
    raise ValueError(f"Unsupported Sinkhorn cost type: {cost_type}")


def _sample_prior(
    rng: jax.random.KeyArray, shape: Sequence[int], prior: str
) -> jnp.ndarray:
    if prior == "gaussian":
        return jax.random.normal(rng, shape)
    if prior == "sphere":
        samples = jax.random.normal(rng, shape)
        norm = jnp.linalg.norm(samples, axis=-1, keepdims=True) + 1e-8
        return samples / norm
    raise ValueError(f"Unsupported Sinkhorn prior: {prior}")


def _sinkhorn_distance(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: float, iters: int, cost_type: str = "l2_sq"
) -> jnp.ndarray:
    epsilon = jnp.maximum(epsilon, 1e-8)
    cost = _cost_matrix(x, y, cost_type)
    log_k = -cost / epsilon
    n = x.shape[0]
    m = y.shape[0]
    log_a = -jnp.log(n)
    log_b = -jnp.log(m)
    log_u = jnp.zeros((n,))
    log_v = jnp.zeros((m,))

    def body(_, state):
        log_u, log_v = state
        log_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_k.T + log_u[None, :], axis=1)
        return log_u, log_v

    log_u, log_v = jax.lax.fori_loop(0, iters, body, (log_u, log_v))
    transport = jnp.exp(log_u[:, None] + log_k + log_v[None, :])
    return jnp.sum(transport * cost)


def make_train_step(model: Autoencoder, config: SinkhornAutoencoderConfig):
    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats, batch: jnp.ndarray, rng: jax.random.KeyArray
    ):
        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            z, enc_updates = model.apply(
                variables, batch, method=Autoencoder.encode, train=True, mutable=["batch_stats"]
            )
            variables = {"params": params, "batch_stats": enc_updates["batch_stats"]}
            recon, dec_updates = model.apply(
                variables, z, method=Autoencoder.decode, train=True, mutable=["batch_stats"]
            )
            recon_loss = _recon_loss(batch, recon, config.loss)
            target = _sample_prior(rng, z.shape, config.prior)
            sinkhorn = _sinkhorn_distance(
                z, target, config.sinkhorn_eps, config.sinkhorn_iters, config.sinkhorn_cost
            )
            weight = config.sinkhorn_weight if config.lambda_sinkhorn is None else config.lambda_sinkhorn
            return recon_loss + weight * sinkhorn, dec_updates["batch_stats"]

        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        return new_state, loss

    return train_step


def fit(
    data: jnp.ndarray, config: SinkhornAutoencoderConfig, rng: jax.random.KeyArray
) -> SinkhornAutoencoderState:
    data = jnp.asarray(data)
    model = Autoencoder(
        latent_dim=config.latent_dim,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        mlp_dim=config.mlp_dim,
        tanh_latent=config.tanh_latent,
    )
    rng, init_rng = jax.random.split(rng)
    variables = init_autoencoder_variables_with_shape(
        init_rng, model, input_shape=data.shape[1:]
    )
    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optax.adam(config.learning_rate),
    )
    train_step = make_train_step(model, config)
    n_samples = data.shape[0]
    steps_per_epoch = math.ceil(n_samples / config.batch_size)

    for _ in range(config.epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
            batch = data[idx]
            rng, step_rng = jax.random.split(rng)
            state, _ = train_step(state, batch, step_rng)

    return SinkhornAutoencoderState(
        params=state.params, batch_stats=state.batch_stats, config=config
    )


def transform(data: jnp.ndarray, state: SinkhornAutoencoderState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = Autoencoder(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        tanh_latent=state.config.tanh_latent,
    )
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    return model.apply(variables, data, method=Autoencoder.encode, train=False)


def fit_transform(
    data: jnp.ndarray, config: SinkhornAutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[SinkhornAutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
