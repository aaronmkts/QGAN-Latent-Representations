from __future__ import annotations

import math
from typing import Sequence

from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .autoencoder import Autoencoder, init_autoencoder_params_with_shape


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
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50


@struct.dataclass
class SinkhornAutoencoderState:
    params: dict
    config: SinkhornAutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _sinkhorn_distance(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: float, iters: int
) -> jnp.ndarray:
    cost = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    k = jnp.exp(-cost / epsilon)
    n = x.shape[0]
    m = y.shape[0]
    a = jnp.ones((n,)) / n
    b = jnp.ones((m,)) / m
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    def body(_, state):
        u, v = state
        u = a / (k @ v + 1e-8)
        v = b / (k.T @ u + 1e-8)
        return u, v

    u, v = jax.lax.fori_loop(0, iters, body, (u, v))
    transport = u[:, None] * k * v[None, :]
    return jnp.sum(transport * cost)


def make_train_step(model: Autoencoder, config: SinkhornAutoencoderConfig):
    @jax.jit
    def train_step(
        state: train_state.TrainState, batch: jnp.ndarray, rng: jax.random.KeyArray
    ):
        def loss_fn(params):
            z = model.apply({"params": params}, batch, method=Autoencoder.encode)
            recon = model.apply({"params": params}, z, method=Autoencoder.decode)
            recon_loss = _recon_loss(batch, recon, config.loss)
            target = jax.random.normal(rng, z.shape)
            sinkhorn = _sinkhorn_distance(z, target, config.sinkhorn_eps, config.sinkhorn_iters)
            return recon_loss + config.sinkhorn_weight * sinkhorn

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
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
    params = init_autoencoder_params_with_shape(init_rng, model, input_shape=data.shape[1:])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
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

    return SinkhornAutoencoderState(params=state.params, config=config)


def transform(data: jnp.ndarray, state: SinkhornAutoencoderState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = Autoencoder(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        tanh_latent=state.config.tanh_latent,
    )
    return model.apply({"params": state.params}, data, method=Autoencoder.encode)


def fit_transform(
    data: jnp.ndarray, config: SinkhornAutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[SinkhornAutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
