from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .autoencoder import Decoder


class VariationalEncoder(nn.Module):
    latent_dim: int
    channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        if self.tanh_latent:
            mu = nn.tanh(mu)
        return mu, logvar


class VariationalAutoencoder(nn.Module):
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True

    def setup(self) -> None:
        self.encoder = VariationalEncoder(
            latent_dim=self.latent_dim,
            channels=self.encoder_channels,
            mlp_dim=self.mlp_dim,
            tanh_latent=self.tanh_latent,
        )
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            channels=self.decoder_channels,
            mlp_dim=self.mlp_dim,
        )

    def __call__(self, x: jnp.ndarray, rng: jax.random.KeyArray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu, logvar = self.encoder(x)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + eps * jnp.exp(0.5 * logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


def init_variational_params(
    init_rng, sample_rng, model: VariationalAutoencoder, input_shape: Sequence[int]
) -> dict:
    dummy = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    variables = model.init(init_rng, dummy, sample_rng)
    return variables["params"]


@struct.dataclass
class VariationalAutoencoderConfig:
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 128
    loss: str = "mse"
    kl_weight: float = 1.0


@struct.dataclass
class VariationalAutoencoderState:
    params: dict
    config: VariationalAutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _kl_loss(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))


def make_train_step(model: VariationalAutoencoder, config: VariationalAutoencoderConfig):
    @jax.jit
    def train_step(
        state: train_state.TrainState, batch: jnp.ndarray, rng: jax.random.KeyArray
    ):
        def loss_fn(params):
            recon, mu, logvar = model.apply({"params": params}, batch, rng)
            recon_loss = _recon_loss(batch, recon, config.loss)
            kl = _kl_loss(mu, logvar)
            return recon_loss + config.kl_weight * kl

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    return train_step


def fit(
    data: jnp.ndarray, config: VariationalAutoencoderConfig, rng: jax.random.KeyArray
) -> VariationalAutoencoderState:
    data = jnp.asarray(data)
    model = VariationalAutoencoder(
        latent_dim=config.latent_dim,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        mlp_dim=config.mlp_dim,
        tanh_latent=config.tanh_latent,
    )
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    params = init_variational_params(init_rng, sample_rng, model, input_shape=data.shape[1:])
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

    return VariationalAutoencoderState(params=state.params, config=config)


def transform(data: jnp.ndarray, state: VariationalAutoencoderState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = VariationalAutoencoder(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        tanh_latent=state.config.tanh_latent,
    )
    return model.apply({"params": state.params}, data, method=VariationalAutoencoder.encode)


def fit_transform(
    data: jnp.ndarray, config: VariationalAutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[VariationalAutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
