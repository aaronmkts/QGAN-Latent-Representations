from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


class Encoder(nn.Module):
    latent_dim: int
    channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        if self.tanh_latent:
            x = nn.tanh(x)
        return x


class Decoder(nn.Module):
    latent_dim: int
    channels: Sequence[int]
    mlp_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.mlp_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(7 * 7 * self.channels[0])(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 7, 7, self.channels[0]))
        for ch in self.channels[1:]:
            x = nn.ConvTranspose(ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(1, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True

    def setup(self) -> None:
        self.encoder = Encoder(
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


def init_autoencoder_params(rng, model: Autoencoder) -> dict:
    return init_autoencoder_params_with_shape(rng, model, input_shape=(28, 28, 1))


def init_autoencoder_params_with_shape(
    rng, model: Autoencoder, input_shape: Sequence[int]
) -> dict:
    dummy = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    variables = model.init(rng, dummy)
    return variables["params"]


@struct.dataclass
class AutoencoderConfig:
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    tanh_latent: bool = True
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 128
    loss: str = "mse"


@struct.dataclass
class AutoencoderState:
    params: dict
    config: AutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def make_train_step(model: Autoencoder, loss_type: str):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: jnp.ndarray):
        def loss_fn(params):
            recon = model.apply({"params": params}, batch)
            loss = _recon_loss(batch, recon, loss_type)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    return train_step


def fit(data: jnp.ndarray, config: AutoencoderConfig, rng: jax.random.KeyArray) -> AutoencoderState:
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
    train_step = make_train_step(model, config.loss)
    n_samples = data.shape[0]
    steps_per_epoch = math.ceil(n_samples / config.batch_size)

    for _ in range(config.epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
            batch = data[idx]
            state, _ = train_step(state, batch)

    return AutoencoderState(params=state.params, config=config)


def transform(data: jnp.ndarray, state: AutoencoderState) -> jnp.ndarray:
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
    data: jnp.ndarray, config: AutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[AutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
