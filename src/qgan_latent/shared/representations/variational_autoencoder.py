from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax

from .autoencoder import Decoder, Encoder
from qgan_latent.shared.utils.train_state import TrainStateWithBatchStats


class VariationalAutoencoder(nn.Module):
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
        self.mu_head = nn.Dense(self.latent_dim)
        self.logvar_head = nn.Dense(self.latent_dim)
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            channels=self.decoder_channels,
            mlp_dim=self.mlp_dim,
        )

    def _encode_stats(self, x: jnp.ndarray, train: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = self.encoder(x, train=train)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def __call__(
        self, x: jnp.ndarray, rng: jax.random.KeyArray, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mu, logvar = self._encode_stats(x, train=train)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + eps * jnp.exp(0.5 * logvar)
        recon = self.decoder(z, train=train)
        return recon, mu, logvar

    def encode(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        mu, _ = self._encode_stats(x, train=train)
        return mu

    def decode(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        return self.decoder(z, train=train)


def init_variational_variables(
    init_rng, sample_rng, model: VariationalAutoencoder, input_shape: Sequence[int]
) -> dict:
    dummy = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    variables = model.init(init_rng, dummy, sample_rng, train=True)
    return variables


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
    beta: float = 1.0
    kl_weight: float | None = None
    kl_anneal_steps: int = 0


@struct.dataclass
class VariationalAutoencoderState:
    params: dict
    batch_stats: dict
    config: VariationalAutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _kl_loss(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))


def _kl_weight(config: VariationalAutoencoderConfig, step: int | None) -> jnp.ndarray:
    weight = config.beta if config.kl_weight is None else config.kl_weight
    if config.kl_anneal_steps and step is not None:
        step = jnp.asarray(step, dtype=jnp.float32)
        return weight * jnp.minimum(1.0, step / config.kl_anneal_steps)
    return jnp.asarray(weight, dtype=jnp.float32)


def make_train_step(model: VariationalAutoencoder, config: VariationalAutoencoderConfig):
    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int = 0,
    ):
        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            outputs, updates = model.apply(
                variables, batch, rng, train=True, mutable=["batch_stats"]
            )
            recon, mu, logvar = outputs
            recon_loss = _recon_loss(batch, recon, config.loss)
            kl = _kl_loss(mu, logvar)
            weight = _kl_weight(config, step)
            return recon_loss + weight * kl, updates["batch_stats"]

        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
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
    variables = init_variational_variables(
        init_rng, sample_rng, model, input_shape=data.shape[1:]
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

    global_step = 0
    for _ in range(config.epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
            batch = data[idx]
            rng, step_rng = jax.random.split(rng)
            step = jnp.asarray(global_step)
            state, _ = train_step(state, batch, step_rng, step)
            global_step += 1

    return VariationalAutoencoderState(
        params=state.params, batch_stats=state.batch_stats, config=config
    )


def transform(data: jnp.ndarray, state: VariationalAutoencoderState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = VariationalAutoencoder(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        tanh_latent=state.config.tanh_latent,
    )
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    return model.apply(
        variables, data, method=VariationalAutoencoder.encode, train=False
    )


def fit_transform(
    data: jnp.ndarray, config: VariationalAutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[VariationalAutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
