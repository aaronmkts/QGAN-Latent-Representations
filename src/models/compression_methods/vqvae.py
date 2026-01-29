from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax

from .autoencoder import Decoder, Encoder
from utils.train_state import TrainStateWithBatchStats


def _flatten_latents(
    z_e: jnp.ndarray, embedding_dim: int
) -> tuple[jnp.ndarray, Sequence[int], bool]:
    if z_e.ndim == 2:
        return z_e, z_e.shape, False
    if z_e.ndim == 4:
        if z_e.shape[-1] == embedding_dim:
            z_perm = z_e
            permuted = False
        elif z_e.shape[1] == embedding_dim:
            z_perm = jnp.transpose(z_e, (0, 2, 3, 1))
            permuted = True
        else:
            raise ValueError(
                "Expected latent channels to match embedding_dim in NHWC or NCHW format."
            )
        return z_perm.reshape((-1, embedding_dim)), z_perm.shape, permuted
    raise ValueError("Vector quantizer expects 2D or 4D latent tensors.")


def _unflatten_latents(
    z_q: jnp.ndarray, shape: Sequence[int], permuted: bool
) -> jnp.ndarray:
    z_q = z_q.reshape(shape)
    if permuted:
        return jnp.transpose(z_q, (0, 3, 1, 2))
    return z_q


class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int

    @nn.compact
    def __call__(
        self, z_e: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        embeddings = self.param(
            "embeddings",
            nn.initializers.uniform(scale=1.0),
            (self.num_embeddings, self.embedding_dim),
        )

        z_flat, z_shape, permuted = _flatten_latents(z_e, self.embedding_dim)
        z_sq = jnp.sum(z_flat**2, axis=1, keepdims=True)
        e_sq = jnp.sum(embeddings**2, axis=1)
        distances = z_sq - 2.0 * (z_flat @ embeddings.T) + e_sq[None, :]

        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)
        quantized_flat = encodings @ embeddings
        quantized = _unflatten_latents(quantized_flat, z_shape, permuted)

        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - quantized) ** 2)
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(quantized)) ** 2)

        z_q = z_e + jax.lax.stop_gradient(quantized - z_e)
        avg_probs = jnp.mean(encodings, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        return z_q, codebook_loss, commitment_loss, perplexity


class VQVAE(nn.Module):
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    num_embeddings: int
    embedding_dim: int
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
        self.quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        self.pre_vq = None
        self.post_vq = None
        if self.embedding_dim != self.latent_dim:
            self.pre_vq = nn.Dense(self.embedding_dim)
            self.post_vq = nn.Dense(self.latent_dim)

    def __call__(
        self, x: jnp.ndarray, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_e = self.encoder(x, train=train)
        if self.pre_vq is not None:
            z_e = self.pre_vq(z_e)
        z_q, codebook_loss, commitment_loss, perplexity = self.quantizer(z_e)
        if self.post_vq is not None:
            z_q = self.post_vq(z_q)
        recon = self.decoder(z_q, train=train)
        return recon, codebook_loss, commitment_loss, perplexity

    def encode(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        z_e = self.encoder(x, train=train)
        if self.pre_vq is not None:
            z_e = self.pre_vq(z_e)
        z_q, _, _, _ = self.quantizer(z_e)
        if self.post_vq is not None:
            z_q = self.post_vq(z_q)
        return z_q

    def decode(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        return self.decoder(z, train=train)


@struct.dataclass
class VQVAEConfig:
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float = 0.25
    codebook_loss_weight: float = 1.0
    tanh_latent: bool = True
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 128
    loss: str = "mse"


@struct.dataclass
class VQVAEState:
    params: dict
    batch_stats: dict
    config: VQVAEConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def init_vqvae_variables_with_shape(
    rng, model: VQVAE, input_shape: Sequence[int]
) -> dict:
    dummy = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=True)
    return variables


def make_train_step(model: VQVAE, config: VQVAEConfig):
    @jax.jit
    def train_step(state: TrainStateWithBatchStats, batch: jnp.ndarray):
        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            outputs, updates = model.apply(
                variables, batch, train=True, mutable=["batch_stats"]
            )
            recon, codebook_loss, commitment_loss, _ = outputs
            recon_loss = _recon_loss(batch, recon, config.loss)
            total = (
                recon_loss
                + config.codebook_loss_weight * codebook_loss
                + config.commitment_cost * commitment_loss
            )
            return total, updates["batch_stats"]

        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        return new_state, loss

    return train_step


def fit(data: jnp.ndarray, config: VQVAEConfig, rng: jax.random.KeyArray) -> VQVAEState:
    data = jnp.asarray(data)
    model = VQVAE(
        latent_dim=config.latent_dim,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        mlp_dim=config.mlp_dim,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        tanh_latent=config.tanh_latent,
    )
    rng, init_rng = jax.random.split(rng)
    variables = init_vqvae_variables_with_shape(
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
            state, _ = train_step(state, batch)

    return VQVAEState(params=state.params, batch_stats=state.batch_stats, config=config)


def transform(data: jnp.ndarray, state: VQVAEState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = VQVAE(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        num_embeddings=state.config.num_embeddings,
        embedding_dim=state.config.embedding_dim,
        tanh_latent=state.config.tanh_latent,
    )
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    return model.apply(variables, data, method=VQVAE.encode, train=False)


def fit_transform(
    data: jnp.ndarray, config: VQVAEConfig, rng: jax.random.KeyArray
) -> tuple[VQVAEState, jnp.ndarray]:
    state = fit(data, config, rng)
    return state, transform(data, state)
