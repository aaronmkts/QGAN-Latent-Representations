from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .vqvae import VectorQuantizer, _flatten_latents


class SpatialEncoder(nn.Module):
    """Encoder that preserves spatial dimensions for spatial VQ-VAE.

    Output: (batch, 7, 7, embedding_dim) for 28x28 input.
    """

    channels: Sequence[int]
    embedding_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Conv1: (batch, 28, 28, 1) -> (batch, 14, 14, channels[0])
        x = nn.Conv(self.channels[0], kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)

        # Conv2: (batch, 14, 14, channels[0]) -> (batch, 7, 7, channels[1])
        x = nn.Conv(self.channels[1], kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)

        # 1x1 conv to embedding_dim: (batch, 7, 7, channels[1]) -> (batch, 7, 7, embedding_dim)
        x = nn.Conv(self.embedding_dim, kernel_size=(1, 1), strides=(1, 1))(x)
        return x


class SpatialDecoder(nn.Module):
    """Decoder that upsamples from spatial codebook grid back to image.

    Input: (batch, 7, 7, embedding_dim)
    Output: (batch, 28, 28, 1)
    """

    channels: Sequence[int]

    @nn.compact
    def __call__(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # z: (batch, 7, 7, embedding_dim)
        # Upsample: (batch, 7, 7, embedding_dim) -> (batch, 14, 14, channels[0])
        x = nn.ConvTranspose(
            self.channels[0], kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(z)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)

        # Upsample: (batch, 14, 14, channels[0]) -> (batch, 28, 28, channels[1])
        x = nn.ConvTranspose(
            self.channels[1], kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)

        # Output: (batch, 28, 28, channels[1]) -> (batch, 28, 28, 1)
        x = nn.ConvTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x


class SpatialVQVAE(nn.Module):
    """VQ-VAE with spatial codebook indices.

    Encodes each image to a (7, 7) grid of codebook indices.
    Each spatial position is independently quantized to one of
    num_embeddings codebook entries.
    """

    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    num_embeddings: int
    embedding_dim: int

    def setup(self) -> None:
        self.encoder = SpatialEncoder(
            channels=self.encoder_channels,
            embedding_dim=self.embedding_dim,
        )
        self.decoder = SpatialDecoder(channels=self.decoder_channels)
        self.quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )

    def __call__(
        self, x: jnp.ndarray, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_e = self.encoder(x, train=train)  # (batch, 7, 7, embedding_dim)
        z_q, codebook_loss, commitment_loss, perplexity = self.quantizer(z_e)
        recon = self.decoder(z_q, train=train)
        return recon, codebook_loss, commitment_loss, perplexity

    def encode(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Encode to quantized spatial latents."""
        z_e = self.encoder(x, train=train)
        z_q, _, _, _ = self.quantizer(z_e)
        return z_q

    def encode_to_indices(
        self, x: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        """Encode images to codebook index grid.

        Returns: (batch, 7, 7) integer indices.
        """
        z_e = self.encoder(x, train=train)  # (batch, 7, 7, embedding_dim)
        batch_size = z_e.shape[0]
        h, w = z_e.shape[1], z_e.shape[2]

        # Flatten spatial dims for distance computation
        z_flat, _, _ = _flatten_latents(z_e, self.embedding_dim)
        # z_flat: (batch*h*w, embedding_dim)

        # Access codebook embeddings from quantizer
        embeddings = self.quantizer.variables["params"]["embeddings"]

        z_sq = jnp.sum(z_flat ** 2, axis=1, keepdims=True)
        e_sq = jnp.sum(embeddings ** 2, axis=1)
        distances = z_sq - 2.0 * (z_flat @ embeddings.T) + e_sq[None, :]
        indices = jnp.argmin(distances, axis=1)  # (batch*h*w,)
        return indices.reshape(batch_size, h, w)

    def decode(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Decode from quantized spatial latents."""
        return self.decoder(z, train=train)

    def decode_from_indices(
        self, indices: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        """Decode from codebook index grid.

        Args:
            indices: (batch, 7, 7) integer indices
        Returns:
            (batch, 28, 28, 1) reconstructed images
        """
        embeddings = self.quantizer.variables["params"]["embeddings"]
        # (batch, 7, 7) -> lookup -> (batch, 7, 7, embedding_dim)
        z_q = embeddings[indices]
        return self.decoder(z_q, train=train)


def init_spatial_vqvae_variables(rng, model: SpatialVQVAE, input_shape: Sequence[int]) -> dict:
    dummy = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    return model.init(rng, dummy, train=True)
