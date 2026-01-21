from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from models.compression_methods.autoencoder import AutoencoderConfig, fit, transform
from models.compression_methods.nmf import NMFConfig, fit as fit_nmf, transform as transform_nmf
from models.compression_methods.pca import PCAConfig, fit as fit_pca, transform as transform_pca


class TestCompressionMethods(unittest.TestCase):
    def test_pca_shapes_and_determinism(self) -> None:
        data = jnp.arange(40, dtype=jnp.float32).reshape((10, 4))
        config = PCAConfig(n_components=2)
        state_a = fit_pca(data, config)
        state_b = fit_pca(data, config)
        self.assertEqual(state_a.components.shape, (4, 2))
        self.assertTrue(jnp.allclose(state_a.components, state_b.components))
        compressed = transform_pca(data, state_a)
        self.assertEqual(compressed.shape, (10, 2))

    def test_autoencoder_shapes_and_determinism(self) -> None:
        rng = jax.random.PRNGKey(0)
        data = jax.random.normal(rng, (8, 28, 28, 1))
        config = AutoencoderConfig(
            latent_dim=4,
            encoder_channels=(4, 8),
            decoder_channels=(8, 4),
            mlp_dim=16,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            loss="mse",
        )
        state_a = fit(data, config, rng)
        state_b = fit(data, config, rng)
        encoded_a = transform(data, state_a)
        encoded_b = transform(data, state_b)
        self.assertEqual(encoded_a.shape, (8, 4))
        self.assertTrue(jnp.allclose(encoded_a, encoded_b, atol=1e-5))

    def test_nmf_shapes_and_determinism(self) -> None:
        rng = jax.random.PRNGKey(123)
        data = jnp.abs(jax.random.normal(rng, (12, 6)))
        config = NMFConfig(n_components=3, max_iters=20, encode_iters=10)
        state_a = fit_nmf(data, config, rng)
        state_b = fit_nmf(data, config, rng)
        self.assertTrue(jnp.allclose(state_a.basis, state_b.basis, atol=1e-5))
        encoded = transform_nmf(data, state_a)
        self.assertEqual(encoded.shape, (12, 3))


if __name__ == "__main__":
    unittest.main()
