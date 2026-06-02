from __future__ import annotations

import jax
import jax.numpy as jnp

from qgan_latent.shared.representations.autoencoder import (
    Autoencoder,
    init_autoencoder_variables_with_shape,
)
from qgan_latent.shared.representations.spatial_vqvae import (
    SpatialVQVAE,
    init_spatial_vqvae_variables,
)
from qgan_latent.workflows.qgan_expectation_values.models.discriminator import Discriminator, init_discriminator_params
from qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps import init_mps_params, mps_nll_loss, mps_sample
from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import build_generator_apply, init_generator_params
from qgan_latent.shared.utils.checkpointing import load_checkpoint, save_checkpoint


def test_model_initializers_and_checkpoint_round_trip(tmp_path) -> None:
    rng = jax.random.PRNGKey(0)

    autoencoder = Autoencoder(
        latent_dim=4,
        encoder_channels=[4, 8],
        decoder_channels=[8, 4],
        mlp_dim=16,
        tanh_latent=True,
    )
    variables = init_autoencoder_variables_with_shape(rng, autoencoder, input_shape=(28, 28, 1))
    checkpoint_path = tmp_path / "autoencoder.ckpt"

    save_checkpoint(checkpoint_path, variables)
    restored = load_checkpoint(checkpoint_path, variables)

    assert restored.keys() == variables.keys()

    vqvae = SpatialVQVAE(
        encoder_channels=[4, 8],
        decoder_channels=[8, 4],
        num_embeddings=4,
        embedding_dim=2,
    )
    vq_vars = init_spatial_vqvae_variables(rng, vqvae, input_shape=(28, 28, 1))
    assert "params" in vq_vars

    discriminator = Discriminator(channels=[4, 8], mlp_dim=16)
    disc_params = init_discriminator_params(rng, discriminator, latent_dim=4)
    assert disc_params


def test_quantum_generator_and_mps_shapes() -> None:
    rng = jax.random.PRNGKey(0)

    gen_params = init_generator_params(rng, n_qubits=2, depth=1, noise_dim=2)
    gen_apply = build_generator_apply(n_qubits=2, depth=1)
    fake_features = gen_apply(gen_params, jnp.zeros((2, 2)))
    assert fake_features.shape == (2, 4)

    mps_params = init_mps_params(rng, n_sites=3, phys_dim=2, bond_dim=2)
    batch = jnp.zeros((2, 3), dtype=jnp.int32)
    loss = mps_nll_loss(mps_params, batch)
    samples = mps_sample(mps_params, rng, n_samples=2)

    assert jnp.isfinite(loss)
    assert samples.shape == (2, 3)
