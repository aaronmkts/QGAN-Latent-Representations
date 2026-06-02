from __future__ import annotations

import importlib


def test_public_package_imports() -> None:
    modules = [
        "qgan_latent",
        "qgan_latent.cli.latent_pretrain",
        "qgan_latent.cli.latent_train",
        "qgan_latent.cli.vqvae_pretrain",
        "qgan_latent.cli.vqvae_train_prior",
        "qgan_latent.shared.datamodules.mnist",
        "qgan_latent.shared.representations.autoencoder",
        "qgan_latent.shared.representations.spatial_vqvae",
        "qgan_latent.workflows.qgan_expectation_values.models.discriminator",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator",
        "qgan_latent.shared.training.pretrain_loop",
        "qgan_latent.workflows.qgan_expectation_values.training.gan_loop",
        "qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop",
        "qgan_latent.shared.utils.checkpointing",
    ]

    for module in modules:
        importlib.import_module(module)
