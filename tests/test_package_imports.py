from __future__ import annotations

import importlib


def test_public_package_imports() -> None:
    modules = [
        "qgan_latent",
        "qgan_latent.datamodules.mnist",
        "qgan_latent.models.compression_methods.autoencoder",
        "qgan_latent.models.compression_methods.spatial_vqvae",
        "qgan_latent.models.discriminator",
        "qgan_latent.models.mps_prior.mps",
        "qgan_latent.models.quantum_generator",
        "qgan_latent.training.pretrain_loop",
        "qgan_latent.training.gan_loop",
        "qgan_latent.training.mps_prior_loop",
        "qgan_latent.utils.checkpointing",
    ]

    for module in modules:
        importlib.import_module(module)


def test_root_wrappers_export_main() -> None:
    for module_name in ["pretrain", "train", "train_prior"]:
        module = importlib.import_module(module_name)
        assert callable(module.main)
