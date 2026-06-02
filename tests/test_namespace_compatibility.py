from __future__ import annotations

import importlib


def test_shared_namespace_imports() -> None:
    modules = [
        "qgan_latent.shared.datamodules",
        "qgan_latent.shared.datamodules.mnist",
        "qgan_latent.shared.representations",
        "qgan_latent.shared.representations.autoencoder",
        "qgan_latent.shared.representations.variational_autoencoder",
        "qgan_latent.shared.representations.sinkhorn_autoencoder",
        "qgan_latent.shared.representations.sinkclass_autoencoder",
        "qgan_latent.shared.representations.vqvae",
        "qgan_latent.shared.representations.spatial_vqvae",
        "qgan_latent.shared.training",
        "qgan_latent.shared.training.pretrain_loop",
        "qgan_latent.shared.smoke",
        "qgan_latent.shared.utils",
        "qgan_latent.shared.utils.checkpointing",
        "qgan_latent.shared.utils.device",
        "qgan_latent.shared.utils.metrics_wrapper",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_workflow_namespace_imports() -> None:
    modules = [
        "qgan_latent.workflows.qgan_expectation_values",
        "qgan_latent.workflows.qgan_expectation_values.models.discriminator",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.circuits",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.generator",
        "qgan_latent.workflows.qgan_expectation_values.training",
        "qgan_latent.workflows.qgan_expectation_values.training.gan_loop",
        "qgan_latent.workflows.tensor_prior_vqvae",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.quimb_mps",
        "qgan_latent.workflows.tensor_prior_vqvae.training",
        "qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_legacy_namespaces_are_removed() -> None:
    removed = [
        "qgan_latent.datamodules",
        "qgan_latent.models",
        "qgan_latent.training",
        "qgan_latent.utils",
    ]

    for module_name in removed:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy namespace still importable: {module_name}")
