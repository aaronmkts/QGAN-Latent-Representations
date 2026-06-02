from __future__ import annotations

import importlib
import sys
import warnings


def _import_with_deprecation(module_name: str):
    sys.modules.pop(module_name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(module_name)
    assert any(item.category is DeprecationWarning for item in caught), module_name
    return module


def test_new_shared_namespace_imports() -> None:
    modules = [
        "qgan_latent.shared.datamodules.mnist",
        "qgan_latent.shared.representations.autoencoder",
        "qgan_latent.shared.representations.variational_autoencoder",
        "qgan_latent.shared.representations.sinkhorn_autoencoder",
        "qgan_latent.shared.representations.sinkclass_autoencoder",
        "qgan_latent.shared.representations.vqvae",
        "qgan_latent.shared.representations.spatial_vqvae",
        "qgan_latent.shared.training.pretrain_loop",
        "qgan_latent.shared.smoke",
        "qgan_latent.shared.utils.checkpointing",
        "qgan_latent.shared.utils.device",
        "qgan_latent.shared.utils.metrics_wrapper",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_new_workflow_namespace_imports() -> None:
    modules = [
        "qgan_latent.workflows.qgan_expectation_values.models.discriminator",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.circuits",
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.generator",
        "qgan_latent.workflows.qgan_expectation_values.training.gan_loop",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps",
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.quimb_mps",
        "qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_legacy_wrappers_reexport_shared_classes() -> None:
    legacy_ae = _import_with_deprecation("qgan_latent.models.compression_methods.autoencoder")
    new_ae = importlib.import_module("qgan_latent.shared.representations.autoencoder")
    assert legacy_ae.Autoencoder is new_ae.Autoencoder
    assert legacy_ae.init_autoencoder_variables_with_shape is new_ae.init_autoencoder_variables_with_shape

    legacy_vqvae = _import_with_deprecation("qgan_latent.models.compression_methods.vqvae")
    new_vqvae = importlib.import_module("qgan_latent.shared.representations.vqvae")
    assert legacy_vqvae.VQVAE is new_vqvae.VQVAE

    legacy_checkpointing = _import_with_deprecation("qgan_latent.utils.checkpointing")
    new_checkpointing = importlib.import_module("qgan_latent.shared.utils.checkpointing")
    assert legacy_checkpointing.save_checkpoint is new_checkpointing.save_checkpoint


def test_legacy_wrappers_reexport_workflow_classes_and_functions() -> None:
    legacy_disc = _import_with_deprecation("qgan_latent.models.discriminator")
    new_disc = importlib.import_module(
        "qgan_latent.workflows.qgan_expectation_values.models.discriminator"
    )
    assert legacy_disc.Discriminator is new_disc.Discriminator
    assert legacy_disc.init_discriminator_params is new_disc.init_discriminator_params

    legacy_qgen = _import_with_deprecation("qgan_latent.models.quantum_generator")
    new_qgen = importlib.import_module(
        "qgan_latent.workflows.qgan_expectation_values.models.quantum_generator"
    )
    assert legacy_qgen.init_generator_params is new_qgen.init_generator_params
    assert legacy_qgen.build_generator_apply is new_qgen.build_generator_apply

    legacy_mps = _import_with_deprecation("qgan_latent.models.mps_prior.mps")
    new_mps = importlib.import_module(
        "qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps"
    )
    assert legacy_mps.init_mps_params is new_mps.init_mps_params
    assert legacy_mps.mps_nll_loss is new_mps.mps_nll_loss


def test_legacy_training_wrappers_reexport_workflow_functions() -> None:
    legacy_pretrain = _import_with_deprecation("qgan_latent.training.pretrain_loop")
    new_pretrain = importlib.import_module("qgan_latent.shared.training.pretrain_loop")
    assert legacy_pretrain.run_pretrain is new_pretrain.run_pretrain

    legacy_gan = _import_with_deprecation("qgan_latent.training.gan_loop")
    new_gan = importlib.import_module(
        "qgan_latent.workflows.qgan_expectation_values.training.gan_loop"
    )
    assert legacy_gan.run_gan is new_gan.run_gan

    legacy_mps_loop = _import_with_deprecation("qgan_latent.training.mps_prior_loop")
    new_mps_loop = importlib.import_module(
        "qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop"
    )
    assert legacy_mps_loop.run_mps_prior is new_mps_loop.run_mps_prior


def test_legacy_package_level_exports_are_preserved() -> None:
    legacy_datamodules = _import_with_deprecation("qgan_latent.datamodules")
    canonical_datamodules = importlib.import_module("qgan_latent.shared.datamodules")

    assert legacy_datamodules.MNISTDataModule is canonical_datamodules.MNISTDataModule
    assert legacy_datamodules.load_mnist is canonical_datamodules.load_mnist

    legacy_training = _import_with_deprecation("qgan_latent.training")
    canonical_pretrain = importlib.import_module("qgan_latent.shared.training")
    canonical_gan = importlib.import_module(
        "qgan_latent.workflows.qgan_expectation_values.training"
    )
    canonical_prior = importlib.import_module(
        "qgan_latent.workflows.tensor_prior_vqvae.training"
    )

    assert legacy_training.run_pretrain is canonical_pretrain.run_pretrain
    assert legacy_training.run_gan is canonical_gan.run_gan
    assert legacy_training.run_mps_prior is canonical_prior.run_mps_prior
