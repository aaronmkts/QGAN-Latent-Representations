from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(name: str, overrides: list[str] | None = None):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=name, overrides=overrides or [])


def test_pretrain_config_composes() -> None:
    cfg = _compose("pretrain")

    assert cfg.model.autoencoder.name == "spatial_vqvae"
    assert cfg.wandb_mode == "disabled"


def test_train_config_composes_without_removed_knobs() -> None:
    cfg = _compose("train")

    assert cfg.model.autoencoder.name == "autoencoder"
    assert "finetune_ae" not in cfg
    assert "ae_lr" not in cfg


def test_train_prior_config_composes() -> None:
    cfg = _compose("train_prior")

    assert cfg.model.vqvae.name == "spatial_vqvae"
    assert cfg.model.mps_prior.name == "mps_prior"
    assert cfg.wandb_mode == "disabled"


def test_stage2_config_taxonomy_files_exist() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    expected_paths = [
        "shared/data/mnist.yaml",
        "shared/metrics/default.yaml",
        "shared/representations/autoencoder.yaml",
        "shared/representations/vae.yaml",
        "shared/representations/sinkhorn_ae.yaml",
        "shared/representations/vqvae.yaml",
        "shared/representations/spatial_vqvae.yaml",
        "projects/qgan_expectation_values/pretrain.yaml",
        "projects/qgan_expectation_values/train.yaml",
        "projects/qgan_expectation_values/models/quantum_generator.yaml",
        "projects/qgan_expectation_values/models/discriminator.yaml",
        "projects/tensor_prior_vqvae/pretrain.yaml",
        "projects/tensor_prior_vqvae/train_prior.yaml",
        "projects/tensor_prior_vqvae/models/mps_prior.yaml",
    ]

    missing = [rel_path for rel_path in expected_paths if not (config_dir / rel_path).exists()]

    assert missing == []


def test_project_qgan_configs_compose_directly() -> None:
    pretrain_cfg = _compose("projects/qgan_expectation_values/pretrain")
    train_cfg = _compose("projects/qgan_expectation_values/train")

    assert pretrain_cfg.model.autoencoder.name == "autoencoder"
    assert train_cfg.model.autoencoder.name == "autoencoder"
    assert train_cfg.model.quantum_generator.n_qubits == 10
    assert train_cfg.model.discriminator.channels == [32, 64]
    assert train_cfg.wandb_mode == "disabled"


def test_project_tensor_prior_configs_compose_directly() -> None:
    pretrain_cfg = _compose("projects/tensor_prior_vqvae/pretrain")
    train_prior_cfg = _compose("projects/tensor_prior_vqvae/train_prior")

    assert pretrain_cfg.model.autoencoder.name == "spatial_vqvae"
    assert train_prior_cfg.model.vqvae.name == "spatial_vqvae"
    assert train_prior_cfg.model.mps_prior.name == "mps_prior"
    assert train_prior_cfg.model.mps_prior.phys_dim == 16
    assert train_prior_cfg.wandb_mode == "disabled"


def test_documented_representation_override_syntax() -> None:
    cfg = _compose(
        "pretrain",
        overrides=["shared/representations@model.autoencoder=vae"],
    )

    assert cfg.model.autoencoder.name == "vae"
    assert cfg.data.data_dir == "data/mnist"


def test_root_config_packages_land_under_runtime_keys() -> None:
    train_cfg = _compose("train")
    train_prior_cfg = _compose("train_prior")

    assert train_cfg.data.data_dir == "data/mnist"
    assert "active_metrics" in train_cfg.metrics
    assert train_cfg.model.quantum_generator.noise_dim == 10
    assert train_cfg.model.discriminator.mlp_dim == 128
    assert train_prior_cfg.data.data_dir == "data/mnist"
    assert train_prior_cfg.model.mps_prior.sample_backend == "jax"
