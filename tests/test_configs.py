from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(name: str):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=name)


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
