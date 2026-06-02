from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import run_gan
from qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop import run_mps_prior
from qgan_latent.shared.training.pretrain_loop import run_pretrain


def _compose(name: str, overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=name, overrides=overrides)


def test_pretrain_smoke_mode_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose(
        "projects/tensor_prior_vqvae/pretrain",
        [
            "smoke_test=true",
            "device=cpu",
            "wandb_mode=disabled",
            "outputs.dir=outputs/pretrain_smoke",
            "checkpoints.dir=checkpoints",
            "model.autoencoder.encoder_channels=[4,8]",
            "model.autoencoder.decoder_channels=[8,4]",
            "model.autoencoder.embedding_dim=2",
            "model.autoencoder.num_embeddings=4",
        ],
    )

    run_pretrain(cfg)

    assert (tmp_path / "checkpoints" / "spatial_vqvae.ckpt").exists()


def test_gan_smoke_mode_runs_without_pretrained_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose(
        "projects/qgan_expectation_values/train",
        [
            "smoke_test=true",
            "device=cpu",
            "wandb_mode=disabled",
            "metrics.active_metrics=[]",
            "outputs.dir=outputs/train_smoke",
            "checkpoints.dir=checkpoints",
            "model.autoencoder.latent_dim=4",
            "model.autoencoder.encoder_channels=[4,8]",
            "model.autoencoder.decoder_channels=[8,4]",
            "model.autoencoder.mlp_dim=16",
            "model.quantum_generator.n_qubits=2",
            "model.quantum_generator.noise_dim=2",
            "model.quantum_generator.depth=1",
        ],
    )

    run_gan(cfg)

    assert (tmp_path / "checkpoints" / "qgan_gen.ckpt").exists()
    assert (tmp_path / "checkpoints" / "qgan_disc.ckpt").exists()


def test_mps_prior_smoke_mode_runs_without_pretrained_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose(
        "projects/tensor_prior_vqvae/train_prior",
        [
            "smoke_test=true",
            "device=cpu",
            "wandb_mode=disabled",
            "outputs.dir=outputs/train_prior_smoke",
            "checkpoints.dir=checkpoints",
            "model.vqvae.encoder_channels=[4,8]",
            "model.vqvae.decoder_channels=[8,4]",
            "model.vqvae.embedding_dim=2",
            "model.vqvae.num_embeddings=4",
            "model.mps_prior.n_sites=49",
            "model.mps_prior.phys_dim=4",
            "model.mps_prior.bond_dim=2",
            "model.mps_prior.epochs=1",
            "model.mps_prior.batch_size=4",
            "model.mps_prior.n_sample_vis=4",
        ],
    )

    run_mps_prior(cfg)

    assert (tmp_path / "checkpoints" / "mps_prior.ckpt").exists()
