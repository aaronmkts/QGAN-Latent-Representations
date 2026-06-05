from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from qgan_latent.shared.latent_cache import cache_latent_representations
from qgan_latent.shared.representations.runtime import (
    build_representation_runtime,
    init_representation_variables,
)
from qgan_latent.cli.cache_latents import run_cache_from_config


def _runtime_and_variables():
    cfg = OmegaConf.create({
        "name": "autoencoder",
        "latent_dim": 4,
        "encoder_channels": [4, 8],
        "decoder_channels": [8, 4],
        "mlp_dim": 16,
        "tanh_latent": True,
    })
    runtime = build_representation_runtime(cfg)
    variables = init_representation_variables(runtime, jax.random.PRNGKey(0), input_shape=(28, 28, 1))
    return runtime, variables


def test_cache_latent_representations_writes_split_artifacts_and_manifest(tmp_path: Path) -> None:
    runtime, variables = _runtime_and_variables()
    splits = {
        "train": jnp.zeros((3, 28, 28, 1), dtype=jnp.float32),
        "validation": jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        "test": jnp.full((1, 28, 28, 1), 0.5, dtype=jnp.float32),
    }

    manifest = cache_latent_representations(
        runtime=runtime,
        variables=variables,
        splits=splits,
        output_dir=tmp_path,
        seed=123,
        checkpoint_path=Path("checkpoints/autoencoder.ckpt"),
        batch_size=2,
    )

    assert (tmp_path / "z_train.npy").exists()
    assert (tmp_path / "z_validation.npy").exists()
    assert (tmp_path / "z_test.npy").exists()
    assert (tmp_path / "latent_stats.json").exists()
    assert (tmp_path / "empirical_quantiles.npz").exists()
    assert (tmp_path / "cache_manifest.json").exists()
    assert np.load(tmp_path / "z_train.npy").shape == (3, 4)
    assert manifest["representation"]["name"] == "autoencoder"
    assert manifest["seed"] == 123
    assert manifest["splits"]["train"]["n_samples"] == 3
    assert manifest["checkpoint_path"] == "checkpoints/autoencoder.ckpt"

    stats = json.loads((tmp_path / "latent_stats.json").read_text())
    assert set(stats) == {"train", "validation", "test"}
    assert stats["train"]["shape"] == [3, 4]
    quantiles = np.load(tmp_path / "empirical_quantiles.npz")
    assert quantiles["train"].shape == (5, 4)


def test_vae_cache_writes_mu_logvar_and_seeded_sample_files(tmp_path: Path) -> None:
    cfg = OmegaConf.create({
        "name": "vae",
        "latent_dim": 4,
        "encoder_channels": [4, 8],
        "decoder_channels": [8, 4],
        "mlp_dim": 16,
        "tanh_latent": True,
    })
    runtime = build_representation_runtime(cfg)
    variables = init_representation_variables(runtime, jax.random.PRNGKey(5), input_shape=(28, 28, 1))
    splits = {"train": jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)}

    cache_latent_representations(
        runtime=runtime,
        variables=variables,
        splits=splits,
        output_dir=tmp_path,
        seed=7,
        checkpoint_path=Path("checkpoints/vae.ckpt"),
        batch_size=2,
    )

    z = np.load(tmp_path / "z_train.npy")
    mu = np.load(tmp_path / "z_mu_train.npy")
    sampled = np.load(tmp_path / "z_sampled_train.npy")
    assert z.shape == (2, 4)
    assert mu.shape == (2, 4)
    assert np.load(tmp_path / "z_logvar_train.npy").shape == (2, 4)
    assert sampled.shape == (2, 4)
    assert np.allclose(z, mu)
    assert not np.allclose(sampled, mu)


def test_qgan_cache_latents_script_is_registered() -> None:
    pyproject = Path("pyproject.toml").read_text()
    assert "qgan-cache-latents = \"qgan_latent.cli.cache_latents:main\"" in pyproject


def test_cache_latents_cli_smoke_mode_writes_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="projects/qgan_expectation_values/train",
            overrides=[
                "smoke_test=true",
                "device=cpu",
                "outputs.dir=outputs/cache_smoke",
                "checkpoints.dir=checkpoints",
                "checkpoints.autoencoder=autoencoder.ckpt",
                "model.autoencoder.latent_dim=4",
                "model.autoencoder.encoder_channels=[4,8]",
                "model.autoencoder.decoder_channels=[8,4]",
                "model.autoencoder.mlp_dim=16",
                "model.quantum_generator.n_qubits=2",
            ],
        )

    run_cache_from_config(cfg)

    cache_dir = tmp_path / "outputs" / "cache_smoke" / "latent_cache"
    assert np.load(cache_dir / "z_train.npy").shape == (8, 4)
    assert (cache_dir / "cache_manifest.json").exists()


def test_cache_cli_prefers_representation_checkpoint_name_for_vae(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="projects/qgan_expectation_values/cache_latents",
            overrides=[
                "shared/representations@model.autoencoder=vae",
                "smoke_test=true",
                "device=cpu",
                "outputs.dir=outputs/cache_smoke",
                "checkpoints.dir=checkpoints",
                "model.autoencoder.latent_dim=4",
                "model.autoencoder.encoder_channels=[4,8]",
                "model.autoencoder.decoder_channels=[8,4]",
                "model.autoencoder.mlp_dim=16",
                "model.quantum_generator.n_qubits=2",
            ],
        )

    manifest = run_cache_from_config(cfg)

    assert manifest["checkpoint_path"].endswith("checkpoints/vae.ckpt")
