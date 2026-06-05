from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from qgan_latent.shared.utils.checkpointing import save_checkpoint
from qgan_latent.shared.representations.runtime import (
    build_representation_runtime,
    init_representation_variables,
    load_representation_checkpoint,
)


def _cfg(name: str):
    base = {
        "name": name,
        "latent_dim": 4,
        "encoder_channels": [4, 8],
        "decoder_channels": [8, 4],
        "mlp_dim": 16,
        "tanh_latent": True,
    }
    if name == "sinkhorn_ae":
        base.update({"sinkhorn_weight": 1.0, "sinkhorn_eps": 0.1, "sinkhorn_iters": 2, "prior": "gaussian"})
    if name == "vqvae":
        base.update({"num_embeddings": 8, "embedding_dim": 4, "commitment_cost": 0.25})
    return OmegaConf.create(base)


def test_autoencoder_runtime_encodes_decodes_and_reports_metadata() -> None:
    runtime = build_representation_runtime(_cfg("autoencoder"))
    variables = init_representation_variables(runtime, jax.random.PRNGKey(0), input_shape=(28, 28, 1))
    batch = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)

    encoded = runtime.encode(variables, batch)
    decoded = runtime.decode(variables, encoded)

    assert encoded.shape == (2, 4)
    assert decoded.shape == batch.shape
    assert runtime.metadata.name == "autoencoder"
    assert runtime.metadata.latent_dim == 4
    assert runtime.metadata.vector_latent is True


def test_vae_runtime_exposes_deterministic_stats_and_decodes_mu() -> None:
    runtime = build_representation_runtime(_cfg("vae"))
    variables = init_representation_variables(runtime, jax.random.PRNGKey(1), input_shape=(28, 28, 1))
    batch = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)

    mu, logvar = runtime.encode_stats(variables, batch)
    encoded = runtime.encode(variables, batch)
    sampled = runtime.sample_latent(variables, batch, jax.random.PRNGKey(99))
    decoded = runtime.decode(variables, sampled)

    assert mu.shape == (2, 4)
    assert logvar.shape == (2, 4)
    assert sampled.shape == (2, 4)
    assert jnp.allclose(encoded, mu)
    assert decoded.shape == batch.shape
    assert runtime.metadata.name == "vae"


def test_sinkhorn_runtime_reuses_autoencoder_architecture_with_loss_metadata() -> None:
    runtime = build_representation_runtime(_cfg("sinkhorn_ae"))
    variables = init_representation_variables(runtime, jax.random.PRNGKey(2), input_shape=(28, 28, 1))
    batch = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)

    encoded = runtime.encode(variables, batch)

    assert encoded.shape == (2, 4)
    assert runtime.metadata.name == "sinkhorn_ae"
    assert runtime.metadata.training_objective == "sinkhorn"


def test_vector_runtime_rejects_vqvae_representation_for_qgan_stage_one() -> None:
    with pytest.raises(NotImplementedError, match="VQ-VAE"):
        build_representation_runtime(_cfg("vqvae"))


def test_vae_checkpoint_round_trip_preserves_encoder_heads(tmp_path: Path) -> None:
    runtime = build_representation_runtime(_cfg("vae"))
    variables = init_representation_variables(runtime, jax.random.PRNGKey(3), input_shape=(28, 28, 1))
    ckpt = tmp_path / "vae.ckpt"

    save_checkpoint(ckpt, variables)
    restored = load_representation_checkpoint(runtime, ckpt, input_shape=(28, 28, 1), rng=jax.random.PRNGKey(4))

    assert "mu_head" in restored["params"]
    assert "logvar_head" in restored["params"]
    assert restored["params"].keys() == variables["params"].keys()
