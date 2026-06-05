from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from qgan_latent.shared.representations.autoencoder import (
    Autoencoder,
    init_autoencoder_variables_with_shape,
)
from qgan_latent.shared.representations.variational_autoencoder import (
    VariationalAutoencoder,
    init_variational_variables,
)
from qgan_latent.shared.utils.checkpointing import load_checkpoint


@dataclass(frozen=True)
class RepresentationMetadata:
    name: str
    latent_dim: int
    vector_latent: bool
    training_objective: str


@dataclass(frozen=True)
class RepresentationRuntime:
    model: Any
    metadata: RepresentationMetadata

    def encode(self, variables: dict, batch: jnp.ndarray) -> jnp.ndarray:
        if self.metadata.name == "vae":
            return self.model.apply(
                variables,
                batch,
                method=VariationalAutoencoder.encode,
                train=False,
            )
        return self.model.apply(
            variables,
            batch,
            method=Autoencoder.encode,
            train=False,
        )

    def encode_stats(self, variables: dict, batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.metadata.name != "vae":
            raise NotImplementedError("encode_stats is only available for VAE runtimes.")
        return self.model.apply(
            variables,
            batch,
            method=VariationalAutoencoder._encode_stats,
            train=False,
        )

    def sample_latent(
        self, variables: dict, batch: jnp.ndarray, rng: jax.random.KeyArray
    ) -> jnp.ndarray:
        if self.metadata.name != "vae":
            return self.encode(variables, batch)
        mu, logvar = self.encode_stats(variables, batch)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * jnp.exp(0.5 * logvar)

    def decode(self, variables: dict, latents: jnp.ndarray) -> jnp.ndarray:
        if self.metadata.name == "vae":
            return self.model.apply(
                variables,
                latents,
                method=VariationalAutoencoder.decode,
                train=False,
            )
        return self.model.apply(
            variables,
            latents,
            method=Autoencoder.decode,
            train=False,
        )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return getattr(cfg, key, default)


def _as_sequence(value: Sequence[int]) -> list[int]:
    return [int(v) for v in value]


def build_representation_runtime(model_cfg: Any) -> RepresentationRuntime:
    name = str(_cfg_get(model_cfg, "name", "autoencoder"))
    if name in {"vqvae", "spatial_vqvae"}:
        raise NotImplementedError(
            "VQ-VAE representations are discrete/spatial and are not supported by "
            "the Stage 1 vector-latent QGAN runtime."
        )
    if name in {"autoencoder", "sinkhorn_ae"}:
        model = Autoencoder(
            latent_dim=int(model_cfg.latent_dim),
            encoder_channels=_as_sequence(model_cfg.encoder_channels),
            decoder_channels=_as_sequence(model_cfg.decoder_channels),
            mlp_dim=int(model_cfg.mlp_dim),
            tanh_latent=bool(_cfg_get(model_cfg, "tanh_latent", True)),
        )
        objective = "sinkhorn" if name == "sinkhorn_ae" else "reconstruction"
    elif name == "vae":
        model = VariationalAutoencoder(
            latent_dim=int(model_cfg.latent_dim),
            encoder_channels=_as_sequence(model_cfg.encoder_channels),
            decoder_channels=_as_sequence(model_cfg.decoder_channels),
            mlp_dim=int(model_cfg.mlp_dim),
            tanh_latent=bool(_cfg_get(model_cfg, "tanh_latent", True)),
        )
        objective = "variational"
    else:
        raise ValueError(f"Unsupported representation runtime: {name}")

    return RepresentationRuntime(
        model=model,
        metadata=RepresentationMetadata(
            name=name,
            latent_dim=int(model_cfg.latent_dim),
            vector_latent=True,
            training_objective=objective,
        ),
    )


def init_representation_variables(
    runtime: RepresentationRuntime,
    rng: jax.random.KeyArray,
    input_shape: Sequence[int],
) -> dict:
    if runtime.metadata.name == "vae":
        init_rng, sample_rng = jax.random.split(rng)
        return init_variational_variables(init_rng, sample_rng, runtime.model, input_shape=input_shape)
    return init_autoencoder_variables_with_shape(rng, runtime.model, input_shape=input_shape)


def load_representation_checkpoint(
    runtime: RepresentationRuntime,
    path: Path,
    input_shape: Sequence[int],
    rng: jax.random.KeyArray,
) -> dict:
    template = init_representation_variables(runtime, rng, input_shape=input_shape)
    return load_checkpoint(path, template)


def representation_checkpoint_path(model_cfg: Any, checkpoints_cfg: Any, run_root: Path) -> Path:
    configured = Path(str(getattr(checkpoints_cfg, "autoencoder")))
    checkpoint_name = getattr(model_cfg, "checkpoint_name", None)
    if checkpoint_name is not None and configured == Path("autoencoder.ckpt"):
        configured = Path(str(checkpoint_name))
    if configured.parent == Path("."):
        return run_root / checkpoints_cfg.dir / configured
    return configured if configured.is_absolute() else run_root / configured
