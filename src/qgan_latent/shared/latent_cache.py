from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from qgan_latent.shared.representations.runtime import RepresentationRuntime


def _encode_split(
    runtime: RepresentationRuntime,
    variables: dict,
    images: jnp.ndarray,
    batch_size: int,
    rng: jax.random.KeyArray | None = None,
) -> np.ndarray:
    images = jnp.asarray(images, dtype=jnp.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[start : start + int(batch_size)]
        if rng is None:
            z = runtime.encode(variables, batch)
        else:
            rng, split_rng = jax.random.split(rng)
            z = runtime.sample_latent(variables, batch, split_rng)
        z = z.reshape((z.shape[0], -1))
        chunks.append(np.asarray(jax.device_get(z), dtype=np.float32))
    if not chunks:
        return np.zeros((0, runtime.metadata.latent_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _encode_logvar_split(
    runtime: RepresentationRuntime,
    variables: dict,
    images: jnp.ndarray,
    batch_size: int,
) -> np.ndarray:
    images = jnp.asarray(images, dtype=jnp.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[start : start + int(batch_size)]
        _, logvar = runtime.encode_stats(variables, batch)
        chunks.append(np.asarray(jax.device_get(logvar), dtype=np.float32))
    if not chunks:
        return np.zeros((0, runtime.metadata.latent_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _stats(latents: np.ndarray) -> dict:
    if latents.size == 0:
        mean = np.zeros((latents.shape[1],), dtype=np.float32)
        std = np.zeros((latents.shape[1],), dtype=np.float32)
    else:
        mean = latents.mean(axis=0)
        std = latents.std(axis=0)
    return {
        "shape": list(latents.shape),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "min": float(latents.min()) if latents.size else None,
        "max": float(latents.max()) if latents.size else None,
    }


def _quantiles(latents: np.ndarray, latent_dim: int) -> np.ndarray:
    if not latents.size:
        return np.zeros((5, latent_dim), dtype=np.float32)
    return np.asarray(np.quantile(latents, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0), dtype=np.float32)


def cache_latent_representations(
    *,
    runtime: RepresentationRuntime,
    variables: dict,
    splits: Mapping[str, jnp.ndarray],
    output_dir: Path,
    seed: int,
    checkpoint_path: Path,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cache_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Latent cache already exists: {manifest_path}")

    split_manifest: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    quantiles: dict[str, np.ndarray] = {}
    sample_rng = jax.random.PRNGKey(int(seed))

    for split_name, images in splits.items():
        if runtime.metadata.name == "vae":
            sample_rng, split_rng = jax.random.split(sample_rng)
            sampled = _encode_split(runtime, variables, images, batch_size=batch_size, rng=split_rng)
            mu = _encode_split(runtime, variables, images, batch_size=batch_size)
            latents = mu
            logvar = _encode_logvar_split(runtime, variables, images, batch_size=batch_size)
            np.save(output_dir / f"z_mu_{split_name}.npy", mu)
            np.save(output_dir / f"z_logvar_{split_name}.npy", logvar)
            np.save(output_dir / f"z_sampled_{split_name}.npy", sampled)
        else:
            latents = _encode_split(runtime, variables, images, batch_size=batch_size)

        artifact_name = f"z_{split_name}.npy"
        np.save(output_dir / artifact_name, latents)
        split_manifest[split_name] = {
            "artifact": artifact_name,
            "n_samples": int(latents.shape[0]),
            "latent_dim": int(latents.shape[1]) if latents.ndim == 2 else 0,
        }
        if runtime.metadata.name == "vae":
            split_manifest[split_name].update(
                {
                    "mu_artifact": f"z_mu_{split_name}.npy",
                    "logvar_artifact": f"z_logvar_{split_name}.npy",
                    "sampled_artifact": f"z_sampled_{split_name}.npy",
                }
            )
        stats[split_name] = _stats(latents)
        quantiles[split_name] = _quantiles(latents, runtime.metadata.latent_dim)

    np.savez(output_dir / "empirical_quantiles.npz", **quantiles)
    (output_dir / "latent_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    manifest = {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "representation": {
            "name": runtime.metadata.name,
            "latent_dim": runtime.metadata.latent_dim,
            "vector_latent": runtime.metadata.vector_latent,
            "training_objective": runtime.metadata.training_objective,
        },
        "splits": split_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
