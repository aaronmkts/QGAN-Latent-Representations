"""Training loop for the MPS Born Machine prior.

Stage 2 of the VQ-VAE pipeline:
1. Load pretrained spatial VQ-VAE
2. Extract codebook indices for all training images
3. Train MPS Born machine to model the prior over index sequences
4. Sample from MPS and decode through VQ-VAE for visualization
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

from datamodules.mnist import MNISTDataModule
from models.compression_methods.spatial_vqvae import (
    SpatialVQVAE,
    init_spatial_vqvae_variables,
)
from models.mps_prior.mps import init_mps_params, mps_nll_loss, mps_sample
from models.mps_prior.quimb_mps import init_quimb_mps, quimb_sample
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.device import select_device
from utils.image_grid import save_image_grid
from utils.logging import log_images, log_metrics, setup_wandb
from utils.seed import set_seed


def _build_vqvae(cfg) -> SpatialVQVAE:
    """Build SpatialVQVAE model from config."""
    return SpatialVQVAE(
        encoder_channels=cfg.model.vqvae.encoder_channels,
        decoder_channels=cfg.model.vqvae.decoder_channels,
        num_embeddings=cfg.model.vqvae.num_embeddings,
        embedding_dim=cfg.model.vqvae.embedding_dim,
    )


def _extract_all_indices(
    vqvae: SpatialVQVAE,
    vqvae_vars: dict,
    images: jnp.ndarray,
    batch_size: int = 256,
) -> jnp.ndarray:
    """Extract codebook indices for all images using pretrained VQ-VAE.

    Args:
        vqvae: SpatialVQVAE model.
        vqvae_vars: {"params": ..., "batch_stats": ...}
        images: (N, 28, 28, 1) training images.
        batch_size: Batch size for encoding.

    Returns:
        (N, 49) integer array of flattened codebook indices.
    """
    n = images.shape[0]
    all_indices = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]
        # encode_to_indices returns (batch, 7, 7)
        indices = vqvae.apply(
            vqvae_vars, batch, method=SpatialVQVAE.encode_to_indices, train=False
        )
        # Flatten spatial dims: (batch, 7, 7) -> (batch, 49)
        indices_flat = indices.reshape(indices.shape[0], -1)
        all_indices.append(indices_flat)

    return jnp.concatenate(all_indices, axis=0)


def _decode_from_indices(
    vqvae: SpatialVQVAE,
    vqvae_vars: dict,
    flat_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Decode flattened codebook indices back to images.

    Args:
        flat_indices: (batch, 49) integer indices.

    Returns:
        (batch, 28, 28, 1) images.
    """
    # Reshape to spatial grid: (batch, 49) -> (batch, 7, 7)
    spatial_indices = flat_indices.reshape(flat_indices.shape[0], 7, 7)
    return vqvae.apply(
        vqvae_vars, spatial_indices,
        method=SpatialVQVAE.decode_from_indices, train=False,
    )


def run_mps_prior(cfg) -> dict:
    """Main training function for MPS prior."""
    select_device(cfg.device)
    set_seed(cfg.seed)

    # --- Data Loading ---
    data = MNISTDataModule(cfg.data.data_dir)
    data.setup()
    print("Moving dataset to GPU...")
    train_images = jax.device_put(jnp.asarray(data.train_images))
    print(f"Dataset on GPU. Shape: {train_images.shape}")

    # --- Load Pretrained Spatial VQ-VAE ---
    vqvae = _build_vqvae(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    vqvae_vars = init_spatial_vqvae_variables(init_rng, vqvae, input_shape=(28, 28, 1))

    orig_cwd = Path(hydra.utils.get_original_cwd())
    ckpt_dir = orig_cwd / cfg.checkpoints.dir
    vqvae_ckpt = ckpt_dir / cfg.checkpoints.vqvae

    if not vqvae_ckpt.exists():
        raise FileNotFoundError(
            f"Pretrained Spatial VQ-VAE checkpoint not found at {vqvae_ckpt}. "
            "Run: python pretrain.py model/spatial_vqvae@model.autoencoder"
        )

    loaded = load_checkpoint(vqvae_ckpt, {
        "params": vqvae_vars["params"],
        "batch_stats": vqvae_vars["batch_stats"],
    })
    vqvae_vars = {"params": loaded["params"], "batch_stats": loaded["batch_stats"]}
    print("Loaded pretrained Spatial VQ-VAE checkpoint.")

    # --- Extract Codebook Indices ---
    print("Extracting codebook indices from training data...")
    all_indices = _extract_all_indices(vqvae, vqvae_vars, train_images)
    print(f"Extracted indices shape: {all_indices.shape}")  # (60000, 49)
    print(f"Index range: [{int(all_indices.min())}, {int(all_indices.max())}]")

    # --- Initialize MPS ---
    mps_cfg = cfg.model.mps_prior
    n_sites = mps_cfg.n_sites
    phys_dim = mps_cfg.phys_dim
    bond_dim = mps_cfg.bond_dim
    use_quimb = mps_cfg.use_quimb

    rng, mps_rng = jax.random.split(rng)

    if use_quimb:
        mps_params = init_quimb_mps(mps_rng, n_sites, phys_dim, bond_dim)
        print(f"Initialized quimb MPS: {n_sites} sites, phys_dim={phys_dim}, bond_dim={bond_dim}")
    else:
        mps_params = init_mps_params(mps_rng, n_sites, phys_dim, bond_dim)
        print(f"Initialized JAX MPS: {n_sites} sites, phys_dim={phys_dim}, bond_dim={bond_dim}")

    # --- Optimizer ---
    # Use a pytree-compatible optimizer: wrap tensor list in a dict
    tx = optax.adam(mps_cfg.learning_rate)
    opt_state = tx.init(mps_params)

    # --- Training ---
    run = setup_wandb(cfg, mode="train_prior")
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = all_indices.shape[0]
    batch_size = mps_cfg.batch_size
    steps_per_epoch = n_samples // batch_size

    @jax.jit
    def train_step(params, batch):
        loss, grads = jax.value_and_grad(mps_nll_loss)(params, batch)
        return loss, grads

    global_step = 0
    epoch_bar = tqdm(range(mps_cfg.epochs), desc="MPS Prior epochs")

    for epoch in epoch_bar:
        rng, perm_rng = jax.random.split(rng)
        perms = jax.random.permutation(perm_rng, n_samples)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        epoch_loss = 0.0
        for i in tqdm(range(steps_per_epoch), desc="Batches", leave=False):
            batch_idx = perms[i]
            batch = all_indices[batch_idx]

            loss, grads = train_step(mps_params, batch)
            updates, opt_state_new = tx.update(grads, opt_state, mps_params)
            mps_params = optax.apply_updates(mps_params, updates)
            opt_state = opt_state_new

            epoch_loss += float(loss)

            if global_step % cfg.log_every == 0:
                log_metrics(run, {"prior/nll": float(loss)}, step=global_step)

            if global_step % cfg.sample_every == 0:
                rng, sample_rng = jax.random.split(rng)
                n_vis = 64
                if use_quimb:
                    sampled = quimb_sample(
                        mps_params["tensors"], sample_rng, n_vis
                    )
                else:
                    sampled = mps_sample(mps_params, sample_rng, n_vis)

                # Decode through VQ-VAE
                images = _decode_from_indices(vqvae, vqvae_vars, sampled)
                sample_path = output_dir / f"mps_samples_step_{global_step:06d}.png"
                save_image_grid(images, sample_path, nrow=8)
                log_images(run, {"prior/samples": images}, step=global_step)

            global_step += 1

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch}: avg NLL = {avg_loss:.4f}")

    # --- Save Checkpoint ---
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_dir / cfg.checkpoints.mps_prior, mps_params)
    print(f"Saved MPS prior checkpoint to {ckpt_dir / cfg.checkpoints.mps_prior}")

    if run is not None:
        run.finish()

    return mps_params
