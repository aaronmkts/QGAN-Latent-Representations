"""Training loop for the MPS Born Machine prior.

Stage 2 of the VQ-VAE pipeline:
1. Load pretrained spatial VQ-VAE
2. Extract codebook indices for all training images
3. Train MPS Born machine to model the prior over index sequences
4. Sample from MPS and decode through VQ-VAE for visualization
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from qgan_latent.datamodules.mnist import MNISTDataModule
from qgan_latent.models.compression_methods.spatial_vqvae import (
    SpatialVQVAE,
    init_spatial_vqvae_variables,
)
from qgan_latent.models.mps_prior.mps import init_mps_params, mps_nll_loss, mps_sample
from qgan_latent.models.mps_prior.quimb_mps import quimb_sample
from qgan_latent.utils.checkpointing import load_checkpoint, save_checkpoint
from qgan_latent.utils.device import select_device
from qgan_latent.utils.image_grid import save_image_grid
from qgan_latent.utils.logging import log_images, log_metrics, setup_wandb
from qgan_latent.utils.seed import set_seed
from qgan_latent.training.smoke import synthetic_mnist_images
from qgan_latent.utils.paths import get_run_root


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


def _sample_from_prior(
    mps_params: dict,
    rng: jax.random.KeyArray,
    n_samples: int,
    sample_backend: str,
) -> jnp.ndarray:
    sample_backend = sample_backend.lower()
    if sample_backend == "jax":
        return mps_sample(mps_params, rng, n_samples)
    if sample_backend == "quimb":
        return quimb_sample(mps_params["tensors"], rng, n_samples)
    if sample_backend == "auto":
        try:
            return quimb_sample(mps_params["tensors"], rng, n_samples)
        except ImportError:
            return mps_sample(mps_params, rng, n_samples)
    raise ValueError(f"Unknown sample backend '{sample_backend}'. Expected one of: jax, quimb, auto.")


def run_mps_prior(cfg) -> dict:
    """Main training function for MPS prior."""
    select_device(cfg.device)
    set_seed(cfg.seed)

    smoke_test = bool(getattr(cfg, "smoke_test", False))
    if smoke_test:
        train_source = synthetic_mnist_images(8)
    else:
        data = MNISTDataModule(cfg.data.data_dir)
        data.setup()
        train_source = jnp.asarray(data.train_images)

    # --- Data Loading ---
    print("Moving dataset to device...")
    train_images = jax.device_put(train_source)
    print(f"Dataset on device. Shape: {train_images.shape}")

    # --- Load Pretrained Spatial VQ-VAE ---
    vqvae = _build_vqvae(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    vqvae_vars = init_spatial_vqvae_variables(init_rng, vqvae, input_shape=(28, 28, 1))

    orig_cwd = get_run_root()
    ckpt_dir = orig_cwd / cfg.checkpoints.dir
    vqvae_ckpt = ckpt_dir / cfg.checkpoints.vqvae

    if not vqvae_ckpt.exists():
        if not smoke_test:
            raise FileNotFoundError(
                f"Pretrained Spatial VQ-VAE checkpoint not found at {vqvae_ckpt}. "
                "Run: python pretrain.py 'model@model.autoencoder=spatial_vqvae'"
            )
    else:
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
    sample_backend = getattr(
        mps_cfg,
        "sample_backend",
        "quimb" if getattr(mps_cfg, "use_quimb", False) else "jax",
    )
    n_sample_vis = getattr(mps_cfg, "n_sample_vis", 64)

    rng, mps_rng = jax.random.split(rng)
    mps_params = init_mps_params(mps_rng, n_sites, phys_dim, bond_dim)
    print(
        f"Initialized JAX MPS: {n_sites} sites, phys_dim={phys_dim}, "
        f"bond_dim={bond_dim}, sample_backend={sample_backend}"
    )

    # --- Optimizer ---
    grad_clip_norm = getattr(mps_cfg, "grad_clip_norm", 0.0)
    weight_decay = getattr(mps_cfg, "weight_decay", 0.0)
    tx_parts = []
    if grad_clip_norm and grad_clip_norm > 0:
        tx_parts.append(optax.clip_by_global_norm(grad_clip_norm))
    tx_parts.append(
        optax.adamw(
            mps_cfg.learning_rate,
            weight_decay=weight_decay,
        )
    )
    tx = optax.chain(*tx_parts)
    opt_state = tx.init(mps_params)

    # --- Training ---
    run = setup_wandb(cfg, mode="train_prior")
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = all_indices.shape[0]
    batch_size = min(int(mps_cfg.batch_size), int(n_samples)) if smoke_test else int(mps_cfg.batch_size)
    steps_per_epoch = max(1, n_samples // batch_size)
    epochs = 1 if smoke_test else int(mps_cfg.epochs)
    log_every = 1 if smoke_test else int(cfg.log_every)
    sample_every = 1 if smoke_test else int(cfg.sample_every)

    @jax.jit
    def train_step(params, batch):
        loss, grads = jax.value_and_grad(mps_nll_loss)(params, batch)
        return loss, grads

    global_step = 0
    epoch_bar = tqdm(range(epochs), desc="MPS Prior epochs")

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

            if global_step % log_every == 0:
                log_metrics(run, {"prior/nll": float(loss)}, step=global_step)

            if global_step % sample_every == 0:
                rng, sample_rng = jax.random.split(rng)
                sampled = _sample_from_prior(
                    mps_params,
                    sample_rng,
                    n_sample_vis,
                    sample_backend,
                )

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
