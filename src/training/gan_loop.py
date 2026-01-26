from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Tuple

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

from datamodules.mnist import MNISTDataModule
from models.compression_methods.autoencoder import (
    Autoencoder,
    init_autoencoder_variables_with_shape,
)
from models.discriminator import Discriminator, init_discriminator_params
from models.quantum_generator import build_generator_apply, init_generator_params, sample_noise
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.image_grid import save_image_grid
from utils.logging import log_images, log_metrics, setup_wandb
from utils.device import select_device
from utils.seed import set_seed
from utils.train_state import TrainStateWithBatchStats


def _gradient_penalty(
    discriminator: Discriminator,
    disc_params: dict,
    real_features: jnp.ndarray,
    fake_features: jnp.ndarray,
    rng: jax.random.KeyArray,
) -> jnp.ndarray:
    """Calculates gradient penalty in the Latent Space."""
    alpha = jax.random.uniform(rng, (real_features.shape[0], 1))
    interpolated = real_features + alpha * (fake_features - real_features)

    def disc_score(x):
        return discriminator.apply({"params": disc_params}, x)

    def single_grad(x):
        return jax.grad(lambda y: disc_score(y[None, ...]).sum())(x)

    grads = jax.vmap(single_grad)(interpolated)
    norm = jnp.linalg.norm(grads, axis=1)
    return jnp.mean((norm - 1.0) ** 2)


def make_disc_step(
    discriminator: Discriminator,
    gen_apply,
    autoencoder: Autoencoder,
    noise_dim: int,
    lambda_gp: float,
):
    # Use partial for JIT to avoid "missing positional argument" error
    @partial(jax.jit, static_argnames=[])
    def disc_step(
        disc_state: train_state.TrainState,
        gen_params: dict,
        ae_params: dict,
        ae_batch_stats: dict,
        batch_images: jnp.ndarray,
        rng: jax.random.KeyArray,
    ):
        rng, noise_rng, gp_rng = jax.random.split(rng, 3)

        # 1. Encode Real Images to Latent Features
        # We stop gradient because the AE is frozen/pretrained
        ae_vars = {"params": ae_params, "batch_stats": ae_batch_stats}
        real_features = autoencoder.apply(
            ae_vars, batch_images, method=Autoencoder.encode, train=False
        )
        real_features = real_features.reshape((real_features.shape[0], -1))
        real_features = jax.lax.stop_gradient(real_features)

        # 2. Generate Fake Latent Features
        noise = sample_noise(noise_rng, batch_images.shape[0], noise_dim)
        fake_features = gen_apply(gen_params, noise)

        def loss_fn(disc_params):
            real_scores = discriminator.apply({"params": disc_params}, real_features)
            fake_scores = discriminator.apply({"params": disc_params}, fake_features)
            
            gp = _gradient_penalty(discriminator, disc_params, real_features, fake_features, gp_rng)
            
            # WGAN Loss: Minimize -D(real) + D(fake) + penalty
            loss = jnp.mean(fake_scores) - jnp.mean(real_scores) + lambda_gp * gp
            
            metrics = {
                "disc_loss": loss,
                "disc_real": jnp.mean(real_scores),
                "disc_fake": jnp.mean(fake_scores),
                "gp": gp,
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_state.params)
        new_state = disc_state.apply_gradients(grads=grads)
        return new_state, metrics

    return disc_step


def make_gen_step(
    discriminator: Discriminator,
    gen_apply,
    noise_dim: int,
):
    # Use partial for JIT and mark batch_size as static
    @partial(jax.jit, static_argnums=(3,))
    def gen_step(
        gen_state: train_state.TrainState,
        disc_params: dict,
        rng: jax.random.KeyArray,
        batch_size: int,
    ):
        noise = sample_noise(rng, batch_size, noise_dim)

        def loss_fn(gen_params):
            fake_features = gen_apply(gen_params, noise)
            fake_scores = discriminator.apply({"params": disc_params}, fake_features)
            # Generator Loss: Minimize -D(fake)
            loss = -jnp.mean(fake_scores)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(gen_state.params)
        gen_state = gen_state.apply_gradients(grads=grads)
        metrics = {"gen_loss": loss}
        
        return gen_state, metrics

    return gen_step


def _prepare_samples(
    gen_apply,
    gen_params: dict,
    autoencoder: Autoencoder,
    ae_params: dict,
    ae_batch_stats: dict,
    rng: jax.random.KeyArray,
    batch_size: int,
    noise_dim: int,
) -> jnp.ndarray:
    """Generates samples by producing latent features and then Decoding them."""
    noise = sample_noise(rng, batch_size, noise_dim)
    fake_features = gen_apply(gen_params, noise)
    
    variables = {"params": ae_params, "batch_stats": ae_batch_stats}
    fake_images = autoencoder.apply(
        variables, fake_features, method=Autoencoder.decode, train=False
    )
    return fake_images


def run_gan(cfg) -> Tuple[dict, train_state.TrainState]:
    select_device(cfg.device)
    set_seed(cfg.seed)
    
    # --- Data Loading (Preload to GPU) ---
    data = MNISTDataModule(cfg.data.data_dir, num_workers=cfg.data.num_workers)
    data.setup()

    print("Moving dataset to GPU...")
    # This matches the efficient pattern from your pretrain loop
    train_images = jax.device_put(jnp.asarray(data.train_images))
    print(f"Dataset on GPU. Shape: {train_images.shape}")

    n_samples = train_images.shape[0]
    steps_per_epoch = n_samples // cfg.batch_size

    # --- Verification of Paper Constraints ---
    expected_dim = 2 * cfg.model.quantum_generator.n_qubits
    if cfg.model.autoencoder.latent_dim != expected_dim:
        raise ValueError(f"Paper Logic Error: Autoencoder latent dim ({cfg.model.autoencoder.latent_dim}) "
                         f"must equal 2 * n_qubits ({expected_dim}) for X+Z measurements.")

    # --- Model Initialization ---
    autoencoder = Autoencoder(
        latent_dim=cfg.model.autoencoder.latent_dim,
        encoder_channels=cfg.model.autoencoder.encoder_channels,
        decoder_channels=cfg.model.autoencoder.decoder_channels,
        mlp_dim=cfg.model.autoencoder.mlp_dim,
        tanh_latent=cfg.model.autoencoder.tanh_latent,
    )
    
    disc_model = Discriminator(
        channels=cfg.model.discriminator.channels,
        mlp_dim=cfg.model.discriminator.mlp_dim,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng, gen_rng, disc_rng = jax.random.split(rng, 4)
    
    # Initialize AE
    ae_variables = init_autoencoder_variables_with_shape(
        init_rng, autoencoder, input_shape=(28, 28, 1)
    )
    ae_params = ae_variables["params"]
    ae_batch_stats = ae_variables["batch_stats"]

    # Load Pretrained AE Checkpoint
    ckpt_dir = Path(hydra.utils.get_original_cwd()) / cfg.checkpoints.dir
    ae_ckpt = ckpt_dir / cfg.checkpoints.autoencoder
    if not ae_ckpt.exists():
        raise FileNotFoundError("Pretrained Autoencoder checkpoint is required for LaSt-QGAN.")
        
    loaded = load_checkpoint(ae_ckpt, {"params": ae_params, "batch_stats": ae_batch_stats})
    ae_params = loaded["params"]
    ae_batch_stats = loaded["batch_stats"]

    # Initialize Generator
    gen_params = init_generator_params(
        gen_rng,
        n_qubits=cfg.model.quantum_generator.n_qubits,
        depth=cfg.model.quantum_generator.depth,
        noise_dim=cfg.model.quantum_generator.noise_dim,
    )
    
    # Initialize Discriminator (Input is latent_dim vector)
    disc_params = init_discriminator_params(
        disc_rng, 
        disc_model, 
        latent_dim=cfg.model.autoencoder.latent_dim
    )

    gen_apply = build_generator_apply(
        n_qubits=cfg.model.quantum_generator.n_qubits,
        depth=cfg.model.quantum_generator.depth,
    )

    # --- Train States ---
    gen_state = train_state.TrainState.create(
        apply_fn=lambda params, noise: gen_apply(params, noise),
        params=gen_params,
        tx=optax.adam(cfg.gen_lr, b1=0.5, b2=0.999), 
    )
    disc_state = train_state.TrainState.create(
        apply_fn=disc_model.apply, 
        params=disc_params, 
        tx=optax.adam(cfg.disc_lr, b1=0.5, b2=0.999) 
    )

    # --- Step Functions ---
    disc_step = make_disc_step(
        disc_model,
        gen_apply,
        autoencoder,
        cfg.model.quantum_generator.noise_dim,
        cfg.lambda_gp, 
    )
    gen_step = make_gen_step(
        disc_model,
        gen_apply,
        cfg.model.quantum_generator.noise_dim,
    )

    run = setup_wandb(cfg, mode="train")
    orig_cwd = Path(hydra.utils.get_original_cwd())
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Epoch-Based Training Loop ---
    global_step = 0
    epoch_bar = tqdm(range(cfg.epochs), desc="Train epochs")
    
    for epoch in epoch_bar:
        # Shuffle data indices every epoch
        rng, perm_rng = jax.random.split(rng)
        perms = jax.random.permutation(perm_rng, n_samples)
        perms = perms[:steps_per_epoch * cfg.batch_size]
        perms = perms.reshape((steps_per_epoch, cfg.batch_size))

        for i in tqdm(range(steps_per_epoch), desc="Batches", leave=False):
            
            # 1. Get Batch
            batch_idx = perms[i]
            batch_images = train_images[batch_idx]
            
            # 2. Update Discriminator (Every Step)
            rng, step_rng = jax.random.split(rng)
            disc_state, disc_metrics = disc_step(
                disc_state,
                gen_state.params,
                ae_params,
                ae_batch_stats,
                batch_images,
                step_rng,
            )

            # 3. Update Generator (Every n_critic steps)
            gen_metrics = {}
            if global_step % cfg.n_critic == 0:
                rng, step_rng = jax.random.split(rng)
                gen_state, gen_metrics = gen_step(
                    gen_state, 
                    disc_state.params, 
                    step_rng, 
                    cfg.batch_size
                )

            # 4. Logging & Sampling
            if global_step % cfg.log_every == 0:
                # Merge metrics (gen_metrics might be empty if we didn't update G this step)
                combined_metrics = {**disc_metrics, **gen_metrics}
                metrics = {f"train/{k}": float(v) for k, v in combined_metrics.items()}
                log_metrics(run, metrics, step=global_step)

            if global_step % cfg.sample_every == 0:
                rng, sample_rng = jax.random.split(rng)
                samples = _prepare_samples(
                    gen_apply,
                    gen_state.params,
                    autoencoder,
                    ae_params,
                    ae_batch_stats,
                    sample_rng,
                    batch_size=64,
                    noise_dim=cfg.model.quantum_generator.noise_dim,
                )
                sample_path = output_dir / f"samples_step_{global_step:06d}.png"
                save_image_grid(samples, sample_path, nrow=8)
                log_images(run, {"train/samples": samples}, step=global_step)
            
            global_step += 1

    epoch_bar.close()
    
    save_checkpoint(ckpt_dir / cfg.checkpoints.generator, gen_state.params)
    save_checkpoint(ckpt_dir / cfg.checkpoints.discriminator, disc_state.params)
    
    if run is not None: 
        run.finish()
    
    return gen_state.params, gen_state