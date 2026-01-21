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
from models.compression_methods.autoencoder import Autoencoder, init_autoencoder_params
from models.discriminator import Discriminator, init_discriminator_params
from models.quantum_generator import build_generator_apply, init_generator_params, sample_noise
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.image_grid import save_image_grid
from utils.logging import log_images, log_metrics, setup_wandb
from utils.device import select_device
from utils.seed import set_seed


def _gradient_penalty(
    discriminator: Discriminator,
    disc_params: dict,
    real: jnp.ndarray,
    fake: jnp.ndarray,
    rng: jax.random.KeyArray,
) -> jnp.ndarray:
    alpha = jax.random.uniform(rng, (real.shape[0], 1, 1, 1))
    interpolated = real + alpha * (fake - real)

    def disc_score(x):
        return discriminator.apply({"params": disc_params}, x)

    def single_grad(x):
        return jax.grad(lambda y: disc_score(y[None, ...]).sum())(x)

    grads = jax.vmap(single_grad)(interpolated)
    grads = grads.reshape((grads.shape[0], -1))
    norm = jnp.linalg.norm(grads, axis=1)
    return jnp.mean((norm - 1.0) ** 2)


def make_disc_step(
    discriminator: Discriminator,
    gen_apply,
    autoencoder: Autoencoder,
    noise_dim: int,
    lambda_gp: float,
):
    @jax.jit
    def disc_step(
        disc_state: train_state.TrainState,
        gen_params: dict,
        ae_params: dict,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
    ):
        rng, noise_rng, gp_rng = jax.random.split(rng, 3)
        noise = sample_noise(noise_rng, batch.shape[0], noise_dim)
        z_fake = gen_apply(gen_params, noise)
        fake_images = autoencoder.apply({"params": ae_params}, z_fake, method=Autoencoder.decode)
        fake_images = jax.lax.stop_gradient(fake_images)

        def loss_fn(disc_params):
            real_scores = discriminator.apply({"params": disc_params}, batch)
            fake_scores = discriminator.apply({"params": disc_params}, fake_images)
            gp = _gradient_penalty(discriminator, disc_params, batch, fake_images, gp_rng)
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
    autoencoder: Autoencoder,
    noise_dim: int,
    finetune_ae: bool,
):
    @partial(jax.jit, static_argnums=(4,))
    def gen_step(
        gen_state: train_state.TrainState,
        disc_params: dict,
        ae_state: train_state.TrainState,
        rng: jax.random.KeyArray,
        batch_size: int,
    ):
        noise = sample_noise(rng, batch_size, noise_dim)

        def loss_fn(gen_params, ae_params):
            z_fake = gen_apply(gen_params, noise)
            fake_images = autoencoder.apply({"params": ae_params}, z_fake, method=Autoencoder.decode)
            fake_scores = discriminator.apply({"params": disc_params}, fake_images)
            loss = -jnp.mean(fake_scores)
            return loss

        if finetune_ae:
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                gen_state.params, ae_state.params
            )
            gen_grads, ae_grads = grads
            gen_state = gen_state.apply_gradients(grads=gen_grads)
            ae_state = ae_state.apply_gradients(grads=ae_grads)
        else:
            loss, grads = jax.value_and_grad(loss_fn)(gen_state.params, ae_state.params)
            gen_state = gen_state.apply_gradients(grads=grads)

        metrics = {"gen_loss": loss}
        return gen_state, ae_state, metrics

    return gen_step


def _prepare_samples(
    gen_apply,
    gen_params: dict,
    autoencoder: Autoencoder,
    ae_params: dict,
    rng: jax.random.KeyArray,
    batch_size: int,
    noise_dim: int,
) -> jnp.ndarray:
    noise = sample_noise(rng, batch_size, noise_dim)
    z_fake = gen_apply(gen_params, noise)
    fake_images = autoencoder.apply({"params": ae_params}, z_fake, method=Autoencoder.decode)
    return fake_images


def run_gan(cfg) -> Tuple[dict, train_state.TrainState]:
    select_device(cfg.device)
    set_seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    data = MNISTDataModule(cfg.data.data_dir, num_workers=cfg.data.num_workers)
    data.setup()

    if cfg.model.quantum_generator.latent_dim != cfg.model.autoencoder.latent_dim:
        raise ValueError("Generator latent_dim must match autoencoder latent_dim.")

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

    rng, init_rng, gen_rng, disc_rng = jax.random.split(rng, 4)
    ae_params = init_autoencoder_params(init_rng, autoencoder)

    ckpt_dir = Path(hydra.utils.get_original_cwd()) / cfg.checkpoints.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ae_ckpt = ckpt_dir / cfg.checkpoints.autoencoder
    if ae_ckpt.exists():
        ae_params = load_checkpoint(ae_ckpt, ae_params)

    gen_params = init_generator_params(
        gen_rng,
        n_qubits=cfg.model.quantum_generator.n_qubits,
        depth=cfg.model.quantum_generator.depth,
        noise_dim=cfg.model.quantum_generator.noise_dim,
        latent_dim=cfg.model.quantum_generator.latent_dim,
    )
    disc_params = init_discriminator_params(disc_rng, disc_model)

    gen_apply = build_generator_apply(
        n_qubits=cfg.model.quantum_generator.n_qubits,
        depth=cfg.model.quantum_generator.depth,
        latent_tanh=cfg.model.quantum_generator.latent_tanh,
        architecture=cfg.model.quantum_generator.architecture,
    )

    gen_state = train_state.TrainState.create(
        apply_fn=lambda params, noise: gen_apply(params, noise),
        params=gen_params,
        tx=optax.adam(cfg.gen_lr),
    )
    disc_state = train_state.TrainState.create(
        apply_fn=disc_model.apply, params=disc_params, tx=optax.adam(cfg.disc_lr)
    )
    ae_state = train_state.TrainState.create(
        apply_fn=autoencoder.apply, params=ae_params, tx=optax.adam(cfg.ae_lr)
    )

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
        autoencoder,
        cfg.model.quantum_generator.noise_dim,
        cfg.finetune_ae,
    )

    run = setup_wandb(cfg, mode="train")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.train_steps
    if cfg.smoke_test:
        total_steps = 10

    data_iter = iter(data.train_batches(cfg.batch_size, shuffle=True, seed=cfg.seed))

    steps_per_epoch = math.ceil(len(data.train_images) / cfg.batch_size)
    num_epochs = math.ceil(total_steps / steps_per_epoch)
    epoch_bar = tqdm(total=num_epochs, desc="Train epochs")
    step_bar = tqdm(total=total_steps, desc="Train steps")
    steps_in_epoch = 0
    for step in range(total_steps):
        for _ in range(cfg.n_critic):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(
                    data.train_batches(cfg.batch_size, shuffle=True, seed=cfg.seed + step)
                )
                batch = next(data_iter)
            images = jnp.asarray(batch["images"])
            if images.shape[0] != cfg.batch_size:
                continue
            rng, step_rng = jax.random.split(rng)
            disc_state, disc_metrics = disc_step(
                disc_state, gen_state.params, ae_state.params, images, step_rng
            )

        rng, step_rng = jax.random.split(rng)
        gen_state, ae_state, gen_metrics = gen_step(
            gen_state, disc_state.params, ae_state, step_rng, cfg.batch_size
        )

        if step % cfg.log_every == 0:
            metrics = {f"train/{k}": float(v) for k, v in {**disc_metrics, **gen_metrics}.items()}
            log_metrics(run, metrics, step=step)

        if step % cfg.sample_every == 0:
            rng, sample_rng = jax.random.split(rng)
            samples = _prepare_samples(
                gen_apply,
                gen_state.params,
                autoencoder,
                ae_state.params,
                sample_rng,
                batch_size=64,
                noise_dim=cfg.model.quantum_generator.noise_dim,
            )
            sample_path = output_dir / f"samples_step_{step:06d}.png"
            save_image_grid(samples, sample_path, nrow=8)
            log_images(run, {"train/samples": samples}, step=step)
        step_bar.update(1)
        steps_in_epoch += 1
        if steps_in_epoch >= steps_per_epoch:
            epoch_bar.update(1)
            steps_in_epoch = 0

    if steps_in_epoch > 0:
        epoch_bar.update(1)
    step_bar.close()
    epoch_bar.close()

    save_checkpoint(ckpt_dir / cfg.checkpoints.generator, gen_state.params)
    save_checkpoint(ckpt_dir / cfg.checkpoints.discriminator, disc_state.params)

    if run is not None:
        run.finish()

    return gen_state.params, gen_state
