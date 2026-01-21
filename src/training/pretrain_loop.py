from __future__ import annotations

import math
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
from utils.checkpointing import save_checkpoint
from utils.image_grid import save_image_grid
from utils.logging import log_images, log_metrics, setup_wandb
from utils.device import select_device
from utils.seed import set_seed


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        loss = -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
        return loss
    return jnp.mean((x - x_hat) ** 2)


def make_train_step(model: Autoencoder, loss_type: str):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: jnp.ndarray):
        def loss_fn(params):
            recon = model.apply({"params": params}, batch)
            loss = _recon_loss(batch, recon, loss_type)
            return loss, recon

        (loss, recon), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, recon

    return train_step


def _prepare_recon_grid(batch: jnp.ndarray, recon: jnp.ndarray, max_items: int = 8) -> jnp.ndarray:
    n = min(batch.shape[0], max_items)
    paired = jnp.stack([batch[:n], recon[:n]], axis=1)
    return paired.reshape((n * 2, batch.shape[1], batch.shape[2], batch.shape[3]))


def run_pretrain(cfg) -> Tuple[dict, train_state.TrainState]:
    select_device(cfg.device)
    set_seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    data = MNISTDataModule(cfg.data.data_dir, num_workers=cfg.data.num_workers)
    data.setup()

    model = Autoencoder(
        latent_dim=cfg.model.autoencoder.latent_dim,
        encoder_channels=cfg.model.autoencoder.encoder_channels,
        decoder_channels=cfg.model.autoencoder.decoder_channels,
        mlp_dim=cfg.model.autoencoder.mlp_dim,
        tanh_latent=cfg.model.autoencoder.tanh_latent,
    )

    rng, init_rng = jax.random.split(rng)
    params = init_autoencoder_params(init_rng, model)
    tx = optax.adam(cfg.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_step = make_train_step(model, cfg.loss)

    run = setup_wandb(cfg, mode="pretrain")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = orig_cwd / cfg.checkpoints.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg.pretrain_epochs
    max_steps = None
    if cfg.smoke_test:
        epochs = 1
        max_steps = 10

    steps_per_epoch = math.ceil(len(data.train_images) / cfg.batch_size)

    global_step = 0
    epoch_bar = tqdm(range(epochs), desc="Pretrain epochs")
    for epoch in epoch_bar:
        rng, epoch_rng = jax.random.split(rng)
        seed = int(jax.random.randint(epoch_rng, (), 0, 1_000_000))
        batch_iter = data.train_batches(cfg.batch_size, shuffle=True, seed=seed)
        batch_bar = tqdm(
            batch_iter,
            total=steps_per_epoch,
            desc="Pretrain batches",
            leave=False,
        )
        for batch in batch_bar:
            images = jnp.asarray(batch["images"])
            state, loss, recon = train_step(state, images)

            if global_step % cfg.log_every == 0:
                log_metrics(run, {"pretrain/loss": float(loss)}, step=global_step)

            if global_step % cfg.sample_every == 0:
                grid = _prepare_recon_grid(images, recon)
                sample_path = output_dir / f"recon_step_{global_step:06d}.png"
                save_image_grid(grid, sample_path, nrow=2)
                log_images(run, {"pretrain/recon": grid}, step=global_step)

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    ckpt_path = ckpt_dir / cfg.checkpoints.autoencoder
    save_checkpoint(ckpt_path, state.params)

    if run is not None:
        run.finish()

    return state.params, state
