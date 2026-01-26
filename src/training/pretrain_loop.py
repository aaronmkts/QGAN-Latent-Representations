from __future__ import annotations
import math
from functools import partial  # <--- FIX 1: Add this import
from pathlib import Path
from typing import Tuple



import hydra

import jax

import jax.numpy as jnp

import optax

from tqdm import tqdm

from datamodules.mnist import MNISTDataModule

from models.compression_methods.autoencoder import (

    Autoencoder,

    init_autoencoder_variables_with_shape,    

)

from utils.checkpointing import save_checkpoint

from utils.image_grid import save_image_grid

from utils.logging import log_images, log_metrics, setup_wandb

from utils.device import select_device

from utils.seed import set_seed

from utils.train_state import TrainStateWithBatchStats





# <--- FIX 2: Use partial to bind static_argnames to jit

@partial(jax.jit, static_argnames=["loss_type"])

def train_step(state: TrainStateWithBatchStats, batch: jnp.ndarray, loss_type: str = "mse"):



    def loss_fn(params):

        variables = {"params": params, "batch_stats": state.batch_stats}

     

        recon, updates = state.apply_fn(

            variables, batch, train=True, mutable=["batch_stats"]

        )

       

        if loss_type == "bce":

            eps = 1e-7

            recon_clipped = jnp.clip(recon, eps, 1.0 - eps)

            loss = -jnp.mean(batch * jnp.log(recon_clipped) + (1.0 - batch) * jnp.log(1.0 - recon_clipped))

        else:

            loss = jnp.mean((batch - recon) ** 2)

           

        return loss, (recon, updates["batch_stats"])



    (loss, (recon, new_batch_stats)), grads = jax.value_and_grad(

        loss_fn, has_aux=True

    )(state.params)



    new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)

    return new_state, loss, recon





def _prepare_recon_grid(batch: jnp.ndarray, recon: jnp.ndarray, max_items: int = 8) -> jnp.ndarray:

    n = min(batch.shape[0], max_items)

    paired = jnp.stack([batch[:n], recon[:n]], axis=1)

    return paired.reshape((n * 2, batch.shape[1], batch.shape[2], batch.shape[3]))





def run_pretrain(cfg) -> Tuple[dict, TrainStateWithBatchStats]:

    select_device(cfg.device)

    set_seed(cfg.seed)

   

    data = MNISTDataModule(cfg.data.data_dir)

    data.setup()



    print("Moving dataset to GPU...")

    train_images = jax.device_put(jnp.asarray(data.train_images))

    print(f"Dataset on GPU. Shape: {train_images.shape}")

   

    n_samples = train_images.shape[0]

    steps_per_epoch = n_samples // cfg.batch_size

 

    model = Autoencoder(

        latent_dim=cfg.model.autoencoder.latent_dim,

        encoder_channels=cfg.model.autoencoder.encoder_channels,

        decoder_channels=cfg.model.autoencoder.decoder_channels,

        mlp_dim=cfg.model.autoencoder.mlp_dim,

        tanh_latent=cfg.model.autoencoder.tanh_latent,

    )



    rng = jax.random.PRNGKey(cfg.seed)

    rng, init_rng = jax.random.split(rng)

   

   

    variables = init_autoencoder_variables_with_shape(init_rng, model, input_shape=(28, 28, 1))



    state = TrainStateWithBatchStats.create(

        apply_fn=model.apply,

        params=variables["params"],

        batch_stats=variables["batch_stats"],

        tx=optax.adam(cfg.learning_rate, b1=0.5, b2=0.999),

    )



    run = setup_wandb(cfg, mode="pretrain")

    orig_cwd = Path(hydra.utils.get_original_cwd())

    output_dir = orig_cwd / cfg.outputs.dir

    output_dir.mkdir(parents=True, exist_ok=True)

   

    global_step = 0

    epoch_bar = tqdm(range(cfg.pretrain_epochs), desc="Pretrain epochs")

   

    loss_type = cfg.loss



    for epoch in epoch_bar:

       

        rng, perm_rng = jax.random.split(rng)

        perms = jax.random.permutation(perm_rng, n_samples)

       

        perms = perms[:steps_per_epoch * cfg.batch_size]  

        perms = perms.reshape((steps_per_epoch, cfg.batch_size))



        for i in tqdm(range(steps_per_epoch), desc="Batches", leave=False):

         

            batch_idx = perms[i]

            batch_images = train_images[batch_idx]

           

            state, loss, recon = train_step(state, batch_images, loss_type)

         

            if global_step % cfg.log_every == 0:

           

                log_metrics(run, {"pretrain/loss": float(loss)}, step=global_step)



            if global_step % cfg.sample_every == 0:

                grid = _prepare_recon_grid(batch_images, recon)

                sample_path = output_dir / f"recon_step_{global_step:06d}.png"

                save_image_grid(grid, sample_path, nrow=2)

                log_images(run, {"pretrain/recon": grid}, step=global_step)



            global_step += 1

           

    # Save Checkpoint

    ckpt_dir = orig_cwd / cfg.checkpoints.dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / cfg.checkpoints.autoencoder

    save_checkpoint(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})



    if run is not None:

        run.finish()



    return state.params, state