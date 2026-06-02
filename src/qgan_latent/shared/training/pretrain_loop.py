from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from qgan_latent.shared.datamodules.mnist import MNISTDataModule
from qgan_latent.shared.representations.autoencoder import (
    Autoencoder,
    init_autoencoder_variables_with_shape,
)
from qgan_latent.shared.representations.sinkhorn_autoencoder import _sample_prior, _sinkhorn_distance
from qgan_latent.shared.representations.variational_autoencoder import (
    VariationalAutoencoder,
    init_variational_variables,
)
from qgan_latent.shared.representations.vqvae import VQVAE, init_vqvae_variables_with_shape
from qgan_latent.shared.representations.spatial_vqvae import SpatialVQVAE, init_spatial_vqvae_variables
from qgan_latent.shared.utils.checkpointing import save_checkpoint
from qgan_latent.shared.utils.image_grid import save_image_grid
from qgan_latent.shared.utils.logging import log_images, log_metrics, setup_wandb
from qgan_latent.shared.utils.device import select_device
from qgan_latent.shared.utils.seed import set_seed
from qgan_latent.shared.smoke import synthetic_mnist_images
from qgan_latent.shared.utils.paths import get_run_root
from qgan_latent.shared.utils.train_state import TrainStateWithBatchStats


@dataclass(frozen=True)
class PretrainModelSpec:
    build_model: Callable
    init_variables: Callable
    make_train_step: Callable

def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _kl_loss(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))

def _kl_weight(model_cfg, step: int) -> jnp.ndarray:
    beta = getattr(model_cfg, "beta", 1.0)
    beta = beta if getattr(model_cfg, "kl_weight", None) is None else model_cfg.kl_weight
    anneal_steps = getattr(model_cfg, "kl_anneal_steps", 0)
    if anneal_steps:
        step = jnp.asarray(step, dtype=jnp.float32)
        return beta * jnp.minimum(1.0, step / anneal_steps)
    return jnp.asarray(beta, dtype=jnp.float32)

def _build_autoencoder(model_cfg) -> Autoencoder:
    return Autoencoder(
        latent_dim=model_cfg.latent_dim,
        encoder_channels=model_cfg.encoder_channels,
        decoder_channels=model_cfg.decoder_channels,
        mlp_dim=model_cfg.mlp_dim,
        tanh_latent=model_cfg.tanh_latent,
    )

def _build_variational_autoencoder(model_cfg) -> VariationalAutoencoder:
    return VariationalAutoencoder(
        latent_dim=model_cfg.latent_dim,
        encoder_channels=model_cfg.encoder_channels,
        decoder_channels=model_cfg.decoder_channels,
        mlp_dim=model_cfg.mlp_dim,
        tanh_latent=model_cfg.tanh_latent,
    )

def _build_vqvae(model_cfg) -> VQVAE:
    return VQVAE(
        latent_dim=model_cfg.latent_dim,
        encoder_channels=model_cfg.encoder_channels,
        decoder_channels=model_cfg.decoder_channels,
        mlp_dim=model_cfg.mlp_dim,
        num_embeddings=model_cfg.num_embeddings,
        embedding_dim=model_cfg.embedding_dim,
        tanh_latent=model_cfg.tanh_latent,
    )

def _init_autoencoder(
    model: Autoencoder,
    init_rng: jax.random.KeyArray,
    input_shape: Tuple[int, ...],
    sample_rng: jax.random.KeyArray,
) -> dict:
    del sample_rng
    return init_autoencoder_variables_with_shape(init_rng, model, input_shape=input_shape)

def _init_variational_autoencoder(
    model: VariationalAutoencoder,
    init_rng: jax.random.KeyArray,
    input_shape: Tuple[int, ...],
    sample_rng: jax.random.KeyArray,
) -> dict:
    return init_variational_variables(init_rng, sample_rng, model, input_shape=input_shape)

def _init_vqvae(
    model: VQVAE,
    init_rng: jax.random.KeyArray,
    input_shape: Tuple[int, ...],
    sample_rng: jax.random.KeyArray,
) -> dict:
    del sample_rng
    return init_vqvae_variables_with_shape(init_rng, model, input_shape=input_shape)

def _make_autoencoder_train_step(model: Autoencoder, loss_type: str, model_cfg=None):
    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int,
    ):
        del rng, step

        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            recon, updates = model.apply(
                variables, batch, train=True, mutable=["batch_stats"]
            )
            recon_loss = _recon_loss(batch, recon, loss_type)
            return recon_loss, (recon, updates["batch_stats"], recon_loss)

        (loss, (recon, new_batch_stats, recon_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        metrics = {"recon_loss": recon_loss}
        return new_state, loss, recon, metrics

    return train_step

def _make_variational_autoencoder_train_step(
    model: VariationalAutoencoder, loss_type: str, model_cfg
):
    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int,
    ):
        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            outputs, updates = model.apply(
                variables, batch, rng, train=True, mutable=["batch_stats"]
            )
            recon, mu, logvar = outputs
            recon_loss = _recon_loss(batch, recon, loss_type)
            kl_loss = _kl_loss(mu, logvar)
            weight = _kl_weight(model_cfg, step)
            total = recon_loss + weight * kl_loss
            mu_abs_mean = jnp.mean(jnp.abs(mu))
            logvar_mean = jnp.mean(logvar)
            return total, (
                recon,
                updates["batch_stats"],
                recon_loss,
                kl_loss,
                weight,
                mu_abs_mean,
                logvar_mean,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        recon, new_batch_stats, recon_loss, kl_loss, weight, mu_abs_mean, logvar_mean = aux
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        metrics = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_weight": weight,
            "mu_abs_mean": mu_abs_mean,
            "logvar_mean": logvar_mean,
        }
        return new_state, loss, recon, metrics

    return train_step

def _make_sinkhorn_autoencoder_train_step(
    model: Autoencoder, loss_type: str, model_cfg
):
    sinkhorn_weight = (
        model_cfg.sinkhorn_weight
        if getattr(model_cfg, "lambda_sinkhorn", None) is None
        else model_cfg.lambda_sinkhorn
    )
    sinkhorn_eps = model_cfg.sinkhorn_eps
    sinkhorn_iters = model_cfg.sinkhorn_iters
    sinkhorn_cost = getattr(model_cfg, "sinkhorn_cost", "l2_sq")
    prior = getattr(model_cfg, "prior", "gaussian")

    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int,
    ):
        del step

        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            z, enc_updates = model.apply(
                variables, batch, method=Autoencoder.encode, train=True, mutable=["batch_stats"]
            )
            variables = {"params": params, "batch_stats": enc_updates["batch_stats"]}
            recon, dec_updates = model.apply(
                variables, z, method=Autoencoder.decode, train=True, mutable=["batch_stats"]
            )
            recon_loss = _recon_loss(batch, recon, loss_type)
            target = _sample_prior(rng, z.shape, prior)
            sinkhorn_loss = _sinkhorn_distance(
                z, target, sinkhorn_eps, sinkhorn_iters, sinkhorn_cost
            )
            total = recon_loss + sinkhorn_weight * sinkhorn_loss
            return total, (recon, dec_updates["batch_stats"], recon_loss, sinkhorn_loss)

        (loss, (recon, new_batch_stats, recon_loss, sinkhorn_loss)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        )
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        metrics = {"recon_loss": recon_loss, "sinkhorn_loss": sinkhorn_loss}
        return new_state, loss, recon, metrics

    return train_step

def _make_vqvae_train_step(model: VQVAE, loss_type: str, model_cfg=None):
    codebook_weight = model_cfg.codebook_loss_weight
    commitment_cost = model_cfg.commitment_cost

    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int,
    ):
        del rng, step

        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            outputs, updates = model.apply(
                variables, batch, train=True, mutable=["batch_stats"]
            )
            recon, codebook_loss, commitment_loss, perplexity = outputs
            recon_loss = _recon_loss(batch, recon, loss_type)
            total = recon_loss + codebook_weight * codebook_loss + commitment_cost * commitment_loss
            return total, (
                recon,
                updates["batch_stats"],
                recon_loss,
                codebook_loss,
                commitment_loss,
                perplexity,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        (
            recon,
            new_batch_stats,
            recon_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
        ) = aux
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        metrics = {
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }
        return new_state, loss, recon, metrics

    return train_step

def _build_spatial_vqvae(model_cfg) -> SpatialVQVAE:
    return SpatialVQVAE(
        encoder_channels=model_cfg.encoder_channels,
        decoder_channels=model_cfg.decoder_channels,
        num_embeddings=model_cfg.num_embeddings,
        embedding_dim=model_cfg.embedding_dim,
    )

def _init_spatial_vqvae(
    model: SpatialVQVAE,
    init_rng: jax.random.KeyArray,
    input_shape: Tuple[int, ...],
    sample_rng: jax.random.KeyArray,
) -> dict:
    del sample_rng
    return init_spatial_vqvae_variables(init_rng, model, input_shape=input_shape)

def _make_spatial_vqvae_train_step(model: SpatialVQVAE, loss_type: str, model_cfg=None):
    codebook_weight = model_cfg.codebook_loss_weight
    commitment_cost = model_cfg.commitment_cost

    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
        step: int,
    ):
        del rng, step

        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            outputs, updates = model.apply(
                variables, batch, train=True, mutable=["batch_stats"]
            )
            recon, codebook_loss, commitment_loss, perplexity = outputs
            recon_loss = _recon_loss(batch, recon, loss_type)
            total = recon_loss + codebook_weight * codebook_loss + commitment_cost * commitment_loss
            return total, (
                recon,
                updates["batch_stats"],
                recon_loss,
                codebook_loss,
                commitment_loss,
                perplexity,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        (
            recon,
            new_batch_stats,
            recon_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
        ) = aux
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        metrics = {
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }
        return new_state, loss, recon, metrics

    return train_step

PRETRAIN_MODEL_REGISTRY = {
    "autoencoder": PretrainModelSpec(
        build_model=_build_autoencoder,
        init_variables=_init_autoencoder,
        make_train_step=_make_autoencoder_train_step,
    ),
    "vae": PretrainModelSpec(
        build_model=_build_variational_autoencoder,
        init_variables=_init_variational_autoencoder,
        make_train_step=_make_variational_autoencoder_train_step,
    ),
    "variational_autoencoder": PretrainModelSpec(
        build_model=_build_variational_autoencoder,
        init_variables=_init_variational_autoencoder,
        make_train_step=_make_variational_autoencoder_train_step,
    ),
    "sinkhorn_ae": PretrainModelSpec(
        build_model=_build_autoencoder,
        init_variables=_init_autoencoder,
        make_train_step=_make_sinkhorn_autoencoder_train_step,
    ),
    "sinkhorn_autoencoder": PretrainModelSpec(
        build_model=_build_autoencoder,
        init_variables=_init_autoencoder,
        make_train_step=_make_sinkhorn_autoencoder_train_step,
    ),
    "vqvae": PretrainModelSpec(
        build_model=_build_vqvae,
        init_variables=_init_vqvae,
        make_train_step=_make_vqvae_train_step,
    ),
    "spatial_vqvae": PretrainModelSpec(
        build_model=_build_spatial_vqvae,
        init_variables=_init_spatial_vqvae,
        make_train_step=_make_spatial_vqvae_train_step,
    ),
}

def _prepare_recon_grid(batch: jnp.ndarray, recon: jnp.ndarray, max_items: int = 8) -> jnp.ndarray:

    n = min(batch.shape[0], max_items)
    paired = jnp.stack([batch[:n], recon[:n]], axis=1)

    return paired.reshape((n * 2, batch.shape[1], batch.shape[2], batch.shape[3]))

def run_pretrain(cfg) -> Tuple[dict, TrainStateWithBatchStats]:

    select_device(cfg.device)
    set_seed(cfg.seed)

    smoke_test = bool(getattr(cfg, "smoke_test", False))
    if smoke_test:
        train_images = synthetic_mnist_images(8)
    else:
        data = MNISTDataModule(cfg.data.data_dir)
        data.setup()
        train_images = jnp.asarray(data.train_images)

    print("Moving dataset to device...")
    train_images = jax.device_put(train_images)
    print(f"Dataset on device. Shape: {train_images.shape}")

    n_samples = train_images.shape[0]
    batch_size = min(int(cfg.batch_size), int(n_samples)) if smoke_test else int(cfg.batch_size)
    steps_per_epoch = max(1, n_samples // batch_size)
    pretrain_epochs = 1 if smoke_test else int(cfg.pretrain_epochs)
    log_every = 1 if smoke_test else int(cfg.log_every)
    sample_every = 1 if smoke_test else int(cfg.sample_every)

    model_cfg = cfg.model.autoencoder
    model_name = getattr(model_cfg, "name", "autoencoder")
    spec = PRETRAIN_MODEL_REGISTRY.get(model_name)
    if spec is None:
        raise ValueError(
            f"Unknown pretrain model '{model_name}'. Available: {sorted(PRETRAIN_MODEL_REGISTRY)}"
        )

    model = spec.build_model(model_cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)

    variables = spec.init_variables(
        model, init_rng, input_shape=(28, 28, 1), sample_rng=sample_rng
    )

    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optax.adam(cfg.learning_rate, b1=0.5, b2=0.999),
    )

    train_step = spec.make_train_step(model, cfg.loss, model_cfg)

    run = setup_wandb(cfg, mode="pretrain")
    orig_cwd = get_run_root()
    output_dir = orig_cwd / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    epoch_bar = tqdm(range(pretrain_epochs), desc="Pretrain epochs")
    for epoch in epoch_bar:
        rng, perm_rng = jax.random.split(rng)
        perms = jax.random.permutation(perm_rng, n_samples)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        for i in tqdm(range(steps_per_epoch), desc="Batches", leave=False):
            batch_idx = perms[i]
            batch_images = train_images[batch_idx]
            rng, step_rng = jax.random.split(rng)
            step = jnp.asarray(global_step)
            state, loss, recon, metrics = train_step(
                state, batch_images, step_rng, step
            )

            if global_step % log_every == 0:
                log_payload = {"pretrain/loss": float(loss)}
                log_payload.update(
                    {f"pretrain/{key}": float(value) for key, value in metrics.items()}
                )
                log_metrics(run, log_payload, step=global_step)

            if global_step % sample_every == 0:
                grid = _prepare_recon_grid(batch_images, recon)
                sample_path = output_dir / f"recon_step_{global_step:06d}.png"
                save_image_grid(grid, sample_path, nrow=2)
                log_images(run, {"pretrain/recon": grid}, step=global_step)
            global_step += 1

    # Save Checkpoint

    ckpt_dir = orig_cwd / cfg.checkpoints.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = getattr(model_cfg, "checkpoint_name", cfg.checkpoints.autoencoder)
    ckpt_path = ckpt_dir / ckpt_name

    save_checkpoint(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})
    if run is not None:

        run.finish()
    return state.params, state
