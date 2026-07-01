from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import jax
import numpy as np
import optax
from flax.training import train_state
from qgan_latent.shared.datamodules.mnist import MNISTDataModule
from qgan_latent.shared.representations.runtime import (
    build_representation_runtime,
    init_representation_variables,
    representation_checkpoint_path,
)
from qgan_latent.shared.smoke import synthetic_mnist_images
from qgan_latent.shared.utils.checkpointing import load_checkpoint, save_checkpoint
from qgan_latent.shared.utils.device import select_device
from qgan_latent.shared.utils.file_logging import EpochMetricLogger
from qgan_latent.shared.utils.image_grid import save_image_grid
from qgan_latent.shared.utils.logging import log_images, log_metrics, setup_wandb
from qgan_latent.shared.utils.metrics_wrapper import MetricsManager
from qgan_latent.shared.utils.paths import get_run_root
from qgan_latent.shared.utils.seed import set_seed
from qgan_latent.workflows.qgan_expectation_values.models.discriminator import Discriminator, init_discriminator_params
from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import build_observable_bank, sample_noise
from qgan_latent.workflows.qgan_expectation_values.models.surrogate import (
    build_fourier_apply,
    fourier_parameter_count,
    init_fourier_params,
    matched_rff_dim,
)
from qgan_latent.workflows.qgan_expectation_values.training.diagnostics import (
    EvsDiagnosticsWriter,
    compute_evs_diagnostics,
    evs_scalar_metrics,
)
from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import (
    _prepare_samples,
    make_disc_step,
    make_gen_step,
)
from qgan_latent.workflows.qgan_expectation_values.training.registry import RunRegistry


def _tree_parameter_count(params: dict) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))


def _surrogate_value(cfg, key: str, default):
    surrogate_cfg = getattr(cfg, "surrogate", None)
    return getattr(surrogate_cfg, key, default) if surrogate_cfg is not None else default


def run_surrogate_gan(cfg, *, matched_run_id: str | None = None) -> Tuple[dict, train_state.TrainState]:
    """Run the smoke-compatible GAN loop with a Fourier/RFF classical generator."""

    select_device(cfg.device)
    set_seed(cfg.seed)

    smoke_test = bool(getattr(cfg, "smoke_test", False))
    if smoke_test:
        train_source = synthetic_mnist_images(8)
        real_images_source = np.array(train_source)
    else:
        data = MNISTDataModule(cfg.data.data_dir, num_workers=cfg.data.num_workers)
        data.setup()
        train_source = np.asarray(data.train_images)
        real_images_source = data.test_images if hasattr(data, "test_images") else data.train_images

    train_images = jax.device_put(train_source)
    metric_total_size = min(8, len(real_images_source)) if smoke_test else 500
    real_images_10k = np.array(real_images_source[:metric_total_size])
    if len(real_images_10k) < metric_total_size:
        real_images_10k = np.array(real_images_source)

    n_samples = train_images.shape[0]
    batch_size = min(int(cfg.batch_size), int(n_samples)) if smoke_test else int(cfg.batch_size)
    steps_per_epoch = max(1, n_samples // batch_size)
    epochs = 1 if smoke_test else int(cfg.epochs)
    log_every = 1 if smoke_test else int(cfg.log_every)
    sample_every = 1 if smoke_test else int(cfg.sample_every)
    eval_epochs = 1 if smoke_test else int(getattr(cfg, "eval_epochs", 10))
    sample_batch_size = 4 if smoke_test else 64

    qcfg = cfg.model.quantum_generator
    observable_bank = build_observable_bank(getattr(qcfg, "observable_bank", None), n_qubits=qcfg.n_qubits)
    expected_dim = observable_bank.output_dim
    if cfg.model.autoencoder.latent_dim != expected_dim:
        raise ValueError(
            f"Generator/representation shape mismatch: autoencoder latent dim ({cfg.model.autoencoder.latent_dim}) "
            f"must equal observable_bank.output_dim ({expected_dim}) for {observable_bank.metadata()['name']} measurements."
        )

    representation = build_representation_runtime(cfg.model.autoencoder)
    disc_model = Discriminator(channels=cfg.model.discriminator.channels, mlp_dim=cfg.model.discriminator.mlp_dim)
    metrics_manager = MetricsManager(cfg.metrics, real_images=real_images_10k)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng, gen_rng, disc_rng = jax.random.split(rng, 4)
    rep_variables = init_representation_variables(representation, init_rng, input_shape=(28, 28, 1))

    run_root = get_run_root()
    ckpt_dir = run_root / cfg.checkpoints.dir
    ae_ckpt = representation_checkpoint_path(cfg.model.autoencoder, cfg.checkpoints, run_root)
    if not ae_ckpt.exists():
        if not smoke_test:
            raise FileNotFoundError("Pretrained representation checkpoint is required for LaSt-QGAN.")
    else:
        rep_variables = load_checkpoint(ae_ckpt, rep_variables)

    rff_setting = _surrogate_value(cfg, "rff_dim", "auto")
    rff_dim = matched_rff_dim(qcfg.n_qubits, qcfg.depth) if str(rff_setting) == "auto" else int(rff_setting)
    hidden_dim = int(_surrogate_value(cfg, "hidden_dim", 64))
    sigma = float(_surrogate_value(cfg, "sigma", 1.0))
    gen_params = init_fourier_params(
        gen_rng,
        noise_dim=qcfg.noise_dim,
        output_dim=expected_dim,
        rff_dim=rff_dim,
        hidden_dim=hidden_dim,
    )
    gen_apply = build_fourier_apply(
        noise_dim=qcfg.noise_dim,
        output_dim=expected_dim,
        rff_dim=rff_dim,
        hidden_dim=hidden_dim,
        sigma=sigma,
        seed=int(cfg.seed),
    )
    disc_params = init_discriminator_params(disc_rng, disc_model, latent_dim=cfg.model.autoencoder.latent_dim)

    gen_state = train_state.TrainState.create(
        apply_fn=lambda params, noise: gen_apply(params, noise),
        params=gen_params,
        tx=optax.adam(cfg.gen_lr, b1=0.5, b2=0.999),
    )
    disc_state = train_state.TrainState.create(
        apply_fn=disc_model.apply,
        params=disc_params,
        tx=optax.adam(cfg.disc_lr, b1=0.5, b2=0.999),
    )
    disc_step = make_disc_step(disc_model, gen_apply, representation, qcfg.noise_dim, cfg.lambda_gp)
    gen_step = make_gen_step(disc_model, gen_apply, qcfg.noise_dim)

    run = setup_wandb(cfg, mode="train")
    output_dir = run_root / cfg.outputs.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_csv_path = output_dir / "metrics.csv"
    diagnostics_path = output_dir / "evs_diagnostics.json"
    metadata_path = output_dir / "observable_bank_metadata.json"
    validation_metric_logger = EpochMetricLogger(metrics_csv_path, metrics_path)
    metadata = observable_bank.metadata()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    evs_diagnostics_writer = EvsDiagnosticsWriter(diagnostics_path)

    surrogate_counts = fourier_parameter_count(qcfg.noise_dim, expected_dim, rff_dim, hidden_dim)
    resource_counts = {
        "n_qubits": int(qcfg.n_qubits),
        "circuit_depth": int(qcfg.depth),
        "noise_dim": int(qcfg.noise_dim),
        "observable_bank_output_dim": int(expected_dim),
        "trainable_quantum_parameters": int(qcfg.noise_dim * matched_rff_dim(qcfg.n_qubits, qcfg.depth) + matched_rff_dim(qcfg.n_qubits, qcfg.depth)),
        "surrogate_rff_dim": int(rff_dim),
        "surrogate_hidden_dim": int(hidden_dim),
        "trainable_surrogate_parameters": _tree_parameter_count(gen_params),
        "discriminator_trainable_parameters": _tree_parameter_count(disc_params),
        "train_images": int(n_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "n_critic": int(cfg.n_critic),
        "gen_lr": float(cfg.gen_lr),
        "disc_lr": float(cfg.disc_lr),
        "lambda_gp": float(cfg.lambda_gp),
        "surrogate_total_parameters_including_fixed_rff": int(surrogate_counts["total"]),
    }
    registry = RunRegistry(output_dir, workflow="fourier_surrogate")
    registry.start(cfg, metadata, resource_counts, parent_run_id=matched_run_id)

    sample_paths: list[Path] = []
    global_step = 0
    for epoch in range(epochs):
        rng, perm_rng = jax.random.split(rng)
        perms = jax.random.permutation(perm_rng, n_samples)[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        for i in range(steps_per_epoch):
            batch_images = train_images[perms[i]]
            rng, step_rng = jax.random.split(rng)
            disc_state, disc_metrics = disc_step(disc_state, gen_state.params, rep_variables, batch_images, step_rng)
            gen_metrics = {}
            if global_step % cfg.n_critic == 0:
                rng, step_rng = jax.random.split(rng)
                gen_state, gen_metrics = gen_step(gen_state, disc_state.params, step_rng, batch_size)
            if global_step % log_every == 0:
                log_metrics(run, {f"train/{k}": float(v) for k, v in {**disc_metrics, **gen_metrics}.items()}, step=global_step)
            if global_step % sample_every == 0:
                rng, sample_rng = jax.random.split(rng)
                samples = _prepare_samples(gen_apply, gen_state.params, representation, rep_variables, sample_rng, sample_batch_size, qcfg.noise_dim)
                sample_path = output_dir / f"samples_step_{global_step:06d}.png"
                save_image_grid(samples, sample_path, nrow=8)
                sample_paths.append(sample_path)
                log_images(run, {"train/samples": samples}, step=global_step)
            global_step += 1

        metric_batch_size = min(4, metric_total_size) if smoke_test else 100
        fake_images_cpu = np.zeros((metric_total_size, 28, 28, 1), dtype=np.float32)
        rng, metric_rng = jax.random.split(rng)
        for start_idx in range(0, metric_total_size, metric_batch_size):
            end_idx = min(start_idx + metric_batch_size, metric_total_size)
            metric_rng, batch_rng = jax.random.split(metric_rng)
            batch_fake = _prepare_samples(gen_apply, gen_state.params, representation, rep_variables, batch_rng, end_idx - start_idx, qcfg.noise_dim)
            fake_images_cpu[start_idx:end_idx] = np.array(batch_fake)
        metrics_manager.update(fake_images_cpu)
        results = metrics_manager.compute()
        if (epoch + 1) % eval_epochs == 0:
            rng, evs_rng, eval_rng = jax.random.split(rng, 3)
            evs_noise = sample_noise(evs_rng, sample_batch_size, qcfg.noise_dim)
            evs_diagnostics = compute_evs_diagnostics(np.array(gen_apply(gen_state.params, evs_noise)))
            results = {**results, **evs_scalar_metrics(evs_diagnostics)}
            evs_diagnostics_writer.log(epoch=epoch, step=global_step, diagnostics=evs_diagnostics)
            eval_samples = _prepare_samples(gen_apply, gen_state.params, representation, rep_variables, eval_rng, sample_batch_size, qcfg.noise_dim)
            eval_sample_path = output_dir / f"samples_epoch_{epoch + 1:04d}.png"
            save_image_grid(eval_samples, eval_sample_path, nrow=8)
            sample_paths.append(eval_sample_path)
            log_images(run, {"eval/samples": eval_samples}, step=global_step)
        log_metrics(run, results, step=global_step)
        validation_metric_logger.log(epoch=epoch, step=global_step, metrics=results)
        metrics_manager.reset()

    gen_ckpt = ckpt_dir / cfg.checkpoints.generator
    disc_ckpt = ckpt_dir / cfg.checkpoints.discriminator
    save_checkpoint(gen_ckpt, gen_state.params)
    save_checkpoint(disc_ckpt, disc_state.params)
    registry.finalize(
        metrics_path=metrics_path,
        diagnostics_path=diagnostics_path,
        observable_bank_metadata_path=metadata_path,
        sample_grid_paths=sample_paths,
        checkpoint_paths={"generator": gen_ckpt, "discriminator": disc_ckpt},
        parent_run_id=matched_run_id,
    )

    if run is not None:
        run.finish()
    return gen_state.params, gen_state
