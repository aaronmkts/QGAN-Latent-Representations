from __future__ import annotations

import logging
from pathlib import Path

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from qgan_latent.cli._common import configure_cpu_parallelism, find_config_dir
from qgan_latent.shared.datamodules.mnist import MNISTDataModule
from qgan_latent.shared.latent_cache import cache_latent_representations
from qgan_latent.shared.representations.runtime import (
    build_representation_runtime,
    init_representation_variables,
    representation_checkpoint_path,
)
from qgan_latent.shared.smoke import synthetic_mnist_images
from qgan_latent.shared.utils.checkpointing import load_checkpoint
from qgan_latent.shared.utils.device import select_device
from qgan_latent.shared.utils.paths import get_run_root
from qgan_latent.shared.utils.seed import set_seed

configure_cpu_parallelism()
log = logging.getLogger(__name__)


def _checkpoint_path(cfg: DictConfig) -> Path:
    return representation_checkpoint_path(cfg.model.autoencoder, cfg.checkpoints, get_run_root())


def _build_splits(cfg: DictConfig) -> dict[str, np.ndarray]:
    if bool(getattr(cfg, "smoke_test", False)):
        images = np.asarray(synthetic_mnist_images(8), dtype=np.float32)
        return {"train": images}

    data = MNISTDataModule(cfg.data.data_dir, num_workers=cfg.data.num_workers)
    data.setup()
    train = np.asarray(data.train_images, dtype=np.float32)
    test = np.asarray(data.test_images, dtype=np.float32) if hasattr(data, "test_images") else train[:0]
    validation_size = min(5000, max(1, train.shape[0] // 10))
    return {
        "train": train[:-validation_size],
        "validation": train[-validation_size:],
        "test": test,
    }


def run_cache_from_config(cfg: DictConfig) -> dict:
    select_device(cfg.device)
    set_seed(cfg.seed)

    runtime = build_representation_runtime(cfg.model.autoencoder)
    rng = jax.random.PRNGKey(int(cfg.seed))
    variables = init_representation_variables(runtime, rng, input_shape=(28, 28, 1))
    ckpt = _checkpoint_path(cfg)
    if ckpt.exists():
        variables = load_checkpoint(ckpt, variables)
    elif not bool(getattr(cfg, "smoke_test", False)):
        raise FileNotFoundError(f"Representation checkpoint not found: {ckpt}")

    output_dir = get_run_root() / cfg.outputs.dir / "latent_cache"
    batch_size = min(int(cfg.batch_size), 8) if bool(getattr(cfg, "smoke_test", False)) else int(cfg.batch_size)
    return cache_latent_representations(
        runtime=runtime,
        variables=variables,
        splits=_build_splits(cfg),
        output_dir=output_dir,
        seed=int(cfg.seed),
        checkpoint_path=ckpt,
        batch_size=batch_size,
        overwrite=bool(getattr(getattr(cfg, "cache", {}), "overwrite", False)),
    )


@hydra.main(
    config_path=str(find_config_dir()),
    config_name="projects/qgan_expectation_values/cache_latents",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    log.info("Configs:\n%s", OmegaConf.to_yaml(cfg))
    manifest = run_cache_from_config(cfg)
    log.info("Cached latent representations: %s", manifest)


if __name__ == "__main__":
    main()
