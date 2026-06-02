from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from qgan_latent.cli._common import configure_cpu_parallelism, find_config_dir
from qgan_latent.training.gan_loop import run_gan

configure_cpu_parallelism()

log = logging.getLogger(__name__)


@hydra.main(config_path=str(find_config_dir()), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Configs:\n%s", OmegaConf.to_yaml(cfg))
    run_gan(cfg)


if __name__ == "__main__":
    main()
