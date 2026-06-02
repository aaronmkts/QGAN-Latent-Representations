from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from qgan_latent.cli._common import find_config_dir
from qgan_latent.shared.training.pretrain_loop import run_pretrain

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(find_config_dir()),
    config_name="projects/qgan_expectation_values/pretrain",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    log.info("Configs:\n%s", OmegaConf.to_yaml(cfg))
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
