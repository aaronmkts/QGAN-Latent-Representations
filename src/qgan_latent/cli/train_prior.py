from __future__ import annotations

import hydra
from omegaconf import DictConfig

from qgan_latent.cli._common import find_config_dir
from qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop import run_mps_prior


@hydra.main(config_path=str(find_config_dir()), config_name="train_prior", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_mps_prior(cfg)


if __name__ == "__main__":
    main()
