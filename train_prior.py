"""Entry point for training the MPS Born Machine prior.

Usage:
    python train_prior.py

Requires a pretrained Spatial VQ-VAE checkpoint. Train one first with:
    python pretrain.py model/spatial_vqvae@model.autoencoder
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


@hydra.main(config_path="configs", config_name="train_prior", version_base=None)
def main(cfg: DictConfig) -> None:
    from training.mps_prior_loop import run_mps_prior

    run_mps_prior(cfg)


if __name__ == "__main__":
    main()
