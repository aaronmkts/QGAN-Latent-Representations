from __future__ import annotations

from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from utils.image_grid import make_grid


def setup_wandb(cfg, mode: str) -> Optional[Any]:
    wandb_mode = getattr(cfg, "wandb_mode", "disabled")
    if wandb_mode == "disabled":
        return None
    import wandb

    config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        name=cfg.wandb.name,
        config=config,
        mode=wandb_mode,
        job_type=mode,
    )
    return run


def log_metrics(run: Optional[Any], metrics: Dict[str, float], step: int) -> None:
    if run is None:
        return
    import wandb

    wandb.log(metrics, step=step)


def log_images(run: Optional[Any], images: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    import numpy as np
    import wandb

    for name, grid in images.items():
        array = np.asarray(grid)
        if array.ndim == 4:
            array = make_grid(array, nrow=8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        wandb.log({name: wandb.Image(array)}, step=step)
