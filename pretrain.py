from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.pretrain_loop import run_pretrain

log = logging.getLogger(__name__)


@hydra.main(config_path=str(ROOT / "configs"), config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Configs:\n%s", OmegaConf.to_yaml(cfg))
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
