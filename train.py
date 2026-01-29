from __future__ import annotations
import os
import multiprocessing
try:
    if hasattr(os, "sched_getaffinity"):
        num_cores = len(os.sched_getaffinity(0))
    else:
        num_cores = multiprocessing.cpu_count()
except Exception:
    num_cores = 4 

num_cores_str = str(num_cores)

# Set environment variables for major math libraries
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] = num_cores_str
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str

print(f"🚀 Enforcing CPU parallelism: OMP_NUM_THREADS={num_cores_str}")

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.gan_loop import run_gan

log = logging.getLogger(__name__)


@hydra.main(config_path=str(ROOT / "configs"), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Configs:\n%s", OmegaConf.to_yaml(cfg))
    run_gan(cfg)


if __name__ == "__main__":
    main()
