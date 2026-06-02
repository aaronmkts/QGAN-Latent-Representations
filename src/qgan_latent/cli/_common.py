from __future__ import annotations

import multiprocessing
import os
from pathlib import Path


def find_config_dir() -> Path:
    """Find configs/ for editable installs, root scripts, and console scripts."""
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        config_dir = candidate / "configs"
        if config_dir.is_dir():
            return config_dir
    raise FileNotFoundError("Could not find configs/ from cwd or package path.")


def configure_cpu_parallelism() -> None:
    """Set common CPU math-library thread counts for training runs."""
    try:
        if hasattr(os, "sched_getaffinity"):
            num_cores = len(os.sched_getaffinity(0))
        else:
            num_cores = multiprocessing.cpu_count()
    except Exception:
        num_cores = 4

    num_cores_str = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = num_cores_str
    os.environ["MKL_NUM_THREADS"] = num_cores_str
    os.environ["OPENBLAS_NUM_THREADS"] = num_cores_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_cores_str
    os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str
