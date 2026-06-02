from __future__ import annotations

from pathlib import Path

import hydra


def get_run_root() -> Path:
    """Return Hydra's original cwd, or cwd when called outside Hydra."""
    try:
        return Path(hydra.utils.get_original_cwd())
    except ValueError:
        return Path.cwd()
