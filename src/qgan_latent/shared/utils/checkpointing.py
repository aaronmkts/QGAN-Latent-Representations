from __future__ import annotations

from pathlib import Path
from typing import Any

from flax import serialization


def save_checkpoint(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(serialization.to_bytes(params))


def load_checkpoint(path: Path, params_template: Any) -> Any:
    with path.open("rb") as f:
        data = f.read()
    return serialization.from_bytes(params_template, data)
