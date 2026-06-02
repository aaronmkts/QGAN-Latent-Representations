from __future__ import annotations

import jax


def select_device(device: str) -> None:
    device = device.lower()
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown device selection: {device}")
    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
        return
    jax.config.update("jax_platform_name", "gpu")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError as exc:
        raise RuntimeError("device=gpu requested but no GPU backend is available.") from exc
    if not gpus:
        raise RuntimeError("device=gpu requested but no GPU devices were found.")
