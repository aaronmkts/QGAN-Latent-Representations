from __future__ import annotations

import warnings

from qgan_latent.shared.datamodules import MNISTDataModule, load_mnist

warnings.warn(
    "qgan_latent.datamodules is deprecated; use qgan_latent.shared.datamodules instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MNISTDataModule", "load_mnist"]
