from __future__ import annotations

import warnings

from qgan_latent.shared.datamodules.mnist import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.datamodules.mnist is deprecated; use qgan_latent.shared.datamodules.mnist instead.",
    DeprecationWarning,
    stacklevel=2,
)
