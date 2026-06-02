from __future__ import annotations

import warnings

from qgan_latent.shared.representations.autoencoder import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.compression_methods.autoencoder is deprecated; use qgan_latent.shared.representations.autoencoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
