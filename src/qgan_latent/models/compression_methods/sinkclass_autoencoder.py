from __future__ import annotations

import warnings

from qgan_latent.shared.representations.sinkclass_autoencoder import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.compression_methods.sinkclass_autoencoder is deprecated; use qgan_latent.shared.representations.sinkclass_autoencoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
