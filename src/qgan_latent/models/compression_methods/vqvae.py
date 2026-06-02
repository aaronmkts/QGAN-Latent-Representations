from __future__ import annotations

import warnings

from qgan_latent.shared.representations.vqvae import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.compression_methods.vqvae is deprecated; use qgan_latent.shared.representations.vqvae instead.",
    DeprecationWarning,
    stacklevel=2,
)
