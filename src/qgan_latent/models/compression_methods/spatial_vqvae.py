from __future__ import annotations

import warnings

from qgan_latent.shared.representations.spatial_vqvae import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.compression_methods.spatial_vqvae is deprecated; use qgan_latent.shared.representations.spatial_vqvae instead.",
    DeprecationWarning,
    stacklevel=2,
)
