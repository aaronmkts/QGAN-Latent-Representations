from __future__ import annotations

import warnings

from qgan_latent.shared.representations.sinkhorn_autoencoder import *  # noqa: F401,F403
from qgan_latent.shared.representations.sinkhorn_autoencoder import _sample_prior as _sample_prior
from qgan_latent.shared.representations.sinkhorn_autoencoder import _sinkhorn_distance as _sinkhorn_distance
warnings.warn(
    "qgan_latent.models.compression_methods.sinkhorn_autoencoder is deprecated; use qgan_latent.shared.representations.sinkhorn_autoencoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
