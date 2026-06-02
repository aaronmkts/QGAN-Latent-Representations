from __future__ import annotations

import warnings

from qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.mps_prior.mps is deprecated; use qgan_latent.workflows.tensor_prior_vqvae.models.mps_prior.mps instead.",
    DeprecationWarning,
    stacklevel=2,
)
