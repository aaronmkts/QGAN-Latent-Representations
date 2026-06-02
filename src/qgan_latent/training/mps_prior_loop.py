from __future__ import annotations

import warnings

from qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.training.mps_prior_loop is deprecated; use qgan_latent.workflows.tensor_prior_vqvae.training.mps_prior_loop instead.",
    DeprecationWarning,
    stacklevel=2,
)
