from __future__ import annotations

import warnings

from qgan_latent.shared.training import run_pretrain
from qgan_latent.workflows.qgan_expectation_values.training import run_gan
from qgan_latent.workflows.tensor_prior_vqvae.training import run_mps_prior

warnings.warn(
    "qgan_latent.training is deprecated; use qgan_latent.shared.training or qgan_latent.workflows.*.training instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run_gan", "run_mps_prior", "run_pretrain"]
