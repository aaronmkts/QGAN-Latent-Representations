from __future__ import annotations

import warnings

from qgan_latent.workflows.qgan_expectation_values.models.discriminator import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.discriminator is deprecated; use qgan_latent.workflows.qgan_expectation_values.models.discriminator instead.",
    DeprecationWarning,
    stacklevel=2,
)
