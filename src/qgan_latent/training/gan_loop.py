from __future__ import annotations

import warnings

from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.training.gan_loop is deprecated; use qgan_latent.workflows.qgan_expectation_values.training.gan_loop instead.",
    DeprecationWarning,
    stacklevel=2,
)
