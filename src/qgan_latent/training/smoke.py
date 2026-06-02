from __future__ import annotations

import warnings

from qgan_latent.shared.smoke import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.training.smoke is deprecated; use qgan_latent.shared.smoke instead.",
    DeprecationWarning,
    stacklevel=2,
)
