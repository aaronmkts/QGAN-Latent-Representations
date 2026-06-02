from __future__ import annotations

import warnings

from qgan_latent.shared.utils.metrics_wrapper import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.metrics_wrapper is deprecated; use qgan_latent.shared.utils.metrics_wrapper instead.",
    DeprecationWarning,
    stacklevel=2,
)
