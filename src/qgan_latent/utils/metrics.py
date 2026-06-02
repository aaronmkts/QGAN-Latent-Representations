from __future__ import annotations

import warnings

from qgan_latent.shared.utils.metrics import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.metrics is deprecated; use qgan_latent.shared.utils.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)
