from __future__ import annotations

import warnings

from qgan_latent.shared.utils.seed import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.seed is deprecated; use qgan_latent.shared.utils.seed instead.",
    DeprecationWarning,
    stacklevel=2,
)
