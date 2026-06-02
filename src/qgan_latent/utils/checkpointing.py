from __future__ import annotations

import warnings

from qgan_latent.shared.utils.checkpointing import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.checkpointing is deprecated; use qgan_latent.shared.utils.checkpointing instead.",
    DeprecationWarning,
    stacklevel=2,
)
