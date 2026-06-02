from __future__ import annotations

import warnings

from qgan_latent.shared.utils.device import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.device is deprecated; use qgan_latent.shared.utils.device instead.",
    DeprecationWarning,
    stacklevel=2,
)
