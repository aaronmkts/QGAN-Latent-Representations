from __future__ import annotations

import warnings

from qgan_latent.shared.utils.image_grid import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.image_grid is deprecated; use qgan_latent.shared.utils.image_grid instead.",
    DeprecationWarning,
    stacklevel=2,
)
