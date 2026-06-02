from __future__ import annotations

import warnings

from qgan_latent.shared.utils.train_state import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.utils.train_state is deprecated; use qgan_latent.shared.utils.train_state instead.",
    DeprecationWarning,
    stacklevel=2,
)
