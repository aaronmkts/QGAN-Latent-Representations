from __future__ import annotations

import warnings

from qgan_latent.shared.training.pretrain_loop import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.training.pretrain_loop is deprecated; use qgan_latent.shared.training.pretrain_loop instead.",
    DeprecationWarning,
    stacklevel=2,
)
