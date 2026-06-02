from __future__ import annotations

import warnings

from qgan_latent.shared.representations import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.compression_methods.__init__ is deprecated; use qgan_latent.shared.representations instead.",
    DeprecationWarning,
    stacklevel=2,
)
