from __future__ import annotations

import warnings

from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.quantum_generator.__init__ is deprecated; use qgan_latent.workflows.qgan_expectation_values.models.quantum_generator instead.",
    DeprecationWarning,
    stacklevel=2,
)
