from __future__ import annotations

import warnings

from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.circuits import *  # noqa: F401,F403
warnings.warn(
    "qgan_latent.models.quantum_generator.circuits is deprecated; use qgan_latent.workflows.qgan_expectation_values.models.quantum_generator.circuits instead.",
    DeprecationWarning,
    stacklevel=2,
)
