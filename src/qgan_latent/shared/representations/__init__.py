from __future__ import annotations

from .autoencoder import (
    Autoencoder,
    AutoencoderConfig,
    AutoencoderState,
    fit as fit_autoencoder,
    fit_transform as fit_transform_autoencoder,
    init_autoencoder_params,
    init_autoencoder_params_with_shape,
    init_autoencoder_variables_with_shape,
    transform as transform_autoencoder,
)
from .sinkclass_autoencoder import (
    SinkclassAutoencoderConfig,
    SinkclassAutoencoderState,
    fit as fit_sinkclass_autoencoder,
    fit_transform as fit_transform_sinkclass_autoencoder,
    transform as transform_sinkclass_autoencoder,
)
from .sinkhorn_autoencoder import (
    SinkhornAutoencoderConfig,
    SinkhornAutoencoderState,
    fit as fit_sinkhorn_autoencoder,
    fit_transform as fit_transform_sinkhorn_autoencoder,
    transform as transform_sinkhorn_autoencoder,
)
from .variational_autoencoder import (
    VariationalAutoencoder,
    VariationalAutoencoderConfig,
    VariationalAutoencoderState,
    fit as fit_variational_autoencoder,
    fit_transform as fit_transform_variational_autoencoder,
    transform as transform_variational_autoencoder,
)
from .vqvae import (
    VQVAE,
    VQVAEConfig,
    VQVAEState,
    fit as fit_vqvae,
    fit_transform as fit_transform_vqvae,
    transform as transform_vqvae,
)

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderState",
    "SinkclassAutoencoderConfig",
    "SinkclassAutoencoderState",
    "SinkhornAutoencoderConfig",
    "SinkhornAutoencoderState",
    "VariationalAutoencoder",
    "VariationalAutoencoderConfig",
    "VariationalAutoencoderState",
    "VQVAE",
    "VQVAEConfig",
    "VQVAEState",
    "fit_autoencoder",
    "fit_sinkclass_autoencoder",
    "fit_sinkhorn_autoencoder",
    "fit_transform_autoencoder",
    "fit_transform_sinkclass_autoencoder",
    "fit_transform_sinkhorn_autoencoder",
    "fit_transform_variational_autoencoder",
    "fit_transform_vqvae",
    "fit_variational_autoencoder",
    "fit_vqvae",
    "init_autoencoder_params",
    "init_autoencoder_params_with_shape",
    "init_autoencoder_variables_with_shape",
    "transform_autoencoder",
    "transform_sinkclass_autoencoder",
    "transform_sinkhorn_autoencoder",
    "transform_variational_autoencoder",
    "transform_vqvae",
]
