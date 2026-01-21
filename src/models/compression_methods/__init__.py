from __future__ import annotations

from .autoencoder import (
    Autoencoder,
    AutoencoderConfig,
    AutoencoderState,
    fit as fit_autoencoder,
    fit_transform as fit_transform_autoencoder,
    init_autoencoder_params,
    init_autoencoder_params_with_shape,
    transform as transform_autoencoder,
)
from .nmf import NMFConfig, NMFState, fit as fit_nmf, fit_transform as fit_transform_nmf, transform as transform_nmf
from .pca import PCAConfig, PCAState, fit as fit_pca, fit_transform as fit_transform_pca, transform as transform_pca
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

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderState",
    "NMFConfig",
    "NMFState",
    "PCAConfig",
    "PCAState",
    "SinkclassAutoencoderConfig",
    "SinkclassAutoencoderState",
    "SinkhornAutoencoderConfig",
    "SinkhornAutoencoderState",
    "VariationalAutoencoder",
    "VariationalAutoencoderConfig",
    "VariationalAutoencoderState",
    "fit_autoencoder",
    "fit_nmf",
    "fit_pca",
    "fit_sinkclass_autoencoder",
    "fit_sinkhorn_autoencoder",
    "fit_transform_autoencoder",
    "fit_transform_nmf",
    "fit_transform_pca",
    "fit_transform_sinkclass_autoencoder",
    "fit_transform_sinkhorn_autoencoder",
    "fit_transform_variational_autoencoder",
    "fit_variational_autoencoder",
    "init_autoencoder_params",
    "init_autoencoder_params_with_shape",
    "transform_autoencoder",
    "transform_nmf",
    "transform_pca",
    "transform_sinkclass_autoencoder",
    "transform_sinkhorn_autoencoder",
    "transform_variational_autoencoder",
]
