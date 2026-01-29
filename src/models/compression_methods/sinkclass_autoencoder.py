from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax

from .autoencoder import Autoencoder, init_autoencoder_variables_with_shape
from .sinkhorn_autoencoder import _sample_prior, _sinkhorn_distance
from utils.train_state import TrainStateWithBatchStats


class LatentClassifier(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


@struct.dataclass
class SinkclassAutoencoderConfig:
    latent_dim: int
    encoder_channels: Sequence[int]
    decoder_channels: Sequence[int]
    mlp_dim: int
    classifier_dim: int = 64
    tanh_latent: bool = True
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 128
    loss: str = "mse"
    sinkhorn_weight: float = 1.0
    lambda_sinkhorn: float | None = None
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    sinkhorn_cost: str = "l2_sq"
    prior: str = "gaussian"
    class_weight: float = 1.0


@struct.dataclass
class SinkclassAutoencoderState:
    params: dict
    batch_stats: dict
    config: SinkclassAutoencoderConfig


def _recon_loss(x: jnp.ndarray, x_hat: jnp.ndarray, loss_type: str) -> jnp.ndarray:
    if loss_type == "bce":
        eps = 1e-7
        x_hat = jnp.clip(x_hat, eps, 1.0 - eps)
        return -jnp.mean(x * jnp.log(x_hat) + (1.0 - x) * jnp.log(1.0 - x_hat))
    return jnp.mean((x - x_hat) ** 2)


def _split_features_labels(data):
    if isinstance(data, (tuple, list)) and len(data) == 2:
        return data[0], data[1]
    if isinstance(data, dict):
        features = data.get("images", data.get("features"))
        labels = data.get("labels")
        return features, labels
    raise ValueError("Sinkclass autoencoder requires (features, labels) data.")


def make_train_step(
    autoencoder: Autoencoder, classifier: LatentClassifier, config: SinkclassAutoencoderConfig
):
    @jax.jit
    def train_step(
        state: TrainStateWithBatchStats,
        batch: jnp.ndarray,
        labels: jnp.ndarray,
        rng: jax.random.KeyArray,
    ):
        def loss_fn(params):
            ae_params = params["autoencoder"]
            cls_params = params["classifier"]
            variables = {"params": ae_params, "batch_stats": state.batch_stats}
            z, enc_updates = autoencoder.apply(
                variables, batch, method=Autoencoder.encode, train=True, mutable=["batch_stats"]
            )
            variables = {"params": ae_params, "batch_stats": enc_updates["batch_stats"]}
            recon, dec_updates = autoencoder.apply(
                variables, z, method=Autoencoder.decode, train=True, mutable=["batch_stats"]
            )
            recon_loss = _recon_loss(batch, recon, config.loss)
            target = _sample_prior(rng, z.shape, config.prior)
            sinkhorn = _sinkhorn_distance(
                z, target, config.sinkhorn_eps, config.sinkhorn_iters, config.sinkhorn_cost
            )
            logits = classifier.apply({"params": cls_params}, z)
            labels = labels.astype(jnp.float32)
            cls_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
            weight = config.sinkhorn_weight if config.lambda_sinkhorn is None else config.lambda_sinkhorn
            total = recon_loss + weight * sinkhorn + config.class_weight * cls_loss
            return total, dec_updates["batch_stats"]

        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
        return new_state, loss

    return train_step


def fit(
    data, config: SinkclassAutoencoderConfig, rng: jax.random.KeyArray
) -> SinkclassAutoencoderState:
    features, labels = _split_features_labels(data)
    if features is None or labels is None:
        raise ValueError("Sinkclass autoencoder requires features and labels.")
    features = jnp.asarray(features)
    labels = jnp.asarray(labels).reshape((-1,))
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must have matching batch size.")

    autoencoder = Autoencoder(
        latent_dim=config.latent_dim,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        mlp_dim=config.mlp_dim,
        tanh_latent=config.tanh_latent,
    )
    classifier = LatentClassifier(hidden_dim=config.classifier_dim)
    rng, ae_rng, cls_rng = jax.random.split(rng, 3)
    ae_variables = init_autoencoder_variables_with_shape(
        ae_rng, autoencoder, features.shape[1:]
    )
    cls_params = classifier.init(cls_rng, jnp.zeros((1, config.latent_dim)))["params"]
    params = {"autoencoder": ae_variables["params"], "classifier": cls_params}
    state = TrainStateWithBatchStats.create(
        apply_fn=lambda p, x: p,
        params=params,
        batch_stats=ae_variables["batch_stats"],
        tx=optax.adam(config.learning_rate),
    )
    train_step = make_train_step(autoencoder, classifier, config)
    n_samples = features.shape[0]
    steps_per_epoch = math.ceil(n_samples / config.batch_size)

    for _ in range(config.epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
            batch = features[idx]
            batch_labels = labels[idx]
            rng, step_rng = jax.random.split(rng)
            state, _ = train_step(state, batch, batch_labels, step_rng)

    return SinkclassAutoencoderState(
        params=state.params, batch_stats=state.batch_stats, config=config
    )


def transform(data: jnp.ndarray, state: SinkclassAutoencoderState) -> jnp.ndarray:
    data = jnp.asarray(data)
    model = Autoencoder(
        latent_dim=state.config.latent_dim,
        encoder_channels=state.config.encoder_channels,
        decoder_channels=state.config.decoder_channels,
        mlp_dim=state.config.mlp_dim,
        tanh_latent=state.config.tanh_latent,
    )
    variables = {"params": state.params["autoencoder"], "batch_stats": state.batch_stats}
    return model.apply(variables, data, method=Autoencoder.encode, train=False)


def fit_transform(
    data, config: SinkclassAutoencoderConfig, rng: jax.random.KeyArray
) -> tuple[SinkclassAutoencoderState, jnp.ndarray]:
    state = fit(data, config, rng)
    features, _ = _split_features_labels(data)
    return state, transform(features, state)
