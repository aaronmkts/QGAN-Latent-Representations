"""Classical MPS Born Machine in pure JAX.

Implements a Matrix Product State (MPS) that defines a probability
distribution via the Born rule:  P(x) = |ψ(x)|^2 / Z
where Z = <ψ|ψ> is the partition function (norm squared).

The MPS is used as an autoregressive prior model over sequences of
discrete tokens (codebook indices from a spatial VQ-VAE).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

_LOG_EPS = 1e-30


def _stable_scale(x: jnp.ndarray) -> jnp.ndarray:
    """Return a finite positive rescaling factor for an intermediate contraction."""
    scale = jnp.max(jnp.abs(x))
    finite_scale = jnp.where(jnp.isfinite(scale) & (scale > _LOG_EPS), scale, 1.0)
    return finite_scale


def init_mps_params(
    rng: jax.random.KeyArray,
    n_sites: int,
    phys_dim: int,
    bond_dim: int,
) -> dict:
    """Initialize MPS tensors with random values.

    Args:
        rng: JAX PRNG key.
        n_sites: Number of sites (e.g. 49 for 7x7 grid).
        phys_dim: Physical dimension per site (codebook size).
        bond_dim: Bond dimension controlling expressiveness.

    Returns:
        Dict with key "tensors": list of arrays with shapes
        site 0: (phys_dim, 1, bond_dim)
        site i: (phys_dim, bond_dim, bond_dim)  for 0 < i < n_sites-1
        site n-1: (phys_dim, bond_dim, 1)
    """
    tensors = []
    for i in range(n_sites):
        rng, sub = jax.random.split(rng)
        dl = 1 if i == 0 else bond_dim
        dr = 1 if i == n_sites - 1 else bond_dim
        # Initialize with small random values; scale ~1/sqrt(bond_dim)
        t = jax.random.normal(sub, (phys_dim, dl, dr)) * (1.0 / jnp.sqrt(bond_dim))
        tensors.append(t)
    return {"tensors": tensors}


def _contract_amplitude(tensors: Sequence[jnp.ndarray], config: jnp.ndarray) -> jnp.ndarray:
    """Contract MPS for a single configuration to get the amplitude ψ(config).

    Args:
        tensors: List of MPS tensors, each (phys_dim, bond_left, bond_right).
        config: 1-D integer array of length n_sites.

    Returns:
        Scalar amplitude ψ(config).
    """
    # Start with the first tensor, selecting the physical index
    vec = tensors[0][config[0]]  # shape: (1, bond_dim)
    for i in range(1, len(tensors)):
        mat = tensors[i][config[i]]  # shape: (bond_dim, bond_dim) or (bond_dim, 1)
        vec = vec @ mat
    # vec is (1, 1), squeeze to scalar
    return vec.squeeze()


def _batch_log_amplitudes(
    tensors: Sequence[jnp.ndarray], configs: jnp.ndarray
) -> jnp.ndarray:
    """Compute log|ψ(config)| for a batch of configurations.

    Args:
        tensors: MPS tensors.
        configs: (batch_size, n_sites) integer array.

    Returns:
        (batch_size,) array of log|ψ(config)|.
    """
    def single_log_amp(config):
        vec = tensors[0][config[0]]
        scale = _stable_scale(vec)
        vec = vec / scale
        log_scale = jnp.log(scale)

        for i in range(1, len(tensors)):
            mat = tensors[i][config[i]]
            vec = vec @ mat
            scale = _stable_scale(vec)
            vec = vec / scale
            log_scale = log_scale + jnp.log(scale)

        return jnp.log(jnp.abs(vec.squeeze()) + _LOG_EPS) + log_scale

    return jax.vmap(single_log_amp)(configs)


def _compute_transfer_matrix(tensor: jnp.ndarray) -> jnp.ndarray:
    """Compute the transfer matrix T = sum_s A[s]^* ⊗ A[s] for a single site.

    For real tensors, this is T[a,b,c,d] = sum_s A[s,a,c] * A[s,b,d]
    which when reshaped to (dl*dl, dr*dr) can be used for efficient norm
    computation.

    Args:
        tensor: (phys_dim, dl, dr)

    Returns:
        Transfer matrix of shape (dl, dl, dr, dr) -> reshaped to (dl*dl, dr*dr)
    """
    # Einstein summation: sum over physical index s
    # T[a,b,c,d] = sum_s A[s,a,c] * A[s,b,d]
    t = jnp.einsum("sac,sbd->abcd", tensor, tensor)
    dl = tensor.shape[1]
    dr = tensor.shape[2]
    return t.reshape(dl * dl, dr * dr)


def compute_log_norm_sq(tensors: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Compute log(<ψ|ψ>) = log(Z) by contracting all transfer matrices.

    Returns:
        Scalar log of the squared norm.
    """
    # Start from the left boundary
    T = _compute_transfer_matrix(tensors[0])  # (1, dr*dr)
    # For boundary: dl=1, so T is (1, dr*dr)
    vec = T  # (1, dr*dr)
    scale = _stable_scale(vec)
    vec = vec / scale
    log_scale = jnp.log(scale)
    for i in range(1, len(tensors)):
        T_i = _compute_transfer_matrix(tensors[i])  # (dl*dl, dr*dr)
        vec = vec @ T_i  # (1, dr*dr)
        scale = _stable_scale(vec)
        vec = vec / scale
        log_scale = log_scale + jnp.log(scale)
    # Final: vec is (1, 1)
    norm_sq = vec.squeeze()
    return jnp.log(jnp.abs(norm_sq) + _LOG_EPS) + log_scale


def mps_nll_loss(params: dict, batch_indices: jnp.ndarray) -> jnp.ndarray:
    """Negative log-likelihood loss for training.

    NLL = -mean(log P(x)) = -mean(2*log|ψ(x)| - log Z)

    Args:
        params: Dict with "tensors" key.
        batch_indices: (batch_size, n_sites) integer indices.

    Returns:
        Scalar NLL loss.
    """
    tensors = params["tensors"]
    log_amps = _batch_log_amplitudes(tensors, batch_indices)
    log_z = compute_log_norm_sq(tensors)
    # log P(x) = 2*log|ψ(x)| - log Z
    log_probs = 2.0 * log_amps - log_z
    return -jnp.mean(log_probs)


def _compute_right_envs(tensors: Sequence[jnp.ndarray]) -> list[jnp.ndarray]:
    """Precompute right environment matrices for autoregressive sampling.

    R[i] represents the contraction of transfer matrices from site i to
    the last site. Used to compute conditional probabilities efficiently.

    Returns:
        List of right environments, where R[i] has shape (bond_dim*bond_dim,)
        or (bond_dim, bond_dim) depending on site.
    """
    n = len(tensors)
    # R[n-1] = transfer matrix of last site, which is (dr*dr, 1) = scalar after squeeze
    rights = [None] * n

    # Start from the rightmost site
    T_last = _compute_transfer_matrix(tensors[n - 1])  # (dl*dl, 1)
    rights[n - 1] = T_last.squeeze(-1)  # (dl*dl,)

    for i in range(n - 2, -1, -1):
        T_i = _compute_transfer_matrix(tensors[i])  # (dl*dl, dr*dr)
        rights[i] = (T_i @ rights[i + 1])  # (dl*dl,)
        # Normalize to prevent overflow
        scale = jnp.max(jnp.abs(rights[i]))
        rights[i] = rights[i] / (scale + 1e-30)

    return rights


def mps_sample(
    params: dict,
    rng: jax.random.KeyArray,
    n_samples: int,
) -> jnp.ndarray:
    """Sample from the MPS Born machine autoregressively.

    For each sample, compute conditional probabilities site by site
    using left contexts and precomputed right environments.

    Args:
        params: Dict with "tensors" key.
        rng: JAX PRNG key.
        n_samples: Number of samples to generate.

    Returns:
        (n_samples, n_sites) integer array of sampled indices.
    """
    tensors = params["tensors"]
    n_sites = len(tensors)
    phys_dim = tensors[0].shape[0]

    # Precompute right environments
    right_envs = _compute_right_envs(tensors)

    def sample_one(rng_key):
        """Sample a single configuration autoregressively."""
        samples = []
        # left_vec represents the contracted state up to current site
        # Initially just a scalar 1 (no left context)
        # We track it as a vector of shape (bond_dim,) for the bra and ket
        # Combined: (bond_dim * bond_dim,) representing |left><left|
        left_vec = jnp.ones((1,))  # (1,) for boundary

        for i in range(n_sites):
            rng_key, sub = jax.random.split(rng_key)
            A = tensors[i]  # (phys_dim, dl, dr)
            dl = A.shape[1]
            dr = A.shape[2]

            # Compute unnormalized probabilities for each physical index
            probs = jnp.zeros(phys_dim)
            for s in range(phys_dim):
                # A[s]: (dl, dr)
                # Bra-ket contraction: left_vec . (A[s] ⊗ A[s]) . right_env
                # left_vec: (dl*dl,)
                mat_s = jnp.kron(A[s], A[s])  # (dl*dl, dr*dr)
                contracted = left_vec @ mat_s  # (dr*dr,)
                if i < n_sites - 1:
                    val = contracted @ right_envs[i + 1]  # scalar
                else:
                    val = contracted.sum()  # last site: dr=1
                probs = probs.at[s].set(jnp.clip(val, a_min=0.0))

            # Normalize
            probs = probs / (jnp.sum(probs) + _LOG_EPS)

            # Sample
            chosen = jax.random.categorical(sub, jnp.log(probs + _LOG_EPS))
            samples.append(chosen)

            # Update left context
            mat_chosen = jnp.kron(A[chosen], A[chosen])  # (dl*dl, dr*dr)
            left_vec = left_vec @ mat_chosen  # (dr*dr,)
            # Normalize to prevent overflow
            scale = jnp.max(jnp.abs(left_vec))
            left_vec = left_vec / (scale + _LOG_EPS)

        return jnp.stack(samples)

    # Sample all configurations
    rngs = jax.random.split(rng, n_samples)
    # Note: not vmapping because of the sequential nature and
    # dynamic indexing; use a simple loop
    all_samples = []
    for i in range(n_samples):
        all_samples.append(sample_one(rngs[i]))
    return jnp.stack(all_samples)
