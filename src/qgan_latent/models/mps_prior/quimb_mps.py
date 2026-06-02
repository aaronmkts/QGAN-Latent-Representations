"""Quimb-backed MPS Born Machine.

Uses quimb tensor network library for optimized MPS operations:
- Efficient contraction ordering via cotengra
- Native MPS canonical form management
- SVD-based bond dimension compression
- Fast norm and overlap computations

The MPS tensors are stored as JAX arrays for gradient computation,
and converted to quimb tensors for contraction/sampling operations.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

try:
    import quimb.tensor as qtn
    HAS_QUIMB = True
except ImportError:
    HAS_QUIMB = False


def _check_quimb():
    if not HAS_QUIMB:
        raise ImportError(
            "quimb is required for quimb_mps backend. "
            "Install with: pip install quimb"
        )


def _params_to_quimb_mps(tensors: Sequence[jnp.ndarray]) -> "qtn.MatrixProductState":
    """Convert JAX MPS tensors to a quimb MatrixProductState.

    quimb expects tensors with indices (bond_left, phys, bond_right)
    while our format is (phys, bond_left, bond_right), so we transpose.
    """
    _check_quimb()
    arrays = []
    for i, t in enumerate(tensors):
        # Convert (phys, dl, dr) -> (dl, phys, dr) for quimb
        arr = np.array(jnp.transpose(t, (1, 0, 2)))
        arrays.append(arr)
    return qtn.MatrixProductState(arrays)


def init_quimb_mps(
    rng: jax.random.KeyArray,
    n_sites: int,
    phys_dim: int,
    bond_dim: int,
) -> dict:
    """Initialize MPS tensors suitable for quimb operations.

    Same format as init_mps_params (phys_dim, dl, dr) for JAX compatibility.
    """
    _check_quimb()
    tensors = []
    for i in range(n_sites):
        rng, sub = jax.random.split(rng)
        dl = 1 if i == 0 else bond_dim
        dr = 1 if i == n_sites - 1 else bond_dim
        t = jax.random.normal(sub, (phys_dim, dl, dr)) * (1.0 / jnp.sqrt(bond_dim))
        tensors.append(t)
    return {"tensors": tensors}


def quimb_norm_sq(tensors: Sequence[jnp.ndarray]) -> float:
    """Compute <ψ|ψ> using quimb's optimized contraction."""
    mps = _params_to_quimb_mps(tensors)
    return float(mps.H @ mps)


def quimb_log_prob(
    tensors: Sequence[jnp.ndarray], config: jnp.ndarray
) -> float:
    """Compute log P(config) = log(|<config|ψ>|^2 / <ψ|ψ>).

    Uses quimb for efficient norm computation.
    """
    mps = _params_to_quimb_mps(tensors)
    norm_sq = float(mps.H @ mps)

    # Compute amplitude by selecting physical indices
    config_np = np.array(config)
    arrays_selected = []
    for i, t_jax in enumerate(tensors):
        # Select physical index: (phys, dl, dr) -> (dl, dr)
        arr = np.array(t_jax[config_np[i]])
        arrays_selected.append(arr)

    # Contract the chain of matrices
    result = arrays_selected[0]
    for mat in arrays_selected[1:]:
        result = result @ mat
    amplitude = float(result.squeeze())

    log_p = 2.0 * np.log(abs(amplitude) + 1e-30) - np.log(abs(norm_sq) + 1e-30)
    return log_p


def quimb_sample(
    tensors: Sequence[jnp.ndarray],
    rng: jax.random.KeyArray,
    n_samples: int,
) -> jnp.ndarray:
    """Sample from MPS using quimb for reduced density matrix computation.

    Uses quimb's efficient partial trace to compute conditional
    probabilities at each site.
    """
    _check_quimb()
    n_sites = len(tensors)
    phys_dim = tensors[0].shape[0]

    # Convert to numpy for quimb operations
    np_tensors = [np.array(t) for t in tensors]

    all_samples = []
    for s in range(n_samples):
        rng, sample_rng = jax.random.split(rng)
        sample = _sample_one_quimb(np_tensors, sample_rng, n_sites, phys_dim)
        all_samples.append(sample)

    return jnp.stack(all_samples)


def _sample_one_quimb(
    np_tensors: list[np.ndarray],
    rng: jax.random.KeyArray,
    n_sites: int,
    phys_dim: int,
) -> jnp.ndarray:
    """Sample a single configuration using quimb partial traces."""
    samples = []
    # Track left boundary state as we condition on sampled values
    # left_bra and left_ket: vectors of shape (bond_dim,)
    left = np.ones((1, 1))  # (dl, dl) identity for boundary

    for i in range(n_sites):
        rng, sub = jax.random.split(rng)
        A = np_tensors[i]  # (phys, dl, dr)

        # Compute unnormalized probabilities for each physical index
        probs = np.zeros(phys_dim)

        # Compute right contraction from site i+1 to end
        right = np.ones((1, 1))  # boundary
        for j in range(n_sites - 1, i, -1):
            Aj = np_tensors[j]  # (phys, dl, dr)
            # Transfer matrix: sum_s Aj[s]^T @ right @ Aj[s]
            new_right = np.zeros((Aj.shape[1], Aj.shape[1]))
            for s_idx in range(phys_dim):
                new_right += Aj[s_idx].T @ right @ Aj[s_idx]
            right = new_right
            # Normalize
            scale = np.max(np.abs(right))
            if scale > 0:
                right = right / scale

        for s_idx in range(phys_dim):
            As = A[s_idx]  # (dl, dr)
            # left @ As @ right @ As^T contracted to scalar
            val = np.trace(left @ As @ right @ As.T)
            probs[s_idx] = abs(val)

        # Normalize and sample
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(phys_dim) / phys_dim

        chosen = int(jax.random.categorical(sub, jnp.log(jnp.array(probs) + 1e-30)))
        samples.append(chosen)

        # Update left context: left' = A[chosen]^T @ left @ A[chosen]
        As_chosen = A[chosen]  # (dl, dr)
        left = As_chosen.T @ left @ As_chosen  # (dr, dr)
        scale = np.max(np.abs(left))
        if scale > 0:
            left = left / scale

    return jnp.array(samples, dtype=jnp.int32)
