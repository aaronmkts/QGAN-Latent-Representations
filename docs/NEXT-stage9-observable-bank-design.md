# NOTE — Next Experiment Design After Stage 8

Stage 8 showed that simple observable-bank substitutions do not yet produce meaningful dequantization-aware separation against the matched Fourier/RFF surrogate. Before spending more GPU time, the next experiment should address the shape and bank-design constraints exposed by the sweep.

## Constraint exposed by Stage 8

The current frozen autoencoder checkpoint has:

```text
latent_dim = 20
```

The QGAN implementation requires:

```text
autoencoder.latent_dim == observable_bank.output_dim
```

This forced awkward dimension-compatible conditions. The mixed condition in particular used 7 qubits to keep `[X,Z]+[XX]` at 20 outputs:

```text
2 * 7 local terms + 6 XX line terms = 20
```

That changes the qubit count and may explain part of the poor mixed-bank result.

## Candidate next designs

### 1. 10-qubit selected mixed bank

Keep the 10-qubit generator and select exactly 20 observables from a larger local-plus-correlator candidate set.

Example design:

```text
10 local X terms
5 local Z terms
5 nearest-neighbour XX terms
= 20 outputs
```

or:

```text
10 local X terms
10 selected nearest-neighbour XX/ZZ terms
= 20 outputs
```

Advantages:

- preserves 10-qubit generator scale;
- avoids the 7-qubit mixed workaround;
- tests correlators while preserving the frozen 20-dimensional encoder;
- directly compares with the existing `fixed_xz_10q` and `local_xy_10q` runs.

Required code support:

- explicit selected-term observable bank;
- deterministic term ordering;
- config validation that selected terms produce exactly `latent_dim` outputs.

### 2. Projected larger bank

Allow the quantum generator to emit a larger observable bank, then apply a fixed projection to the encoder latent dimension.

Example:

```text
30-60 observable terms -> fixed projection -> 20 latent dimensions
```

Projection options:

- random orthogonal projection with fixed seed;
- PCA-style projection learned only from frozen encoder latents;
- structured block projection preserving local/correlator groups.

Advantages:

- tests genuinely richer banks without retraining the encoder;
- separates measurement expressivity from latent shape;
- keeps comparison to current frozen checkpoint possible.

Risks:

- projection may become the real model component;
- the matched classical surrogate must use the same projection;
- reviewer concern: projection could hide the role of quantum measurements.

### 3. Latent-dim-expanded encoder

Train and freeze a new autoencoder with latent dimension matching richer banks directly.

Example targets:

```text
latent_dim = 30 for local [X,Y,Z] on 10 qubits
latent_dim = 38 for mixed [X,Z] + [XX,ZZ] on 10 qubits
```

Advantages:

- avoids forced term selection;
- gives the observable bank its natural dimensionality;
- scientifically cleaner if the goal is to test richer EVS representations.

Risks:

- introduces a new encoder checkpoint and possible confound;
- requires re-running baselines under the new latent dimension;
- more compute expensive than selected-bank or projected-bank tests.

## Recommendation

Do not run repeated seeds yet. First implement a cleaner 20-dimensional 10-qubit mixed-bank design.

Recommended next stage:

```text
Stage 9 — Selected-Term 10-Qubit Mixed Observable Bank
```

Acceptance criteria:

1. Add a selected-term observable-bank implementation.
2. Keep `n_qubits=10` and `output_dim=20`.
3. Match the Fourier/RFF surrogate using the same Stage 7/8 comparison infrastructure.
4. Compare against `fixed_xz_10q` and `local_xy_10q` from Stage 8.
5. Only then decide whether repeated seeds are worth the compute.
