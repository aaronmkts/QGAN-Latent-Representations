# E2X — Stage 2 Local Pauli Observable Banks + Corridor Diagnostics

## Repository state

- Repository: `/root/orion-workspaces/qgan-stage2`
- Branch: `feat/local-pauli-observable-banks`
- Base: current `origin/main` at branch creation
- Commit status: not committed

## Implemented scope

Implemented Stage 2 according to `docs/PPACK-stage2-local-pauli-observable-banks.md`:

1. Preserved default `FixedPauliBank [X,Z]` baseline.
2. Added config-selectable fixed observable banks:
   - `local_pauli`
   - `two_body_pauli`
   - `mixed_pauli`
3. Added deterministic term ordering:
   - local terms: Pauli label block first, then qubit index;
   - two-body line terms: Pauli term block first, then nearest-neighbour line edge;
   - mixed terms: local block first, then two-body block.
4. Added validation errors for invalid Pauli labels, malformed two-body terms, unsupported topology, and invalid qubit counts.
5. Added observable-bank metadata:
   - `name`
   - `output_dim`
   - `n_qubits`
   - `num_terms`
   - `max_locality`
   - `terms`
   - `topology`
   - `measurement_groups_estimate`
6. Added EVS Parseval/non-concentration diagnostics:
   - per-dim mean
   - per-dim second moment
   - per-dim variance
   - per-dim parseval norm squared
   - total parseval norm squared
   - mean/min/max variance
   - num samples
7. Added run artifact:
   - `observable_bank_metadata.json`
8. Added per-evaluation diagnostics artifact:
   - `evs_diagnostics.json`
9. Added aggregate EVS metrics to existing metrics CSV/JSON via scalar keys:
   - `evs/total_parseval_norm_sq`
   - `evs/mean_variance`
   - `evs/min_variance`
   - `evs/max_variance`
   - `evs/num_samples`

## Files changed

- `configs/projects/qgan_expectation_values/models/quantum_generator.yaml`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/__init__.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/circuits.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/generator.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py`
- `tests/test_configs.py`
- `tests/test_evs_diagnostics.py`
- `tests/test_observable_bank.py`
- `tests/test_smoke_modes.py`
- `docs/PPACK-stage2-local-pauli-observable-banks.md`

## Verification evidence

### Targeted tests

Command:

```bash
PYTHONPATH=src pytest -q tests/test_observable_bank.py tests/test_evs_diagnostics.py tests/test_configs.py tests/test_smoke_modes.py --tb=short
```

Result:

```text
28 passed, 14 warnings
```

### Full suite

Command:

```bash
PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check
```

Result:

```text
60 passed, 14 warnings
compileall passed
git diff --check clean
```

### Direct acceptance probe

Ran direct generator probes for:

- `fixed_pauli [X,Z]`, 4 qubits → shape `(5, 8)`
- `local_pauli [X,Y,Z]`, 4 qubits → shape `(5, 12)`
- `two_body_pauli [XX,ZZ]`, 4 qubits → shape `(5, 6)`
- `mixed_pauli [X,Z] + [XX,ZZ]`, 4 qubits → shape `(5, 14)`

Probe also confirmed metadata and malformed two-body term validation.

### Mixed-bank smoke run

Ran an actual `run_gan` smoke with:

- `observable_bank.name=mixed_pauli`
- `paulis=[X,Z]`
- `two_body_paulis=[XX,ZZ]`
- `topology=line`
- `n_qubits=4`
- `model.autoencoder.latent_dim=14`

Result:

```text
metrics_records 1 has_evs True
diag_records 1 vector_len 14 num_samples 4
sample_grid_exists True
```

This proves the non-default mixed-bank training path completes and emits diagnostics.

## Warnings

Warnings are pre-existing third-party warnings from matplotlib / pyparsing and JAX complex128 truncation in direct probes. No test failures.

## Notes

The clean checkout is not installed as an editable package, so tests were run with `PYTHONPATH=src`. This is equivalent to the source-layout import path and avoids requiring conda environment creation inside this local worktree.

## Non-goals preserved

No implementation of:

- trainable observables;
- graph-RL / PPO;
- candidate registry;
- classical Fourier/RFF surrogate;
- calibration maps;
- VAE/Sinkhorn integration;
- shot-noise simulation.
