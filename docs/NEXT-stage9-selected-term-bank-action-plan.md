# NEXT ACTION PLAN — Stage 9 Selected-Term 10-Qubit Mixed Observable Bank

## Current status

Stage 8 is complete. The tracked result document is:

- `docs/RESULTS-stage8-observable-bank-sweep.md`

The next-stage design document is:

- `docs/NEXT-stage9-observable-bank-design.md`

Inspection of the current code/config/test tree shows Stage 9 has **not** yet been implemented. The code currently supports these observable bank types:

- `fixed_pauli`
- `local_pauli`
- `two_body_pauli`
- `mixed_pauli`

There is no selected-term observable bank implementation yet.

## Stage 8 conclusion to carry forward

Stage 8 tested richer observable-bank substitutions against a matched Fourier/RFF surrogate. The best result was `local_xy_10q`, but the observed separation was very small:

```text
Delta_sep = 0.0172 ~= log(1.017x)
```

This is not strong evidence of dequantization-resistant quantum signal. The mixed 7-qubit condition was actively poor because it reduced qubit count to satisfy the frozen encoder's 20-dimensional latent shape.

The key constraint is:

```text
autoencoder.latent_dim == observable_bank.output_dim
```

The frozen encoder has:

```text
latent_dim = 20
```

Therefore Stage 9 should test a better 20-output bank without changing the 10-qubit generator scale.

## Recommended objective

Implement and test:

```text
Stage 9 — Selected-Term 10-Qubit Mixed Observable Bank
```

Goal: keep `n_qubits=10` and `output_dim=20` while selecting a deterministic mix of local and two-body observables from a richer candidate set.

## Proposed selected bank

Primary candidate:

```text
10 local X terms
5 local Z terms
5 nearest-neighbour XX terms
= 20 outputs
```

Concrete deterministic term order:

```text
X_0, X_1, ..., X_9,
Z_0, Z_1, Z_2, Z_3, Z_4,
XX_0_1, XX_1_2, XX_2_3, XX_3_4, XX_4_5
```

Alternative candidate after primary implementation is verified:

```text
10 local X terms
10 selected nearest-neighbour XX/ZZ terms
= 20 outputs
```

Do not run repeated seeds yet. First establish whether this cleaner 10-qubit mixed bank gives any larger single-seed signal than the Stage 8 conditions.

## Implementation plan

### 1. Add selected-term observable bank

Target file:

- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py`

Add a new bank class, e.g.:

```python
SelectedTermPauliBank
```

Expected behaviour:

- accepts explicit term strings such as `X_0`, `Z_4`, `XX_0_1`;
- validates Pauli labels;
- validates qubit indices are in range;
- validates one-body vs two-body term shape;
- preserves deterministic term order exactly as configured;
- exposes existing `ObservableBank` protocol fields:
  - `output_dim`
  - `num_terms`
  - `max_locality`
  - `terms`
  - `topology`
  - `measurement_groups_estimate`
  - `metadata()`
  - `measurements()`

Also update:

```python
build_observable_bank(...)
```

to dispatch:

```text
name: selected_pauli
```

### 2. Add config support

Target config:

- `configs/projects/qgan_expectation_values/models/quantum_generator.yaml`

Add support for a `terms:` list under `observable_bank`, while preserving existing defaults.

Example Stage 9 override:

```yaml
model:
  quantum_generator:
    n_qubits: 10
    noise_dim: 10
    observable_bank:
      name: selected_pauli
      terms:
        - X_0
        - X_1
        - X_2
        - X_3
        - X_4
        - X_5
        - X_6
        - X_7
        - X_8
        - X_9
        - Z_0
        - Z_1
        - Z_2
        - Z_3
        - Z_4
        - XX_0_1
        - XX_1_2
        - XX_2_3
        - XX_3_4
        - XX_4_5
```

### 3. Add tests before implementation

Target tests:

- `tests/test_observable_bank.py`
- `tests/test_configs.py`
- optionally `tests/test_stage8_sweep.py` or a new Stage 9-specific test file

Required tests:

1. Selected bank preserves configured term order.
2. Selected bank output dimension equals number of terms.
3. Selected bank metadata records the exact selected terms.
4. Invalid terms fail clearly:
   - invalid Pauli label;
   - missing wire index;
   - out-of-range qubit index;
   - malformed two-body term.
5. `build_observable_bank({"name": "selected_pauli", "terms": [...]}, n_qubits=10)` dispatches correctly.
6. GAN validation accepts selected 20-term bank when autoencoder latent dimension is 20.
7. GAN validation rejects mismatched selected bank dimensions.

### 4. Add experiment condition

Use the existing Stage 8 comparison infrastructure rather than inventing a new runner.

Candidate locations:

- extend `src/qgan_latent/workflows/qgan_expectation_values/training/stage8.py` with a selected-bank condition, or
- introduce a small Stage 9 condition module that reuses the same runner/aggregation primitives.

The cleaner option is probably to generalise naming away from Stage 8 in a small follow-up refactor, because `scripts/observable_bank_sweep.py` is now generic but the package module remains named `stage8.py`.

### 5. Run minimal verification

After implementation:

```bash
PYTHONPATH=src python3 -m pytest tests/test_observable_bank.py tests/test_configs.py tests/test_stage8_sweep.py -q
PYTHONPATH=src python3 -m pytest -q
```

### 6. Run one Stage 9 sweep condition

Use the frozen encoder:

```text
outputs/experiments/baseline_mnist_last_20260701_124132/checkpoints/autoencoder.ckpt
```

Run one selected-bank condition against the matched surrogate. Compare against Stage 8:

- `local_xy_10q`
- `two_body_xx_zz_11q`
- `fixed_xz_10q`
- `mixed_xz_xx_7q`

Decision threshold:

- If `Delta_sep` is still near zero, do not spend compute on repeated seeds yet.
- If `Delta_sep` improves materially over `0.0172`, then plan repeated seeds and confidence intervals.

## Acceptance criteria

- Selected-term observable bank implemented and tested.
- `n_qubits=10`, `output_dim=20` selected bank config validated.
- Existing Stage 8 tests remain green.
- Full test suite remains green.
- A single Stage 9 selected-bank QGAN/surrogate comparison is produced.
- Result is documented with the same beta / Delta_sep interpretation used for Stage 8.

## Recommended next engineering route

Use the standard full workflow:

```text
Engineering Director plan
→ Implementation Engineer with TDD
→ Code Reviewer verifies tests and dimension/metadata handling
→ draft PR if code changes are worth preserving
```

Do not start repeated GPU sweeps until the selected-term bank implementation and one-condition comparison are verified.
