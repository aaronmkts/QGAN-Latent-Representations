# REV-CODE-stage2-local-pauli-observable-banks

Verdict: APPROVED

## Review scope

Reviewed Stage 2 implementation against:

- `docs/PPACK-stage2-local-pauli-observable-banks.md`
- `docs/E2X-stage2-local-pauli-observable-banks.md`
- current git diff on branch `feat/local-pauli-observable-banks`

`graphify-out/graph.json` is not present in this repository, so review used direct diff/source inspection.

## Decision

Approved. I found no blocking correctness, safety, or scope issues.

The implementation preserves the fixed `[X,Z]` baseline behavior, adds deterministic fixed local/two-body/mixed Pauli banks, computes EVS diagnostics on raw generator expectation vectors at evaluation epochs, separates scalar metrics from per-dimension diagnostic vectors, and writes observable-bank metadata. The changed tests cover the principal behavior contracts and I independently reran targeted and full verification.

## Blocking findings

None.

## Non-blocking notes

1. `configs/projects/qgan_expectation_values/models/quantum_generator.yaml:7` adds `two_body_paulis` and `topology` as active config keys under the default fixed bank. This is harmless because `fixed_pauli` ignores them and tests assert they parse, but the PPACK wording said these optional keys should be documented "in a comment only." If strict config minimalism matters, move the example into comments only; otherwise this is acceptable as discoverable optional config.

2. `measurement_groups_estimate` remains a heuristic, not exact QWC grouping. This matches the PPACK risk note and is named as an estimate, so it is not a blocker.

3. The EVS diagnostic probe uses `sample_batch_size` samples at eval epochs (`4` in smoke, `64` normally). That is consistent with the implementation brief, but later experiment reports should treat these as diagnostic estimates over the eval noise sample, not as full-distribution statistics.

## Scope verification

- FixedPauliBank `[X,Z]` baseline preserved:
  - Default config remains `observable_bank.name=fixed_pauli`, `paulis=[X,Z]`.
  - `FixedPauliBank` still orders local measurements as all `X_i`, then all `Z_i`.
  - Targeted tests include the original shape/order checks.

- New observable banks:
  - `LocalPauliBank`, `TwoBodyPauliBank`, and `MixedPauliBank` are frozen dataclasses with deterministic term construction in `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py`.
  - Local ordering is Pauli block first, then qubit index.
  - Two-body ordering is Pauli-term block first, then nearest-neighbour line edge.
  - Mixed ordering is local block first, then two-body block.
  - Invalid Pauli labels, malformed two-body terms, unsupported topology, and invalid qubit counts raise `ValueError`.
  - `build_observable_bank` dispatches `fixed_pauli`, `local_pauli`, `two_body_pauli`, and `mixed_pauli` from config.

- Metadata:
  - `metadata()` includes `name`, `output_dim`, `n_qubits`, `num_terms`, `max_locality`, `terms`, `topology`, and `measurement_groups_estimate`.
  - `run_gan` writes `observable_bank_metadata.json` before training epochs.

- EVS diagnostics:
  - `compute_evs_diagnostics` computes per-dim mean, second moment, variance, parseval norm squared, and aggregate total Parseval norm squared / variance summaries in `training/diagnostics.py`.
  - `gan_loop.py` computes diagnostics from `gen_apply(gen_state.params, evs_noise)` before representation decode/sample-grid generation, so diagnostics are on raw generator expectation outputs.
  - Aggregate scalar keys are merged into the existing metrics logger as `evs/...` values.
  - Per-dim vectors are written to `evs_diagnostics.json` via `EvsDiagnosticsWriter` and are not forced into CSV.

- Out-of-scope exclusions:
  - I found no implementation of trainable observables, graph-RL, PPO, candidate registry, classical surrogate, calibration maps, VAE/Sinkhorn integration, or shot-noise simulation in the changed qgan expectation-values code.

## Test adequacy assessment

Meaningful and sufficient for this stage.

Coverage includes:

- default fixed-bank preservation;
- local/two-body/mixed bank term ordering and output dimensions;
- metadata schema;
- config dispatch and config composition;
- invalid bank input validation;
- generator output shape matching bank output dimensions;
- EVS diagnostic numeric values;
- scalar/vector artifact separation;
- smoke `run_gan` artifact emission.

One useful future addition would be a smoke run specifically committed as a test for non-default `mixed_pauli` end-to-end training, but ORION already ran that acceptance probe and the direct generator/config paths are covered. I do not consider this a blocker for Stage 2.

## Independent verification evidence

Commands run from `/root/orion-workspaces/qgan-stage2` with `PYTHONPATH=src`.

1. Custom acceptance probe:

```text
python - <<'PY'
# Probed fixed/local/two_body/mixed config dispatch, term ordering,
# generator output shape, malformed two-body validation, EVS numeric values,
# scalar metric logging, and evs_diagnostics.json vector writing.
PY
```

Result:

```text
custom_acceptance_probe passed
```

2. Targeted Stage 2 and smoke tests:

```text
pytest -q tests/test_observable_bank.py tests/test_evs_diagnostics.py tests/test_configs.py tests/test_smoke_modes.py --tb=short
```

Result:

```text
28 passed, 14 warnings in 30.19s
```

3. Full compile/test/diff check:

```text
python -m compileall -q src
pytest -q
git diff --check
```

Result:

```text
60 passed, 14 warnings in 33.75s
git diff --check exited 0
```

Warnings observed were third-party matplotlib/pyparsing deprecations and JAX complex128 truncation warnings in direct probes; they did not indicate Stage 2 regressions.
