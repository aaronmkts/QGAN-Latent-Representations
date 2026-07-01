# Stage 2 PPACK — Local Pauli Observable Banks + Corridor Diagnostics

Extend the fixed observable-bank layer with config-selectable `local_pauli`, `two_body_pauli`, and `mixed_pauli` banks, attach rich bank metadata, and add EVS Parseval / non-concentration diagnostics on raw generator expectation outputs at evaluation epochs. The existing `FixedPauliBank` `[X,Z]` baseline (defaults, ordering, validation) is preserved byte-for-byte. Strictly TDD: each behavior gets a RED test before implementation. Out of scope: trainable observables, graph-RL, PPO, candidate registry, classical surrogate, calibration maps, VAE/Sinkhorn, shot noise.

## Affected files

- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py` — add `local_pauli`/`two_body_pauli`/`mixed_pauli` banks, shared metadata surface, extend `build_observable_bank` dispatch. `FixedPauliBank` unchanged except additive metadata properties.
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/circuits.py` — make `CircuitConfig.__post_init__` bank-type-agnostic (validate `n_qubits`, stop reconstructing as `FixedPauliBank` via `.paulis`).
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/generator.py` — widen `observable_bank` type hint to the bank protocol/union; behavior unchanged.
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/__init__.py` — export new bank classes + `ObservableBank` protocol.
- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py` — NEW: `compute_evs_diagnostics(expectations)` + `EvsDiagnosticsWriter`.
- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py` — at eval epochs, collect raw `gen_apply` expectation vectors, compute diagnostics, log aggregate scalars via existing `EpochMetricLogger` (prefixed `evs/`), write per-dim vectors via `EvsDiagnosticsWriter`.
- `configs/projects/qgan_expectation_values/models/quantum_generator.yaml` — keep default exactly `fixed_pauli` / `[X,Z]`; document optional `two_body_paulis` / `topology` keys in a comment only.
- Tests: `tests/test_observable_bank.py` (extend), `tests/test_evs_diagnostics.py` (NEW), `tests/test_configs.py` (assert default unchanged + optional keys parse).

## Test plan

1. `test_fixed_pauli_bank_output_dim_and_validation` — unchanged, must still pass.
2. `test_default_fixed_pauli_bank_preserves_x_then_z_readout` — unchanged, must still pass.
3. `test_local_pauli_bank_terms_and_metadata` — `paulis=(X,Y,Z)`, n=3 → `output_dim==9`, `num_terms==9`, `max_locality==1`, ordering X_0..X_2,Y_0..,Z_0..; `terms`, `topology`, `measurement_groups_estimate` present.
4. `test_two_body_pauli_bank_line_topology_ordering` — `two_body_paulis=(XX,ZZ)`, n=4, line → pairs (0,1),(1,2),(2,3); `output_dim==6`, `max_locality==2`; measurement wires match term ordering.
5. `test_mixed_pauli_bank_local_then_two_body_order` — `[X,Z]`+`[XX,ZZ]`, n=4 → local terms first, then two-body; `output_dim==8+6`.
6. `test_bank_metadata_schema` — every bank exposes `name,output_dim,n_qubits,num_terms,max_locality,terms,topology,measurement_groups_estimate` via `metadata()`.
7. `test_invalid_pauli_label_raises` / `test_malformed_two_body_term_raises` (len≠2) / `test_unsupported_topology_raises`.
8. `test_build_observable_bank_dispatch` — each `name` returns the correct class; unknown name raises.
9. `test_generator_output_dim_matches_bank` — `build_generator_apply` output width equals `bank.output_dim` and column order equals `bank.terms` order for a two-body bank.
10. `test_compute_evs_diagnostics_values` — synthetic array → verify per-dim mean/second-moment/variance, `parseval_norm_sq`, `total_parseval_norm_sq`, `mean/min/max_variance`, `num_samples`.
11. `test_evs_diagnostics_writer_scalars_in_csv_vectors_in_json` — aggregate scalars floatify into `metrics.csv`; per-dim vectors land in `evs_diagnostics.json`.
12. `test_gan_smoke_emits_evs_diagnostics` — smoke `run_gan` writes `evs/total_parseval_norm_sq` to metrics and an `evs_diagnostics.json` record.
13. `test_quantum_generator_config_declares_default_observable_bank` — unchanged; add optional-key parse assertion.

## Implementation sequence

1. RED tests 3–8 for banks + metadata + validation (`tests/test_observable_bank.py`).
2. GREEN: add `ObservableBank` protocol and `_term_measurements(terms)` helper in `observables.py`; add metadata properties to `FixedPauliBank` (no behavior change).
3. GREEN: implement `LocalPauliBank`, `TwoBodyPauliBank` (line topology), `MixedPauliBank` (frozen dataclasses) with deterministic `terms`, `measurements()`, metadata, and validation.
4. GREEN: extend `build_observable_bank` dispatch on `name`, reading `paulis`, `two_body_paulis`, `topology`.
5. RED test 9 → make `CircuitConfig.__post_init__` bank-agnostic; widen generator type hints; export new symbols.
6. RED tests 10–11 → implement `training/diagnostics.py` (`compute_evs_diagnostics`, `EvsDiagnosticsWriter`).
7. RED test 12 → wire diagnostics into `gan_loop.py` eval block: gather raw `gen_apply(gen_params, noise)` vectors (pre-decode), compute, log scalars via `EpochMetricLogger`, write vectors via `EvsDiagnosticsWriter`.
8. Update config comment + `tests/test_configs.py`; run targeted then full suite.

## Acceptance criteria

- All existing tests pass unchanged; `FixedPauliBank` default output_dim=20 and X-then-Z ordering intact.
- `local_pauli`, `two_body_pauli`, `mixed_pauli` selectable via config with deterministic, documented term ordering; generator output column order equals `bank.terms`.
- `bank.metadata()` returns all eight required fields for every bank.
- Invalid Pauli labels, malformed two-body terms, and non-`line` topology raise clear `ValueError`s.
- Eval epochs emit aggregate EVS scalars (`evs/total_parseval_norm_sq`, `evs/mean_variance`, `evs/min_variance`, `evs/max_variance`, `evs/num_samples`) into `metrics.csv`/`metrics.json` and full per-dim vectors into `evs_diagnostics.json`.
- Smoke `run_gan` (fixed_pauli default) runs end-to-end and produces a diagnostics record.

## Smoke-run commands

- `pytest tests/test_observable_bank.py -v`
- `pytest tests/test_evs_diagnostics.py -v`
- `pytest tests/test_configs.py tests/test_smoke_modes.py tests/test_models_and_checkpointing.py -v`
- `pytest -q`
- `python -m qgan_latent.workflows.qgan_expectation_values.training.gan_loop` (or existing smoke entrypoint) with `smoke_test=true device=cpu wandb_mode=disabled`

## Risks / non-goals

- Risk: ordering ambiguity between local and two-body terms in `mixed_pauli` — mitigated by fixed rule (local block first in `paulis` order, then two-body in `two_body_paulis` order over line pairs) locked by tests 3–5,9.
- Risk: `CircuitConfig.__post_init__` currently reconstructs `FixedPauliBank` via `.paulis`, which breaks two-body banks — fixed in step 5 with a bank-agnostic n_qubits check.
- Risk: `EpochMetricLogger._floatify` silently drops vector metrics — mitigated by routing per-dim vectors to a dedicated `evs_diagnostics.json` writer, scalars only to CSV.
- Risk: `measurement_groups_estimate` is a heuristic (distinct Pauli-basis count), not an exact QWC grouping — documented as an estimate.
- Risk: capturing raw expectations pre-decode adds an extra generator call per eval epoch — bounded by eval batch size, no training-loop cost.
- Non-goals: trainable observables, graph-RL/PPO, candidate registry, classical surrogate, calibration maps, VAE/Sinkhorn integration, shot-noise simulation, topologies other than `line`.

---

## Model / planning metadata

- Planning model: claude-opus-4-8
- Input tokens: 0
- Output tokens: 0

### Structured plan summary

```json
{
  "affected_files": [
    "src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py",
    "src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/circuits.py",
    "src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/generator.py",
    "src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/__init__.py",
    "src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py",
    "src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py",
    "configs/projects/qgan_expectation_values/models/quantum_generator.yaml",
    "tests/test_observable_bank.py",
    "tests/test_evs_diagnostics.py",
    "tests/test_configs.py"
  ],
  "test_plan": [
    "test_fixed_pauli_bank_output_dim_and_validation (unchanged, must pass)",
    "test_default_fixed_pauli_bank_preserves_x_then_z_readout (unchanged, must pass)",
    "test_local_pauli_bank_terms_and_metadata",
    "test_two_body_pauli_bank_line_topology_ordering",
    "test_mixed_pauli_bank_local_then_two_body_order",
    "test_bank_metadata_schema",
    "test_invalid_pauli_label_raises",
    "test_malformed_two_body_term_raises",
    "test_unsupported_topology_raises",
    "test_build_observable_bank_dispatch",
    "test_generator_output_dim_matches_bank",
    "test_compute_evs_diagnostics_values",
    "test_evs_diagnostics_writer_scalars_in_csv_vectors_in_json",
    "test_gan_smoke_emits_evs_diagnostics",
    "test_quantum_generator_config_declares_default_observable_bank (extended: optional keys parse)"
  ],
  "implementation_sequence": [
    "RED: add bank/metadata/validation tests (tests 3-8) to tests/test_observable_bank.py",
    "GREEN: add ObservableBank protocol + _term_measurements helper; add additive metadata properties to FixedPauliBank (no behavior change)",
    "GREEN: implement LocalPauliBank, TwoBodyPauliBank (line topology), MixedPauliBank with deterministic terms/measurements/metadata/validation",
    "GREEN: extend build_observable_bank dispatch on name reading paulis/two_body_paulis/topology",
    "RED test_generator_output_dim_matches_bank -> make CircuitConfig.__post_init__ bank-agnostic; widen generator type hints; export new symbols in __init__.py",
    "RED tests 10-11 -> implement training/diagnostics.py (compute_evs_diagnostics, EvsDiagnosticsWriter)",
    "RED test_gan_smoke_emits_evs_diagnostics -> wire raw pre-decode expectations + diagnostics into gan_loop.py eval block",
    "Update quantum_generator.yaml comment + tests/test_configs.py; run targeted then full pytest"
  ],
  "acceptance_criteria": [
    "All existing tests pass unchanged; FixedPauliBank default output_dim=20 and X-then-Z ordering intact",
    "local_pauli / two_body_pauli / mixed_pauli are config-selectable with deterministic documented term ordering",
    "Generator output column order equals bank.terms order",
    "bank.metadata() returns name, output_dim, n_qubits, num_terms, max_locality, terms, topology, measurement_groups_estimate for every bank",
    "Invalid Pauli labels, malformed two-body terms, and unsupported topology raise clear ValueErrors",
    "Eval epochs write aggregate EVS scalars (evs/total_parseval_norm_sq, evs/mean_variance, evs/min_variance, evs/max_variance, evs/num_samples) to metrics.csv/metrics.json",
    "Per-dim EVS vectors (mean, second_moment, variance, parseval_norm_sq) written to evs_diagnostics.json",
    "Smoke run_gan with default fixed_pauli completes end-to-end and emits a diagnostics record"
  ],
  "smoke_run_commands": [
    "pytest tests/test_observable_bank.py -v",
    "pytest tests/test_evs_diagnostics.py -v",
    "pytest tests/test_configs.py tests/test_smoke_modes.py tests/test_models_and_checkpointing.py -v",
    "pytest -q",
    "python -m qgan_latent.workflows.qgan_expectation_values.training.gan_loop smoke_test=true device=cpu wandb_mode=disabled"
  ],
  "risks_non_goals": [
    "Risk: ordering ambiguity between local and two-body terms in mixed_pauli -> fixed rule (local block first in paulis order, then two-body over line pairs) locked by tests",
    "Risk: CircuitConfig.__post_init__ reconstructs FixedPauliBank via .paulis and would break non-fixed banks -> replaced with bank-agnostic n_qubits validation",
    "Risk: EpochMetricLogger._floatify silently drops vector metrics -> per-dim vectors routed to dedicated evs_diagnostics.json writer, scalars only to CSV/JSON",
    "Risk: measurement_groups_estimate is a heuristic (distinct Pauli-basis count), not exact QWC grouping -> documented as an estimate",
    "Risk: capturing raw pre-decode expectations adds one extra generator call per eval epoch -> bounded by eval batch size, no training-loop cost",
    "Non-goal: trainable observables",
    "Non-goal: graph-RL / PPO",
    "Non-goal: candidate registry",
    "Non-goal: classical surrogate",
    "Non-goal: calibration maps",
    "Non-goal: VAE/Sinkhorn integration",
    "Non-goal: shot-noise simulation",
    "Non-goal: topologies other than line"
  ]
}
```
