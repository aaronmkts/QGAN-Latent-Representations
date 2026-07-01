# ENG-FINAL — Stage 2 Local Pauli Observable Banks + Corridor Diagnostics

## Verdict: READY

Stage 2 is complete against the agreed PPACK and is approved for commit and PR.

## Completion against PPACK

| PPACK acceptance criterion | Evidence |
|---|---|
| All existing tests pass unchanged; `FixedPauliBank` default `output_dim=20` and X-then-Z ordering intact | Full test suite: 60 passed. `test_default_fixed_pauli_bank_preserves_x_then_z_readout` and `test_fixed_pauli_bank_output_dim_and_validation` pass. `FixedPauliBank` default remains `n_qubits=10`, `paulis=(X,Z)`. |
| `local_pauli`, `two_body_pauli`, `mixed_pauli` selectable via config with deterministic, documented term ordering | Implemented in `observables.py`. Ordering is Pauli-block then qubit/edge; mixed is local block then two-body block. Tests `test_local_pauli_bank_terms_and_metadata`, `test_two_body_pauli_bank_line_topology_ordering`, `test_mixed_pauli_bank_local_then_two_body_order` lock the order. |
| Generator output column order equals `bank.terms` order | `TwoBodyPauliBank` and `MixedPauliBank` use `_term_measurements(self.terms)`; `test_generator_output_dim_matches_bank` asserts column order matches `bank.terms`. |
| `bank.metadata()` returns all eight required fields for every bank | `_metadata` helper plus tests in `test_bank_metadata_schema`. |
| Invalid labels, malformed two-body terms, and non-`line` topology raise clear `ValueError`s | `_normalise_paulis`, `_normalise_two_body_paulis`, and `TwoBodyPauliBank.__post_init__` raise explicit messages. Covered by `test_invalid_pauli_label_raises`, `test_malformed_two_body_term_raises`, `test_unsupported_topology_raises`. |
| Eval epochs emit aggregate EVS scalars (`evs/total_parseval_norm_sq`, `evs/mean_variance`, `evs/min_variance`, `evs/max_variance`, `evs/num_samples`) into `metrics.csv`/`metrics.json` | `diagnostics.py:compute_evs_diagnostics` plus `gan_loop.py` logging. `test_gan_smoke_emits_evs_diagnostics` and `test_evs_diagnostics_writer_scalars_in_csv_vectors_in_json` verify. |
| Smoke `run_gan` with default `fixed_pauli` completes and produces diagnostics | Full suite smoke tests pass; smoke run emitted `evs/total_parseval_norm_sq` and an `evs_diagnostics.json` record. |

## Required changes before commit/PR

None blocking.

### Non-blocking carry-forward notes

1. The YAML currently exposes `two_body_paulis` and `topology` as active keys under the default `fixed_pauli` bank. This is harmless (`fixed_pauli` ignores them) and aids discoverability, but it slightly departs from the PPACK wording that these keys be documented "in a comment only." If config minimalism is preferred, move them into a commented example before PR. This does not block engineering integration.
2. `measurement_groups_estimate` is documented as a heuristic. No further action.
3. EVS diagnostics are estimates over the eval noise batch; future experiment reports should not treat them as full-distribution statistics.

## Repository state summary

- **Repository:** `/root/orion-workspaces/qgan-stage2`
- **Branch:** `feat/local-pauli-observable-banks`
- **Status:** working tree contains all Stage 2 changes; nothing staged.
- **Verification:** 60 passed, 14 warnings (third-party pre-existing warnings only), `git diff --check` clean.

### Files changed (9)

- `configs/projects/qgan_expectation_values/models/quantum_generator.yaml`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/__init__.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/circuits.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/generator.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/quantum_generator/observables.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py`
- `tests/test_configs.py`
- `tests/test_observable_bank.py`
- `tests/test_smoke_modes.py`

### Files added (4)

- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py`
- `tests/test_evs_diagnostics.py`
- `docs/E2X-stage2-local-pauli-observable-banks.md`
- `docs/REV-CODE-stage2-local-pauli-observable-banks.md`

### Diff stat

571 insertions, 35 deletions across the 9 changed source/test/config files.

## Recommended next phase

Stage 2 engineering integration is finished. Recommended next steps in order:

1. **Commit and open PR** on `feat/local-pauli-observable-banks`.
2. **After merge**, begin Stage 3 scoping. Likely candidates from the PPACK non-goals list: trainable observables, candidate registry / bank selection, or calibration maps. The choice should be driven by the downstream experiment plan, not by implementation availability.
3. **Short-term experiment hygiene**: ensure any runs using `two_body_pauli` or `mixed_pauli` record `observable_bank_metadata.json` and `evs_diagnostics.json` in run artifacts for reproducibility.

## PR title/body guidance

- **Title:** `feat: local, two-body, and mixed Pauli observable banks + EVS corridor diagnostics`
- **Body outline:**
  - What: extend fixed observable-bank layer with config-selectable `local_pauli`, `two_body_pauli`, and `mixed_pauli` banks; add `bank.metadata()`; add eval-epoch EVS Parseval/non-concentration diagnostics.
  - Why: supports richer generator readouts and gives experimenters visibility into raw expectation-vector concentration.
  - Backwards compatibility: default `fixed_pauli [X,Z]` preserved byte-for-byte; existing tests pass unchanged.
  - Scope boundaries: no trainable observables, graph-RL/PPO, candidate registry, classical surrogate, calibration maps, VAE/Sinkhorn, or shot noise.
  - Verification: 60 tests pass; targeted smoke run for `mixed_pauli` completed; `git diff --check` clean.
  - Artifacts: closes `PPACK-stage2-local-pauli-observable-banks.md` and `REV-CODE-stage2-local-pauli-observable-banks.md`.

## Engineering Director sign-off

Stage 2 implementation satisfies the PPACK, passes independent code review, passes the full test suite, and is ready for commit/PR with no blocking changes.

Verdict: **READY**.
