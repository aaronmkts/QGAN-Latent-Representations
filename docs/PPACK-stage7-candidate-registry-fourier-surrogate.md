# Stage 7 PPACK — Candidate Registry + Fourier/RFF Classical Surrogate

> Artifact: PPACK-stage7-candidate-registry-fourier-surrogate  
> Branch: `feat/candidate-registry-fourier-surrogate` (off current main, with Stage 6 merged)  
> Base: current main at merge of `feat/local-pauli-observable-banks` (commit `eb1fa3d`)

Stage 7 adds a structured run/candidate registry to every QGAN experiment, records all run artifacts and provenance, implements a matched classical Fourier/RFF surrogate baseline, compares the QGAN EVS signature against the surrogate under matched spectrum/feature/parameter/training budgets, and computes the dequantization-aware separation metric `Delta_sep`. All work is TDD; existing Stage 6 behavior and tests are preserved.

## Objectives

1. **Run/candidate registry** — for every QGAN training run, write a single JSON record capturing experiment identity, observable-bank metadata, resource counts, config snapshot, metrics/diagnostics/sample-grid paths, and git commit hash.
2. **Classical Fourier/RFF surrogate** — build a deterministic non-quantum generator that maps the same noise dimension to the same latent dimension using a fixed random Fourier feature layer followed by a small dense network, and train it with the same GAN loop against the same discriminator/representation budget.
3. **Matched comparison** — under identical spectrum, feature, parameter, training-step, and batch budgets, collect the surrogate EVS diagnostics and compare to the QGAN EVS diagnostics.
4. **Separation metric** — compute `Delta_sep = log(||beta_Q|| / (||beta_cls|| + eps))` where `beta` is the aggregate beta-style energy captured by the Fourier/Parseval norm of the learned generator.

## Non-goals

Out of scope and forbidden in this stage:

- Graph-RL / PPO
- Trainable observables
- Full QAS search
- Hardware shot-noise simulation
- Real dataset dependency in smoke mode
- W&B or checkpoint behavior changes
- Any change to the Stage 6 observable bank, EVS diagnostics math, or generator circuit

## Affected files

### Add

- `src/qgan_latent/workflows/qgan_expectation_values/training/registry.py` — `RunRegistry`, `RunRecord`, `build_run_record`, git/commit helpers.
- `src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/fourier_generator.py` — `FourierRFFSurrogate`, `init_fourier_params`, `build_fourier_apply`, `fourier_spectrum_counts`.
- `src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/__init__.py` — exports for surrogate module.
- `src/qgan_latent/workflows/qgan_expectation_values/training/surrogate_loop.py` — `run_surrogate_gan`, a thin wrapper that reuses `gan_loop.py` logic or calls the same training primitives with the Fourier surrogate substituted for the quantum generator.
- `tests/test_run_registry.py` — registry schema, git hash capture, round-trip I/O.
- `tests/test_fourier_surrogate.py` — surrogate shape, deterministic features, parameter counts, matching policy, smoke training.
- `tests/test_delta_sep.py` — numeric separation metric under synthetic and smoke data.

### Modify

- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py`:
  - Write `run_record.json` at start of training and update/finalize it at end.
  - Optionally emit a `comparison.json` when both QGAN and surrogate have run (the surrogate may be run separately and the registry used to pair runs).
- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py`:
  - Expose a single helper `parseval_norm_from_diagnostics(diagnostics) -> float` returning `total_parseval_norm_sq` for downstream separation math.
- `pyproject.toml`:
  - Add optional entry point `qgan-latent-train-surrogate` if a clean CLI is needed; otherwise use the existing `qgan-latent-train` with an override.
- `tests/test_smoke_modes.py`:
  - Extend the existing smoke test to assert `run_record.json` exists and contains required keys.

### Config changes

- Add optional `surrogate` section to `configs/projects/qgan_expectation_values/train.yaml`:
  - `enabled: false` by default
  - `rff_dim: auto` (defaults to `model.quantum_generator.depth * n_qubits * 2` to match quantum angle count)
  - `sigma: 1.0`
  - `hidden_dim: 64`

No changes to the default `fixed_pauli` bank or EVS diagnostics.

## API designs

### Registry API

```python
# src/qgan_latent/workflows/qgan_expectation_values/training/registry.py
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class RunRecord:
    run_id: str                # e.g. "qgan_autoencoder_<timestamp>" or user-supplied
    git_commit: str            # full sha, or "unknown" if not a git repo
    git_dirty: bool            # whether the working tree had uncommitted changes
    workflow: str              # "qgan_expectation_values" or "fourier_surrogate"
    observable_bank: dict[str, Any]
    resource_counts: dict[str, int]
    config_snapshot: dict[str, Any]   # OmegaConf resolved container
    metrics_path: str
    diagnostics_path: str
    observable_bank_metadata_path: str
    sample_grid_paths: list[str]
    checkpoint_paths: dict[str, str]
    status: str                # "started" / "completed" / "failed"
    parent_run_id: str | None  # for surrogate pointing to matched QGAN run

class RunRegistry:
    def __init__(self, output_dir: Path, run_id: str | None = None): ...
    def start(self, cfg, observable_bank_metadata: dict[str, Any]) -> RunRecord: ...
    def finalize(
        self,
        metrics_path: Path,
        diagnostics_path: Path,
        sample_grid_paths: list[Path],
        checkpoint_paths: dict[str, Path],
        parent_run_id: str | None = None,
    ) -> RunRecord: ...
    def write(self, record: RunRecord) -> Path: ...
    def load(self) -> RunRecord | None: ...
```

### Surrogate API

```python
# src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/fourier_generator.py
from typing import Callable
import jax.numpy as jnp

class FourierRFFSurrogate:
    """Random Fourier Feature + MLP classical generator.

    Matches:
      - output_dim == observable_bank.output_dim
      - noise_dim == model.quantum_generator.noise_dim
      - trainable_parameter_count approximately equals quantum style-w + style-b count
    """
    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        rff_dim: int,
        hidden_dim: int,
        sigma: float,
        seed: int,
    ): ...

def init_fourier_params(rng, *, noise_dim: int, output_dim: int, rff_dim: int, hidden_dim: int) -> dict: ...

def build_fourier_apply(
    *,
    noise_dim: int,
    output_dim: int,
    rff_dim: int,
    hidden_dim: int,
    sigma: float,
    seed: int,
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]: ...

def fourier_parameter_count(noise_dim: int, output_dim: int, rff_dim: int, hidden_dim: int) -> dict[str, int]: ...

def matched_rff_dim(n_qubits: int, depth: int) -> int:
    """Default RFF count equals the quantum circuit angle count: depth * n_qubits * 2."""
    return depth * n_qubits * 2
```

### Surrogate training entry point

```python
# src/qgan_latent/workflows/qgan_expectation_values/training/surrogate_loop.py
def run_surrogate_gan(cfg, *, matched_run_id: str | None = None) -> tuple[dict, Any]:
    """Run the GAN training loop substituting the Fourier/RFF surrogate generator.

    Reuses the same discriminator, representation, metrics manager, and smoke path
    as `run_gan`. Writes the same artifacts plus a run_record whose
    workflow == "fourier_surrogate" and parent_run_id points to the matched QGAN run.
    """
```

### Diagnostics helper

```python
# src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py
def parseval_norm_from_diagnostics(diagnostics: Mapping[str, Any]) -> float:
    return float(diagnostics["total_parseval_norm_sq"])
```

## JSON schemas and example records

### run_record.json

```json
{
  "run_id": "qgan_autoencoder_2026-07-01T12-34-56",
  "git_commit": "eb1fa3d4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0",
  "git_dirty": false,
  "workflow": "qgan_expectation_values",
  "observable_bank": {
    "name": "fixed_pauli",
    "output_dim": 4,
    "n_qubits": 2,
    "num_terms": 4,
    "max_locality": 1,
    "terms": ["X_0", "X_1", "Z_0", "Z_1"],
    "topology": "local",
    "measurement_groups_estimate": 2
  },
  "resource_counts": {
    "n_qubits": 2,
    "circuit_depth": 1,
    "noise_dim": 2,
    "observable_bank_output_dim": 4,
    "trainable_quantum_parameters": 4,
    "discriminator_trainable_parameters": 151,
    "train_images": 8,
    "epochs": 1,
    "batch_size": 8,
    "steps_per_epoch": 1
  },
  "config_snapshot": {
    "seed": 42,
    "batch_size": 128,
    "epochs": 100,
    "model": {
      "quantum_generator": {
        "n_qubits": 10,
        "depth": 4,
        "noise_dim": 10,
        "observable_bank": {
          "name": "fixed_pauli",
          "paulis": ["X", "Z"]
        }
      }
    }
  },
  "metrics_path": "outputs/qgan_expectation_values/train/metrics.json",
  "diagnostics_path": "outputs/qgan_expectation_values/train/evs_diagnostics.json",
  "observable_bank_metadata_path": "outputs/qgan_expectation_values/train/observable_bank_metadata.json",
  "sample_grid_paths": [
    "outputs/qgan_expectation_values/train/samples_step_000000.png",
    "outputs/qgan_expectation_values/train/samples_epoch_0001.png"
  ],
  "checkpoint_paths": {
    "generator": "checkpoints/qgan_gen.ckpt",
    "discriminator": "checkpoints/qgan_disc.ckpt"
  },
  "status": "completed",
  "parent_run_id": null
}
```

### comparison.json (produced after paired QGAN + surrogate run)

```json
{
  "qgan_run_id": "qgan_autoencoder_2026-07-01T12-34-56",
  "surrogate_run_id": "fourier_surrogate_2026-07-01T12-35-10",
  "matched_policy": {
    "spectrum": "rff_dim == depth * n_qubits * 2",
    "feature_dim": "output_dim == observable_bank.output_dim",
    "noise_dim": "same",
    "discriminator": "same architecture and initialization seed",
    "representation": "same frozen autoencoder checkpoint",
    "epochs": "same",
    "batch_size": "same",
    "n_critic": "same",
    "learning_rates": "same"
  },
  "beta_q": 12.345,
  "beta_cls": 8.901,
  "eps": 1e-12,
  "delta_sep": 0.318
}
```

## Mathematical definition of beta and Delta_sep

For this implementation stage, the Fourier/Parseval energy of a learned generator is defined directly from the existing EVS diagnostics on raw generator outputs.

Given raw expectation vectors `v_i in R^D` for `i = 1..N` produced by a generator (QGAN or surrogate), the per-dimension second moment is

```
S_j = (1/N) sum_i v_{i,j}^2
```

and the aggregate Parseval-style norm squared is

```
||beta||^2 = sum_{j=1..D} S_j
```

Equivalently,

```
||beta|| = sqrt( sum_j (1/N) sum_i v_{i,j}^2 )
         = sqrt( diagnostics["total_parseval_norm_sq"] )
```

For the QGAN generator:

```
beta_Q  := sqrt( parseval_norm_qgan )
```

For the classical Fourier/RFF surrogate generator:

```
beta_cls := sqrt( parseval_norm_surrogate )
```

The dequantization-aware separation metric is

```
Delta_sep := log( beta_Q / (beta_cls + eps) )
```

with `eps = 1e-12` and the natural logarithm. When `beta_Q > beta_cls`, `Delta_sep > 0`; when `beta_cls > beta_Q`, `Delta_sep < 0`. In the degenerate case `beta_cls == 0`, `Delta_sep := log(beta_Q / eps)`. For small nonzero classical beta values, the denominator remains `beta_cls + eps`, not `max(beta_cls, eps)`.

Implementation helper:

```python
def compute_delta_sep(beta_q: float, beta_cls: float, eps: float = 1e-12):
    denom = beta_cls + eps
    return float(np.log(beta_q / denom))
```

This stage intentionally does **not** define beta as the l2 norm of trainable parameters; it is defined from the output-space Parseval energy so that QGAN and surrogate are compared on the same feature-domain footing.

## Matching policy

Every surrogate run must be matched to a QGAN run by the following rules. The registry record stores these as a `matched_policy` block, and tests assert the match.

| Budget | QGAN | Surrogate | Match rule |
|---|---|---|---|
| Spectrum | circuit angles `depth * n_qubits * 2` | RFF dimension `rff_dim` | `rff_dim = depth * n_qubits * 2` |
| Feature/output dim | `observable_bank.output_dim` | surrogate output dim | identical |
| Noise dim | `model.quantum_generator.noise_dim` | surrogate input dim | identical |
| Discriminator | same Flax module and same seed | same Flax module and same seed | identical architecture + seed |
| Representation | frozen pretrained autoencoder | frozen pretrained autoencoder | same checkpoint path |
| Training epochs | `cfg.epochs` | `cfg.epochs` | identical |
| Batch size | `cfg.batch_size` | `cfg.batch_size` | identical |
| Critic updates | `cfg.n_critic` | `cfg.n_critic` | identical |
| Learning rates | `cfg.gen_lr`, `cfg.disc_lr` | surrogate uses `cfg.gen_lr` | identical |
| Loss function | WGAN-GP | WGAN-GP | identical |
| Sample noise | `jax.random.normal` | `jax.random.normal` | identical distribution and seed chain |
| Smoke data | synthetic MNIST | synthetic MNIST | identical |

The surrogate is **not** required to match the QGAN trainable parameter count exactly; the default `rff_dim` gives a comparable count, and the registry records both counts for transparency.

## Test plan (failure-first)

Write these tests before implementation. Each test must fail with a clear missing-symbol or assertion error before the corresponding code is added.

### Registry tests (`tests/test_run_registry.py`)

1. `test_run_record_schema_roundtrip` — construct a `RunRecord`, write via `RunRegistry`, load back, assert all top-level keys equal.
2. `test_registry_captures_git_commit_and_dirty_state` — start a record in the repo; assert `git_commit` is a 40-char hex string and `git_dirty` is a bool; assert `git diff --quiet` result matches `git_dirty`.
3. `test_registry_start_and_finalize_update_status` — `start` yields `"started"`; `finalize` yields `"completed"` and fills metrics/diagnostics/sample/checkpoint paths.
4. `test_registry_parent_run_id_links_surrogate_to_qgan` — create a QGAN record, then finalize a surrogate record with `parent_run_id=qgan_id`; assert loaded surrogate record has correct parent and workflow `fourier_surrogate`.

### Surrogate tests (`tests/test_fourier_surrogate.py`)

5. `test_fourier_surrogate_output_shape_matches_bank` — for `noise_dim=2`, `output_dim=4`, `rff_dim=8`, a forward pass on `(3, 2)` returns `(3, 4)`.
6. `test_fourier_features_are_deterministic_given_seed` — two calls with the same seed produce identical RFF cosine/sine features; different seeds differ.
7. `test_matched_rff_dim_equals_quantum_angle_count` — assert `matched_rff_dim(4, 3) == 24`.
8. `test_fourier_parameter_count_has_expected_shape` — returned dict has keys `rff_weights`, `rff_bias`, `w1`, `b1`, `w2`, `b2`, `total` and `total > 0`.
9. `test_surrogate_smoke_train_runs_without_pretrained_checkpoint` — compose smoke config with `surrogate.enabled=true`, call `run_surrogate_gan`, assert `metrics.json`, `evs_diagnostics.json`, and `run_record.json` exist, and that the record workflow is `fourier_surrogate`.
10. `test_surrogate_runs_with_mixed_pauli_bank` — smoke config with `observable_bank.name=mixed_pauli`, `n_qubits=4`, `latent_dim=14`, surrogate enabled; assert output shape and diagnostics vector length are 14.

### Separation metric tests (`tests/test_delta_sep.py`)

11. `test_compute_delta_sep_positive_when_qgan_larger` — `beta_q=2.0`, `beta_cls=1.0` → positive `log(2)`.
12. `test_compute_delta_sep_negative_when_classical_larger` — `beta_q=1.0`, `beta_cls=2.0` → negative.
13. `test_compute_delta_sep_clips_at_eps` — `beta_cls=0.0`, `beta_q=1.0`, `eps=1e-12` → `log(1 / 1e-12)`.
14. `test_delta_sep_from_smoke_qgan_and_surrogate` — run smoke QGAN and smoke surrogate with matching policy, then compute `Delta_sep` from their diagnostics; assert it is finite and stored in `comparison.json`.

### Smoke integration test (`tests/test_smoke_modes.py`)

15. Extend `test_gan_smoke_mode_runs_without_pretrained_checkpoint` to assert:
    - `(tmp_path / "outputs" / "train_smoke" / "run_record.json").exists()`
    - loaded record has `status == "completed"`, `workflow == "qgan_expectation_values"`, non-empty `git_commit`, and all paths resolve.

### Existing tests

All existing tests from Stage 6 must pass unchanged:

- `tests/test_observable_bank.py`
- `tests/test_evs_diagnostics.py`
- `tests/test_configs.py`
- `tests/test_smoke_modes.py`
- `tests/test_stage5_experiment_runner.py`
- `tests/test_models_and_checkpointing.py`
- `tests/test_latent_representation_runtime.py`
- `tests/test_latent_cache.py`
- `tests/test_file_logging.py`
- `tests/test_package_imports.py`
- `tests/test_namespace_compatibility.py`
- `tests/test_stage4_cli_contract.py`

## Implementation sequence

1. **Registry skeleton (RED first):** add failing tests 1–4. Then create `registry.py` with `RunRecord`, `RunRegistry`, git helpers, JSON write/load. Update `gan_loop.py` to instantiate registry, call `start()` before epochs and `finalize()` after checkpoints. Update `test_smoke_modes.py` assertion.
2. **Surrogate model (RED first):** add failing tests 5–8. Implement `FourierRFFSurrogate`, `init_fourier_params`, `build_fourier_apply`, `fourier_parameter_count`, and `matched_rff_dim` under `models/surrogate/`.
3. **Surrogate training loop (RED first):** add failing test 9. Implement `run_surrogate_gan` that reuses the same data, discriminator, representation, metrics, and smoke path as `run_gan`, but substitutes the surrogate generator and its train state.
4. **Matching + separation metric (RED first):** add failing tests 11–14. Add `compute_delta_sep`, `write_comparison_json`, and a pairing helper that reads two `run_record.json` files and produces `comparison.json`.
5. **Config + CLI wiring:** add `surrogate` section to `train.yaml`, add optional entry point or override, and add tests 10 and 15.
6. **Full verification:** run targeted tests, then `pytest -q`, then the smoke commands below.

## Smoke commands

```bash
# Run from repo root /root/orion-workspaces/qgan-stage2

# 1. Existing QGAN smoke (must still work and emit run_record.json)
PYTHONPATH=src python -m qgan_latent.cli.latent_train \
  smoke_test=true device=cpu wandb_mode=disabled \
  metrics.active_metrics=[] \
  outputs.dir=outputs/train_smoke \
  checkpoints.dir=checkpoints \
  model.autoencoder.latent_dim=4 \
  model.autoencoder.encoder_channels=[4,8] \
  model.autoencoder.decoder_channels=[8,4] \
  model.autoencoder.mlp_dim=16 \
  model.quantum_generator.n_qubits=2 \
  model.quantum_generator.noise_dim=2 \
  model.quantum_generator.depth=1

# 2. Fourier/RFF surrogate smoke (if CLI added)
PYTHONPATH=src qgan-latent-train-surrogate \
  smoke_test=true device=cpu wandb_mode=disabled \
  metrics.active_metrics=[] \
  outputs.dir=outputs/surrogate_smoke \
  checkpoints.dir=checkpoints \
  model.autoencoder.latent_dim=4 \
  model.autoencoder.encoder_channels=[4,8] \
  model.autoencoder.decoder_channels=[8,4] \
  model.autoencoder.mlp_dim=16 \
  model.quantum_generator.n_qubits=2 \
  model.quantum_generator.noise_dim=2 \
  model.quantum_generator.depth=1 \
  surrogate.enabled=true

# 3. Targeted tests
PYTHONPATH=src pytest -q tests/test_run_registry.py tests/test_fourier_surrogate.py tests/test_delta_sep.py tests/test_smoke_modes.py --tb=short

# 4. Full suite
PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check
```

## Acceptance criteria

- `RunRegistry` writes a valid `run_record.json` for every QGAN and surrogate run, capturing all required fields and git provenance.
- `FourierRFFSurrogate` produces the correct output dimension, deterministic RFF features, and a matched default `rff_dim`.
- `run_surrogate_gan` completes smoke mode without a pretrained checkpoint and writes the same artifact set as `run_gan`.
- The matching policy is explicit, recorded in `comparison.json`, and verified by tests.
- `Delta_sep` is computed from `total_parseval_norm_sq` of paired QGAN/surrogate diagnostics, using natural log and `eps=1e-12`, and stored in `comparison.json`.
- All existing tests pass unchanged; Stage 6 artifacts (`observable_bank_metadata.json`, `evs_diagnostics.json`, `metrics.csv/json`, sample grids) continue to be produced.
- No new runtime dependencies beyond numpy/JAX already in `pyproject.toml`; no graph-RL/PPO/trainable observables/QAS/shot noise introduced.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Registry adds I/O that could fail mid-run | Write `run_record.json` atomically (temp file + rename) and only finalize after successful checkpoint writes. |
| Surrogate forward pass differs in shape handling from quantum generator | Keep the same `build_fourier_apply` signature `(params, noise)` and output shape `(batch, output_dim)`; test shape parity with `build_generator_apply`. |
| Matching policy becomes ambiguous | Lock the six match rules in `comparison.json` and a dedicated test; default `rff_dim` derived from `depth * n_qubits * 2`. |
| `Delta_sep` numerically unstable | Use the defined `beta_cls + eps` denominator; assert finite in tests; log not sqrt-of-zero because `total_parseval_norm_sq >= 0`. |
| Smoke mode slow because two full loops run | Tests 9 and 14 should be separate and may reuse the same small smoke config used by existing `test_gan_smoke_mode_runs_without_pretrained_checkpoint`; do not add real MNIST smoke. |
| Git hash unavailable in CI | `RunRegistry` falls back to `"unknown"` and `git_dirty=false` without crashing; test only requires the key to be present and a string. |
| Config override for surrogate breaks existing configs | `surrogate.enabled` defaults to `false`; the surrogate module is only imported when enabled or tested. |

## Implementation Engineer instructions

1. Read this PPACK and the three key Stage 6 files: `gan_loop.py`, `diagnostics.py`, `observables.py`.
2. Do not change any existing test behavior unless this PPACK explicitly says to extend it.
3. Implement tests in the order listed; ensure each fails before writing the production code.
4. Keep files small and focused: registry in `training/registry.py`, surrogate model in `models/surrogate/`, surrogate loop in `training/surrogate_loop.py`.
5. Reuse `gan_loop.py` as much as possible rather than duplicating the full training loop. The cleanest approach is to refactor `run_gan` into a generator-agnostic inner function that accepts `gen_apply`, `gen_params`, `gen_state`, and the existing discriminator/representation/data setup, then call it from both `run_gan` and `run_surrogate_gan`. If that refactor is large, stop and ask for direction.
6. Do not implement the CLI entry point until the model, loop, and tests are green.
7. After all targeted tests pass, run the full suite and the smoke commands above. Report any failures.
8. Produce an E2X artifact at `docs/E2X-stage7-candidate-registry-fourier-surrogate.md` summarizing files changed, verification commands, and results.

---

## Model / planning metadata

- Planning model: kimi-k2.7-code
- Input tokens: N/A
- Output tokens: N/A

### Structured plan summary

```json
{
  "affected_files": [
    "src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py",
    "src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py",
    "src/qgan_latent/workflows/qgan_expectation_values/training/registry.py",
    "src/qgan_latent/workflows/qgan_expectation_values/training/surrogate_loop.py",
    "src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/fourier_generator.py",
    "src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/__init__.py",
    "configs/projects/qgan_expectation_values/train.yaml",
    "pyproject.toml",
    "tests/test_run_registry.py",
    "tests/test_fourier_surrogate.py",
    "tests/test_delta_sep.py",
    "tests/test_smoke_modes.py"
  ],
  "test_plan": [
    "test_run_record_schema_roundtrip",
    "test_registry_captures_git_commit_and_dirty_state",
    "test_registry_start_and_finalize_update_status",
    "test_registry_parent_run_id_links_surrogate_to_qgan",
    "test_fourier_surrogate_output_shape_matches_bank",
    "test_fourier_features_are_deterministic_given_seed",
    "test_matched_rff_dim_equals_quantum_angle_count",
    "test_fourier_parameter_count_has_expected_shape",
    "test_surrogate_smoke_train_runs_without_pretrained_checkpoint",
    "test_surrogate_runs_with_mixed_pauli_bank",
    "test_compute_delta_sep_positive_when_qgan_larger",
    "test_compute_delta_sep_negative_when_classical_larger",
    "test_compute_delta_sep_clips_at_eps",
    "test_delta_sep_from_smoke_qgan_and_surrogate",
    "test_gan_smoke_emits_run_record (extension in test_smoke_modes.py)"
  ],
  "implementation_sequence": [
    "RED tests 1-4 -> implement registry.py and wire into gan_loop.py",
    "RED tests 5-8 -> implement models/surrogate/fourier_generator.py",
    "RED test 9 -> implement training/surrogate_loop.py reusing gan_loop primitives",
    "RED tests 11-14 -> implement compute_delta_sep and comparison.json writer",
    "Add surrogate config section to train.yaml and optional CLI in pyproject.toml",
    "RED test 10 and extend test_smoke_modes.py",
    "Full pytest and smoke commands"
  ],
  "acceptance_criteria": [
    "run_record.json emitted for every QGAN and surrogate run with all required fields",
    "FourierRFFSurrogate deterministic, correct output dim, matched default rff_dim",
    "run_surrogate_gan completes smoke mode without pretrained checkpoint",
    "Matching policy explicit and verified by tests",
    "Delta_sep computed from total_parseval_norm_sq with natural log and eps=1e-12",
    "All existing Stage 6 tests pass unchanged",
    "No new dependencies beyond numpy/JAX; no forbidden scope introduced"
  ],
  "smoke_run_commands": [
    "PYTHONPATH=src python -m qgan_latent.cli.latent_train smoke_test=true device=cpu wandb_mode=disabled metrics.active_metrics=[] outputs.dir=outputs/train_smoke checkpoints.dir=checkpoints model.autoencoder.latent_dim=4 model.autoencoder.encoder_channels=[4,8] model.autoencoder.decoder_channels=[8,4] model.autoencoder.mlp_dim=16 model.quantum_generator.n_qubits=2 model.quantum_generator.noise_dim=2 model.quantum_generator.depth=1",
    "PYTHONPATH=src qgan-latent-train-surrogate smoke_test=true device=cpu wandb_mode=disabled metrics.active_metrics=[] outputs.dir=outputs/surrogate_smoke checkpoints.dir=checkpoints model.autoencoder.latent_dim=4 model.autoencoder.encoder_channels=[4,8] model.autoencoder.decoder_channels=[8,4] model.autoencoder.mlp_dim=16 model.quantum_generator.n_qubits=2 model.quantum_generator.noise_dim=2 model.quantum_generator.depth=1 surrogate.enabled=true",
    "PYTHONPATH=src pytest -q tests/test_run_registry.py tests/test_fourier_surrogate.py tests/test_delta_sep.py tests/test_smoke_modes.py --tb=short",
    "PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check"
  ],
  "risks_non_goals": [
    "Risk: registry I/O fails mid-run -> atomic temp-file write and finalize only after checkpoints",
    "Risk: surrogate shape handling differs from quantum generator -> same (params, noise) signature and (batch, output_dim) output; test parity",
    "Risk: matching policy ambiguity -> lock six rules in comparison.json and dedicated test",
    "Risk: Delta_sep numerical instability -> clip beta_cls to eps, assert finite",
    "Risk: smoke mode slow -> keep same tiny smoke config as existing test",
    "Risk: git hash unavailable in CI -> fallback to 'unknown' without crash",
    "Risk: surrogate config breaks existing configs -> default enabled=false",
    "Non-goal: graph-RL / PPO",
    "Non-goal: trainable observables",
    "Non-goal: full QAS search",
    "Non-goal: hardware shot noise",
    "Non-goal: real dataset dependency in smoke mode",
    "Non-goal: changing Stage 6 observable bank or EVS math"
  ]
}
```
