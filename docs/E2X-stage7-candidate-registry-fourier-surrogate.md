# E2X — Stage 7 Candidate Registry + Fourier/RFF Surrogate

## Repository state

- Repository: `/root/orion-workspaces/qgan-stage2`
- Branch: `feat/candidate-registry-fourier-surrogate`
- Base: current `main` after Stage 6 merge (`eb1fa3d`)
- Commit status: not committed

## Implemented scope

Implemented Stage 7 according to `docs/PPACK-stage7-candidate-registry-fourier-surrogate.md`:

1. Structured run/candidate registry for QGAN and surrogate runs.
2. Run records with:
   - run ID
   - git commit
   - git dirty flag
   - workflow name
   - observable-bank metadata
   - resource counts
   - resolved config snapshot
   - metrics path
   - EVS diagnostics path
   - observable-bank metadata path
   - sample grid paths
   - checkpoint paths
   - status
   - parent run ID for surrogate-to-QGAN linkage
3. `run_record.json` emission from `run_gan`.
4. Classical Fourier/RFF surrogate generator.
5. Surrogate GAN training loop using the same representation/discriminator/WGAN-GP shape as the QGAN path.
6. Surrogate CLI module and console-script entry point.
7. Parseval-norm helper for diagnostics.
8. QGAN-vs-surrogate comparison helper and `comparison.json` writer.
9. Dequantization-aware separation metric:
10. Matched-policy validation before writing `comparison.json`, including parent-run linkage, workflow names, noise/output dimensions, training budget fields, learning rates, and default RFF spectrum rule.
11. Resource-count type preservation so floating-point learning rates remain auditable in `run_record.json` and `validated_match`.

```text
Delta_sep = log(beta_Q / (beta_cls + eps))
```

where:

```text
beta = sqrt(total_parseval_norm_sq)
eps = 1e-12
```

## Files changed / added

### Added

- `src/qgan_latent/workflows/qgan_expectation_values/training/registry.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/comparison.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/surrogate_loop.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/__init__.py`
- `src/qgan_latent/workflows/qgan_expectation_values/models/surrogate/fourier_generator.py`
- `src/qgan_latent/cli/latent_train_surrogate.py`
- `tests/test_run_registry.py`
- `tests/test_fourier_surrogate.py`
- `tests/test_delta_sep.py`
- `docs/PPACK-stage7-candidate-registry-fourier-surrogate.md`

### Modified

- `configs/projects/qgan_expectation_values/train.yaml`
- `pyproject.toml`
- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py`
- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py`
- `tests/test_smoke_modes.py`
- `tests/test_stage4_cli_contract.py`

## Verification evidence

### Targeted Stage 7 tests

Command:

```bash
PYTHONPATH=src pytest -q tests/test_run_registry.py tests/test_fourier_surrogate.py tests/test_delta_sep.py tests/test_smoke_modes.py --tb=short
```

Result:

```text
Initial implementation: 17 passed, 14 warnings.

After Code Reviewer changes for exact `beta_cls + eps`, matched-policy validation, and learning-rate type preservation:

27 passed, 14 warnings
```

### Full suite

Command:

```bash
PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check
```

Result:

```text
Initial implementation: 74 passed, 14 warnings.

After review fixes:

84 passed, 14 warnings
compileall passed
git diff --check clean
```

### Paired QGAN + surrogate comparison smoke

Ran smoke QGAN and smoke Fourier/RFF surrogate with matching config, then wrote comparison JSON.

Result:

```text
qgan_record_status completed qgan_expectation_values
surrogate_record_status completed
comparison_delta_finite True
comparison_keys ['beta_cls', 'beta_q', 'delta_sep', 'eps', 'matched_policy', 'qgan_run_id', 'surrogate_run_id']
```

After matched-policy validation was added, paired smoke also confirmed:

```text
qgan_gen_lr 0.001
validated_gen_lr 0.001
delta_sep 0.4996857472161421
```

### Surrogate CLI module smoke

Command shape:

```bash
PYTHONPATH=src python -m qgan_latent.cli.latent_train_surrogate \
  smoke_test=true device=cpu wandb_mode=disabled \
  metrics.active_metrics=[] \
  outputs.dir=<tmp>/outputs/surrogate_cli \
  checkpoints.dir=<tmp>/checkpoints \
  model.autoencoder.latent_dim=4 \
  model.autoencoder.encoder_channels=[4,8] \
  model.autoencoder.decoder_channels=[8,4] \
  model.autoencoder.mlp_dim=16 \
  model.quantum_generator.n_qubits=2 \
  model.quantum_generator.noise_dim=2 \
  model.quantum_generator.depth=1 \
  surrogate.enabled=true \
  surrogate.hidden_dim=8
```

Result:

```text
workflow fourier_surrogate
status completed
diag_len 4
sample_grid_exists True
```

## Notes / caveats

- The surrogate loop intentionally mirrors `run_gan` rather than performing a larger generator-agnostic training-loop refactor. This avoids destabilising the already-tested QGAN loop in Stage 7. A future cleanup can deduplicate the common training body after the science gate is validated.
- `rff_weights` and `rff_bias` are fixed random features generated from the configured seed. `fourier_parameter_count(...)["total"]` records fixed + trainable RFF-surrogate parameters; `trainable_surrogate_parameters` records the actual trainable tree size.
- Registry git state reports the active branch as dirty while this uncommitted implementation is present. This is expected pre-commit and will become clean once committed.
- Warnings observed are existing third-party matplotlib/pyparsing warnings plus JAX backend info during CLI smoke. They are not Stage 7 failures.

## Non-goals preserved

No implementation of:

- graph-RL / PPO;
- trainable observables;
- full QAS search;
- hardware shot-noise simulation;
- new runtime dependencies.
