# ENG-FINAL — Stage 7 Candidate Registry + Fourier/RFF Surrogate

Verdict: READY

## Integration judgement

### 1. Complete against PPACK and Aaron's request?
Yes.

All PPACK acceptance criteria are met:
- `RunRegistry` emits `run_record.json` from both `run_gan` and `run_surrogate_gan` with the required schema, git provenance, resource counts, config snapshot, artifact paths, status, and parent-run linkage.
- `FourierRFFSurrogate` has deterministic RFF features, correct output shape, matched default `rff_dim = depth * n_qubits * 2`, and an auditable parameter count.
- `run_surrogate_gan` completes smoke mode without a pretrained checkpoint and produces the same artifact set as the QGAN path.
- QGAN/surrogate matched policy is validated before `comparison.json` is written, and the `validated_match` block is included in the artifact.
- `Delta_sep` is computed from `sqrt(total_parseval_norm_sq)` with natural log and denominator `beta_cls + eps` (`eps = 1e-12`), exactly as specified.
- All existing Stage 6 tests pass unchanged (84 passed).
- Forbidden scope (graph-RL/PPO, trainable observables, QAS, shot-noise simulation, new runtime deps) is not present.

### 2. Required changes before commit/PR?
No blocking changes.

The PPACK risk-table wording flagged during code review has been cleaned up; it now matches the implemented `beta_cls + eps` denominator. Recommended pre-PR action: commit the PPACK/E2X/REV/ENG-FINAL docs along with code. They are currently untracked.

### 3. Repository state summary

Branch: `feat/candidate-registry-fourier-surrogate`
Base: `eb1fa3d` (Stage 6 merge)
Working tree: uncommitted changes + untracked new files.

Modified tracked files:
- `configs/projects/qgan_expectation_values/train.yaml` — added optional `surrogate` section.
- `pyproject.toml` — added `qgan-latent-train-surrogate` console-script entry point.
- `src/qgan_latent/workflows/qgan_expectation_values/training/diagnostics.py` — added `parseval_norm_from_diagnostics`.
- `src/qgan_latent/workflows/qgan_expectation_values/training/gan_loop.py` — integrated `RunRegistry` start/finalize.
- `tests/test_smoke_modes.py` — extended smoke assertions for `run_record.json`.
- `tests/test_stage4_cli_contract.py` — accommodated new entry point.

New untracked files:
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
- `docs/E2X-stage7-candidate-registry-fourier-surrogate.md`
- `docs/REV-CODE-stage7-candidate-registry-fourier-surrogate.md`

### 4. Verification evidence (fresh)

Targeted Stage 7 tests:
```bash
PYTHONPATH=src pytest -q tests/test_run_registry.py tests/test_fourier_surrogate.py tests/test_delta_sep.py tests/test_smoke_modes.py --tb=short
```
Result: `27 passed, 14 warnings in 33.58s`

Full suite:
```bash
PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check
```
Result:
- `compileall` passed.
- `84 passed, 14 warnings in 40.48s`
- `git diff --check` clean.

### 5. Recommended next phase

1. Commit this branch with the Stage 7 code + docs as a single coherent commit.
2. Push to origin and open a PR against `main`.
3. PR title/body guidance below.
4. After merge, update the project graphify knowledge graph (`graphify-pr-merge-update` skill) so downstream code inspection remains graph-first.

### 6. PR title/body guidance

Title:
```
Stage 7: candidate run registry + Fourier/RFF classical surrogate + Delta_sep comparison
```

Body:
```markdown
## What
Implements Stage 7: structured run/candidate registry, a matched classical Fourier/RFF surrogate baseline, and the dequantization-aware separation metric `Delta_sep`.

## Why
Every QGAN experiment now produces auditable provenance (`run_record.json`) and a deterministic non-quantum baseline for comparison, which is required for downstream dequantization analysis.

## Scope
- Adds `RunRegistry` + `RunRecord` with git commit/dirty state, resource counts, config snapshot, artifact paths, and status.
- Adds `FourierRFFSurrogate` with matched default spectrum `rff_dim = depth * n_qubits * 2`.
- Adds `run_surrogate_gan` and `qgan-latent-train-surrogate` CLI entry point.
- Adds `compute_delta_sep`, `write_comparison_json`, and matched-policy validation.
- Emits `run_record.json` from QGAN smoke path and `comparison.json` for paired runs.

## Verification
- Targeted Stage 7 tests: 27 passed.
- Full suite: 84 passed, compileall clean, `git diff --check` clean.
- Paired QGAN + surrogate smoke comparison produces finite `Delta_sep`.

## Risks / notes
- Surrogate training loop mirrors `run_gan` rather than a full generator-agnostic refactor; this is intentional to keep the existing QGAN loop stable.
- `git_dirty` uses `git diff --quiet` and therefore does not flag untracked-only changes.
- PPACK risk-table wording has been cleaned up to match the implemented `beta_cls + eps` denominator.

## Docs
- docs/PPACK-stage7-candidate-registry-fourier-surrogate.md
- docs/E2X-stage7-candidate-registry-fourier-surrogate.md
- docs/REV-CODE-stage7-candidate-registry-fourier-surrogate.md
- docs/ENG-FINAL-stage7-candidate-registry-fourier-surrogate.md
```

---

Handoff:
- Artifact: `docs/ENG-FINAL-stage7-candidate-registry-fourier-surrogate.md`
- Verdict: READY
- Recommended owner for next action: Implementation Engineer or Aaron for commit/PR creation.
