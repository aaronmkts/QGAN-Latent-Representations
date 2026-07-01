# REV-CODE-stage7-candidate-registry-fourier-surrogate

Verdict: APPROVED

Reviewer: Code Reviewer profile
Repository: `/root/orion-workspaces/qgan-stage2`
Branch: `feat/candidate-registry-fourier-surrogate`
Reviewed against:
- `docs/PPACK-stage7-candidate-registry-fourier-surrogate.md`
- `docs/E2X-stage7-candidate-registry-fourier-surrogate.md`
- current git diff / working tree

## Summary

The Stage 7 re-review fixes the two prior blocking issues.

`compute_delta_sep()` now implements the stated metric exactly as `log(beta_Q / (beta_cls + eps))` in `src/qgan_latent/workflows/qgan_expectation_values/training/comparison.py:24`, and the regression test covers the small-nonzero `0 < beta_cls < eps` case in `tests/test_delta_sep.py:56`.

`comparison_payload()` now validates the QGAN/surrogate match before computing or writing a comparison artifact in `src/qgan_latent/workflows/qgan_expectation_values/training/comparison.py:68`. The validated artifact includes an auditable `validated_match` block, rejects mismatched workflows/parent run/budget fields/RFF dimension, and preserves floating learning rates in registry resource counts.

No blocking findings remain.

## Blocking findings

None.

## Non-blocking notes

- `docs/PPACK-stage7-candidate-registry-fourier-surrogate.md:448` still says “Clip `beta_cls` to `eps` in the denominator” in the risk table. The mathematical definition and implementation are correct elsewhere (`beta_cls + eps`), so this is not blocking, but that one risk-table sentence should be cleaned up before using the PPACK as a future implementation template.
- `RunRegistry.current_git_state()` still uses `git diff --quiet`, so untracked files alone do not set `git_dirty=True`. This was already noted in the prior review and is acceptable for this stage because the tests lock that behavior explicitly.
- The surrogate loop intentionally duplicates a substantial part of `run_gan`. This remains acceptable for Stage 7 stability, but a future generator-agnostic training-loop refactor would reduce drift risk.

## Verification evidence

Repository state inspected:

```text
git status --short && git branch --show-current && test -f graphify-out/graph.json && echo GRAPHIFY_PRESENT || echo GRAPHIFY_MISSING
```

Result summary:

```text
Branch: feat/candidate-registry-fourier-surrogate
Graphify: GRAPHIFY_MISSING
Working tree contains Stage 7 modified/untracked files under review.
```

Targeted Stage 7 tests run independently:

```text
PYTHONPATH=src pytest -q tests/test_delta_sep.py tests/test_run_registry.py tests/test_fourier_surrogate.py tests/test_smoke_modes.py --tb=short
```

Result:

```text
27 passed, 14 warnings in 33.98s
```

Full verification run independently:

```text
PYTHONPATH=src python -m compileall -q src && PYTHONPATH=src pytest -q && git diff --check
```

Result:

```text
84 passed, 14 warnings in 40.30s
compileall exited 0
git diff --check exited 0
```

Additional reviewer spot-check for prior blockers:

```text
PYTHONPATH=src python - <<'PY'
# Checked exact plus-eps formula, validated_match inclusion, float gen_lr preservation,
# and rejection of parent_run_id, workflow, noise_dim, observable_bank_output_dim,
# steps_per_epoch, disc_lr, lambda_gp, and surrogate_rff_dim mismatches.
PY
```

Observed result:

```text
delta_matches_plus_eps True
validated_gen_lr 0.001
has_validated_match True
reject parent_run_id True
reject workflow True
reject noise_dim True
reject observable_bank_output_dim True
reject steps_per_epoch True
reject disc_lr True
reject lambda_gp True
reject surrogate_rff_dim True
```

## Scope assessment

- Structured registry: present.
- Registry metadata: present for observable bank, resource counts, config snapshot, metrics/diagnostics/sample-grid/checkpoint paths, and git commit hash.
- Resource-count type preservation: present; float learning rates remain floats.
- Fourier/RFF surrogate: deterministic, shape-correct, and smoke-trainable under current tests.
- Delta separation: computed from `sqrt(total_parseval_norm_sq)` using `beta_cls + eps` exactly.
- Matched QGAN-vs-surrogate policy: enforced before comparison artifact writing and recorded via `validated_match`.
- Stage 6 tests: preserved under full suite.
- Forbidden scope: no evidence found of graph-RL/PPO, trainable observables, full QAS search, shot-noise simulation, or new runtime dependencies.

## Reviewer decision

APPROVED for Stage 7 acceptance/merge, subject only to the non-blocking documentation cleanup noted above.
