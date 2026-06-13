# Implementation Plan Status and Local Branch Audit

This note records the cleanup decision for local-only implementation branches that remained after the staged QGAN latent-representation work landed on `main`.

## Source plans

- ORION/Obsidian roadmap: `QGAN Latent Representations — Implementation Roadmap and Fair Evaluation`.
- ORION/Obsidian detailed plan: `QGAN Latent Representations — Steps 1–2 Representation Interface and Latent Cache Implementation Plan`.
- Repository-facing boundary: `docs/research-memory-boundary.md`.

## Branch mapping

| Local branch | Plan stage | Decision |
| --- | --- | --- |
| `orion/stage2-config-separation` | Stage 2: shared/project Hydra config taxonomy | Already represented on `main`; `git cherry main <branch>` reports patch-equivalent. |
| `orion/stage3-namespace-separation` | Stage 3: canonical source namespaces | Already represented on `main`; `git cherry main <branch>` reports patch-equivalent. |
| `orion/stage4-explicit-cli` | Stage 4: explicit workflow CLIs and removal of legacy wrappers | Already represented on `main`; `git cherry main <branch>` reports patch-equivalent. |
| `feat/stage5-test-experiments` | Stage 5: short experiment runner and GPU execution | Useful functionality is already on `main`; the stale branch predates later representation-runtime and latent-cache work and should not be merged directly. |

## Current `main` coverage

`main` already contains the durable implementation pieces from the old local branches:

- Stage 2 config taxonomy under `configs/shared/` and `configs/projects/`.
- Stage 3/4 canonical namespaces under `qgan_latent.shared` and `qgan_latent.workflows.*`.
- Explicit console scripts:
  - `qgan-latent-pretrain`
  - `qgan-latent-train`
  - `qgan-vqvae-pretrain`
  - `qgan-vqvae-train-prior`
  - `qgan-stage5-test-run`
  - `qgan-cache-latents`
- Stage 5 experiment runner in `src/qgan_latent/stage5.py`.
- CUDA-oriented dependency declaration via `jax[cuda12]`.
- GPU default for Stage 5 command generation.
- Separate train checkpoint directories for latent QGAN and MPS-prior stages.
- JSON summary and Obsidian handoff export for VPS-to-vault filing.
- Representation-aware GAN runtime and latent-cache boundary for roadmap Steps 1–2.

## Why the stale Stage 5 branch is not merged

The local `feat/stage5-test-experiments` branch contains two commits:

- `f081591 feat: add stage 5 experiment test runner`
- `bb2fb0a fix: run stage 5 experiments on CUDA JAX`

Those concepts are already present on `main`, but the branch forked before later work. Merging it directly would regress current `main` by removing or reverting:

- `qgan-cache-latents` entry point;
- representation-runtime imports in the latent-QGAN training loop;
- checkpoint-path handling for AE/VAE/SinkhornAE runtime loading;
- the broader latent-cache implementation and tests.

Therefore the correct cleanup action is to preserve `main`, record this audit, and delete the stale local branches after tests pass.

## Verification expectation

Before deleting the stale branches, run:

```bash
python -m compileall -q src
pytest -q
```

Acceptance criterion: full tests pass on current `main` plus this documentation note.
