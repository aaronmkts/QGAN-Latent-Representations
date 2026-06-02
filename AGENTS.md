# AGENTS.md

This repository is a research testbed for latent-representation generative modelling on MNIST. It currently supports two first-class project workflows:

- Latent-space QGAN: representation pretraining followed by a latent-space QGAN with a classical discriminator.
- Tensor-prior VQ-VAE: Spatial VQ-VAE pretraining followed by an MPS Born-machine prior over codebook indices.

Stage 4 uses explicit workflow CLIs and canonical source namespaces only. Legacy console commands, root scripts, and Stage 3 compatibility wrapper packages have been removed.

## Structure

- `src/qgan_latent/` is the Python package root.
- `src/qgan_latent/cli/` contains stable Hydra command entrypoints with explicit workflow names.
- `src/qgan_latent/shared/` contains reusable datamodules, representation models, pretraining, smoke helpers, and utilities.
- `src/qgan_latent/workflows/qgan_expectation_values/` contains the latent-space QGAN generator, discriminator, and GAN training loop.
- `src/qgan_latent/workflows/tensor_prior_vqvae/` contains the tensor-prior/MPS prior models and training loop.
- `configs/shared/` contains reusable Hydra groups.
- `configs/projects/qgan_expectation_values/` contains latent-space QGAN configs.
- `configs/projects/tensor_prior_vqvae/` contains tensor-prior VQ-VAE configs.
- `tests/` contains pytest coverage for configs, package imports, CLI contract, model initialization, checkpoints, and smoke modes.

## Coding Conventions

- Use package-qualified imports from canonical namespaces: `qgan_latent.shared...` or `qgan_latent.workflows...`.
- Do not add `sys.path` mutation to production code or scripts.
- Do not restore legacy wrapper packages under `qgan_latent.datamodules`, `qgan_latent.models`, `qgan_latent.training`, or `qgan_latent.utils`.
- Do not restore root compatibility scripts (`pretrain.py`, `train.py`, `train_prior.py`).
- Prefer small helper modules when behavior is shared across training loops.
- Keep training defaults research-oriented, but keep smoke paths short and deterministic. Smoke checks verify wiring and checkpoints, not sample quality.
- W&B must be optional. Default configs and smoke commands should run with `wandb_mode=disabled`.
- Preserve ignored runtime artifacts unless the user explicitly asks to clean them.

## Environment

- Conda is the source of truth: `environment.yml`.
- Python target: 3.11.
- `requirements.txt` is only a pip compatibility list.
- Install/editable package behavior is expected; console scripts are preferred.

## Commands

- Create/update environment: `conda env update -f environment.yml`
- Activate: `conda activate qlatent`
- Run tests: `pytest -q`
- Compile check: `python -m compileall -q src`
- Pretrain latent-space QGAN representation: `qgan-latent-pretrain`
- Train latent-space QGAN: `qgan-latent-train`
- Pretrain tensor-prior Spatial VQ-VAE: `qgan-vqvae-pretrain`
- Train tensor-prior/MPS prior: `qgan-vqvae-train-prior`
- Compose direct QGAN project configs with names such as `projects/qgan_expectation_values/pretrain` and `projects/qgan_expectation_values/train`.
- Compose direct tensor-prior project configs with names such as `projects/tensor_prior_vqvae/pretrain` and `projects/tensor_prior_vqvae/train_prior`.
- Override representation groups with syntax such as `shared/representations@model.autoencoder=autoencoder`.

## Artifacts

These paths are local runtime artifacts and should stay ignored:

- `outputs/`
- `checkpoints/`
- `data/mnist/`
- `wandb/`
- `lancedb/`
