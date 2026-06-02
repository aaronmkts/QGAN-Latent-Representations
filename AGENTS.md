# AGENTS.md

This repository is a research testbed for latent-representation generative modelling on MNIST. It currently supports two first-class project workflows:

- Latent-space QGAN: representation pretraining followed by a latent-space QGAN with a classical discriminator.
- Tensor-prior VQ-VAE: Spatial VQ-VAE pretraining followed by an MPS Born-machine prior over codebook indices.

Stage 3 separates Python source namespaces into shared components and workflow-specific packages. Keep legacy import wrappers thin and compatibility-only; new production imports should use canonical Stage 3 namespaces.

## Structure

- `src/qgan_latent/` is the Python package root.
- `src/qgan_latent/cli/` contains stable Hydra command entrypoints.
- `src/qgan_latent/shared/` contains reusable datamodules, representation models, pretraining, smoke helpers, and utilities.
- `src/qgan_latent/workflows/qgan_expectation_values/` contains the latent-space QGAN generator, discriminator, and GAN training loop.
- `src/qgan_latent/workflows/tensor_prior_vqvae/` contains the tensor-prior/MPS prior models and training loop.
- `src/qgan_latent/datamodules/`, `src/qgan_latent/models/`, `src/qgan_latent/training/`, and `src/qgan_latent/utils/` are legacy compatibility wrappers. Do not add new implementation logic there.
- `configs/` contains Hydra configs. Root configs (`pretrain`, `train`, `train_prior`) are compatibility entrypoints. Shared groups live under `configs/shared/`; project-specific groups live under `configs/projects/qgan_expectation_values/` and `configs/projects/tensor_prior_vqvae/`.
- `tests/` contains pytest coverage for configs, package imports, model initialization, checkpoints, and smoke modes.

## Coding Conventions

- Use package-qualified imports from canonical Stage 3 namespaces: `qgan_latent.shared...` or `qgan_latent.workflows...`.
- Do not add `sys.path` mutation to production code or scripts.
- Keep root scripts (`pretrain.py`, `train.py`, `train_prior.py`) as thin compatibility wrappers around `qgan_latent.cli`.
- Prefer small helper modules when behavior is shared across training loops.
- Keep training defaults research-oriented, but keep smoke paths short and deterministic. Smoke checks verify wiring and checkpoints, not sample quality.
- W&B must be optional. Default configs and smoke commands should run with `wandb_mode=disabled`.
- Preserve ignored runtime artifacts unless the user explicitly asks to clean them.

## Environment

- Conda is the source of truth: `environment.yml`.
- Python target: 3.11.
- `requirements.txt` is only a pip compatibility list.
- Install/editable package behavior is expected; console scripts are preferred over root scripts.

## Commands

- Create/update environment: `conda env update -f environment.yml`
- Activate: `conda activate qlatent`
- Run tests: `pytest -q`
- Compile check: `python -m compileall -q src pretrain.py train.py train_prior.py`
- Pretrain representation models: `qgan-pretrain`
- Train the latent-space QGAN workflow: `qgan-train`
- Train the tensor-prior VQ-VAE workflow: `qgan-train-prior`
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
