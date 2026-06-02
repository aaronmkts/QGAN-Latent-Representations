# AGENTS.md

This repository is a research testbed for latent-representation generative modelling on MNIST. It currently supports two first-class project workflows:

- Latent-space QGAN: representation pretraining followed by a latent-space QGAN with a classical discriminator.
- Tensor-prior VQ-VAE: Spatial VQ-VAE pretraining followed by an MPS Born-machine prior over codebook indices.

Stage 2 separates Hydra configs into shared and project-specific groups. Do not move Python namespaces or production modules until the later namespace-refactor stage.

## Structure

- `src/qgan_latent/` is the Python package root.
- `src/qgan_latent/cli/` contains Hydra command entrypoints.
- `src/qgan_latent/datamodules/` contains dataset loading code.
- `src/qgan_latent/models/` contains representation models, quantum generator code, discriminators, and MPS priors.
- `src/qgan_latent/training/` contains training loops and smoke-test helpers.
- `src/qgan_latent/utils/` contains checkpointing, device, image-grid, logging, metrics, path, seed, and train-state helpers.
- `configs/` contains Hydra configs. Root configs (`pretrain`, `train`, `train_prior`) are compatibility entrypoints. Shared groups live under `configs/shared/`; project-specific groups live under `configs/projects/qgan_expectation_values/` and `configs/projects/tensor_prior_vqvae/`.
- `tests/` contains pytest coverage for configs, package imports, model initialization, checkpoints, and smoke modes.

## Coding Conventions

- Use package-qualified imports: `from qgan_latent...`.
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
