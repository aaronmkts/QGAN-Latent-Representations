# QGAN Latent Representations Testbed

This repository is a JAX/Flax/PennyLane testbed for latent-representation generative modelling on MNIST. It treats two research workflows as first-class projects:

- **Latent-space QGAN workflow:** pretrain a representation model, then train a latent-space quantum GAN with a classical discriminator.
- **Tensor-prior VQ-VAE workflow:** pretrain a Spatial VQ-VAE, then train an MPS Born-machine prior over its codebook-index sequences.

The current goal is high-level functionality, clear wiring, and reproducible smoke checks before research-quality tuning.

## Setup

Conda is the source of truth for the environment.

```bash
conda env create -f environment.yml
conda activate qlatent
```

For an existing environment:

```bash
conda env update -f environment.yml
conda activate qlatent
```

The environment installs the package in editable mode. Stage 4 uses explicit workflow console scripts only:

- `qgan-latent-pretrain`
- `qgan-latent-train`
- `qgan-vqvae-pretrain`
- `qgan-vqvae-train-prior`

Legacy commands (`qgan-pretrain`, `qgan-train`, `qgan-train-prior`) and root scripts (`pretrain.py`, `train.py`, `train_prior.py`) have been removed.

## Smoke Checks

Smoke mode uses deterministic synthetic MNIST-shaped data, disables W&B, and avoids requiring pretrained checkpoints unless the command explicitly points to one. These commands check wiring, config composition, checkpoint creation, and entrypoint behaviour; they do not establish sample quality.

```bash
qgan-latent-pretrain smoke_test=true device=cpu wandb_mode=disabled
```

```bash
qgan-latent-train smoke_test=true device=cpu wandb_mode=disabled metrics.active_metrics=[] \
  model.autoencoder.latent_dim=4 model.quantum_generator.n_qubits=2 \
  model.quantum_generator.noise_dim=2 model.quantum_generator.depth=1
```

```bash
qgan-vqvae-pretrain smoke_test=true device=cpu wandb_mode=disabled
```

```bash
qgan-vqvae-train-prior smoke_test=true device=cpu wandb_mode=disabled \
  model.vqvae.num_embeddings=4 model.vqvae.embedding_dim=2 \
  model.mps_prior.phys_dim=4 model.mps_prior.bond_dim=2 \
  model.mps_prior.epochs=1 model.mps_prior.batch_size=4
```

Run the test suite with:

```bash
pytest -q
```

## Workflow 1: Latent-Space QGAN

This workflow trains an autoencoder representation and then trains a latent-space QGAN against the learned representation.

Pretrain the default QGAN autoencoder representation:

```bash
qgan-latent-pretrain
```

Other representation configs remain available through explicit Hydra overrides:

```bash
qgan-latent-pretrain 'shared/representations@model.autoencoder=vae'
qgan-latent-pretrain 'shared/representations@model.autoencoder=sinkhorn_ae'
qgan-latent-pretrain 'shared/representations@model.autoencoder=vqvae'
qgan-latent-pretrain 'shared/representations@model.autoencoder=spatial_vqvae'
```

Then train the latent-space QGAN using the configured autoencoder checkpoint path:

```bash
qgan-latent-train
```

Outputs:

- Representation checkpoints in `checkpoints/`
- QGAN generator/discriminator checkpoints in `checkpoints/`
- Reconstructions and sample grids in `outputs/`

## Workflow 2: Tensor-Prior VQ-VAE

This workflow trains a Spatial VQ-VAE and then trains an MPS Born-machine prior over the model's discrete codebook-index sequences.

Pretrain the default Spatial VQ-VAE:

```bash
qgan-vqvae-pretrain
```

Train the MPS prior over the Spatial VQ-VAE codebook indices:

```bash
qgan-vqvae-train-prior
```

Outputs:

- Spatial VQ-VAE checkpoint in `checkpoints/spatial_vqvae.ckpt`
- MPS prior checkpoint in `checkpoints/mps_prior.ckpt`
- Decoded MPS samples in `outputs/tensor_prior_vqvae/train_prior/`

## Architecture

- `qgan_latent.shared.datamodules`: MNIST download/loading.
- `qgan_latent.shared.representations`: AE, VAE, Sinkhorn AE, VQ-VAE, Spatial VQ-VAE.
- `qgan_latent.shared.training` and `qgan_latent.shared.smoke`: reusable pretraining and smoke-test helpers.
- `qgan_latent.shared.utils`: checkpoints, device selection, logging, metrics, paths, seeds, image grids.
- `qgan_latent.workflows.qgan_expectation_values`: latent-space QGAN generator, discriminator, and GAN training loop.
- `qgan_latent.workflows.tensor_prior_vqvae`: tensor-prior/MPS prior models and training loop.

Stage 4 removed the legacy `qgan_latent.datamodules`, `qgan_latent.models`, `qgan_latent.training`, and `qgan_latent.utils` wrapper packages. New code should use only canonical `qgan_latent.shared` and `qgan_latent.workflows` imports.

## Configuration Layout

Hydra configs are split into shared and project-specific groups. Explicit workflow commands use project-specific configs by default.

- `configs/shared/data/`: dataset configs such as MNIST.
- `configs/shared/metrics/`: reusable metric configs.
- `configs/shared/representations/`: reusable representation models including AE, VAE, VQ-VAE, and Spatial VQ-VAE.
- `configs/projects/qgan_expectation_values/`: latent-space QGAN pretraining/training configs plus QGAN-specific generator/discriminator configs.
- `configs/projects/tensor_prior_vqvae/`: Spatial VQ-VAE plus tensor-prior/MPS configs.

Direct project configs can be composed with Hydra names such as `projects/qgan_expectation_values/train` and `projects/tensor_prior_vqvae/train_prior`.

## Documentation Scope

Repository documentation should stay operational and reviewable: setup, commands, smoke checks, architecture, configuration, and reproducibility. Long-form research notes, paper extractions, theory development, experiment interpretation, negative results, and cross-stream synthesis belong in Aaron's Obsidian PhD research graph.

Canonical research note: `Research/PhD/QGAN Latent Representations.md`.

## Notes

- W&B is optional and disabled by default in configs.
- Full MNIST training downloads data into `data/mnist/`.
- Runtime artifacts are ignored under `outputs/`, `checkpoints/`, `data/mnist/`, `wandb/`, and `lancedb/`.
- The quantum circuit runs on a simulator and can be slow for large batch sizes.
