# QGAN Latent Representations Testbed

This repository is a JAX/Flax/PennyLane testbed for latent-representation generative modelling on MNIST. It currently treats two research workflows as first-class projects:

- **Latent-space QGAN workflow:** train a representation model, then train a latent-space quantum GAN with a classical discriminator.
- **Tensor-prior VQ-VAE workflow:** train a Spatial VQ-VAE, then train an MPS Born-machine prior over its codebook-index sequences.

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

The environment installs the package in editable mode. Preferred commands are the console scripts:

- `qgan-pretrain`
- `qgan-train`
- `qgan-train-prior`

The root scripts `pretrain.py`, `train.py`, and `train_prior.py` remain as compatibility wrappers after editable installation.

## Smoke Checks

Smoke mode uses deterministic synthetic MNIST-shaped data, disables W&B, and avoids requiring pretrained checkpoints. These commands check wiring, config composition, checkpoint creation, and entrypoint behaviour; they do not establish sample quality.

```bash
qgan-pretrain smoke_test=true device=cpu wandb_mode=disabled
```

```bash
qgan-train smoke_test=true device=cpu wandb_mode=disabled metrics.active_metrics=[] \
  model.autoencoder.latent_dim=4 model.quantum_generator.n_qubits=2 \
  model.quantum_generator.noise_dim=2 model.quantum_generator.depth=1
```

```bash
qgan-train-prior smoke_test=true device=cpu wandb_mode=disabled \
  model.vqvae.num_embeddings=4 model.vqvae.embedding_dim=2 \
  model.mps_prior.phys_dim=4 model.mps_prior.bond_dim=2 \
  model.mps_prior.epochs=1 model.mps_prior.batch_size=4
```

Run the test suite with:

```bash
pytest -q
```

## Workflow 1: Latent-Space QGAN

This workflow trains a representation model and then trains a latent-space QGAN against the learned representation.

Pretrain the default Spatial VQ-VAE representation:

```bash
qgan-pretrain
```

Pretrain an autoencoder for the QGAN path:

```bash
qgan-pretrain 'shared/representations@model.autoencoder=autoencoder'
```

Other representation configs:

```bash
qgan-pretrain 'shared/representations@model.autoencoder=vae'
qgan-pretrain 'shared/representations@model.autoencoder=sinkhorn_ae'
qgan-pretrain 'shared/representations@model.autoencoder=vqvae'
qgan-pretrain 'shared/representations@model.autoencoder=spatial_vqvae'
```

Then train the latent-space QGAN using the configured autoencoder checkpoint path:

```bash
qgan-train
```

Outputs:

- Representation checkpoints in `checkpoints/`
- QGAN generator/discriminator checkpoints in `checkpoints/`
- Reconstructions and sample grids in `outputs/`

## Workflow 2: Tensor-Prior VQ-VAE

This workflow trains a Spatial VQ-VAE and then trains an MPS Born-machine prior over the model's discrete codebook-index sequences.

Train the Spatial VQ-VAE:

```bash
qgan-pretrain 'shared/representations@model.autoencoder=spatial_vqvae'
```

Train the MPS prior over the Spatial VQ-VAE codebook indices:

```bash
qgan-train-prior
```

Outputs:

- Spatial VQ-VAE checkpoint in `checkpoints/spatial_vqvae.ckpt`
- MPS prior checkpoint in `checkpoints/mps_prior.ckpt`
- Decoded MPS samples in `outputs/train_prior/`

## Architecture

- `qgan_latent.datamodules`: MNIST download/loading.
- `qgan_latent.models.compression_methods`: AE, VAE, Sinkhorn AE, VQ-VAE, Spatial VQ-VAE.
- `qgan_latent.models.quantum_generator`: PennyLane/JAX style-based quantum generator.
- `qgan_latent.models.mps_prior`: JAX and optional quimb MPS prior utilities.
- `qgan_latent.training`: pretraining, QGAN, MPS-prior loops, and smoke helpers.
- `qgan_latent.utils`: checkpoints, device selection, logging, metrics, paths, seeds, image grids.

## Configuration Layout

Hydra configs are split into shared and project-specific groups while preserving the root compatibility entrypoints `pretrain`, `train`, and `train_prior`.

- `configs/shared/data/`: dataset configs such as MNIST.
- `configs/shared/metrics/`: reusable metric configs.
- `configs/shared/representations/`: reusable representation models including AE, VAE, VQ-VAE, and Spatial VQ-VAE.
- `configs/projects/qgan_expectation_values/`: latent-space QGAN pretraining/training configs plus QGAN-specific generator/discriminator configs.
- `configs/projects/tensor_prior_vqvae/`: Spatial VQ-VAE plus tensor-prior/MPS configs.

Direct project configs can be composed with Hydra names such as `projects/qgan_expectation_values/train` and `projects/tensor_prior_vqvae/train_prior`. The console scripts continue to use the root compatibility configs by default.

## Documentation Scope

Repository documentation should stay operational and reviewable: setup, commands, smoke checks, architecture, configuration, and reproducibility. Long-form research notes, paper extractions, theory development, experiment interpretation, negative results, and cross-stream synthesis belong in Aaron's Obsidian PhD research graph.

Canonical research note: `Research/PhD/QGAN Latent Representations.md`.

## Notes

- W&B is optional and disabled by default in configs.
- Full MNIST training downloads data into `data/mnist/`.
- Runtime artifacts are ignored under `outputs/`, `checkpoints/`, `data/mnist/`, `wandb/`, and `lancedb/`.
- The quantum circuit runs on a simulator and can be slow for large batch sizes.
