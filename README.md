# QGAN Latent Representations

This repository contains a JAX/Flax/PennyLane research pipeline for generative modeling in learned MNIST latent spaces. It currently treats two workflows as first-class:

- Train a representation model, then train a latent-space QGAN with a classical discriminator.
- Train a Spatial VQ-VAE, then train an MPS Born-machine prior over its codebook-index sequences.

The current goal is high-level functionality and clear wiring before research-quality tuning.

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

Smoke mode uses deterministic synthetic MNIST-shaped data, disables W&B, and avoids requiring pretrained checkpoints. These commands check wiring, not sample quality.

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

## Workflow 1: Representation + Latent QGAN

Pretrain the default Spatial VQ-VAE representation:

```bash
qgan-pretrain
```

Pretrain an autoencoder for the QGAN path:

```bash
qgan-pretrain 'model@model.autoencoder=autoencoder'
```

Other representation configs:

```bash
qgan-pretrain 'model@model.autoencoder=vae'
qgan-pretrain 'model@model.autoencoder=sinkhorn_ae'
qgan-pretrain 'model@model.autoencoder=vqvae'
qgan-pretrain 'model@model.autoencoder=spatial_vqvae'
```

Then train the latent-space QGAN:

```bash
qgan-train
```

Outputs:

- Representation checkpoints in `checkpoints/`
- QGAN generator/discriminator checkpoints in `checkpoints/`
- Reconstructions and sample grids in `outputs/`

## Workflow 2: Spatial VQ-VAE + MPS Prior

Train the Spatial VQ-VAE:

```bash
qgan-pretrain 'model@model.autoencoder=spatial_vqvae'
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

## Notes

- W&B is optional and disabled by default in configs.
- Full MNIST training downloads data into `data/mnist/`.
- Runtime artifacts are ignored under `outputs/`, `checkpoints/`, `data/mnist/`, `wandb/`, and `lancedb/`.
- The quantum circuit runs on a simulator and can be slow for large batch sizes.
