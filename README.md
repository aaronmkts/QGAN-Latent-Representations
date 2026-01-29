# Latent Style-based Quantum GAN (Minimal JAX + PennyLane)

This repo contains a minimal, runnable pipeline inspired by the LaSt-QGAN paper. It trains a convolutional autoencoder to learn a low-dimensional latent space, then trains a quantum generator in that latent space against a classical discriminator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- MNIST is downloaded from Yann LeCun's site on first run.
- W&B logging is optional and controlled in the config.

## Pretrain a representation model

Default (autoencoder):

```bash
python pretrain.py
```

VAE:

```bash
python pretrain.py model/vae@model.autoencoder
```

Sinkhorn AE:

```bash
python pretrain.py model/sinkhorn_ae@model.autoencoder
```

VQ-VAE:

```bash
python pretrain.py model/vqvae@model.autoencoder
```

Outputs:
- Pretrained checkpoint: `checkpoints/autoencoder.ckpt` (or `model.autoencoder.checkpoint_name`)
- Reconstructions: `outputs/pretrain/`

Model-specific knobs (Hydra config):
- VAE: `model.autoencoder.beta`, `model.autoencoder.kl_anneal_steps`
- Sinkhorn AE: `model.autoencoder.lambda_sinkhorn`, `model.autoencoder.sinkhorn_eps`, `model.autoencoder.sinkhorn_iters`, `model.autoencoder.sinkhorn_cost`
- VQ-VAE: `model.autoencoder.num_embeddings`, `model.autoencoder.embedding_dim`, `model.autoencoder.commitment_cost`, `model.autoencoder.codebook_loss_weight`

Checkpoint naming uses `model.autoencoder.checkpoint_name` when set; otherwise it falls back to `checkpoints.autoencoder`.

Loss summaries:
- VAE: reconstruction + `beta * KL(q(z|x) || p(z))`
- Sinkhorn AE: reconstruction + `lambda_sinkhorn * Sinkhorn(z, z_prior)`
- VQ-VAE: reconstruction + codebook loss + `commitment_cost * commitment`

## Train the QGAN

```bash
python train.py
```

Outputs:
- Generator checkpoint: `checkpoints/qgan_gen.ckpt`
- Discriminator checkpoint: `checkpoints/qgan_disc.ckpt`
- Sample grids: `outputs/train/`

## Architecture overview

1. **Autoencoder** (JAX/Flax): CNN encoder maps 1x28x28 MNIST images to a low-dimensional latent vector. Decoder reconstructs images from the latent code.
2. **Quantum generator** (PennyLane + JAX): noise -> parameterized quantum circuit -> Pauli-Z expectations -> linear projection -> latent code.
3. **Discriminator** (JAX/Flax): small CNN that scores real vs fake images.
4. **Training**: WGAN-GP on images. Generator outputs latent vectors; decoder maps them to images.

## Configs

Hydra manages configs in `configs/`. Key defaults:
- Autoencoder latent dim: 20
- Quantum generator: 8 qubits, depth 4
- WGAN-GP: `lambda_gp = 10`
- Batch size: 128

Use `smoke_test=true` to run a short wiring check.

## Known limitations

- This is a minimal reference implementation; it is not tuned for sample quality.
- The quantum circuit runs on a simulator and can be slow for large batch sizes.
- MNIST download requires network access.
