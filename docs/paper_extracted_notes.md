# Paper extracted notes (LaSt-QGAN)

Source: `Latent Style-based Quantum GAN for high-quality Image Generation.pdf` (local-rag)
Accessed: 2026-01-20

## Extracted

- Latent-space training pipeline:
  - Quote: "This novel approach relies on powerful classical auto-encoders to map a high-dimensional original image dataset into a latent representation. The hybrid classical-quantum GAN operates in this latent space to generate an arbitrary number of fake features, which are then passed back to the auto-encoder to reconstruct the original data." (chunk 2)
- WGAN-GP usage:
  - Quote: "Additionally, Wasserstein loss with gradient penalty ... is used for better convergence in the model." (chunk 44)
- Latent dimension for MNIST/FashionMNIST autoencoder:
  - Quote: "... autoencoder architecture with latent space of dimension 20 for MNIST and FashionMNIST datasets." (chunk 312)
- Latent range constraint:
  - Quote: "We apply the Tanh activation function at the end of the encoder to ensure that the latent features are confined within the range of [ -1 , 1 ]." (chunk 314)
- MNIST qubit note:
  - Quote: "... generate high-quality MNIST images on discretized latent space ... using 8 qubits." (chunk 310)

## Assumptions (not explicitly specified in the paper text we extracted)

- Training steps, batch size, and learning rates: defaults chosen for a minimal runnable baseline (see `configs/pretrain.yaml` and `configs/train.yaml`).
- Autoencoder architecture: small CNN encoder/decoder used to keep runtime light; only the latent dimension and tanh constraint are matched to the paper.
- Generator circuit depth: set to 4 for a minimal circuit; paper uses multiple circuit variants and depths, but exact choices are not fully specified in extracted text.
