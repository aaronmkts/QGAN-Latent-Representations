# RESULTS — Stage 8 Observable-Bank Sweep

## Status

Completed on the Orion GPU host using the Stage 7 registry, EVS diagnostics, matched Fourier/RFF surrogate, and dequantization-aware comparison pipeline.

- Run ID: `stage8_gpu_bank_sweep_20260701_1730c`
- Remote output root: `/srv/orion/workspace/PhD/QGAN-Latent-Representations/outputs/experiments/stage8_gpu_bank_sweep_20260701_1730c`
- Summary JSON: `outputs/experiments/stage8_gpu_bank_sweep_20260701_1730c/stage8_summary.json`
- Summary CSV: `outputs/experiments/stage8_gpu_bank_sweep_20260701_1730c/stage8_summary.csv`
- Frozen encoder checkpoint: `outputs/experiments/baseline_mnist_last_20260701_124132/checkpoints/autoencoder.ckpt`

## Fixed settings

| Setting | Value |
|---|---:|
| Device | GPU |
| Seed | 42 |
| Epochs | 100 |
| Batch size | 128 |
| `n_critic` | 5 |
| Generator LR | `1e-3` |
| Discriminator LR | `1e-3` |
| `lambda_gp` | 10.0 |
| WandB | disabled |

The frozen encoder is 20-dimensional, so every condition was constrained to `observable_bank.output_dim = 20`.

## Result table

| Rank | Condition | Observable bank | QGAN β | Surrogate β | Δ_sep | QGAN JS | Surrogate JS | QGAN NDB/K | Surrogate NDB/K |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `local_xy_10q` | `local_pauli [X,Y]`, 10 qubits | 1.3881 | 1.3644 | **0.0172** | 0.0894 | 0.0746 | 0.12 | 0.07 |
| 2 | `two_body_xx_zz_11q` | `two_body_pauli [XX,ZZ]`, 11 qubits | 1.4096 | 1.3968 | **0.0091** | 0.0814 | 0.0639 | 0.09 | 0.04 |
| 3 | `fixed_xz_10q` | `fixed_pauli [X,Z]`, 10 qubits | 1.3702 | 1.3644 | **0.0043** | 0.0795 | 0.0746 | 0.05 | 0.07 |
| 4 | `mixed_xz_xx_7q` | `mixed_pauli [X,Z]+[XX]`, 7 qubits | 1.1792 | 1.4528 | **-0.2086** | 0.1018 | 0.0655 | 0.08 | 0.05 |

## Metric definition

For each QGAN/surrogate pair:

```text
β = sqrt(total_parseval_norm_sq)
Δ_sep = log(β_QGAN / (β_surrogate + ε))
ε = 1e-12
```

`Δ_sep > 0` means the QGAN has a larger EVS signal norm than the matched Fourier/RFF surrogate. `Δ_sep < 0` means the surrogate has the larger EVS signal norm.

## Interpretation

The best condition was `local_xy_10q`, but its separation is only:

```text
Δ_sep = 0.0172 ≈ log(1.017×)
```

This is a tiny positive separation, not strong evidence of dequantization-resistant quantum signal. The two-body bank is also positive but smaller. The baseline is essentially matched. The mixed 7-qubit bank is poor: the surrogate substantially exceeds the QGAN in EVS norm.

Current conclusion:

> Under the current frozen encoder, optimizer settings, training budget, and single seed, the tested richer observable banks do not produce meaningful dequantization-aware separation.

## Operational note

The first run stopped after completing `fixed_xz_10q/qgan` because the ad-hoc runner used `qgan_record[run_id]` instead of `qgan_record["run_id"]` when printing the surrogate handoff. The run was resumed after patching the runner to reuse completed `run_record.json` artifacts. The completed QGAN row was not rerun.

This motivated committing a hardened resumable Stage 8 runner and aggregation CLI.
