# Research Memory Boundary

This repository should describe the current runnable QGAN latent-representation testbed: package layout, commands, configuration, tests, and reproducibility instructions.

Durable research memory belongs outside the repository unless it directly supports reproduction or implementation.

## Keep in this repository

- Current command-line entry points and smoke-test instructions.
- Current package and configuration structure.
- Reproducibility notes required to run the code.
- Minimal design notes that explain why code is structured the way it is.
- Test, environment, and artifact conventions needed by future contributors.

## Keep in ORION / Obsidian research memory

- Stage history and abandoned implementation plans.
- Speculative research directions.
- Literature synthesis and paper-positioning notes.
- Experiment interpretation that is not needed to reproduce a run.
- Long-form rationale, reviewer-risk analysis, and cross-project research links.

## Stale-branch cleanup note

The former `orion/stage1-research-memory-boundary` branch captured a useful boundary idea, but its implementation notes predated the Stage 4 namespace and CLI refactor. Do not restore obsolete root scripts such as `pretrain.py`, `train.py`, or `train_prior.py`, and do not restore compatibility namespaces such as `qgan_latent.datamodules`, `qgan_latent.models`, `qgan_latent.training`, or `qgan_latent.utils`.

Current repo docs should follow `main`: explicit workflow CLIs, `qgan_latent.shared`, and `qgan_latent.workflows.*` namespaces.
