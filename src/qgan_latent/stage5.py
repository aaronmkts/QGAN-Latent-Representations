from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

LATENT_QGAN_REPRESENTATIONS = ("autoencoder", "vae", "sinkhorn_ae")
"""Latent-QGAN representation variants.

Vector-quantized variants are intentionally excluded: the latent-QGAN training
loop is designed around continuous Autoencoder-style encode/decode semantics.
VQ-VAE and Spatial VQ-VAE are reserved for the tensor-prior/MPS workflow.
"""

TENSOR_PRIOR_REPRESENTATION = "spatial_vqvae"
DEFAULT_OUTPUT_ROOT = Path("outputs/experiments/stage5_test_run")
DEFAULT_DEVICE = "gpu"
DEFAULT_OBSIDIAN_RELATIVE_PATH = Path(
    "Research/PhD/Experiments/QGAN Latent Representations - Stage 5 Test Run.md"
)
DEFAULT_OBSIDIAN_HANDOFF_DIRNAME = "obsidian_handoff"
DEFAULT_OBSIDIAN_HANDOFF_FILENAME = "QGAN Latent Representations - Stage 5 Test Run.md"


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    workflow: str
    representation: str
    phase: str
    command: tuple[str, ...]
    output_dir: Path
    log_path: Path
    expected_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class Stage5Plan:
    run_id: str
    epochs: int
    output_root: Path
    summary_path: Path
    runs: tuple[ExperimentRun, ...]


@dataclass(frozen=True)
class ExperimentStatus:
    run: ExperimentRun
    returncode: int
    log_path: Path
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _path_override(key: str, path: Path) -> str:
    return f"{key}={path.as_posix()}"


def _latent_pretrain_run(output_root: Path, representation: str, epochs: int) -> ExperimentRun:
    output_dir = output_root / "latent_qgan" / representation / "pretrain"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_name = f"{representation}.ckpt"
    command = (
        "qgan-latent-pretrain",
        f"shared/representations@model.autoencoder={representation}",
        f"pretrain_epochs={epochs}",
        f"device={DEFAULT_DEVICE}",
        "wandb_mode=disabled",
        _path_override("outputs.dir", output_dir),
        _path_override("checkpoints.dir", checkpoint_dir),
        f"checkpoints.autoencoder={checkpoint_name}",
    )
    return ExperimentRun(
        name=f"latent_{representation}_pretrain_{epochs}ep",
        workflow="latent_qgan",
        representation=representation,
        phase="pretrain",
        command=command,
        output_dir=output_dir,
        log_path=output_dir / "stage5.log",
        expected_outputs=(checkpoint_dir / checkpoint_name,),
    )


def _latent_qgan_run(output_root: Path, representation: str, epochs: int) -> ExperimentRun:
    output_dir = output_root / "latent_qgan" / representation / "qgan_train"
    pretrain_checkpoint_dir = output_root / "latent_qgan" / representation / "pretrain" / "checkpoints"
    train_checkpoint_dir = output_dir / "checkpoints"
    checkpoint_name = f"{representation}.ckpt"
    pretrained_checkpoint = pretrain_checkpoint_dir / checkpoint_name
    command = (
        "qgan-latent-train",
        f"shared/representations@model.autoencoder={representation}",
        f"epochs={epochs}",
        f"device={DEFAULT_DEVICE}",
        "wandb_mode=disabled",
        _path_override("outputs.dir", output_dir),
        _path_override("checkpoints.dir", train_checkpoint_dir),
        _path_override("checkpoints.autoencoder", pretrained_checkpoint),
    )
    return ExperimentRun(
        name=f"latent_{representation}_qgan_{epochs}ep",
        workflow="latent_qgan",
        representation=representation,
        phase="qgan_train",
        command=command,
        output_dir=output_dir,
        log_path=output_dir / "stage5.log",
        expected_outputs=(
            train_checkpoint_dir / "qgan_gen.ckpt",
            train_checkpoint_dir / "qgan_disc.ckpt",
        ),
    )


def _vqvae_pretrain_run(output_root: Path, epochs: int) -> ExperimentRun:
    output_dir = output_root / "tensor_prior_vqvae" / "spatial_vqvae_pretrain"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_name = "spatial_vqvae.ckpt"
    command = (
        "qgan-vqvae-pretrain",
        "shared/representations@model.autoencoder=spatial_vqvae",
        f"pretrain_epochs={epochs}",
        f"device={DEFAULT_DEVICE}",
        "wandb_mode=disabled",
        _path_override("outputs.dir", output_dir),
        _path_override("checkpoints.dir", checkpoint_dir),
        f"checkpoints.autoencoder={checkpoint_name}",
    )
    return ExperimentRun(
        name=f"spatial_vqvae_pretrain_{epochs}ep",
        workflow="tensor_prior_vqvae",
        representation=TENSOR_PRIOR_REPRESENTATION,
        phase="pretrain",
        command=command,
        output_dir=output_dir,
        log_path=output_dir / "stage5.log",
        expected_outputs=(checkpoint_dir / checkpoint_name,),
    )


def _mps_prior_run(output_root: Path, epochs: int) -> ExperimentRun:
    output_dir = output_root / "tensor_prior_vqvae" / "mps_prior_train"
    pretrain_checkpoint_dir = output_root / "tensor_prior_vqvae" / "spatial_vqvae_pretrain" / "checkpoints"
    train_checkpoint_dir = output_dir / "checkpoints"
    pretrained_vqvae_checkpoint = pretrain_checkpoint_dir / "spatial_vqvae.ckpt"
    command = (
        "qgan-vqvae-train-prior",
        "shared/representations@model.vqvae=spatial_vqvae",
        f"model.mps_prior.epochs={epochs}",
        f"device={DEFAULT_DEVICE}",
        "wandb_mode=disabled",
        _path_override("outputs.dir", output_dir),
        _path_override("checkpoints.dir", train_checkpoint_dir),
        _path_override("checkpoints.vqvae", pretrained_vqvae_checkpoint),
    )
    return ExperimentRun(
        name=f"mps_prior_train_{epochs}ep",
        workflow="tensor_prior_vqvae",
        representation=TENSOR_PRIOR_REPRESENTATION,
        phase="mps_prior_train",
        command=command,
        output_dir=output_dir,
        log_path=output_dir / "stage5.log",
        expected_outputs=(train_checkpoint_dir / "mps_prior.ckpt",),
    )


def build_stage5_plan(
    *,
    run_id: str | None = None,
    epochs: int = 5,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Stage5Plan:
    if epochs < 1:
        raise ValueError("Stage 5 test runs require at least one epoch.")
    resolved_run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    root = Path(output_root) / resolved_run_id

    runs: list[ExperimentRun] = []
    for representation in LATENT_QGAN_REPRESENTATIONS:
        runs.append(_latent_pretrain_run(root, representation, epochs))
        runs.append(_latent_qgan_run(root, representation, epochs))
    runs.append(_vqvae_pretrain_run(root, epochs))
    runs.append(_mps_prior_run(root, epochs))

    return Stage5Plan(
        run_id=resolved_run_id,
        epochs=epochs,
        output_root=root,
        summary_path=root / "stage5_test_run_summary.md",
        runs=tuple(runs),
    )


def render_command(command: Sequence[str]) -> str:
    return " ".join(command)


def render_summary_markdown(
    plan: Stage5Plan,
    statuses: Sequence[ExperimentStatus],
    *,
    git_commit: str | None = None,
) -> str:
    lines = [
        "# Stage 5 Test Run Summary",
        "",
        f"Run ID: {plan.run_id}",
        f"Epochs: {plan.epochs}",
        f"Output root: {plan.output_root.as_posix()}",
        f"Git commit: {git_commit or 'unknown'}",
        "",
        "## Scope",
        "",
        "- Latent-QGAN runs use only continuous latent representations: autoencoder, vae, sinkhorn_ae.",
        f"- Device override: {DEFAULT_DEVICE}.",
        "- VQ-VAE variants are excluded from latent-QGAN tests by design.",
        "- The MPS-prior workflow uses only Spatial VQ-VAE codebook outputs.",
        "",
        "## Results",
        "",
    ]
    for status in statuses:
        verdict = "PASS" if status.passed else f"FAIL({status.returncode})"
        lines.extend(
            [
                f"- {status.run.name}: {verdict}",
                f"  - workflow: {status.run.workflow}",
                f"  - representation: {status.run.representation}",
                f"  - output: {status.run.output_dir.as_posix()}",
                f"  - log: {status.log_path.as_posix()}",
                f"  - elapsed_seconds: {status.elapsed_seconds:.2f}",
            ]
        )
    lines.extend(["", "## Commands", ""])
    for run in plan.runs:
        lines.append(f"- {run.name}: `{render_command(run.command)}`")
    return "\n".join(lines) + "\n"


def render_obsidian_note(
    plan: Stage5Plan,
    statuses: Sequence[ExperimentStatus],
    *,
    git_commit: str | None = None,
) -> str:
    passed = sum(status.passed for status in statuses)
    total = len(statuses)
    lines = [
        "# QGAN Latent Representations - Stage 5 Test Run",
        "",
        "Related project: [[QGAN Latent Representations]]",
        "",
        "## Purpose",
        "",
        "Short execution test for Stage 5: verify representation pretraining, latent QGAN training, Spatial VQ-VAE pretraining, MPS-prior training, output routing, and checkpoint creation.",
        "",
        "## Run metadata",
        "",
        f"- Run ID: {plan.run_id}",
        f"- Epochs: {plan.epochs}",
        f"- Git commit: {git_commit or 'unknown'}",
        f"- Output root: {plan.output_root.as_posix()}",
        f"- Result: {passed}/{total} commands passed",
        "",
        "## Scope rule",
        "",
        "No VQ-VAE variants are used for latent-QGAN runs. Vector-quantized representations are reserved for the Spatial VQ-VAE plus MPS-prior workflow.",
        "",
        "## Results",
        "",
    ]
    for status in statuses:
        verdict = "PASS" if status.passed else f"FAIL({status.returncode})"
        lines.append(f"- {status.run.name}: {verdict} — {status.run.log_path.as_posix()}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are intentionally not interpreted in this run.",
            "- This note records wiring, output routing, and execution status only.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_obsidian_note(
    note: str,
    *,
    vault_path: Path | str | None = None,
    relative_path: Path | str = DEFAULT_OBSIDIAN_RELATIVE_PATH,
) -> Path | None:
    resolved_vault = Path(vault_path or os.environ.get("OBSIDIAN_VAULT_PATH", ""))
    if not str(resolved_vault):
        return None
    note_path = resolved_vault / Path(relative_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")
    return note_path

def write_obsidian_handoff(
    note: str,
    *,
    output_root: Path,
    filename: str = DEFAULT_OBSIDIAN_HANDOFF_FILENAME,
) -> Path:
    handoff_dir = output_root / DEFAULT_OBSIDIAN_HANDOFF_DIRNAME
    handoff_dir.mkdir(parents=True, exist_ok=True)
    note_path = handoff_dir / filename
    note_path.write_text(note, encoding="utf-8")
    return note_path


def render_summary_json(
    plan: Stage5Plan,
    statuses: Sequence[ExperimentStatus],
    *,
    git_commit: str | None = None,
) -> str:
    payload = {
        "run_id": plan.run_id,
        "epochs": plan.epochs,
        "output_root": plan.output_root.as_posix(),
        "git_commit": git_commit or "unknown",
        "device": DEFAULT_DEVICE,
        "results": [
            {
                "name": status.run.name,
                "workflow": status.run.workflow,
                "representation": status.run.representation,
                "phase": status.run.phase,
                "passed": status.passed,
                "returncode": status.returncode,
                "output_dir": status.run.output_dir.as_posix(),
                "log_path": status.log_path.as_posix(),
                "elapsed_seconds": round(status.elapsed_seconds, 2),
                "expected_outputs": [path.as_posix() for path in status.run.expected_outputs],
            }
            for status in statuses
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def run_experiment(run: ExperimentRun, *, dry_run: bool = False) -> ExperimentStatus:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if dry_run:
        run.log_path.write_text(render_command(run.command) + "\n", encoding="utf-8")
        return ExperimentStatus(run=run, returncode=0, log_path=run.log_path, elapsed_seconds=0.0)

    with run.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {render_command(run.command)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            run.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - start
    return ExperimentStatus(
        run=run,
        returncode=completed.returncode,
        log_path=run.log_path,
        elapsed_seconds=elapsed,
    )


def run_stage5_plan(plan: Stage5Plan, *, dry_run: bool = False) -> tuple[ExperimentStatus, ...]:
    statuses: list[ExperimentStatus] = []
    for run in plan.runs:
        status = run_experiment(run, dry_run=dry_run)
        statuses.append(status)
        if not status.passed:
            break
    return tuple(statuses)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 5 short experiment matrix.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Write commands/logs without executing training.")
    parser.add_argument("--write-obsidian", action="store_true", help="Write a compact Obsidian experiment note directly when the vault is locally accessible.")
    parser.add_argument("--obsidian-vault", type=Path, default=None)
    parser.add_argument("--obsidian-handoff", action="store_true", help="Write a compact Obsidian note artifact under the run output for ORION/VPS filing.")
    args = parser.parse_args(argv)

    plan = build_stage5_plan(run_id=args.run_id, epochs=args.epochs, output_root=args.output_root)
    statuses = run_stage5_plan(plan, dry_run=args.dry_run)
    commit = _git_commit()
    plan.output_root.mkdir(parents=True, exist_ok=True)
    plan.summary_path.write_text(
        render_summary_markdown(plan, statuses, git_commit=commit),
        encoding="utf-8",
    )
    (plan.output_root / "stage5_test_run_summary.json").write_text(
        render_summary_json(plan, statuses, git_commit=commit),
        encoding="utf-8",
    )
    obsidian_note = render_obsidian_note(plan, statuses, git_commit=commit)
    if args.write_obsidian:
        write_obsidian_note(
            obsidian_note,
            vault_path=args.obsidian_vault,
        )
    if args.obsidian_handoff:
        write_obsidian_handoff(obsidian_note, output_root=plan.output_root)
    return 0 if all(status.passed for status in statuses) and len(statuses) == len(plan.runs) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
