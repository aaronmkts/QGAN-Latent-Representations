from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from qgan_latent.workflows.qgan_expectation_values.training.registry import (
    RunRecord,
    RunRegistry,
    current_git_state,
)


REQUIRED_KEYS = {
    "run_id",
    "git_commit",
    "git_dirty",
    "workflow",
    "observable_bank",
    "resource_counts",
    "config_snapshot",
    "metrics_path",
    "diagnostics_path",
    "observable_bank_metadata_path",
    "sample_grid_paths",
    "checkpoint_paths",
    "status",
    "parent_run_id",
}


def _cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "batch_size": 8,
            "epochs": 1,
            "n_critic": 1,
            "model": {
                "autoencoder": {"latent_dim": 4},
                "quantum_generator": {"n_qubits": 2, "depth": 1, "noise_dim": 2},
            },
        }
    )


def test_run_record_schema_roundtrip(tmp_path: Path) -> None:
    record = RunRecord(
        run_id="test_run",
        git_commit="unknown",
        git_dirty=False,
        workflow="qgan_expectation_values",
        observable_bank={"name": "fixed_pauli", "output_dim": 4},
        resource_counts={"n_qubits": 2},
        config_snapshot={"seed": 42},
        metrics_path="metrics.json",
        diagnostics_path="evs_diagnostics.json",
        observable_bank_metadata_path="observable_bank_metadata.json",
        sample_grid_paths=["samples.png"],
        checkpoint_paths={"generator": "qgan_gen.ckpt"},
        status="completed",
        parent_run_id=None,
    )
    registry = RunRegistry(tmp_path, run_id="test_run", workflow="qgan_expectation_values")
    path = registry.write(record)

    assert path == tmp_path / "run_record.json"
    loaded = registry.load()
    assert loaded == record
    assert set(json.loads(path.read_text(encoding="utf-8"))) == REQUIRED_KEYS


def test_registry_captures_git_commit_and_dirty_state(tmp_path: Path) -> None:
    commit, dirty = current_git_state(repo_root=Path(__file__).resolve().parents[1])
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert isinstance(dirty, bool)

    expected_dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    ).returncode != 0
    assert dirty == expected_dirty


def test_registry_start_and_finalize_update_status(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path, run_id="run-1", workflow="qgan_expectation_values")
    started = registry.start(
        _cfg(),
        observable_bank_metadata={"name": "fixed_pauli", "output_dim": 4, "n_qubits": 2},
        resource_counts={"n_qubits": 2, "steps_per_epoch": 1, "gen_lr": 0.001},
    )
    assert started.status == "started"
    assert started.workflow == "qgan_expectation_values"
    assert started.resource_counts["gen_lr"] == 0.001

    completed = registry.finalize(
        metrics_path=tmp_path / "metrics.json",
        diagnostics_path=tmp_path / "evs_diagnostics.json",
        observable_bank_metadata_path=tmp_path / "observable_bank_metadata.json",
        sample_grid_paths=[tmp_path / "samples.png"],
        checkpoint_paths={"generator": tmp_path / "qgan_gen.ckpt"},
    )

    assert completed.status == "completed"
    assert completed.metrics_path.endswith("metrics.json")
    assert completed.diagnostics_path.endswith("evs_diagnostics.json")
    assert completed.sample_grid_paths == [str(tmp_path / "samples.png")]
    assert completed.checkpoint_paths["generator"].endswith("qgan_gen.ckpt")
    assert registry.load() == completed


def test_registry_parent_run_id_links_surrogate_to_qgan(tmp_path: Path) -> None:
    qgan = RunRegistry(tmp_path / "qgan", run_id="qgan-1", workflow="qgan_expectation_values")
    qgan.start(_cfg(), {"name": "fixed_pauli"}, {"n_qubits": 2})
    qgan_record = qgan.finalize(
        metrics_path=tmp_path / "qgan" / "metrics.json",
        diagnostics_path=tmp_path / "qgan" / "evs_diagnostics.json",
        observable_bank_metadata_path=tmp_path / "qgan" / "observable_bank_metadata.json",
        sample_grid_paths=[],
        checkpoint_paths={},
    )

    surrogate = RunRegistry(tmp_path / "surrogate", run_id="surrogate-1", workflow="fourier_surrogate")
    surrogate.start(_cfg(), {"name": "fixed_pauli"}, {"n_qubits": 2})
    surrogate_record = surrogate.finalize(
        metrics_path=tmp_path / "surrogate" / "metrics.json",
        diagnostics_path=tmp_path / "surrogate" / "evs_diagnostics.json",
        observable_bank_metadata_path=tmp_path / "surrogate" / "observable_bank_metadata.json",
        sample_grid_paths=[],
        checkpoint_paths={},
        parent_run_id=qgan_record.run_id,
    )

    assert surrogate_record.workflow == "fourier_surrogate"
    assert surrogate_record.parent_run_id == "qgan-1"
