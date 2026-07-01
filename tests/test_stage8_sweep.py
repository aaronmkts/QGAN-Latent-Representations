from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from qgan_latent.workflows.qgan_expectation_values.training.stage8 import (
    STAGE8_CONDITIONS,
    Stage8Paths,
    Stage8SweepConfig,
    aggregate_stage8_results,
    assert_preflight_valid,
    preflight_conditions,
    run_stage8_condition,
    write_stage8_summary,
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _diagnostics(path: Path, total: float) -> Path:
    return _write_json(path, [{"total_parseval_norm_sq": total}])


def _metrics(path: Path, js: float = 0.1, ndb_k: float = 0.05) -> Path:
    return _write_json(
        path,
        [
            {
                "epoch": 0,
                "step": 1,
                "val/JS": js,
                "val/NDB_K": ndb_k,
                "evs/total_parseval_norm_sq": 1.0,
            }
        ],
    )


def _record(
    root: Path,
    *,
    run_id: str,
    workflow: str,
    parent_run_id: str | None,
    total_parseval_norm_sq: float,
    output_dim: int = 4,
) -> dict[str, Any]:
    diagnostics = _diagnostics(root / "evs_diagnostics.json", total_parseval_norm_sq)
    metrics = _metrics(root / "metrics.json")
    is_surrogate = workflow == "fourier_surrogate"
    return {
        "run_id": run_id,
        "git_commit": "unknown",
        "git_dirty": False,
        "workflow": workflow,
        "observable_bank": {
            "name": "fixed_pauli",
            "output_dim": output_dim,
            "n_qubits": 2,
            "num_terms": output_dim,
            "max_locality": 1,
            "measurement_groups_estimate": 2,
            "terms": [f"X_{i}" for i in range(output_dim)],
            "topology": "local",
        },
        "resource_counts": {
            "n_qubits": 2,
            "circuit_depth": 1,
            "noise_dim": 2,
            "observable_bank_output_dim": output_dim,
            "epochs": 1,
            "batch_size": 8,
            "steps_per_epoch": 1,
            "n_critic": 1,
            "gen_lr": 0.001,
            "disc_lr": 0.001,
            "lambda_gp": 10.0,
            **({"surrogate_rff_dim": 4} if is_surrogate else {}),
        },
        "config_snapshot": {},
        "metrics_path": str(metrics),
        "diagnostics_path": str(diagnostics),
        "observable_bank_metadata_path": str(root / "observable_bank_metadata.json"),
        "sample_grid_paths": [],
        "checkpoint_paths": {},
        "status": "completed",
        "parent_run_id": parent_run_id,
    }


def _condition() -> dict[str, Any]:
    return {
        "name": "fixed_xz_2q",
        "label": "fixed_pauli [X,Z], 2 qubits",
        "overrides": [
            "smoke_test=true",
            "device=cpu",
            "wandb_mode=disabled",
            "metrics.active_metrics=[]",
            "model.autoencoder.latent_dim=4",
            "model.autoencoder.encoder_channels=[4,8]",
            "model.autoencoder.decoder_channels=[8,4]",
            "model.autoencoder.mlp_dim=16",
            "model.quantum_generator.n_qubits=2",
            "model.quantum_generator.noise_dim=2",
            "model.quantum_generator.depth=1",
        ],
    }


def _stage8_config(tmp_path: Path) -> Stage8SweepConfig:
    repo = Path(__file__).resolve().parents[1]
    return Stage8SweepConfig(
        repo_root=repo,
        output_root=tmp_path / "stage8",
        config_dir=repo / "configs",
        frozen_encoder=None,
        run_id="test-stage8",
        common_overrides=[],
        conditions=[_condition()],
        skip_completed=True,
    )


def test_completed_run_reuse_skips_qgan_and_surrogate(tmp_path: Path) -> None:
    cfg = _stage8_config(tmp_path)
    paths = Stage8Paths(cfg.output_root, "fixed_xz_2q")
    _write_json(paths.qgan_record_path, _record(paths.qgan_output_dir, run_id="qgan-1", workflow="qgan_expectation_values", parent_run_id=None, total_parseval_norm_sq=4.0))
    _write_json(paths.surrogate_record_path, _record(paths.surrogate_output_dir, run_id="surrogate-1", workflow="fourier_surrogate", parent_run_id="qgan-1", total_parseval_norm_sq=1.0))

    def fail_qgan(_cfg):
        raise AssertionError("QGAN should have been reused")

    def fail_surrogate(_cfg, *, matched_run_id: str | None = None):
        raise AssertionError("surrogate should have been reused")

    result = run_stage8_condition(_condition(), cfg, run_qgan=fail_qgan, run_surrogate=fail_surrogate)

    assert result["qgan_run_id"] == "qgan-1"
    assert result["surrogate_run_id"] == "surrogate-1"
    assert result["delta_sep"] == math.log(2.0 / (1.0 + 1e-12))
    assert paths.comparison_path.exists()


def test_missing_surrogate_resume_reuses_qgan_and_runs_only_surrogate(tmp_path: Path) -> None:
    cfg = _stage8_config(tmp_path)
    paths = Stage8Paths(cfg.output_root, "fixed_xz_2q")
    _write_json(paths.qgan_record_path, _record(paths.qgan_output_dir, run_id="qgan-1", workflow="qgan_expectation_values", parent_run_id=None, total_parseval_norm_sq=4.0))
    calls: list[str] = []

    def fail_qgan(_cfg):
        raise AssertionError("QGAN should have been reused")

    def write_surrogate(run_cfg, *, matched_run_id: str | None = None):
        calls.append(str(matched_run_id))
        out = Path(str(run_cfg.outputs.dir))
        _write_json(out / "run_record.json", _record(out, run_id="surrogate-1", workflow="fourier_surrogate", parent_run_id=matched_run_id, total_parseval_norm_sq=1.0))

    result = run_stage8_condition(_condition(), cfg, run_qgan=fail_qgan, run_surrogate=write_surrogate)

    assert calls == ["qgan-1"]
    assert result["surrogate_run_id"] == "surrogate-1"
    assert paths.surrogate_record_path.exists()


def test_aggregate_stage8_results_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "stage8"
    for name, q_total, s_total in [
        ("fixed_xz_10q", 4.0, 1.0),
        ("local_xy_10q", 9.0, 4.0),
    ]:
        paths = Stage8Paths(root, name)
        _write_json(paths.qgan_record_path, _record(paths.qgan_output_dir, run_id=f"{name}-q", workflow="qgan_expectation_values", parent_run_id=None, total_parseval_norm_sq=q_total))
        _write_json(paths.surrogate_record_path, _record(paths.surrogate_output_dir, run_id=f"{name}-s", workflow="fourier_surrogate", parent_run_id=f"{name}-q", total_parseval_norm_sq=s_total))

    summary = aggregate_stage8_results(root, run_id="aggregate-test")
    json_path, csv_path, md_path = write_stage8_summary(summary, root)

    assert len(summary["results"]) == 2
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == "aggregate-test"
    assert "delta_sep" in csv_path.read_text(encoding="utf-8")
    assert "Stage 8 Summary" in md_path.read_text(encoding="utf-8")


def test_latent_dimension_preflight_rejects_mismatched_bank(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = Stage8SweepConfig(
        repo_root=repo,
        output_root=tmp_path / "stage8",
        config_dir=repo / "configs",
        frozen_encoder=None,
        run_id="preflight-test",
        common_overrides=[
            "device=cpu",
            "smoke_test=true",
            "model.autoencoder.latent_dim=20",
        ],
        conditions=[
            {
                "name": "bad_local_xyz",
                "label": "bad local XYZ",
                "overrides": [
                    "model.quantum_generator.n_qubits=10",
                    "model.quantum_generator.observable_bank.name=local_pauli",
                    "model.quantum_generator.observable_bank.paulis=[X,Y,Z]",
                ],
            }
        ],
    )

    rows = preflight_conditions(cfg)

    assert rows[0].latent_dim == 20
    assert rows[0].output_dim == 30
    assert rows[0].valid is False
    with pytest.raises(ValueError, match="latent_dim 20 != observable_bank.output_dim 30"):
        assert_preflight_valid(rows)


def test_default_stage8_conditions_preflight_match_latent_dim(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = Stage8SweepConfig(
        repo_root=repo,
        output_root=tmp_path / "stage8",
        config_dir=repo / "configs",
        frozen_encoder=None,
        run_id="preflight-ok",
        common_overrides=["device=cpu", "smoke_test=true"],
        conditions=[dict(condition) for condition in STAGE8_CONDITIONS],
    )

    rows = preflight_conditions(cfg)

    assert len(rows) == 4
    assert {row.output_dim for row in rows} == {20}
    assert all(row.valid for row in rows)
