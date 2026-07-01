from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.training.comparison import (
    comparison_payload,
    compute_delta_sep,
    write_comparison_json,
)
from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import run_gan
from qgan_latent.workflows.qgan_expectation_values.training.surrogate_loop import run_surrogate_gan


def _compose_train(overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="projects/qgan_expectation_values/train", overrides=overrides)


def _smoke(output_dir: str) -> list[str]:
    return [
        "smoke_test=true",
        "device=cpu",
        "wandb_mode=disabled",
        "metrics.active_metrics=[]",
        f"outputs.dir={output_dir}",
        "checkpoints.dir=checkpoints",
        "model.autoencoder.latent_dim=4",
        "model.autoencoder.encoder_channels=[4,8]",
        "model.autoencoder.decoder_channels=[8,4]",
        "model.autoencoder.mlp_dim=16",
        "model.quantum_generator.n_qubits=2",
        "model.quantum_generator.noise_dim=2",
        "model.quantum_generator.depth=1",
        "surrogate.hidden_dim=8",
    ]


def test_compute_delta_sep_positive_when_qgan_larger() -> None:
    assert compute_delta_sep(2.0, 1.0) == math.log(2.0 / (1.0 + 1e-12))


def test_compute_delta_sep_negative_when_classical_larger() -> None:
    assert compute_delta_sep(1.0, 2.0) == math.log(1.0 / (2.0 + 1e-12))


def test_compute_delta_sep_clips_at_eps() -> None:
    assert compute_delta_sep(1.0, 0.0, eps=1e-12) == math.log(1.0 / 1e-12)


def test_compute_delta_sep_uses_plus_eps_for_small_nonzero_classical_beta() -> None:
    assert compute_delta_sep(1.0, 1e-13, eps=1e-12) == math.log(1.0 / (1e-13 + 1e-12))


def _write_diagnostics(path: Path, total_parseval_norm_sq: float) -> None:
    path.write_text(
        json.dumps([{"total_parseval_norm_sq": total_parseval_norm_sq}]),
        encoding="utf-8",
    )


def _record(tmp_path: Path, *, run_id: str, workflow: str, diagnostics_name: str, parent_run_id: str | None = None) -> dict:
    diagnostics_path = tmp_path / diagnostics_name
    _write_diagnostics(diagnostics_path, 4.0 if workflow == "qgan_expectation_values" else 1.0)
    return {
        "run_id": run_id,
        "workflow": workflow,
        "parent_run_id": parent_run_id,
        "diagnostics_path": str(diagnostics_path),
        "observable_bank": {"output_dim": 4},
        "resource_counts": {
            "n_qubits": 2,
            "circuit_depth": 1,
            "noise_dim": 2,
            "observable_bank_output_dim": 4,
            "epochs": 1,
            "batch_size": 8,
            "steps_per_epoch": 1,
            "n_critic": 5,
            "gen_lr": 0.001,
            "disc_lr": 0.001,
            "lambda_gp": 10.0,
            "surrogate_rff_dim": 4,
        },
    }


def test_comparison_payload_records_validated_match(tmp_path: Path) -> None:
    qgan = _record(tmp_path, run_id="qgan-1", workflow="qgan_expectation_values", diagnostics_name="qgan_diag.json")
    surrogate = _record(
        tmp_path,
        run_id="surrogate-1",
        workflow="fourier_surrogate",
        diagnostics_name="surrogate_diag.json",
        parent_run_id="qgan-1",
    )

    payload = comparison_payload(qgan, surrogate)

    assert payload["validated_match"]["qgan_run_id"] == "qgan-1"
    assert payload["validated_match"]["surrogate_rff_dim"] == 4
    assert payload["delta_sep"] == math.log(2.0 / (1.0 + 1e-12))


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("parent_run_id", "wrong-parent", "parent_run_id"),
        ("noise_dim", 999, "noise_dim"),
        ("observable_bank_output_dim", 999, "observable_bank_output_dim"),
        ("epochs", 10, "epochs"),
        ("batch_size", 16, "batch_size"),
        ("n_critic", 1, "n_critic"),
        ("gen_lr", 0.123, "gen_lr"),
        ("surrogate_rff_dim", 999, "surrogate_rff_dim"),
    ],
)
def test_comparison_payload_rejects_unmatched_records(
    tmp_path: Path, field: str, value, expected_message: str
) -> None:
    qgan = _record(tmp_path, run_id="qgan-1", workflow="qgan_expectation_values", diagnostics_name="qgan_diag.json")
    surrogate = _record(
        tmp_path,
        run_id="surrogate-1",
        workflow="fourier_surrogate",
        diagnostics_name="surrogate_diag.json",
        parent_run_id="qgan-1",
    )
    if field == "parent_run_id":
        surrogate["parent_run_id"] = value
    else:
        surrogate["resource_counts"][field] = value

    with pytest.raises(ValueError, match=expected_message):
        comparison_payload(qgan, surrogate)


def test_delta_sep_from_smoke_qgan_and_surrogate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    qgan_cfg = _compose_train(_smoke("outputs/qgan_smoke"))
    run_gan(qgan_cfg)
    qgan_record = json.loads((tmp_path / "outputs" / "qgan_smoke" / "run_record.json").read_text(encoding="utf-8"))

    surrogate_cfg = _compose_train(_smoke("outputs/surrogate_smoke") + ["surrogate.enabled=true"])
    run_surrogate_gan(surrogate_cfg, matched_run_id=qgan_record["run_id"])

    comparison_path = write_comparison_json(
        qgan_record_path=tmp_path / "outputs" / "qgan_smoke" / "run_record.json",
        surrogate_record_path=tmp_path / "outputs" / "surrogate_smoke" / "run_record.json",
        output_path=tmp_path / "outputs" / "comparison.json",
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert math.isfinite(comparison["delta_sep"])
    assert comparison["qgan_run_id"] == qgan_record["run_id"]
    assert comparison["surrogate_run_id"]
    assert comparison["matched_policy"]["spectrum"] == "rff_dim == depth * n_qubits * 2"
