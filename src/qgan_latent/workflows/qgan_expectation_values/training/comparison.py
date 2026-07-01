from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from qgan_latent.workflows.qgan_expectation_values.training.diagnostics import parseval_norm_from_diagnostics


MATCHED_POLICY = {
    "spectrum": "rff_dim == depth * n_qubits * 2",
    "feature_dim": "output_dim == observable_bank.output_dim",
    "noise_dim": "same",
    "discriminator": "same architecture and initialization seed",
    "representation": "same frozen autoencoder checkpoint",
    "epochs": "same",
    "batch_size": "same",
    "n_critic": "same",
    "learning_rates": "same",
}


def compute_delta_sep(beta_q: float, beta_cls: float, eps: float = 1e-12) -> float:
    denom = float(beta_cls) + float(eps)
    return float(math.log(float(beta_q) / denom))


def _load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _last_diagnostics(path: Path | str) -> Mapping[str, Any]:
    diagnostics = _load_json(path)
    if isinstance(diagnostics, list):
        if not diagnostics:
            raise ValueError(f"No diagnostics records in {path}.")
        diagnostics = diagnostics[-1]
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"Diagnostics at {path} must be a mapping or non-empty list.")
    return diagnostics


def beta_from_diagnostics_path(path: Path | str) -> float:
    return math.sqrt(max(parseval_norm_from_diagnostics(_last_diagnostics(path)), 0.0))


def _resource_counts(record: Mapping[str, Any]) -> Mapping[str, Any]:
    resource_counts = record.get("resource_counts", {})
    if not isinstance(resource_counts, Mapping):
        raise ValueError(f"Record {record.get('run_id', '<unknown>')} has invalid resource_counts.")
    return resource_counts


def _require_equal(
    qgan_counts: Mapping[str, Any],
    surrogate_counts: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> None:
    if qgan_counts.get(key) != surrogate_counts.get(key):
        errors.append(
            f"resource_counts.{key} mismatch: "
            f"qgan={qgan_counts.get(key)!r}, surrogate={surrogate_counts.get(key)!r}"
        )


def validate_matched_records(
    qgan_record: Mapping[str, Any],
    surrogate_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the Stage 7 QGAN/surrogate matching contract."""

    errors: list[str] = []
    if qgan_record.get("workflow") != "qgan_expectation_values":
        errors.append(
            f"qgan workflow must be 'qgan_expectation_values', got {qgan_record.get('workflow')!r}"
        )
    if surrogate_record.get("workflow") != "fourier_surrogate":
        errors.append(
            f"surrogate workflow must be 'fourier_surrogate', got {surrogate_record.get('workflow')!r}"
        )
    if surrogate_record.get("parent_run_id") != qgan_record.get("run_id"):
        errors.append(
            "surrogate parent_run_id must equal qgan run_id: "
            f"parent_run_id={surrogate_record.get('parent_run_id')!r}, "
            f"qgan_run_id={qgan_record.get('run_id')!r}"
        )

    qgan_counts = _resource_counts(qgan_record)
    surrogate_counts = _resource_counts(surrogate_record)
    for key in (
        "noise_dim",
        "observable_bank_output_dim",
        "epochs",
        "batch_size",
        "steps_per_epoch",
        "n_critic",
        "gen_lr",
        "disc_lr",
        "lambda_gp",
    ):
        _require_equal(qgan_counts, surrogate_counts, key, errors)

    expected_rff_dim = None
    if "circuit_depth" in qgan_counts and "n_qubits" in qgan_counts:
        expected_rff_dim = int(qgan_counts["circuit_depth"]) * int(qgan_counts["n_qubits"]) * 2
        if surrogate_counts.get("surrogate_rff_dim") != expected_rff_dim:
            errors.append(
                "surrogate_rff_dim must equal circuit_depth * n_qubits * 2: "
                f"expected={expected_rff_dim!r}, got={surrogate_counts.get('surrogate_rff_dim')!r}"
            )

    qgan_bank = qgan_record.get("observable_bank", {})
    surrogate_bank = surrogate_record.get("observable_bank", {})
    if isinstance(qgan_bank, Mapping) and isinstance(surrogate_bank, Mapping):
        if qgan_bank.get("output_dim") != surrogate_bank.get("output_dim"):
            errors.append(
                f"observable_bank.output_dim mismatch: qgan={qgan_bank.get('output_dim')!r}, "
                f"surrogate={surrogate_bank.get('output_dim')!r}"
            )

    if errors:
        raise ValueError("Cannot compare unmatched QGAN/surrogate records: " + "; ".join(errors))

    return {
        "qgan_run_id": qgan_record.get("run_id"),
        "surrogate_run_id": surrogate_record.get("run_id"),
        "parent_run_id": surrogate_record.get("parent_run_id"),
        "noise_dim": qgan_counts.get("noise_dim"),
        "observable_bank_output_dim": qgan_counts.get("observable_bank_output_dim"),
        "epochs": qgan_counts.get("epochs"),
        "batch_size": qgan_counts.get("batch_size"),
        "steps_per_epoch": qgan_counts.get("steps_per_epoch"),
        "n_critic": qgan_counts.get("n_critic"),
        "gen_lr": qgan_counts.get("gen_lr"),
        "disc_lr": qgan_counts.get("disc_lr"),
        "lambda_gp": qgan_counts.get("lambda_gp"),
        "expected_rff_dim": expected_rff_dim,
        "surrogate_rff_dim": surrogate_counts.get("surrogate_rff_dim"),
    }


def comparison_payload(qgan_record: Mapping[str, Any], surrogate_record: Mapping[str, Any], *, eps: float = 1e-12) -> dict[str, Any]:
    validated_match = validate_matched_records(qgan_record, surrogate_record)
    beta_q = beta_from_diagnostics_path(qgan_record["diagnostics_path"])
    beta_cls = beta_from_diagnostics_path(surrogate_record["diagnostics_path"])
    return {
        "qgan_run_id": qgan_record["run_id"],
        "surrogate_run_id": surrogate_record["run_id"],
        "matched_policy": dict(MATCHED_POLICY),
        "validated_match": validated_match,
        "beta_q": beta_q,
        "beta_cls": beta_cls,
        "eps": float(eps),
        "delta_sep": compute_delta_sep(beta_q, beta_cls, eps=eps),
    }


def write_comparison_json(
    *,
    qgan_record_path: Path | str,
    surrogate_record_path: Path | str,
    output_path: Path | str,
    eps: float = 1e-12,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = comparison_payload(_load_json(qgan_record_path), _load_json(surrogate_record_path), eps=eps)
    tmp_path = output.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(output)
    return output
