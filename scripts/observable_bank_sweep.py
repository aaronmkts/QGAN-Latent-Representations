
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import run_gan
from qgan_latent.workflows.qgan_expectation_values.training.surrogate_loop import run_surrogate_gan
from qgan_latent.workflows.qgan_expectation_values.training.comparison import write_comparison_json

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
FROZEN_ENCODER = REPO / "outputs/experiments/baseline_mnist_last_20260701_124132/checkpoints/autoencoder.ckpt"
RUN_ID = (
    os.environ.get("OBSERVABLE_BANK_SWEEP_RUN_ID")
    or os.environ.get("STAGE8_RUN_ID")  # Backward-compatible with existing Stage 8 resumes.
    or datetime.utcnow().strftime("observable_bank_sweep_%Y%m%d_%H%M%S")
)
OUT_ROOT = REPO / "outputs/experiments" / RUN_ID

COMMON = [
    "device=GPU",
    "wandb_mode=disabled",
    "smoke_test=false",
    "seed=42",
    "batch_size=128",
    "epochs=100",
    "n_critic=5",
    "gen_lr=1e-3",
    "disc_lr=1e-3",
    "lambda_gp=10.0",
    "sample_every=200",
    "eval_epochs=10",
    "log_every=50",
    f"checkpoints.autoencoder={FROZEN_ENCODER}",
]

CONDITIONS = [
    {
        "name": "fixed_xz_10q",
        "label": "fixed_pauli [X,Z], 10 qubits",
        "overrides": [
            "model.quantum_generator.n_qubits=10",
            "model.quantum_generator.noise_dim=10",
            "model.quantum_generator.depth=4",
            "model.quantum_generator.observable_bank.name=fixed_pauli",
            "model.quantum_generator.observable_bank.paulis=[X,Z]",
            "model.quantum_generator.observable_bank.two_body_paulis=[XX,ZZ]",
        ],
    },
    {
        "name": "local_xy_10q",
        "label": "local_pauli [X,Y], 10 qubits",
        "overrides": [
            "model.quantum_generator.n_qubits=10",
            "model.quantum_generator.noise_dim=10",
            "model.quantum_generator.depth=4",
            "model.quantum_generator.observable_bank.name=local_pauli",
            "model.quantum_generator.observable_bank.paulis=[X,Y]",
            "model.quantum_generator.observable_bank.two_body_paulis=[XX,ZZ]",
        ],
    },
    {
        "name": "two_body_xx_zz_11q",
        "label": "two_body_pauli [XX,ZZ], 11 qubits",
        "overrides": [
            "model.quantum_generator.n_qubits=11",
            "model.quantum_generator.noise_dim=10",
            "model.quantum_generator.depth=4",
            "model.quantum_generator.observable_bank.name=two_body_pauli",
            "model.quantum_generator.observable_bank.paulis=[X,Z]",
            "model.quantum_generator.observable_bank.two_body_paulis=[XX,ZZ]",
            "model.quantum_generator.observable_bank.topology=line",
        ],
    },
    {
        "name": "mixed_xz_xx_7q",
        "label": "mixed_pauli [X,Z]+[XX], 7 qubits",
        "overrides": [
            "model.quantum_generator.n_qubits=7",
            "model.quantum_generator.noise_dim=10",
            "model.quantum_generator.depth=4",
            "model.quantum_generator.observable_bank.name=mixed_pauli",
            "model.quantum_generator.observable_bank.paulis=[X,Z]",
            "model.quantum_generator.observable_bank.two_body_paulis=[XX]",
            "model.quantum_generator.observable_bank.topology=line",
        ],
    },
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def last_metric(metrics_path: Path) -> dict:
    data = load_json(metrics_path)
    if isinstance(data, list):
        return data[-1] if data else {}
    return data


def compose_cfg(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="projects/qgan_expectation_values/train", overrides=overrides)


def _record_completed(path: Path, workflow: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        record = load_json(path)
    except Exception:
        return False
    if record.get("status") != "completed":
        return False
    if workflow is not None and record.get("workflow") != workflow:
        return False
    return True


def run_condition(condition: dict) -> dict:
    name = condition["name"]
    label = condition["label"]
    condition_root = OUT_ROOT / name
    qgan_out = condition_root / "qgan"
    surrogate_out = condition_root / "surrogate"
    comparison_path = condition_root / "comparison.json"
    qgan_record_path = qgan_out / "run_record.json"
    surrogate_record_path = surrogate_out / "run_record.json"
    qgan_ckpt_dir = qgan_out / "checkpoints"
    surrogate_ckpt_dir = surrogate_out / "checkpoints"

    qgan_overrides = COMMON + condition["overrides"] + [
        f"outputs.dir={qgan_out}",
        f"checkpoints.dir={qgan_ckpt_dir}",
    ]
    if _record_completed(qgan_record_path, "qgan_expectation_values"):
        print(f"=== REUSE QGAN {name}: {qgan_record_path} ===", flush=True)
    else:
        print(f"=== QGAN {name}: {label} ===", flush=True)
        qgan_cfg = compose_cfg(qgan_overrides)
        run_gan(qgan_cfg)
    qgan_record = load_json(qgan_record_path)

    surrogate_overrides = COMMON + condition["overrides"] + [
        "surrogate.enabled=true",
        "surrogate.rff_dim=auto",
        "surrogate.sigma=1.0",
        "surrogate.hidden_dim=64",
        f"outputs.dir={surrogate_out}",
        f"checkpoints.dir={surrogate_ckpt_dir}",
    ]
    if _record_completed(surrogate_record_path, "fourier_surrogate"):
        print(f"=== REUSE SURROGATE {name}: {surrogate_record_path} ===", flush=True)
    else:
        print(f"=== SURROGATE {name}: matched to {qgan_record['run_id']} ===", flush=True)
        surrogate_cfg = compose_cfg(surrogate_overrides)
        run_surrogate_gan(surrogate_cfg, matched_run_id=qgan_record["run_id"])
    surrogate_record = load_json(surrogate_record_path)

    write_comparison_json(
        qgan_record_path=qgan_record_path,
        surrogate_record_path=surrogate_record_path,
        output_path=comparison_path,
    )
    comparison = load_json(comparison_path)
    qgan_metrics = last_metric(Path(qgan_record["metrics_path"]))
    surrogate_metrics = last_metric(Path(surrogate_record["metrics_path"]))
    result = {
        "condition": name,
        "label": label,
        "qgan_run_id": qgan_record["run_id"],
        "surrogate_run_id": surrogate_record["run_id"],
        "qgan_record_path": str(qgan_record_path),
        "surrogate_record_path": str(surrogate_record_path),
        "comparison_path": str(comparison_path),
        "qgan_beta": comparison["beta_q"],
        "surrogate_beta": comparison["beta_cls"],
        "delta_sep": comparison["delta_sep"],
        "validated_match": comparison["validated_match"],
        "observable_bank": qgan_record["observable_bank"],
        "qgan_resource_counts": qgan_record["resource_counts"],
        "surrogate_resource_counts": surrogate_record["resource_counts"],
        "qgan_last_metrics": qgan_metrics,
        "surrogate_last_metrics": surrogate_metrics,
    }
    (condition_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"=== DONE {name}: Delta_sep={result['delta_sep']:.6f} ===", flush=True)
    return result

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not FROZEN_ENCODER.exists():
        raise FileNotFoundError(f"Frozen encoder checkpoint not found: {FROZEN_ENCODER}")
    print(json.dumps({
        "run_id": RUN_ID,
        "out_root": str(OUT_ROOT),
        "frozen_encoder": str(FROZEN_ENCODER),
        "common": COMMON,
        "conditions": CONDITIONS,
    }, indent=2), flush=True)

    results = []
    for condition in CONDITIONS:
        results.append(run_condition(condition))
        (OUT_ROOT / "partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_path = OUT_ROOT / "observable_bank_sweep_summary.json"
    summary_path.write_text(json.dumps({"run_id": RUN_ID, "out_root": str(OUT_ROOT), "results": results}, indent=2), encoding="utf-8")

    csv_path = OUT_ROOT / "observable_bank_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "label", "qgan_beta", "surrogate_beta", "delta_sep",
            "n_qubits", "output_dim", "num_terms", "max_locality",
            "measurement_groups_estimate", "qgan_run_id", "surrogate_run_id", "comparison_path"
        ])
        writer.writeheader()
        for r in results:
            bank = r["observable_bank"]
            writer.writerow({
                "condition": r["condition"],
                "label": r["label"],
                "qgan_beta": r["qgan_beta"],
                "surrogate_beta": r["surrogate_beta"],
                "delta_sep": r["delta_sep"],
                "n_qubits": bank.get("n_qubits"),
                "output_dim": bank.get("output_dim"),
                "num_terms": bank.get("num_terms"),
                "max_locality": bank.get("max_locality"),
                "measurement_groups_estimate": bank.get("measurement_groups_estimate"),
                "qgan_run_id": r["qgan_run_id"],
                "surrogate_run_id": r["surrogate_run_id"],
                "comparison_path": r["comparison_path"],
            })
    print(f"OBSERVABLE_BANK_SWEEP_SUMMARY_JSON={summary_path}", flush=True)
    print(f"OBSERVABLE_BANK_SWEEP_SUMMARY_CSV={csv_path}", flush=True)


if __name__ == "__main__":
    main()
