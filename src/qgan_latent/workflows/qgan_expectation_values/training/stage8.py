from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import build_observable_bank
from qgan_latent.workflows.qgan_expectation_values.training.comparison import write_comparison_json
from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import run_gan
from qgan_latent.workflows.qgan_expectation_values.training.surrogate_loop import run_surrogate_gan

QGAN_WORKFLOW = "qgan_expectation_values"
SURROGATE_WORKFLOW = "fourier_surrogate"
DEFAULT_RUN_PREFIX = "stage8_gpu_bank_sweep"

STAGE8_CONDITIONS: tuple[dict[str, Any], ...] = (
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
)


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(output)
    return output


def last_metric(metrics_path: Path | str) -> dict[str, Any]:
    data = load_json(metrics_path)
    if isinstance(data, list):
        return data[-1] if data else {}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Metrics at {metrics_path} must be a JSON list or object.")


def record_completed(path: Path | str, workflow: str | None = None) -> bool:
    path = Path(path)
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


@dataclass(frozen=True)
class Stage8Paths:
    root: Path
    condition: str

    @property
    def condition_root(self) -> Path:
        return self.root / self.condition

    @property
    def qgan_output_dir(self) -> Path:
        return self.condition_root / "qgan"

    @property
    def surrogate_output_dir(self) -> Path:
        return self.condition_root / "surrogate"

    @property
    def qgan_record_path(self) -> Path:
        return self.qgan_output_dir / "run_record.json"

    @property
    def surrogate_record_path(self) -> Path:
        return self.surrogate_output_dir / "run_record.json"

    @property
    def comparison_path(self) -> Path:
        return self.condition_root / "comparison.json"

    @property
    def summary_path(self) -> Path:
        return self.condition_root / "summary.json"


@dataclass(frozen=True)
class PreflightRow:
    condition: str
    label: str
    bank_name: str
    n_qubits: int
    latent_dim: int
    output_dim: int
    valid: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "label": self.label,
            "bank_name": self.bank_name,
            "n_qubits": self.n_qubits,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
            "valid": self.valid,
            "message": self.message,
        }


@dataclass
class Stage8SweepConfig:
    repo_root: Path
    output_root: Path
    config_dir: Path
    frozen_encoder: Path | None = None
    run_id: str | None = None
    common_overrides: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=lambda: [dict(c) for c in STAGE8_CONDITIONS])
    selected_conditions: set[str] | None = None
    skip_completed: bool = True
    dry_run: bool = False

    @classmethod
    def default(
        cls,
        repo_root: Path | str,
        *,
        run_id: str | None = None,
        frozen_encoder: Path | str | None = None,
        output_root: Path | str | None = None,
        device: str = "GPU",
        smoke_test: bool = False,
    ) -> "Stage8SweepConfig":
        repo = Path(repo_root).resolve()
        if run_id is None:
            run_id = f"{DEFAULT_RUN_PREFIX}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        frozen = Path(frozen_encoder) if frozen_encoder is not None else repo / "outputs/experiments/baseline_mnist_last_20260701_124132/checkpoints/autoencoder.ckpt"
        out = Path(output_root) if output_root is not None else repo / "outputs/experiments" / run_id
        common = [
            f"device={device}",
            "wandb_mode=disabled",
            f"smoke_test={str(smoke_test).lower()}",
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
            f"checkpoints.autoencoder={frozen}",
        ]
        return cls(
            repo_root=repo,
            output_root=out,
            config_dir=repo / "configs",
            frozen_encoder=frozen,
            run_id=run_id,
            common_overrides=common,
        )

    def filtered_conditions(self) -> list[dict[str, Any]]:
        if self.selected_conditions is None:
            return self.conditions
        missing = self.selected_conditions - {str(c["name"]) for c in self.conditions}
        if missing:
            raise ValueError(f"Unknown Stage 8 condition(s): {sorted(missing)}")
        return [c for c in self.conditions if str(c["name"]) in self.selected_conditions]


def compose_stage8_cfg(config_dir: Path | str, overrides: Sequence[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base="1.3"):
        return compose(config_name="projects/qgan_expectation_values/train", overrides=list(overrides))


def preflight_conditions(config: Stage8SweepConfig) -> list[PreflightRow]:
    rows: list[PreflightRow] = []
    for condition in config.filtered_conditions():
        cfg = compose_stage8_cfg(config.config_dir, config.common_overrides + list(condition["overrides"]))
        bank = build_observable_bank(
            getattr(cfg.model.quantum_generator, "observable_bank", None),
            n_qubits=int(cfg.model.quantum_generator.n_qubits),
        )
        latent_dim = int(cfg.model.autoencoder.latent_dim)
        output_dim = int(bank.output_dim)
        valid = latent_dim == output_dim
        message = "ok" if valid else f"latent_dim {latent_dim} != observable_bank.output_dim {output_dim}"
        rows.append(
            PreflightRow(
                condition=str(condition["name"]),
                label=str(condition.get("label", condition["name"])),
                bank_name=str(bank.metadata()["name"]),
                n_qubits=int(bank.n_qubits),
                latent_dim=latent_dim,
                output_dim=output_dim,
                valid=valid,
                message=message,
            )
        )
    return rows


def assert_preflight_valid(rows: Iterable[PreflightRow]) -> None:
    invalid = [row for row in rows if not row.valid]
    if invalid:
        details = "; ".join(f"{row.condition}: {row.message}" for row in invalid)
        raise ValueError(f"Stage 8 preflight failed: {details}")


def condition_result(condition: Mapping[str, Any], paths: Stage8Paths) -> dict[str, Any]:
    qgan_record = load_json(paths.qgan_record_path)
    surrogate_record = load_json(paths.surrogate_record_path)
    write_comparison_json(
        qgan_record_path=paths.qgan_record_path,
        surrogate_record_path=paths.surrogate_record_path,
        output_path=paths.comparison_path,
    )
    comparison = load_json(paths.comparison_path)
    result = {
        "condition": str(condition["name"]),
        "label": str(condition.get("label", condition["name"])),
        "qgan_run_id": qgan_record["run_id"],
        "surrogate_run_id": surrogate_record["run_id"],
        "qgan_record_path": str(paths.qgan_record_path),
        "surrogate_record_path": str(paths.surrogate_record_path),
        "comparison_path": str(paths.comparison_path),
        "qgan_beta": comparison["beta_q"],
        "surrogate_beta": comparison["beta_cls"],
        "delta_sep": comparison["delta_sep"],
        "validated_match": comparison["validated_match"],
        "observable_bank": qgan_record["observable_bank"],
        "qgan_resource_counts": qgan_record["resource_counts"],
        "surrogate_resource_counts": surrogate_record["resource_counts"],
        "qgan_last_metrics": last_metric(qgan_record["metrics_path"]),
        "surrogate_last_metrics": last_metric(surrogate_record["metrics_path"]),
    }
    write_json(paths.summary_path, result)
    return result


def run_stage8_condition(
    condition: Mapping[str, Any],
    config: Stage8SweepConfig,
    *,
    run_qgan: Callable[[Any], Any] = run_gan,
    run_surrogate: Callable[..., Any] = run_surrogate_gan,
) -> dict[str, Any]:
    name = str(condition["name"])
    paths = Stage8Paths(config.output_root, name)
    qgan_overrides = config.common_overrides + list(condition["overrides"]) + [
        f"outputs.dir={paths.qgan_output_dir}",
        f"checkpoints.dir={paths.qgan_output_dir / 'checkpoints'}",
    ]
    if not (config.skip_completed and record_completed(paths.qgan_record_path, QGAN_WORKFLOW)):
        qgan_cfg = compose_stage8_cfg(config.config_dir, qgan_overrides)
        run_qgan(qgan_cfg)
    qgan_record = load_json(paths.qgan_record_path)

    surrogate_overrides = config.common_overrides + list(condition["overrides"]) + [
        "surrogate.enabled=true",
        "surrogate.rff_dim=auto",
        "surrogate.sigma=1.0",
        "surrogate.hidden_dim=64",
        f"outputs.dir={paths.surrogate_output_dir}",
        f"checkpoints.dir={paths.surrogate_output_dir / 'checkpoints'}",
    ]
    if not (config.skip_completed and record_completed(paths.surrogate_record_path, SURROGATE_WORKFLOW)):
        surrogate_cfg = compose_stage8_cfg(config.config_dir, surrogate_overrides)
        run_surrogate(surrogate_cfg, matched_run_id=qgan_record["run_id"])
    return condition_result(condition, paths)


def aggregate_stage8_results(output_root: Path | str, *, run_id: str | None = None) -> dict[str, Any]:
    root = Path(output_root)
    results: list[dict[str, Any]] = []
    by_name = {str(c["name"]): c for c in STAGE8_CONDITIONS}
    condition_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
    ordered = [name for name in by_name if name in condition_names] + [name for name in condition_names if name not in by_name]
    for name in ordered:
        paths = Stage8Paths(root, name)
        if not (paths.qgan_record_path.exists() and paths.surrogate_record_path.exists()):
            continue
        condition = by_name.get(name, {"name": name, "label": name})
        results.append(condition_result(condition, paths))
    return {"run_id": run_id or root.name, "out_root": str(root), "results": results}


def write_stage8_summary(summary: Mapping[str, Any], output_root: Path | str) -> tuple[Path, Path, Path]:
    root = Path(output_root)
    json_path = write_json(root / "stage8_summary.json", summary)
    csv_path = root / "stage8_summary.csv"
    md_path = root / "stage8_summary.md"
    fields = [
        "condition",
        "label",
        "qgan_beta",
        "surrogate_beta",
        "delta_sep",
        "n_qubits",
        "output_dim",
        "num_terms",
        "max_locality",
        "measurement_groups_estimate",
        "qgan_js",
        "surrogate_js",
        "qgan_ndb_k",
        "surrogate_ndb_k",
        "qgan_run_id",
        "surrogate_run_id",
        "comparison_path",
    ]
    rows = [_flat_summary_row(r) for r in summary.get("results", [])]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(stage8_markdown_table(rows, title=f"Stage 8 Summary — {summary.get('run_id', root.name)}"), encoding="utf-8")
    return json_path, csv_path, md_path


def _metric(metrics: Mapping[str, Any], key: str) -> Any:
    return metrics.get(key) if key in metrics else metrics.get(key.removeprefix("val/").upper())


def _flat_summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    bank = result.get("observable_bank", {}) or {}
    q_metrics = result.get("qgan_last_metrics", {}) or {}
    s_metrics = result.get("surrogate_last_metrics", {}) or {}
    return {
        "condition": result.get("condition"),
        "label": result.get("label"),
        "qgan_beta": result.get("qgan_beta"),
        "surrogate_beta": result.get("surrogate_beta"),
        "delta_sep": result.get("delta_sep"),
        "n_qubits": bank.get("n_qubits"),
        "output_dim": bank.get("output_dim"),
        "num_terms": bank.get("num_terms"),
        "max_locality": bank.get("max_locality"),
        "measurement_groups_estimate": bank.get("measurement_groups_estimate"),
        "qgan_js": _metric(q_metrics, "val/JS"),
        "surrogate_js": _metric(s_metrics, "val/JS"),
        "qgan_ndb_k": _metric(q_metrics, "val/NDB_K"),
        "surrogate_ndb_k": _metric(s_metrics, "val/NDB_K"),
        "qgan_run_id": result.get("qgan_run_id"),
        "surrogate_run_id": result.get("surrogate_run_id"),
        "comparison_path": result.get("comparison_path"),
    }


def stage8_markdown_table(rows: Sequence[Mapping[str, Any]], *, title: str = "Stage 8 Summary") -> str:
    ordered = sorted(rows, key=lambda r: float(r.get("delta_sep") or 0.0), reverse=True)
    lines = [f"# {title}", "", "| Rank | Condition | Bank | QGAN β | Surrogate β | Δ_sep | QGAN JS | Surrogate JS |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for idx, row in enumerate(ordered, start=1):
        lines.append(
            f"| {idx} | `{row.get('condition')}` | {row.get('label')} | {_fmt(row.get('qgan_beta'))} | {_fmt(row.get('surrogate_beta'))} | {_fmt(row.get('delta_sep'))} | {_fmt(row.get('qgan_js'))} | {_fmt(row.get('surrogate_js'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_stage8_sweep(config: Stage8SweepConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.frozen_encoder is not None and not Path(config.frozen_encoder).exists() and not config.dry_run:
        raise FileNotFoundError(f"Frozen encoder checkpoint not found: {config.frozen_encoder}")
    preflight = preflight_conditions(config)
    write_json(config.output_root / "preflight.json", [row.as_dict() for row in preflight])
    assert_preflight_valid(preflight)
    if config.dry_run:
        return {"run_id": config.run_id or config.output_root.name, "out_root": str(config.output_root), "preflight": [r.as_dict() for r in preflight], "results": []}
    results = []
    for condition in config.filtered_conditions():
        results.append(run_stage8_condition(condition, config))
        write_json(config.output_root / "partial_results.json", results)
    summary = {"run_id": config.run_id or config.output_root.name, "out_root": str(config.output_root), "results": results}
    write_stage8_summary(summary, config.output_root)
    return summary
