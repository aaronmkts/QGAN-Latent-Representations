from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    git_commit: str
    git_dirty: bool
    workflow: str
    observable_bank: dict[str, Any]
    resource_counts: dict[str, Any]
    config_snapshot: dict[str, Any]
    metrics_path: str
    diagnostics_path: str
    observable_bank_metadata_path: str
    sample_grid_paths: list[str]
    checkpoint_paths: dict[str, str]
    status: str
    parent_run_id: str | None


def current_git_state(repo_root: Path | None = None) -> tuple[str, bool]:
    """Return (full commit sha or 'unknown', dirty flag) without failing outside git."""

    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = commit_proc.stdout.strip() if commit_proc.returncode == 0 else "unknown"
        if not commit:
            commit = "unknown"

        dirty_proc = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = dirty_proc.returncode != 0 if commit != "unknown" else False
        return commit, bool(dirty)
    except Exception:
        return "unknown", False


def _resolved_config(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        resolved = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        resolved = dict(cfg)
    else:
        resolved = {
            key: getattr(cfg, key)
            for key in dir(cfg)
            if not key.startswith("_") and not callable(getattr(cfg, key))
        }
    return resolved if isinstance(resolved, dict) else {"config": resolved}


def _default_run_id(workflow: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{workflow}_{stamp}"


class RunRegistry:
    def __init__(self, output_dir: Path | str, run_id: str | None = None, workflow: str = "qgan_expectation_values") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or _default_run_id(workflow)
        self.workflow = workflow
        self.path = self.output_dir / "run_record.json"
        self._record: RunRecord | None = None

    def start(
        self,
        cfg: Any,
        observable_bank_metadata: dict[str, Any],
        resource_counts: dict[str, Any] | None = None,
        parent_run_id: str | None = None,
    ) -> RunRecord:
        commit, dirty = current_git_state()
        record = RunRecord(
            run_id=self.run_id,
            git_commit=commit,
            git_dirty=dirty,
            workflow=self.workflow,
            observable_bank=dict(observable_bank_metadata),
            resource_counts={str(k): v for k, v in (resource_counts or {}).items()},
            config_snapshot=_resolved_config(cfg),
            metrics_path="",
            diagnostics_path="",
            observable_bank_metadata_path="",
            sample_grid_paths=[],
            checkpoint_paths={},
            status="started",
            parent_run_id=parent_run_id,
        )
        self._record = record
        self.write(record)
        return record

    def finalize(
        self,
        metrics_path: Path | str,
        diagnostics_path: Path | str,
        observable_bank_metadata_path: Path | str,
        sample_grid_paths: list[Path | str],
        checkpoint_paths: dict[str, Path | str],
        parent_run_id: str | None = None,
        status: str = "completed",
    ) -> RunRecord:
        base = self._record or self.load()
        if base is None:
            raise RuntimeError("RunRegistry.finalize called before start and no run_record.json exists.")
        record = RunRecord(
            run_id=base.run_id,
            git_commit=base.git_commit,
            git_dirty=base.git_dirty,
            workflow=base.workflow,
            observable_bank=base.observable_bank,
            resource_counts=base.resource_counts,
            config_snapshot=base.config_snapshot,
            metrics_path=str(metrics_path),
            diagnostics_path=str(diagnostics_path),
            observable_bank_metadata_path=str(observable_bank_metadata_path),
            sample_grid_paths=[str(path) for path in sample_grid_paths],
            checkpoint_paths={key: str(path) for key, path in checkpoint_paths.items()},
            status=status,
            parent_run_id=parent_run_id if parent_run_id is not None else base.parent_run_id,
        )
        self._record = record
        self.write(record)
        return record

    def write(self, record: RunRecord) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
        return self.path

    def load(self) -> RunRecord | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return RunRecord(**data)
