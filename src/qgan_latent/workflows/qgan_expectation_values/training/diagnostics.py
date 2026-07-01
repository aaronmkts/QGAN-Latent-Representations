from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_VECTOR_KEYS = ("mean", "second_moment", "variance", "parseval_norm_sq")
_SCALAR_KEYS = (
    "total_parseval_norm_sq",
    "mean_variance",
    "min_variance",
    "max_variance",
    "num_samples",
)


def compute_evs_diagnostics(expectations: Any) -> dict[str, Any]:
    """Compute EVS Parseval and non-concentration diagnostics.

    Args:
        expectations: Array-like generator expectation vectors with shape
            ``(num_samples, output_dim)``.

    Returns:
        A JSON-serialisable mapping containing per-dimension vectors and
        aggregate scalar summaries.
    """

    values = np.asarray(expectations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"expectations must have shape (num_samples, output_dim), got {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError("expectations must contain at least one sample.")

    mean = np.mean(values, axis=0)
    second_moment = np.mean(np.square(values), axis=0)
    variance = np.var(values, axis=0)
    parseval_norm_sq = second_moment

    return {
        "mean": mean.tolist(),
        "second_moment": second_moment.tolist(),
        "variance": variance.tolist(),
        "parseval_norm_sq": parseval_norm_sq.tolist(),
        "total_parseval_norm_sq": float(np.sum(parseval_norm_sq)),
        "mean_variance": float(np.mean(variance)),
        "min_variance": float(np.min(variance)),
        "max_variance": float(np.max(variance)),
        "num_samples": int(values.shape[0]),
    }


def evs_scalar_metrics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    """Return only aggregate scalar EVS diagnostics with ``evs/`` prefixes."""

    return {f"evs/{key}": float(diagnostics[key]) for key in _SCALAR_KEYS}


def parseval_norm_from_diagnostics(diagnostics: Mapping[str, Any]) -> float:
    """Return the aggregate EVS Parseval norm squared for comparison metrics."""

    return float(diagnostics["total_parseval_norm_sq"])

class EvsDiagnosticsWriter:
    """Append per-evaluation EVS diagnostics records to JSON."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def log(self, epoch: int, step: int, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"epoch": int(epoch), "step": int(step)}
        for key in (*_VECTOR_KEYS, *_SCALAR_KEYS):
            value = diagnostics[key]
            if key in _VECTOR_KEYS:
                record[key] = [float(item) for item in value]
            elif key == "num_samples":
                record[key] = int(value)
            else:
                record[key] = float(value)

        records = self._load_records()
        records.append(record)
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return record
