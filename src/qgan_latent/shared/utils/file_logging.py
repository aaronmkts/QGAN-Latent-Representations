from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EpochMetricLogger:
    """Persist per-epoch metric records to CSV and cumulative JSON files."""

    def __init__(self, csv_path: Path | str, json_path: Path | str) -> None:
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def _floatify(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                continue
        return clean

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.json_path.exists() or self.json_path.stat().st_size == 0:
            return []
        try:
            records = json.loads(self.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _write_csv(self, records: list[dict[str, Any]]) -> None:
        fieldnames: list[str] = []
        for preferred in ("epoch", "step"):
            if any(preferred in record for record in records):
                fieldnames.append(preferred)
        for record in records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def log(self, epoch: int, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"epoch": int(epoch), "step": int(step)}
        record.update(self._floatify(metrics))

        records = self._load_records()
        records.append(record)
        self.json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        self._write_csv(records)
        return record
