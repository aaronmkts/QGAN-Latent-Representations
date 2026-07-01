from __future__ import annotations

import json

import numpy as np

from qgan_latent.shared.utils.file_logging import EpochMetricLogger
from qgan_latent.workflows.qgan_expectation_values.training.diagnostics import (
    EvsDiagnosticsWriter,
    compute_evs_diagnostics,
)


def test_compute_evs_diagnostics_values() -> None:
    expectations = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=np.float32,
    )

    diagnostics = compute_evs_diagnostics(expectations)

    np.testing.assert_allclose(diagnostics["mean"], [0.5, 1.0, -1.0])
    np.testing.assert_allclose(diagnostics["second_moment"], [0.5, 2.0, 1.0])
    np.testing.assert_allclose(diagnostics["variance"], [0.25, 1.0, 0.0])
    np.testing.assert_allclose(diagnostics["parseval_norm_sq"], [0.5, 2.0, 1.0])
    assert diagnostics["total_parseval_norm_sq"] == 3.5
    assert diagnostics["mean_variance"] == float(np.mean([0.25, 1.0, 0.0]))
    assert diagnostics["min_variance"] == 0.0
    assert diagnostics["max_variance"] == 1.0
    assert diagnostics["num_samples"] == 2


def test_evs_diagnostics_writer_scalars_in_csv_vectors_in_json(tmp_path) -> None:
    diagnostics = compute_evs_diagnostics(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    metrics_logger = EpochMetricLogger(tmp_path / "metrics.csv", tmp_path / "metrics.json")
    writer = EvsDiagnosticsWriter(tmp_path / "evs_diagnostics.json")

    scalar_record = metrics_logger.log(
        epoch=0,
        step=7,
        metrics={f"evs/{key}": value for key, value in diagnostics.items()},
    )
    vector_record = writer.log(epoch=0, step=7, diagnostics=diagnostics)

    assert "evs/total_parseval_norm_sq" in scalar_record
    assert "evs/mean_variance" in scalar_record
    assert "evs/mean" not in scalar_record
    metrics_csv = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    csv_header = metrics_csv.splitlines()[0].split(",")
    assert "evs/total_parseval_norm_sq" in csv_header
    assert "evs/mean" not in csv_header

    records = json.loads((tmp_path / "evs_diagnostics.json").read_text(encoding="utf-8"))
    assert records == [vector_record]
    assert records[0]["epoch"] == 0
    assert records[0]["step"] == 7
    assert records[0]["mean"] == [0.5, 0.5]
    assert records[0]["parseval_norm_sq"] == [0.5, 0.5]
    assert records[0]["total_parseval_norm_sq"] == 1.0
