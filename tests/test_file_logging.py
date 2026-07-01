from __future__ import annotations

import json

from qgan_latent.shared.utils.file_logging import EpochMetricLogger


def test_epoch_metric_logger_writes_csv_and_json(tmp_path) -> None:
    logger = EpochMetricLogger(tmp_path / "metrics.csv", tmp_path / "metrics.json")

    first = logger.log(0, 10, {"fid": 1.5, "ignored": object()})
    second = logger.log(1, 20, {"fid": 0.5, "kid": 2.0})

    assert first == {"epoch": 0, "step": 10, "fid": 1.5}
    assert second == {"epoch": 1, "step": 20, "fid": 0.5, "kid": 2.0}

    csv_text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert "epoch,step,fid,kid" in csv_text
    assert "0,10,1.5" in csv_text
    assert "1,20,0.5,2.0" in csv_text

    records = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert records == [first, second]
