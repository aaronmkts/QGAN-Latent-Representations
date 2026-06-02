from __future__ import annotations

from typing import Any

from flax.training import train_state


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: Any
