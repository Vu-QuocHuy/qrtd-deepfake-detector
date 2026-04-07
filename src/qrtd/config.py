from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    experiment_name: str
    output_dir: str
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    n_frames: int
    image_size: int
    anti_compression: bool
    contrastive_enabled: bool
    reliability_enabled: bool


def load_train_config(path: str | Path) -> TrainConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return TrainConfig(**raw)
