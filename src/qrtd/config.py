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
    model_name: str
    pretrained: bool
    anti_compression: bool
    contrastive_enabled: bool
    reliability_enabled: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    train_real: list[str]
    train_fake: list[str]
    val_real: list[str]
    val_fake: list[str]


def load_train_config(path: str | Path) -> TrainConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return TrainConfig(**raw)
