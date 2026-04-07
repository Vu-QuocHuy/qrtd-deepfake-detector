from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetSpec:
    train_real: list[str]
    train_fake: list[str]
    val_real: list[str]
    val_fake: list[str]


def build_dataloaders() -> dict[str, object]:
    """
    Placeholder for QRTD data pipeline.

    This function is intentionally separated from the legacy project data code.
    """
    return {"train": None, "val": None}
