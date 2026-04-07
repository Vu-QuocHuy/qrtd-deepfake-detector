#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "qrtd" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrtd.config import load_train_config
from qrtd.trainer import QRTDTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QRTD scaffold")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    trainer = QRTDTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
