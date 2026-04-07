#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not SRC.exists():
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "qrtd" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrtd.config import load_train_config
from qrtd.model import QRTDDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and validate a QRTD checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = QRTDDetector(
        model_name=cfg.model_name,
        pretrained=False,
        reliability_enabled=cfg.reliability_enabled,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print("[QRTD] Loaded checkpoint successfully.")
    print(f"[QRTD] epoch={ckpt.get('epoch', 'n/a')} val_acc={ckpt.get('val_acc', 'n/a')}")


if __name__ == "__main__":
    main()
