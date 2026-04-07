#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "qrtd" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrtd.config import load_train_config
from qrtd.model import QRTDDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Test QRTD scaffold")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    _ = load_train_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = QRTDDetector()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print("[QRTD] Loaded checkpoint successfully.")
    print(f"[QRTD] scaffold_only={ckpt.get('scaffold_only', False)}")


if __name__ == "__main__":
    main()
