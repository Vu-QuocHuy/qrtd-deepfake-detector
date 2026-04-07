from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .config import TrainConfig
from .model import QRTDDetector


class QRTDTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QRTDDetector().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def save_scaffold_artifacts(self) -> None:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (out / "run_config.txt").open("w", encoding="utf-8") as f:
            for k, v in asdict(self.cfg).items():
                f.write(f"{k}: {v}\n")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "experiment_name": self.cfg.experiment_name,
                "scaffold_only": True,
            },
            out / "checkpoints" / "scaffold_init.pth",
        )

    def fit(self) -> None:
        """
        Placeholder training loop for QRTD track.
        Real data/loss wiring should be implemented experiment by experiment.
        """
        self.save_scaffold_artifacts()
        print("[QRTD] Scaffold initialized.")
        print(f"[QRTD] Device: {self.device}")
        print(f"[QRTD] anti_compression={self.cfg.anti_compression}")
        print(f"[QRTD] contrastive_enabled={self.cfg.contrastive_enabled}")
        print(f"[QRTD] reliability_enabled={self.cfg.reliability_enabled}")
