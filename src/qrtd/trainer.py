from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import TrainConfig
from .data import DatasetSpec, build_dataloaders
from .model import QRTDDetector


class QRTDTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._set_seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QRTDDetector(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            reliability_enabled=cfg.reliability_enabled,
            frame_chunk_size=cfg.frame_chunk_size,
            grad_checkpoint=cfg.grad_checkpoint,
            grad_ckpt_segments=cfg.grad_ckpt_segments,
        ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_acc = -1.0
        self.no_improve_count = 0
        self.out = Path(self.cfg.output_dir)
        self.ckpt_dir = self.out / "checkpoints"
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_run_config(self) -> None:
        with (self.out / "run_config.txt").open("w", encoding="utf-8") as f:
            for k, v in asdict(self.cfg).items():
                f.write(f"{k}: {v}\n")

    def _save_checkpoint(self, name: str, epoch: int, val_acc: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "experiment_name": self.cfg.experiment_name,
            },
            self.ckpt_dir / name,
        )

    def _run_one_epoch(self, loader, train: bool) -> tuple[float, float]:
        self.model.train(mode=train)
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
        for seqs, labels in pbar:
            seqs = seqs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.model(seqs)
                    logits = out["logits"]
                    loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            running_loss += float(loss.item()) * labels.numel()
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.numel())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(1, running_total)
        mean_acc = running_correct / max(1, running_total)
        return mean_loss, mean_acc

    def fit(self) -> None:
        spec = DatasetSpec(
            train_real=self.cfg.train_real,
            train_fake=self.cfg.train_fake,
            val_real=self.cfg.val_real,
            val_fake=self.cfg.val_fake,
        )
        loaders = build_dataloaders(
            spec=spec,
            n_frames=self.cfg.n_frames,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            anti_compression=self.cfg.anti_compression,
        )
        self._save_run_config()

        metrics_path = self.out / "metrics.txt"
        with metrics_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        print(f"[QRTD] Device: {self.device}")
        print(f"[QRTD] anti_compression={self.cfg.anti_compression}")
        print(f"[QRTD] contrastive_enabled={self.cfg.contrastive_enabled}")
        print(f"[QRTD] reliability_enabled={self.cfg.reliability_enabled}")
        print(
            f"[QRTD] use_amp={self.use_amp} frame_chunk_size={self.cfg.frame_chunk_size} "
            f"grad_checkpoint={self.cfg.grad_checkpoint} segments={self.cfg.grad_ckpt_segments}"
        )

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.epochs}")
            train_loss, train_acc = self._run_one_epoch(loaders["train"], train=True)
            val_loss, val_acc = self._run_one_epoch(loaders["val"], train=False)
            self.scheduler.step()

            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

            self._save_checkpoint(name=f"epoch_{epoch:03d}.pth", epoch=epoch, val_acc=val_acc)
            improved = val_acc > (self.best_val_acc + self.cfg.early_stopping_min_delta)
            if improved:
                self.best_val_acc = val_acc
                self.no_improve_count = 0
                self._save_checkpoint(name="best.pth", epoch=epoch, val_acc=val_acc)
                print(f"[QRTD] New best val_acc={val_acc:.4f}")
            else:
                self.no_improve_count += 1

            print(
                f"[QRTD] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if self.no_improve_count >= self.cfg.early_stopping_patience:
                print(
                    "[QRTD] Early stopping triggered: "
                    f"no val_acc improvement >= {self.cfg.early_stopping_min_delta} "
                    f"for {self.cfg.early_stopping_patience} epochs."
                )
                break
