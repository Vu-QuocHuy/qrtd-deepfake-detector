#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not SRC.exists():
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "qrtd" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrtd.config import load_train_config
from qrtd.data import build_test_loader
from qrtd.model import QRTDDetector


@torch.no_grad()
def evaluate(model: QRTDDetector, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for seqs, labels in tqdm(loader, desc="Testing"):
        seqs = seqs.to(device, non_blocking=True)
        logits = model(seqs)["logits"]
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().astype(np.int64).tolist())
    return np.asarray(all_probs, dtype=np.float64), np.asarray(all_labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QRTD checkpoint on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test-real", nargs="+", required=True)
    parser.add_argument("--test-fake", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="test_results_qrtd")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = QRTDDetector(
        model_name=cfg.model_name,
        pretrained=False,
        reliability_enabled=cfg.reliability_enabled,
        frame_chunk_size=cfg.frame_chunk_size,
        grad_checkpoint=False,
        grad_ckpt_segments=cfg.grad_ckpt_segments,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)

    test_loader = build_test_loader(
        test_real=args.test_real,
        test_fake=args.test_fake,
        n_frames=cfg.n_frames,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    probs, labels = evaluate(model, test_loader, device)
    preds = (probs >= 0.5).astype(np.int64)

    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, zero_division=0))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"acc: {acc:.6f}\n")
        f.write(f"auc: {auc:.6f}\n")
        f.write(f"f1: {f1:.6f}\n")
        f.write(f"precision: {prec:.6f}\n")
        f.write(f"recall: {rec:.6f}\n")
        f.write(f"samples: {len(labels)}\n")

    pred_path = out_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "prob_real"])
        for y, p, pr in zip(labels.tolist(), preds.tolist(), probs.tolist()):
            writer.writerow([y, p, pr])

    print("[QRTD] Loaded checkpoint successfully.")
    print(f"[QRTD] epoch={ckpt.get('epoch', 'n/a')} val_acc={ckpt.get('val_acc', 'n/a')}")
    print(f"[QRTD] acc={acc:.4f} auc={auc:.4f} f1={f1:.4f} precision={prec:.4f} recall={rec:.4f}")
    print(f"[QRTD] Saved metrics -> {metrics_path}")
    print(f"[QRTD] Saved predictions -> {pred_path}")


if __name__ == "__main__":
    main()
