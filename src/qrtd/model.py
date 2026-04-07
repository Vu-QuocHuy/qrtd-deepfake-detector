from __future__ import annotations

import torch
from torch import nn


class QRTDDetector(nn.Module):
    """
    QRTD model scaffold for:
    - anti-compression training
    - optional contrastive branch
    - optional reliability-aware temporal aggregation
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, 1)
        self.reliability_head = nn.Linear(feature_dim, 1)
        self.projection_head = nn.Linear(feature_dim, 128)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.classifier(feat).squeeze(-1)
        reliability = torch.sigmoid(self.reliability_head(feat).squeeze(-1))
        proj = self.projection_head(feat)
        return {"logits": logits, "reliability": reliability, "projection": proj}
