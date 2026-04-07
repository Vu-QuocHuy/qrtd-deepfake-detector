from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


class QRTDDetector(nn.Module):
    """
    QRTD model scaffold for:
    - anti-compression training
    - optional contrastive branch
    - optional reliability-aware temporal aggregation
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b4",
        pretrained: bool = True,
        reliability_enabled: bool = False,
        frame_chunk_size: int = 4,
        feature_dim: int = 1792,
    ) -> None:
        super().__init__()
        if model_name != "efficientnet-b4":
            raise ValueError(f"Unsupported model_name: {model_name}")
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        base = efficientnet_b4(weights=weights)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, 1)
        self.reliability_enabled = reliability_enabled
        self.frame_chunk_size = max(1, int(frame_chunk_size))
        self.reliability_head = nn.Linear(feature_dim, 1)
        self.projection_head = nn.Linear(feature_dim, 128)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, C, H, W]
        bsz, timesteps = x.shape[:2]
        chunks = []
        # Process frames in chunks to reduce peak VRAM usage.
        for start in range(0, timesteps, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, timesteps)
            x_chunk = x[:, start:end].contiguous().view(bsz * (end - start), *x.shape[2:])
            feat_map = self.backbone(x_chunk)
            feat_chunk = self.pool(feat_map).flatten(1)
            feat_chunk = feat_chunk.view(bsz, end - start, -1)
            chunks.append(feat_chunk)
        feat = torch.cat(chunks, dim=1)  # [B, T, D]

        frame_logits = self.classifier(feat).squeeze(-1)  # [B, T]
        if self.reliability_enabled:
            reliability = torch.sigmoid(self.reliability_head(feat).squeeze(-1))  # [B, T]
            logits = (frame_logits * reliability).sum(dim=1) / reliability.sum(dim=1).clamp_min(1e-6)
        else:
            reliability = torch.ones_like(frame_logits)
            logits = frame_logits.mean(dim=1)

        proj = self.projection_head(feat.mean(dim=1))
        return {"logits": logits, "reliability": reliability, "projection": proj}
