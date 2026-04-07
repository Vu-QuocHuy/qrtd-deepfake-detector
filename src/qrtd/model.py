from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
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
        grad_checkpoint: bool = False,
        grad_ckpt_segments: int = 4,
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
        self.grad_checkpoint = bool(grad_checkpoint)
        self.grad_ckpt_segments = max(1, int(grad_ckpt_segments))
        self.reliability_head = nn.Linear(feature_dim, 1)
        self.projection_head = nn.Linear(feature_dim, 128)

    def _backbone_forward(self, x_chunk: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            try:
                return checkpoint_sequential(
                    self.backbone,
                    self.grad_ckpt_segments,
                    x_chunk,
                    use_reentrant=False,
                )
            except TypeError:
                return checkpoint_sequential(self.backbone, self.grad_ckpt_segments, x_chunk)
        return self.backbone(x_chunk)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, C, H, W]
        bsz, timesteps = x.shape[:2]
        feat_sum = None
        logit_sum = None
        weighted_logit_sum = None
        weight_sum = None

        # Process frames in chunks to reduce peak VRAM usage and avoid
        # materializing [B, T, D] for the full sequence.
        for start in range(0, timesteps, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, timesteps)
            x_chunk = x[:, start:end].contiguous().view(bsz * (end - start), *x.shape[2:])
            feat_map = self._backbone_forward(x_chunk)
            feat_chunk = self.pool(feat_map).flatten(1)
            feat_chunk = feat_chunk.view(bsz, end - start, -1)

            chunk_len = feat_chunk.shape[1]
            chunk_feat_sum = feat_chunk.sum(dim=1)  # [B, D]
            feat_sum = chunk_feat_sum if feat_sum is None else (feat_sum + chunk_feat_sum)

            frame_logits = self.classifier(feat_chunk).squeeze(-1)  # [B, Tc]
            if self.reliability_enabled:
                rel = torch.sigmoid(self.reliability_head(feat_chunk).squeeze(-1))  # [B, Tc]
                wl = (frame_logits * rel).sum(dim=1)
                ws = rel.sum(dim=1)
                weighted_logit_sum = wl if weighted_logit_sum is None else (weighted_logit_sum + wl)
                weight_sum = ws if weight_sum is None else (weight_sum + ws)
            else:
                ls = frame_logits.sum(dim=1)
                logit_sum = ls if logit_sum is None else (logit_sum + ls)

        if self.reliability_enabled:
            logits = weighted_logit_sum / weight_sum.clamp_min(1e-6)
            reliability = weight_sum / float(timesteps)
        else:
            logits = logit_sum / float(timesteps)
            reliability = torch.ones_like(logits)

        feat_mean = feat_sum / float(timesteps)
        proj = self.projection_head(feat_mean)
        return {"logits": logits, "reliability": reliability, "projection": proj}
