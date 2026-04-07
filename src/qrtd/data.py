from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DatasetSpec:
    train_real: list[str]
    train_fake: list[str]
    val_real: list[str]
    val_fake: list[str]


def _extract_video_id(filename: str) -> str:
    stem = Path(filename).stem
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return stem


def _list_frames_by_video(directory: str) -> dict[str, list[str]]:
    p = Path(directory)
    if not p.exists():
        return {}
    video_frames: dict[str, list[str]] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in p.glob(ext):
            vid = _extract_video_id(img_path.name)
            video_frames.setdefault(vid, []).append(str(img_path))
    for vid in video_frames:
        video_frames[vid].sort()
    return video_frames


class SequenceDataset(Dataset):
    """
    Dataset that returns tensor [T, 3, H, W] and label.

    Input folders must contain extracted face frames, not raw video files.
    """

    def __init__(
        self,
        roots: list[str],
        label: int,
        n_frames: int,
        transform: transforms.Compose,
        random_sampling: bool,
    ) -> None:
        self.label = label
        self.n_frames = n_frames
        self.transform = transform
        self.random_sampling = random_sampling
        self.samples: list[list[str]] = []

        for root in roots:
            grouped = _list_frames_by_video(root)
            for frames in grouped.values():
                if len(frames) >= 2:
                    self.samples.append(frames)

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_indices(self, n: int) -> list[int]:
        if n <= self.n_frames:
            return [int(i) for i in torch.linspace(0, n - 1, self.n_frames)]
        if self.random_sampling:
            start = random.randint(0, n - self.n_frames)
            return list(range(start, start + self.n_frames))
        return [int(i) for i in torch.linspace(0, n - 1, self.n_frames)]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.samples[idx]
        indices = self._pick_indices(len(frames))
        seq = []
        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            seq.append(self.transform(img))
        x = torch.stack(seq, dim=0)  # [T, C, H, W]
        y = torch.tensor(float(self.label), dtype=torch.float32)
        return x, y


class ConcatDataset(Dataset):
    def __init__(self, a: Dataset, b: Dataset) -> None:
        self.a = a
        self.b = b
        self.a_len = len(a)

    def __len__(self) -> int:
        return self.a_len + len(self.b)

    def __getitem__(self, idx: int):
        if idx < self.a_len:
            return self.a[idx]
        return self.b[idx - self.a_len]


class RandomJPEGCompression:
    def __init__(self, qmin: int = 35, qmax: int = 95, p: float = 0.5) -> None:
        self.qmin = qmin
        self.qmax = qmax
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_transforms(image_size: int, anti_compression: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[object] = [
        transforms.Resize((image_size, image_size)),
    ]
    if anti_compression:
        train_ops.extend(
            [
                RandomJPEGCompression(35, 95, p=0.7),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            ]
        )
    train_ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_ops = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(train_ops), val_ops


def build_dataloaders(
    spec: DatasetSpec,
    n_frames: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anti_compression: bool,
) -> dict[str, DataLoader]:
    train_tfm, val_tfm = build_transforms(image_size=image_size, anti_compression=anti_compression)

    train_real_ds = SequenceDataset(spec.train_real, label=1, n_frames=n_frames, transform=train_tfm, random_sampling=True)
    train_fake_ds = SequenceDataset(spec.train_fake, label=0, n_frames=n_frames, transform=train_tfm, random_sampling=True)
    val_real_ds = SequenceDataset(spec.val_real, label=1, n_frames=n_frames, transform=val_tfm, random_sampling=False)
    val_fake_ds = SequenceDataset(spec.val_fake, label=0, n_frames=n_frames, transform=val_tfm, random_sampling=False)

    train_ds = ConcatDataset(train_real_ds, train_fake_ds)
    val_ds = ConcatDataset(val_real_ds, val_fake_ds)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    return {"train": train_loader, "val": val_loader}
