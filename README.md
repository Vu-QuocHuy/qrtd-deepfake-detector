# QRTD Workspace

This folder is an isolated workspace for the new cross-dataset direction.
It is intentionally separated from the legacy `deepfake_detector` package.

## Goals

- Keep experimentation for the new paper track fully isolated.
- Avoid accidental coupling with old training and inference pipelines.
- Provide a clean, maintainable structure for iterative ablations.

## Structure

- `src/qrtd/`: Python package for the QRTD track.
- `scripts/`: standalone train/test entrypoints.
- `configs/`: experiment configuration files.

## First experiments

- `expA`: anti-compression augmentation only.
- `expB`: expA + quality-consistency contrastive objective.
- `expC`: expB + reliability-aware temporal aggregation.

## Usage

Run from repository root:

```bash
python qrtd/scripts/train_qrtd.py --config qrtd/configs/expA.yaml
python qrtd/scripts/test_qrtd.py --checkpoint /path/to/best.pth --config qrtd/configs/expA.yaml
```

For a standalone repository layout (where `scripts/` and `configs/` are at root), use:

```bash
python scripts/train_qrtd.py --config configs/expA.yaml
python scripts/test_qrtd.py --checkpoint /path/to/best.pth --config configs/expA.yaml
```

## Notes

- Current code includes a working train/val loop with EfficientNet-B4.
- Input data must be extracted face frames grouped by video id via filename.
- It does not depend on legacy model classes.
