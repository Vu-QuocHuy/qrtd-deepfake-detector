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
python qrtd/scripts/test_qrtd.py --checkpoint /path/to/model.pth --config qrtd/configs/expA.yaml
```

## Notes

- Current code is a clean scaffold with deterministic structure and APIs.
- It does not depend on legacy model classes.
- Team can implement modules incrementally without touching old code paths.
