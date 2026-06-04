# Repository Structure

This repository is organized as the official code release for MethaneFuse.

## Top-level directories

- `src/`: MethaneFuse model code.
- `baselines/`: baseline implementations, including per-ViT, single-sensor, AnySat, SatMAE, and UNet baselines.
- `thirdparty/dinov2/`: inherited Panopticon/DINOv2 code kept as third-party code.
- `configs/`: configuration files for training and evaluation.
- `scripts/train/`: placeholder training launchers with user-provided paths.
- `figures/`: figures suitable for the official repository.
- `docs/`: documentation beyond the README.
- `environments/`: optional development or extra dependency files.

## Local-only directories

These are intentionally ignored by Git: `data/`, `data_csv/`, `datasets/`, `examples/`, `logs/`, `weights/`, `checkpoints/`, `results/reports/generated/`, `experiments_tryout/`, `tools/`, non-training `scripts/`, `universal_models/`, and `universal_models_fusion/`.

## Notes

Dataset preparation, report generation, release packaging, cluster maintenance, and other local utilities are kept out of the official repository. The remote repository keeps training-facing code plus lightweight launch templates only.
