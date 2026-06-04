# Repository Structure

This repository is organized as the official code release for MethaneFuse.

## Top-level directories

- `src/`: MethaneFuse model code.
- `baselines/`: baseline implementations.
- `dinov2/`: inherited Panopticon/DINOv2 code kept in place for compatibility for now.
- `configs/`: configuration files for training, evaluation, data, and figures.
- `scripts/`: command-line entry points and utility scripts.
- `figures/`: figures suitable for the official repository.
- `docs/`: documentation beyond the README.
- `environments/`: optional development or extra dependency files.

## Local-only directories

These are intentionally ignored by Git: `data/`, `data_csv/`, `datasets/`, `examples/`, `logs/`, `weights/`, `checkpoints/`, `results/reports/generated/`, `experiments_tryout/`, `universal_models/`, and `universal_models_fusion/`.

## Notes

The `dinov2/` directory remains at its current import path during this stage. Moving it to `third_party/` will require coordinated import updates and is left for a later cleanup stage.
