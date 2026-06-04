# MethaneFuse

Official code repository for MethaneFuse, a multi-sensor methane plume detection framework.

This README is intentionally a short placeholder while the repository is being cleaned for release. The final README will include installation, checkpoint download, Hugging Face dataset links, evaluation commands, reproduction instructions, and citation information.

## Repository Layout

- `src/`: MethaneFuse project code.
  - `models/pretrain_multisensor.py`: pretraining model code.
  - `models/finetune_loramoe_adapter.py`: fine-tuning model code.
  - `models/segmentation.py`: segmentation model code.
- `baselines/per_vit/`: per-sensor ViT baseline code.
- `baselines/unet/`: UNet baseline code.
- `baselines/single_sensor/`: other single-sensor baselines.
- `dinov2/`: inherited Panopticon/DINOv2 code retained for compatibility. This will be reorganized in a later cleanup stage.
- `configs/`: training, evaluation, data, and figure configs.
- `scripts/`: runnable entry points for training, evaluation, data preparation, figures, release, and cluster jobs.
- `figures/`: figures suitable for the official repository.
- `docs/`: detailed documentation pages.
- `environments/`: optional development or extra dependency files.

## Local-only Files

The dataset is not stored in this repository. It is maintained separately and hosted on Hugging Face.

The following local directories are ignored by Git and should not be pushed to the official repository: `data/`, `data_csv/`, `datasets/`, `examples/`, `logs/`, `weights/`, `checkpoints/`, `results/reports/generated/`, and `experiments_tryout/`.

## Baselines

Panopticon is used as a baseline/backbone dependency in this project. It is not the identity of this repository.
