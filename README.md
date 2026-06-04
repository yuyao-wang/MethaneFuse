# MethaneFuse

Official code repository for MethaneFuse, a multi-sensor methane plume detection framework.

This README is intentionally a short placeholder while the repository is being cleaned for release. The final README will include installation, checkpoint download, Hugging Face dataset links, evaluation commands, reproduction instructions, and citation information.

## Repository Layout

- `src/methanefuse/`: MethaneFuse project code.
  - `models/pretrain_multisensor.py`: pretraining model code.
  - `models/finetune_loramoe_adapter.py`: fine-tuning model code.
  - `models/segmentation.py`: segmentation model code.
- `baselines/per_vit/`: per-sensor ViT baseline code.
- `baselines/unet/`: UNet baseline code.
- `baselines/single_sensor/`: other single-sensor baselines.
- `dinov2/`: inherited Panopticon/DINOv2 code retained for compatibility.
- `configs/`: training, evaluation, data, and figure configs.
- `scripts/`: runnable entry points for training, evaluation, data preparation, figures, release, and cluster jobs.
- `figures/`: paper and generated figures that are suitable for the official repository.
- `results/`: small reproducibility artifacts and result summaries suitable for the official repository.
- `docs/`: detailed documentation pages.

## Data

The dataset is not stored in this repository. It is maintained separately and hosted on Hugging Face. Local directories such as `data/`, `data_csv/`, `datasets/`, `examples/`, and `logs/` are ignored by Git.

## Baselines

Panopticon is used as a baseline/backbone dependency in this project. It is not the identity of this repository.
