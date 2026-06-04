# MethaneFuse

Official code repository for MethaneFuse, a multi-sensor methane plume detection framework.

This README is intentionally a short placeholder while the repository is being cleaned for release. The final README will include installation, checkpoint download, Hugging Face dataset links, evaluation commands, reproduction instructions, and citation information.

## Repository Layout

- `src/`: MethaneFuse project code.
  - `models/pretrain_multisensor.py`: pretraining model code.
  - `models/finetune_loramoe_adapter.py`: fine-tuning model code.
  - `models/segmentation.py`: segmentation model code.
- `baselines/per_vit/`: per-sensor ViT baseline code.
- `baselines/single_sensor/`: ResNet/ViT single-sensor baselines.
- `baselines/anysat_ft_avg_fusion_480m.py`: AnySat average-fusion baseline.
- `baselines/satmae_ft_avg_fusion_480m.py`: SatMAE average-fusion baseline.
- `baselines/unet_multisensor_baseline_iou_plus.py`: UNet segmentation baseline.
- `thirdparty/dinov2/`: inherited Panopticon/DINOv2 code kept as inherited third-party code. This will be reorganized in a later cleanup stage.
- `configs/`: training, evaluation, data, and figure configs.
- `scripts/train/`: training launchers with recommended hyperparameters and user-provided paths.
- `figures/`: figures suitable for the official repository.
- `docs/`: detailed documentation pages.
- `environments/`: optional development or extra dependency files.


## Training Launchers

The scripts in `scripts/train/` include recommended training hyperparameters. To run them, set dataset and checkpoint paths through environment variables rather than editing private paths into the repository. For example:

```bash
TRAIN_CSV=/path/to/manifest_time_train.csv \
TEST_CSV=/path/to/manifest_time_test.csv \
WEIGHTS=weights/panopticon_vitb14_teacher.pth \
bash scripts/train/run_query.sh
```

The main MethaneFuse launchers use relative default checkpoint paths under `weights/` and `checkpoints/`. External baselines such as AnySat and SatMAE additionally require their own local repository or pretrained checkpoint paths.

## Local-only Files

The dataset is not stored in this repository. It is maintained separately and hosted on Hugging Face.

The following local directories are ignored by Git and should not be pushed to the official repository: `data/`, `data_csv/`, `datasets/`, `examples/`, `logs/`, `weights/`, `checkpoints/`, `results/reports/generated/`, `experiments_tryout/`, `tools/`, and non-training `scripts/`.

## Baselines

Panopticon is used as a baseline/backbone dependency in this project. It is not the identity of this repository.
