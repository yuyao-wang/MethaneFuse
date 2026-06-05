<p align="center">
  <img src="assets/methanefuse_overview.png" alt="MethaneFuse overview" width="100%">
</p>

**MethaneFuse** is a two-stage learning framework for methane plume detection from naturally available multi-sensor satellite observations. MethaneFuse is designed for partial multi-sensor observation settings, where each plume event may only have a subset of Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P observations available at the target location and time. The repository contains the implementation of MethaneFuse, dataset preparation scripts for MethaneUnion, training and evaluation code, baseline implementations, and released checkpoints.

## Overview

Satellite methane plume detection is constrained by incomplete sensor availability. Sentinel-2 provides useful fine-resolution SWIR evidence, but many reported plume events do not have valid Sentinel-2 observations because of revisit timing, cloud coverage, acquisition quality, and the transient lifetime of methane emissions.

MethaneFuse addresses this setting by learning from the irregular union of available satellite observations. It combines sensor-native temporal encoding, masked sensor-set fusion, and lightweight sensor-aware adaptation for query-level methane plume classification and segmentation.

## Method

MethaneFuse consists of two stages:

* **Stage 1: Sensor-native pretraining.**
  The model learns methane-aware representations from sensor-native temporal observations. Each available sensor is tokenized and encoded independently, and available sensor representations are aggregated through masked sensor-set fusion.

* **Stage 2: Query-level adaptation.**
  The pretrained representation is adapted to scale-controlled plume classification and segmentation. The encoder is frozen, while lightweight sensor-aware LoRA experts and task heads are trained for downstream prediction.

<p align="center">
  <img src="assets/methanefuse_model.png" alt="MethaneFuse model" width="100%">
</p>

## MethaneUnion Dataset

MethaneUnion is an event-centered partial multi-sensor dataset constructed from Carbon Mapper plume reports and matched satellite observations from:

* Sentinel-2 Level-2A surface reflectance
* Landsat 8/9 Collection 2 Level-2 surface reflectance
* EMIT Level-2A hyperspectral surface reflectance
* Sentinel-5P Level-2 methane products

Datasets are available at https://huggingface.co/datasets/yuyao42/MethaneUnion.

<p align="center">
  <img src="assets/methaneunion_pipeline.png" alt="MethaneUnion dataset pipeline" width="85%">
</p>

## Main Results

At the 480 m query footprint, MethaneFuse improves over independently trained per-sensor predictors, heuristic score fusion, and generic Earth observation representation transfer.

| Method          |      F1 ↑ | Acc ↑ |     FPR ↓ |  Recall ↑ |   AUROC ↑ |
| --------------- | --------: | -------: | --------: | --------: | --------: |
| Per-sensor ViT  |     79.22 |      78.00 |     23.06 |     78.94 |     85.32 |
| SatMAE-FT       |     67.60 |      63.33 |     48.41 |     74.44 |     67.61 |
| AnySat-FT       |     58.90 |      58.96 |     37.53 |     55.82 |     62.48 |
| Panopticon-FT   |     77.65 |      75.61 |     29.13 |     79.80 |     83.28 |
| **MethaneFuse** | **84.87** |  **84.21** | **14.87** | **83.40** | **93.62** |

## Repository Structure

```text
MethaneFuse/
├── src/
│   ├── models/              # MethaneFuse model components
│   ├── data/                # Dataset loading and preprocessing utilities
│   └── utils/               # Shared utilities
├── baselines/               # Per-sensor ViT, U-Net, SatMAE, AnySat, and Panopticon baselines
├── configs/                 # Training and evaluation configs
├── scripts/
│   ├── train/               # Training launchers
│   └── eval/                # Evaluation launchers
├── figures/                 # Paper and README figures
├── docs/                    # Additional documentation
├── environments/            # Environment files
└── README.md
```

## Installation

```bash
git clone https://github.com/yuyao-wang/MethaneFuse.git
cd MethaneFuse

conda create -n methanefuse python=3.10
conda activate methanefuse
pip install -r requirements.txt
```

## Training

Stage 1 pretraining:

```bash
bash scripts/train/run_fuse.sh
```

Stage 2 query-level classification:

```bash
bash scripts/train/run_query.sh
```

Segmentation training:

```bash
bash scripts/train/run_seg.sh
```

## Evaluation

Classification evaluation:

```bash
python scripts/eval/evaluate_classification.py \
  --config configs/eval_480m.yaml
```

Segmentation evaluation:

```bash
python scripts/eval/evaluate_segmentation.py \
  --config configs/eval_segmentation.yaml
```

## Data and Checkpoints

The processed MethaneUnion dataset is maintained separately because the repository does not store raw satellite products or local experiment files. Dataset preparation scripts, processed manifests, and trained checkpoints will be released after cleanup, subject to the redistribution policies of the original data providers.

Expected checkpoint structure:

```text
checkpoints/
├── stage1_pretrained.pt
├── stage2_classification_120m.pt
├── stage2_classification_360m.pt
├── stage2_classification_480m.pt
├── stage2_classification_960m.pt
└── segmentation/
```
