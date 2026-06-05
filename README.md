<p align="center">
  <img src="assets/methanefuse_overview.png" alt="MethaneFuse overview" width="100%">
</p>

# MethaneFuse

**MethaneFuse** is a two-stage learning framework for methane plume detection from partial multi-sensor satellite observations. It learns from whichever satellite observations are available for a plume event, including Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P.

MethaneFuse is built for real methane monitoring settings where satellite coverage is incomplete because of revisit timing, cloud coverage, acquisition quality, mission coverage, and the transient lifetime of methane emissions.

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#method-overview">Method</a> •
  <a href="#main-results">Results</a> •
  <a href="#methaneunion-dataset">Dataset</a> •
  <a href="#training-and-evaluation">Training & Evaluation</a>
</p>

---

## Highlights

* **Partial multi-sensor learning:** supports plume events with different available sensor sets.
* **Two-stage training framework:** combines sensor-native pretraining and query-level adaptation.
* **Multi-sensor satellite inputs:** Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P.
* **Classification and segmentation:** supports query-level methane plume detection and plume mask prediction.
* **Reproducible evaluation:** includes training, evaluation, baseline, and dataset preparation code.

---

## Method Overview

MethaneFuse contains two stages.

**Stage 1: Sensor-native pretraining.**
The model learns methane-aware representations from sensor-native temporal observations. Each available sensor is tokenized and encoded independently, and available sensor representations are aggregated through masked sensor-set fusion.

**Stage 2: Query-level adaptation.**
The pretrained representation is adapted to scale-controlled plume classification and segmentation. The encoder is frozen, while lightweight sensor-aware LoRA experts and task heads are trained for downstream prediction.

<p align="center">
  <img src="assets/methanefuse_model.png" alt="MethaneFuse model" width="100%">
</p>

---

## Main Results

At the 480 m query footprint, MethaneFuse improves over independently trained per-sensor predictors, heuristic score fusion, and generic Earth observation representation transfer.

| Method          |      F1 ↑ |     Acc ↑ |     FPR ↓ |  Recall ↑ |   AUROC ↑ |
| --------------- | --------: | --------: | --------: | --------: | --------: |
| Per-sensor ViT  |     79.22 |     78.00 |     23.06 |     78.94 |     85.32 |
| SatMAE-FT       |     67.60 |     63.33 |     48.41 |     74.44 |     67.61 |
| AnySat-FT       |     58.90 |     58.96 |     37.53 |     55.82 |     62.48 |
| Panopticon-FT   |     77.65 |     75.61 |     29.13 |     79.80 |     83.28 |
| **MethaneFuse** | **84.87** | **84.21** | **14.87** | **83.40** | **93.62** |

---

## MethaneUnion Dataset

MethaneFuse is trained and evaluated on **MethaneUnion**, an event-centered partial multi-sensor dataset constructed from Carbon Mapper plume reports and matched satellite observations.

MethaneUnion includes observations from:

* Sentinel-2 Level-2A surface reflectance
* Landsat 8/9 Collection 2 Level-2 surface reflectance
* EMIT Level-2A hyperspectral surface reflectance
* Sentinel-5P Level-2 methane products

<p align="center">
  <img src="assets/methaneunion_pipeline.png" alt="MethaneUnion dataset pipeline" width="85%">
</p>

Dataset and preprocessing resources:

* Dataset: https://huggingface.co/datasets/yuyao42/MethaneUnion
* Dataset construction pipeline: https://github.com/yuyao-wang/MethaneUnion

---

## Quick Start

The commands below assume a fresh machine with `conda`, `git`, and `huggingface_hub` installed.

```bash
# 1. Clone the repository.
git clone https://github.com/yuyao-wang/MethaneFuse.git
cd MethaneFuse

# 2. Create the Python environment.
conda create -n methanefuse python=3.10 -y
conda activate methanefuse
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"

# 3. Download MethaneUnion.
mkdir -p data/MethaneUnion
huggingface-cli download yuyao42/MethaneUnion \
  --repo-type dataset \
  --local-dir data/MethaneUnion

# 4. Download released MethaneFuse checkpoints.
mkdir -p checkpoints
huggingface-cli download yuyao42/MethaneFuse-checkpoints \
  --repo-type model \
  --local-dir checkpoints
```

Run a 480 m classification evaluation:

```bash
python scripts/eval/evaluate_classification.py \
  --eval_csv data/MethaneUnion/datasets/temporal_split/480m_GSD/test.csv \
  --checkpoint checkpoints/stage2_classification_480m.pt \
  --stage b \
  --batch_size 16 \
  --num_workers 8 \
  --row_fusion_mode max \
  --output_json results/eval/classification_480m.json
```

Inspect a MethaneUnion sample:

```bash
python examples/load_methaneunion_sample.py \
  --csv data/MethaneUnion/datasets/temporal_split/480m_GSD/test.csv \
  --index 0
```

Visualize a prediction row:

```bash
python examples/visualize_prediction.py \
  --csv data/manifests/manifest_time_test_480m_all_samples_with_pred_metrics.csv \
  --index 0 \
  --output results/examples/prediction_row0.png
```

---

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
├── examples/                # Inference, visualization, and data-loading examples
├── assets/                  # README figures
├── docs/                    # Additional documentation
├── environments/            # Environment files
└── README.md
```

---

## Training and Evaluation

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

---

## Data and Checkpoints

* MethaneUnion dataset: https://huggingface.co/datasets/yuyao42/MethaneUnion
* MethaneUnion construction pipeline: https://github.com/yuyao-wang/MethaneUnion
* MethaneFuse checkpoints: https://huggingface.co/yuyao42/MethaneFuse-checkpoints

---

## License

This repository is released for research use. Please see the license file for details.

---

## Contact

For questions, please open an issue or contact the maintainer.
