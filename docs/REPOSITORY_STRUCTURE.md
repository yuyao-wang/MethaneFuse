# Repository Structure

This repository is organized around reproducibility for the paper rather than around intermediate experiment outputs.

## Top-level directories

- `configs/`: YAML or JSON configuration files for data, training, evaluation, and figure generation.
- `scripts/`: command-line entry points. Core reusable logic should live in Python modules, not in shell scripts.
- `data/`: lightweight manifests, split files, or examples. Full datasets must not be committed here.
- `external/`: archived helpers for external dataset/model release workflows.
- `figures/`: paper figures and generated figure assets.
- `results/`: small metrics, logs, and report artifacts used to document reproduced results.
- `docs/`: documentation beyond the README.
- `baseline/`: baseline implementations.
- `dinov2/`: inherited Panopticon/DINOv2 code kept in place for compatibility.
- `universal_models/` and `universal_models_fusion/`: current project model implementations.

## Current compatibility note

The `dinov2/`, `baseline/`, `universal_models/`, and `universal_models_fusion/` directories are intentionally kept at their current import paths during the first cleanup pass. A later refactor can move them under a new project package once scripts and imports are updated together.
