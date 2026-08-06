# Historical S2 Results Audit

## Bottom Line

- Historical S2 performance is consistently high across multiple independent runs. The signal is real and is not explained only by one lucky experiment.
- The strongest clean evidence is the old standalone Panopticon checkpoint: it still reaches `accuracy=0.9032`, `F1=0.9014`, and `AUROC=0.9534` on test rows whose canonical events do not occur in its training set.
- The old random-initialized ResNet18 experiment also reported high accuracy, but its original split is not a clean generalization benchmark. The notebook first merged the old train/test CSVs and then randomly split individual crop rows. Reconstructing that split shows essentially every Permian test crop has a sibling crop from the same plume and event in training.
- On the current pure-GEE cohort, controlled `t0-only`, old-equivalent `3-time`, and `6-time` Panopticon experiments all plateau near `0.81` best-threshold F1. Therefore six temporal visits are not the cause of the gap.
- The current result is abnormal relative to the historical S2 guardrail. The evidence points primarily to a changed data cohort, crop-generation contract, and evaluation split rather than a broken optimizer, double normalization, or an intrinsically weak S2 signal.

## Historical Standalone S2 Panopticon

| Scale/data | W&B run | Test accuracy | Test AUROC |
|---|---:|---:|---:|
| Legacy 120 m | `run-20260424_043603-1j0ks9hf` | `0.8782` | `0.9553` |
| Legacy 360 m | `run-20260422_185210-7c09ty1p` | `0.8934` | `0.9558` |
| Legacy 480 m | `run-20260424_173618-qzkl321t` | `0.8755` | `0.9509` |
| Legacy 480 m variant | `run-20260428_211123-l7wnewfp` | `0.8869` | `0.9514` |

The old 360 m checkpoint was evaluated again after removing every test row whose canonical event appears in training:

| Subset | Rows | Accuracy | F1 | AUROC |
|---|---:|---:|---:|---:|
| Overlapping events | `12,319` | `0.8913` | `0.8943` | `0.9562` |
| Non-overlapping events | `2,635` | `0.9032` | `0.9014` | `0.9534` |

This result rejects the explanation that the old near-`0.90` score was mainly caused by event leakage.

## Historical S2-Specific Heads

Multi-sensor experiments independently show the same S2 strength:

| W&B run | S2 evaluation stratum | Accuracy | AUROC |
|---|---|---:|---:|
| `run-20260411_030850-cl4gxba8` | reference head, S2-only rows | `0.9260` | `0.9730` |
| `run-20260411_030850-cl4gxba8` | fused head, S2-only rows | `0.9265` | `0.9715` |
| `run-20260421_201624-mqvrit3j` | reference head, S2-only rows | `0.9003` | `0.9626` |
| `run-20260421_201624-mqvrit3j` | fused head, all S2 rows | `0.9006` | `0.9630` |
| `run-20260401_053813-j71m081c` | reference head, S2-only rows | `0.9162` | `0.9663` |

## Original ResNet18 Experiment

The preserved configuration shows:

- Randomly initialized `ResNet18`.
- Twelve S2 bands.
- `200` epochs, batch size `256`, AdamW learning rate `5e-4`.
- `data_range=all_time`, which randomly chooses one of `t0`, `t-90`, or `t-360` for each sample access.
- Permian-only geographic filtering.
- Input CSVs named `train_shuffled.csv` and `test_shuffled.csv`.

The original output directory and W&B summary are no longer present locally, so the exact reported final accuracy cannot be independently recovered from the current disks. The data-generation code is preserved, however, and it reveals a major benchmark flaw:

1. Read the old `train.csv`.
2. Read the old `test.csv`.
3. Concatenate them.
4. Run `train_test_split(..., test_size=0.1, random_state=43, shuffle=True)` on individual crop rows.

Reconstructing that exact split on the preserved old S2 CSV gives:

| Dataset | Scope | Test rows | Test rows with same plume in train | Test rows with same event in train |
|---|---|---:|---:|---:|
| `plume_s2_90360_32` | All | `5,870` | `5,862` (`99.864%`) | `5,868` (`99.966%`) |
| `plume_s2_90360_32` | Permian | `1,889` | `1,889` (`100%`) | `1,889` (`100%`) |
| `plume_s2_90360_32_fixed_512` | Permian | `9,435` | `9,434` (`99.989%`) | `9,435` (`100%`) |
| `s2_90360_temporal_CDSE0_gee90360_2025_16` | Permian | `1,993` | `1,993` (`100%`) | `1,993` (`100%`) |

The row-random ResNet18 score therefore measures recognition of new crops from almost entirely already-seen plume/event sites, not unseen-event temporal generalization. It remains useful as a data-signal smoke test, but it cannot be compared directly with the current event-disjoint chronological split.

## Current Pure-GEE Controlled Ablation

All three runs use the same current cohort and strict split:

| Temporal input | Best train F1 | Best-threshold test F1 | Test AUROC |
|---|---:|---:|---:|
| `t0-only` | `0.8839` | `0.8118` | `0.8801` |
| Legacy-equivalent `3-time` | `0.9357` | `0.8147` | `0.8845` |
| Full `6-time` | `0.8865` | `0.8124` | `0.8797` |

The near-identical `1/3/6-time` results rule out the number of visits as the main cause.

## Material Benchmark Differences

- The old near-`0.90` experiment and current pure-GEE experiment share zero exact plume IDs in their trained cohorts.
- The old benchmark uses a `360 m` S2 field of view (`36×36` native pixels), while the current GEE pipeline uses `320 m` (`32×32` native pixels).
- Old positive crop centers have a median radial offset of about `140 m`; current positive crops have a median radial offset of about `51 m`.
- The current split is canonical-event disjoint and chronological: training ends on `2025-12-22`, while testing starts on `2025-12-24` and extends to `2026-05-24`.
- Current channel normalization occurs once as `(image - mean) / std`; it is not followed by an additional forced `[0,1]` scaling.
- The current model reaches high training F1, so the optimizer and backbone can fit the input. The failure is primarily out-of-sample transfer.

## Root-Cause Ranking

1. **Changed cohort and data-generation contract:** strongest explanation because the old and new benchmarks contain different plume populations and use different crop geometry.
2. **Much stricter chronological/event-disjoint evaluation:** contributes to the drop, although it cannot alone explain why the old checkpoint remains near `0.90` on its own event-nonoverlap subset.
3. **Current temporal fusion implementation:** repeated channel concatenation discards explicit visit identity and explains why extra visits do not help, but not why `t0-only` is also only about `0.81`.
4. **Normalization or optimizer failure:** not supported by the code or training curves.
5. **Six-time input itself:** rejected by the controlled ablation.

## Decisive Next Experiment

Use the same old legacy plume cohort and the same `360 m` crop coordinates, regenerate only the S2 imagery through the current GEE source, and evaluate both:

1. the historical row-random split as a signal smoke test; and
2. a canonical-event-disjoint split as the real benchmark.

This matched-cohort experiment isolates imagery/crop processing from cohort and split effects. Adding more epochs to the current unmatched benchmark cannot answer that question.

## Evidence Files

- `Upgraded_dataset/diagnostics/s2_old_vs_new_benchmark_audit.json`
- `Upgraded_dataset/diagnostics/s2_old_near90_event_overlap/old_checkpoint_nonoverlap_fp32.json`
- `Upgraded_dataset/diagnostics/s2_old_near90_event_overlap/old_checkpoint_overlap_fp32.json`
- `wandb/run-20260422_185210-7c09ty1p/files/wandb-summary.json`
- `wandb/run-20260424_043603-1j0ks9hf/files/wandb-summary.json`
- `wandb/run-20260424_173618-qzkl321t/files/wandb-summary.json`
- `wandb/run-20260428_211123-l7wnewfp/files/wandb-summary.json`
- `wandb/run-20260411_030850-cl4gxba8/files/wandb-summary.json`
- `wandb/run-20260421_201624-mqvrit3j/files/wandb-summary.json`
- `wandb/run-20260401_053813-j71m081c/files/wandb-summary.json`
- `/home/yuyao/methane_train/train/train_config/finetune/byol_resnet18_l2a_finetune_32_resize128.yaml`
- `/home/yuyao/methane_train/data_preprocess/data_preprocess.ipynb`
- `/home/yuyao/methane_train/train/dataset/MethaneGEEL2AClassificationDataset.py`
