# Methane pretraining research log — 2026-07-27

The original sensor test CSVs remain sealed during model and input selection.
Single-sensor rows below use deterministic event-disjoint
train/recent-validation manifests unless explicitly marked as a historical
result.

> **Joint-protocol invalidation (2026-07-27):** A later global audit found 124
> canonical events in `union(sensor train) ∩ union(sensor validation)`.  The
> per-sensor splits are individually disjoint but are not a valid split for a
> model whose encoder is shared across sensors.  Every four-sensor
> shared/independent result produced under that original split is retained only
> as a systems diagnostic and must not be used for model selection, research
> claims, or sealed-test evaluation.  Joint experiments must use one global
> event assignment.  The later `four_sensor_global_val_purge_v1` MAE/scratch
> comparison explicitly satisfies this requirement and is labeled separately
> as valid negative evidence.

> **Registry enforcement (2026-07-27):** `collect_results.py` now excludes any
> history carrying `protocol_status=invalid_diagnostic_only`,
> `do_not_use_for_model_selection=true`, `protocol_valid=false`, or a
> `PROTOCOL_INVALID.json` sidecar.  Its current audit discovered 20 metric
> histories, four sealed-result JSONs, and one promotion-gate JSON; it excluded
> all five invalid joint histories and recorded zero parse failures.  The
> registry contains 14 validation records, one globally purged pretraining
> record, four separately named `sealed_test` records, and one
> `promotion_gate` record.  The failed gate remains indexed because
> `do_not_promote` is a valid negative decision, not a protocol exclusion;
> sealed metrics use only `sealed_*` fields.

## Historical reference results

| Sensor | Historical protocol | F1 | AUROC | Caveat |
|---|---|---:|---:|---|
| S2 | six-time Panopticon full fine-tune | 0.764 | 0.836 | checkpoint selected on the historical test split |
| L89 | hard-event scratch | 0.754 | 0.853 | hard-event test was conditioned on an earlier model |
| EMIT32 | six-time full set | 0.567 | 0.711 | historical sensor-specific split |
| S5P | six-time new-data run | 0.638 | 0.650 | historical sensor-specific split |

These numbers are context only and are not directly comparable with the new
recent-validation protocol.

## New controlled experiments

| Run | Train / val rows | Epoch | F1@0.5 | Best val F1 | AP | AUROC | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| S2 current+residual concat pilot | 2,048 / 2,048 | 1 | 0.392 | not recorded in early runner | 0.680 | 0.736 | stopped: residuals should not share the reflectance stem naively |
| S2 current-only pilot | 2,048 / 2,048 | 1 | 0.742 | not recorded in early runner | 0.786 | 0.767 | promoted |
| S2 raw-six-time pilot | 2,048 / 2,048 | 1 | 0.688 | not recorded in early runner | 0.822 | 0.840 | promoted |
| S2 current+residual temporal-token pilot | 2,048 / 2,048 | 1 | 0.000 | 0.726 | 0.731 | 0.753 | better than naive concat, still below raw/current |
| S2 current-only cap8 | 18,747 / 16,546 | 1 | 0.771 | 0.775 | 0.839 | 0.842 | continued |
| S2 current-only cap8 | 18,747 / 16,546 | 2 | 0.779 | 0.780 | **0.863** | 0.859 | AP-selected checkpoint |
| S2 current-only cap8 | 18,747 / 16,546 | 3 | **0.787** | **0.788** | 0.862 | 0.859 | finished; epoch 2 had already been selected by validation AP |
| S5P current-only pilot | 2,048 / 2,048 | 1 | 0.000 | 0.668 | 0.556 | 0.533 | near-random ranking; do not scale Panopticon branch |
| S5P current-only Panopticon pilot | 2,048 / 2,048 | 3 | 0.000 | 0.668 | 0.557 | 0.536 | stopped |
| S5P current+residual Panopticon pilot | 2,048 / 2,048 | 1 | 0.000 | 0.668 | 0.540 | 0.517 | stopped |
| S5P current+residual ResNet18 | 18,772 / 3,805 | 3 | 0.537 | 0.685 | 0.559 | 0.537 | best AP 0.565 at epoch 1; residual branch rejected |
| S5P raw-six-time ResNet18 | 18,772 / 3,805 | 3 | 0.604 | 0.685 | **0.568** | 0.554 | weak but better than residual ResNet |
| S5P raw-six-time Panopticon | 18,772 / 3,805 | 1 | 0.641 | **0.686** | 0.596 | 0.568 | continued |
| S5P raw-six-time Panopticon | 18,772 / 3,805 | 2 | **0.663** | 0.685 | 0.641 | 0.611 | continued |
| S5P raw-six-time Panopticon | 18,772 / 3,805 | 3 | 0.637 | 0.685 | **0.644** | **0.622** | finished; AP-selected checkpoint is epoch 3 |
| S5P Copernicus-FM frozen proxy screen | 2,048 / 2,048 | 1 | 0.582 | 0.668 | 0.572 | 0.566 | stopped by the prespecified reference gate; no full run |
| S2 raw-six-time cap8 | 18,747 / 16,546 | 1 | 0.701 | 0.769 | 0.811 | 0.823 | early-stopped after safe checkpoint write |
| S2 UniverSat frozen current-only | 18,747 / 16,546 | 3 | 0.671 | 0.723 | 0.731 | 0.755 | stopped: substantially below Panopticon current-only |
| S2 UniverSat frozen raw-six-time | 18,747 / 16,546 | 3 | 0.646 | 0.731 | 0.735 | 0.763 | stopped: temporal input did not close the domain/task gap |
| EMIT raw-three-time frozen Panopticon pilot | 2,048 / 2,048 | 1 | 0.191 | 0.685 | 0.681 | 0.670 | matched pilot baseline |
| EMIT residual frozen Panopticon pilot | 2,048 / 2,048 | 1 | 0.427 | 0.685 | **0.705** | **0.696** | promoted: AP +0.024 and AUROC +0.026 vs matched raw input |
| EMIT residual full Panopticon | 11,688 / 5,201 | 1 | 0.506 | 0.617 | 0.674 | 0.728 | continued; matched full raw run is concurrent |
| EMIT residual full Panopticon | 11,688 / 5,201 | 2 | 0.511 | 0.619 | 0.668 | 0.730 | stopped: AP declined; epoch 1 remains selected |
| EMIT raw-three-time full Panopticon | 11,688 / 5,201 | 1 | 0.339 | **0.639** | 0.666 | **0.734** | continued one matched epoch; residual has AP +0.008 but raw has best-F1 +0.022 |
| EMIT raw-three-time full Panopticon | 11,688 / 5,201 | 2 | 0.336 | 0.600 | 0.566 | 0.635 | stopped: unfreezing collapsed ranking; epoch 1 remains the raw comparator |
| L89 raw-three-time frozen Panopticon pilot | 2,048 / 2,048 | 1 | 0.643 | not recorded | **0.735** | **0.715** | promoted as the matched L89 input baseline |
| L89 residual frozen Panopticon pilot | 2,048 / 2,048 | 1 | 0.663 | not recorded | 0.688 | 0.685 | stopped: AP -0.046 and AUROC -0.030 vs matched raw input |
| L89 raw-three-time full Panopticon | 10,049 / 9,614 | 1 | 0.589 | not recorded | 0.696 | 0.736 | continued; resumed epochs 2–3 at batch 24 after the safe epoch-1 checkpoint |
| L89 raw-three-time full Panopticon | 10,049 / 9,614 | 2 | 0.457 | not recorded | 0.677 | 0.727 | stopped: full-backbone fine-tuning reduced AP by 0.019; epoch 1 remains selected |

### Copernicus-FM S5P frozen screen

This screen used the official Copernicus-FM ViT-B checkpoint with all
139,452,544 backbone parameters frozen.  Because the released variable
embeddings contain S5P CO, NO2, SO2, and O3 but not CH4, the methane proxy was
predeclared as the arithmetic mean of exactly those four 2,048-dimensional
embeddings.  The proxy was fixed before training and was never selected on
validation results.

- deterministic balanced selection: 1,024 negatives and 1,024 positives in
  each split, seed `20260727`;
- normalization: selected-training-only mean `1896.829995`, standard deviation
  `37.756537`;
- official variable-input geometry: each of six frames resized from 224 to 56,
  kernel size 4;
- metadata: all unknown (`NaN`), invoking the official learned unknown tokens
  and preventing a location shortcut;
- cached feature shape: six frozen 768-dimensional frame embeddings per row;
- head: temporal mean and max concatenation, LayerNorm, and one linear
  two-class layer;
- three head epochs maximum; complete 2,048-row selected validation evaluated
  after every epoch.

| Head epoch | Train loss | F1@0.5 | Best F1 | AP | AUROC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7328 | **0.5815** | 0.6678 | **0.5720** | 0.5661 |
| 2 | 0.7219 | 0.6667 | **0.6680** | 0.5637 | **0.5696** |
| 3 | 0.6923 | 0.5249 | 0.6673 | 0.5589 | 0.5686 |

Validation AP selects epoch 1.  Its best-F1 threshold is `0.425912`, with
confusion counts `TN=7, FP=1017, FN=1, TP=1023`.  Thus the nominal best F1 is
essentially the balanced-set all-positive baseline (`2/3`) and is not evidence
of useful discrimination.  Ranking metrics are the correct screen signal:
best AP is `0.572046` and AUROC is `0.566066`.  The prespecified screen gate
used the promoted Panopticon full-validation AP `0.644` as a reference and
fired (reference difference `-0.071954`), so no full-data Copernicus-FM run
was launched.  Because Copernicus-FM used a deterministic balanced 2,048-row
validation subset while `0.644` uses all 3,805 rows, this difference is a
resource-allocation gate, not a formal matched head-to-head effect size.

Exact command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/yuyao/miniconda3/envs/train/bin/python \
  research/pretraining_20260727/copernicusfm_s5p_probe.py \
  --train_csv /diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/s5p/train.csv \
  --eval_csv /diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/s5p/val.csv \
  --repo /diniuvol/yuyao/methanefuse_research_20260727/external/Copernicus-FM/Copernicus-FM \
  --checkpoint /diniuvol/yuyao/methanefuse_research_20260727/external/Copernicus-FM/Copernicus-FM/weights/CopernicusFM_ViT_base_varlang_e100.pth \
  --variable_embeddings /diniuvol/yuyao/methanefuse_research_20260727/external/Copernicus-FM/Copernicus-FM/weights/varname_embed_llama3.2_1B.pt \
  --cache_root /diniuvol/yuyao/methanefuse_research_20260727/cache/copernicusfm_s5p_probe \
  --output_json /home/yuyao/panopticon/research/pretraining_20260727/copernicusfm_s5p_screen_result.json \
  --max_train 2048 --max_eval 2048 --seed 20260727 \
  --stats_workers 1 --num_workers 1 --extract_batch_size 16 \
  --head_batch_size 256 --epochs 3 --amp_dtype bfloat16 \
  --device cuda:0 --progress_every 20 --promotion_ap 0.644
```

The first implementation smoke attempts did not reach model inference:
the official source needed both its repository root and `src/` on
`PYTHONPATH`, and the `methane` environment lacked `einops`.  The runner now
loads both official import roots and the successful run uses the existing
`train` environment; no dependency was installed.  The successful 8/8
end-to-end smoke preceded the formal screen.

Compact result:
`research/pretraining_20260727/copernicusfm_s5p_screen_result.json`.
Frozen feature cache:
`/diniuvol/yuyao/methanefuse_research_20260727/cache/copernicusfm_s5p_probe/1e69e3b0f781bbed`.

### Four-sensor shared-residual screen — protocol-invalid diagnostic

The metrics in this subsection are **protocol-invalid diagnostics**.  Both
models use normalized `t0`, `t0-prev1`, and `t0-seasonal` streams,
sensor-specific patch stems and heads, balanced round-robin sensor batches,
30 train batches per sensor, and the same 30-batch validation prefixes.
Because the S5P validation manifest is label ordered, its prefix contains only
positives; S5P AP/AUROC and the screen macro AP/AUROC therefore exclude S5P.
A full-validation run is required before drawing a four-sensor conclusion.

| Model | Parameters | Macro F1@0.5 | Macro best F1 | Macro AP | Macro AUROC | Epoch wall time |
|---|---:|---:|---:|---:|---:|---:|
| Shared transformer + sensor stems | 23,319,556 | 0.3720 | **0.7238** | **0.5226** | 0.6013 | 482.2 s |
| Independent transformer per sensor | 47,066,116 | **0.4087** | 0.7195 | 0.5206 | **0.6033** | 526.3 s |

The shared model uses 50.5% fewer parameters with effectively matched ranking
metrics (`AP +0.0020`, `AUROC -0.0020`).  Its AP relative to the independent
control is `-0.0017` on S2, `-0.0058` on L89, and `+0.0135` on EMIT.  This is
evidence of parameter-efficient non-degradation, not yet evidence that sharing
improves every sensor.  The first independent attempt was deliberately
interrupted during validation to relieve CPU oversubscription; the table uses
the clean three-worker retry.

| Sensor | Shared AP | Independent AP | Shared AUROC | Independent AUROC | Shared best F1 | Independent best F1 |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.6155 | 0.6171 | 0.6500 | 0.6585 | 0.7013 | 0.6993 |
| L89 | 0.4202 | 0.4260 | 0.5014 | 0.5058 | 0.5714 | 0.5714 |
| EMIT | 0.5321 | 0.5187 | 0.6524 | 0.6455 | 0.6226 | 0.6074 |
| S5P prefix | invalid | invalid | invalid | invalid | 1.0000 | 1.0000 |

The promoted shared model then completed one full balanced epoch (158 batches
per sensor; 632 optimizer steps) and evaluated every recent-validation row.
The epoch took 1,709.3 seconds: approximately 15.8 minutes for training and
12.7 minutes for full validation.

| Sensor | Val rows | Positive rate | AP | AUROC | F1@0.5 | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 16,546 | 0.5021 | **0.7364** | **0.7458** | **0.6987** | **0.7216** |
| L89 | 9,614 | 0.4327 | 0.5454 | 0.6365 | 0.5716 | 0.6184 |
| EMIT | 5,201 | 0.3957 | 0.5111 | 0.6216 | 0.5361 | 0.5763 |
| S5P | 3,805 | 0.5209 | 0.5779 | 0.5336 | 0.1249 | 0.6851 |
| Macro over sensors | — | — | 0.5927 | 0.6344 | 0.4828 | 0.6504 |

Every sensor ranked above its positive-rate baseline and had AUROC above 0.5,
but the cross-sensor event leakage prevents interpreting that observation as
generalization.  Epoch 2 training completed, then validation was stopped as
soon as the global leakage audit became available.  No epoch-2 metrics are
valid or recorded.  All affected artifact directories contain an explicit
`PROTOCOL_INVALID.json` marker and their `metrics_history.json` files are
tagged `protocol_status=invalid_diagnostic_only`.

### Shared Panopticon 30-round gate — protocol-invalid diagnostic

The frozen official Panopticon shared-backbone gate used 30 balanced rounds,
seed 20260727, EMIT residual input, full validation, and batches S2 64, L89
24, S5P 4, and EMIT 12. S5P was reduced from the proposed 32 to the
dedicated-run-proven batch 4 because six 224×224 frames use standard
attention without xFormers. Training completed without OOM or bad-sample
replacement. S2 full validation completed on 16,546 rows (positive rate
0.5021, AP 0.6867, AUROC 0.7308, F1@0.5 0.4711, best F1 0.7172), after which
the run was interrupted during L89 validation immediately when the 124-event
global leakage audit became available. These S2 values are retained only to
prove the pipeline ran; they are not a baseline and must not be compared for
model selection. No checkpoint was written. The run directory is marked
`PROTOCOL_INVALID.json` and `run_status.json` says
`protocol_invalid_diagnostic_only`.

Machine-readable records are regenerated under
`/diniuvol/yuyao/methanefuse_research_20260727/results/experiment_registry.*`.

## Unified-data preparation audit

The S2 canonical-mask production run resolved all 4,097 selected source rows.
Nearest-neighbour reprojection wrote 4,090 masks.  Seven non-empty raw masks at
0.6–1.2 m resolution were missed only because their positive source pixels
were sub-pixel relative to the 20 m destination grid.  An explicit opt-in
fallback quantized real positive source-pixel centres into their containing
minimum target cells; it did not dilate points or rasterize an invented
polygon.  The seven fallback outputs contain 3–6 positive cells each and all
overlap both raw and catalogue plume bounds.

A subsequent no-overwrite full readback validated 4,097/4,097 outputs.  The
canonical artifacts are:

```text
/diniuvol/yuyao/methanefuse_research_20260727/cache/s2_512_masks
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_subpixel_fallback.audit.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/s2_mask_final_readback.audit.json
```

The unified-crop metadata smoke then selected two globally split plumes with
sensor masks `1100` (`S2|L89`) and `1111` (all four sensors).  It planned four
tasks (two per label), with no dropped sensor and no crop files materialized.
The planner uses real t0 image georeferencing for L89/EMIT and the repaired
mask grid only for identity-array S2; mask diagnostics are independently
mapped by query longitude/latitude.  This corrected a 15 m L89 mask/image
half-pixel offset exposed by the first smoke.
Two positive tasks were read across all six source times.  The partial-sensor
case passed with S2/L89 finite-data masks `111111`.  The all-four case
correctly emitted a warning: S2/L89 were `111111`, EMIT was `010011`
(`t0`, `prev2`, and `prev3` all-NaN at the query), and S5P recomputed a
different nearest mapping independently from each timepoint's geolocation
array.  File presence is therefore kept separate from query-level finite-data
presence before any full crop is authorized.

```text
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.audit.json
/diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.validation.csv
```

## Frozen sealed-test evaluations

The following threshold was fixed on validation before opening the original
test manifest.  No test-time threshold search or checkpoint selection was
performed.

| Sensor / selected model | Rows | Fixed val threshold | Positive F1 | Macro-F1 | AP | AUROC | Recall | FPR | R@FPR 5% | Confusion (`TP/FP/FN/TN`) | Development-event overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 current-only Panopticon, val-AP epoch 2 | 23,413 | 0.520996 | 0.738 | 0.702 | 0.815 | 0.805 | 0.822 | 0.411 | 0.383 | 9,678 / 4,781 / 2,093 / 6,861 | not recorded in legacy result artifact |
| L89 raw-three-time Panopticon, val-AP epoch 1 | 7,942 | 0.314209 | 0.737 | 0.736 | 0.775 | 0.816 | 0.836 | 0.344 | 0.347 | 2,942 / 1,521 / 578 / 2,901 | **0** |
| EMIT residual-three-time Panopticon, val-AP epoch 1 | 6,138 | 0.228882 | 0.611 | 0.591 | 0.666 | 0.691 | 0.742 | 0.522 | 0.232 | 1,963 / 1,823 / 681 / 1,671 | **0** |
| S5P raw-six-time Panopticon, val-AP epoch 3 | 3,213 | 0.118713 | 0.687 | 0.343 | 0.657 | 0.638 | 1.000 | 1.000 | 0.154 | 1,680 / 1,533 / 0 / 0 | **0** |

For L89, the exact transferred threshold was `0.314208984375`.  The sealed
metrics were positive F1 `0.737066`, macro-F1 `0.735702`, AP `0.774746`,
AUROC `0.816375`, recall `0.835795`, FPR `0.343962`, and recall at FPR 5%
`0.347443`.  The confusion matrix was `TP=2942, FP=1521, FN=578, TN=2901`.

For EMIT, the exact transferred threshold was `0.2288818359375`.  The sealed
metrics were positive F1 `0.610575`, macro-F1 `0.591124`, AP `0.666042`,
AUROC `0.691294`, recall `0.742436`, FPR `0.521752`, and recall at FPR 5%
`0.231846`.  The confusion matrix was `TP=1963, FP=1823, FN=681, TN=1671`.
The large FPR is material: positive F1 alone would overstate the quality of
the fixed operating point.

Both new result artifacts report zero canonical-event overlap with development
data.  Their checkpoints were selected by maximum validation AP and their
operating thresholds were frozen on validation before sealed inference.  No
test-time threshold or model selection was performed.  These L89 and EMIT
numbers are **not directly comparable** with the historical L89 `0.754` and
EMIT `0.567` F1 values: the old splits, inputs, data versions, and
test-conditioned/test-selected checkpoint protocols differ.

### S5P calibration caveat

The S5P checkpoint and operating threshold were frozen before the original
test manifest was opened.  The full-validation F1 optimum was already
pathological: at threshold `0.11871337890625`, it predicted 3,804 of 3,805
validation rows positive (`TP=1982, FP=1822, FN=0, TN=1`).  Transferring that
threshold unchanged to the sealed test predicted all 3,213 rows positive
(`TP=1680, FP=1533, FN=0, TN=0`).  The resulting positive-class F1 `0.686695`
and recall `1.0` are therefore essentially the all-positive baseline, not
evidence of a useful operating point.  The corresponding macro-F1 is
`0.343348`, accuracy `0.522876`, and FPR `1.0`.

The defensible S5P signal is limited to modest ranking skill (sealed AP
`0.657061`, AUROC `0.638205`, recall at FPR 5% `0.153571`).  Future S5P
claims must report macro-F1, FPR, and the confusion matrix alongside positive
F1, and must improve calibration/discrimination rather than exploiting class
prevalence.  No test-time threshold adjustment was made.

Reproducible records:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_frozen_threshold.json
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_sealed_test.json
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_frozen_threshold.json
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_sealed_test.json
/diniuvol/yuyao/methanefuse_research_20260727/results/s5p_frozen_threshold.json
/diniuvol/yuyao/methanefuse_research_20260727/results/s5p_sealed_test.json
/diniuvol/yuyao/methanefuse_research_20260727/results/s5p_sealed_test_predictions.csv
```

## Residual-MAE transfer and split-protocol audit — 2026-07-27

This audit did not start a training process, allocate a GPU, or read a sealed
test manifest.  It covered the real `pretrain -> supervised` checkpoint path
in `multisensor_residual_runner.py`, the four staged train/recent-validation
manifests, normalization reuse, resume semantics, and the matched scratch
control.

### Blocking findings and fixes

The original `strict=False` initialization was not a defensible MAE transfer.
It loaded every matching state key.  Legacy pretrain models contained random
classification heads that were instantiated but never used by the MAE loss,
so those random heads were silently copied into finetuning.  It also did not
prove complete encoder coverage, validate that the source was a pretrain
checkpoint, or bind the checkpoint to the same architecture and data.

The audited runner now enforces the following invariants:

- pretrain models do not instantiate classification heads;
- legacy `heads.*`, all `decoder*` components, and `mask_token` are explicitly
  excluded during transfer;
- the source must declare `mode=pretrain`, and every target encoder key must
  be present with the exact shape (100% encoder parameter coverage);
- encoder architecture and effective data signatures must match exactly;
- the target head SHA-256 must be unchanged by transfer, and can be compared
  directly with the same-seed scratch run's recorded initial head SHA-256;
- `--init-checkpoint` and `--resume` are mutually exclusive;
- resume requires the original output directory and history and an exact
  architecture/data/optimizer/update-budget/evaluation signature;
- normalization statistics are bound to each effective training CSV SHA-256,
  sampling seed, sample count, and S5P data key;
- a fresh invocation refuses to overwrite existing metrics/checkpoints.

CPU validation with the `panopticon` environment passed.  The self-test read
zero manifests, loaded 100% of encoder parameters, excluded 35 MAE decoder
state keys, left the head identical to matched scratch, produced finite S2 and
S5P pretrain/supervised forwards, and rejected a source relabelled as
supervised:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_residual_runner.py --self-test
```

### Cross-sensor event leakage

Per-sensor train/validation event overlap is zero, but that is insufficient
for a shared encoder.  Using S2 `event_group_id` and, for the other manifests,
`plume_id` with the final hyphen suffix removed, the union of all sensor
training events intersected the union of all validation events by **124
canonical events**:

| Cross-sensor path | Overlapping events |
|---|---:|
| S2 train -> L89 val | 25 |
| S2 train -> EMIT val | 16 |
| L89 train -> EMIT val | 1 |
| S5P train -> S2 val | 16 |
| S5P train -> L89 val | 41 |
| S5P train -> EMIT val | 41 |

Consequently, the earlier shared full-validation result is exploratory and
must not be used as a globally event-held-out claim.  The new default is to
error.  The explicit `purge` protocol removes the union of all 996 validation
events from every sensor's training rows before normalization or training and
materializes write-once manifests.  A real-manifest CPU audit produced:

| Sensor | Original train | Purged train | Removed rows | Batches at 64 |
|---|---:|---:|---:|---:|
| S2 | 18,747 | 18,107 | 640 | 283 |
| L89 | 10,049 | 10,033 | 16 | 157 |
| EMIT | 11,688 | 11,688 | 0 | 183 |
| S5P | 18,772 | 18,271 | 501 | 286 |

Post-purge global train/validation event overlap is zero.  Balanced training
therefore has **157 rounds per epoch** and **628 optimizer steps per epoch**
(one step for each of four sensors per round).

### Minimal interpretable screen

Use one common protocol-manifest directory and one common normalization file
for all three runs.  The actual pretrain was launched once in session 3055
with full training (157 balanced rounds / 628 optimizer steps) and 30
validation batches per sensor.  Its exact process command is:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_residual_runner.py \
  --mode pretrain --sharing shared --sensors s2,l89,emit,s5p \
  --device cuda:0 --epochs 1 --batch-size 64 --num-workers 2 \
  --stats-samples 128 --stats-workers 2 --seed 20260727 \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir /diniuvol/yuyao/methanefuse_research_20260727/manifests_protocol/four_sensor_global_val_purge_v1 \
  --max-train-rounds 0 --max-val-batches 30 \
  --log-interval-rounds 10 \
  --output-dir /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727
```

No `--stats-json` override was supplied, so the common stats artifact for all
subsequent matched runs is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/normalization_stats.json
```

The scratch control can start on the second GPU after the pretrain process has
printed its model line (the atomic stats write is then complete); finetuning
starts only after the pretrain checkpoint exists.

Encoder-only finetune, one full epoch:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_residual_runner.py \
  --mode supervised --sharing shared --sensors s2,l89,emit,s5p \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir /diniuvol/yuyao/methanefuse_research_20260727/manifests_protocol/four_sensor_global_val_purge_v1 \
  --stats-json /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/normalization_stats.json \
  --stats-samples 128 --stats-workers 2 \
  --epochs 1 --batch-size 64 --num-workers 2 --device cuda:0 \
  --image-size 112 --patch-size 14 --embed-dim 256 --depth 6 \
  --num-heads 8 --mlp-ratio 4 --fuse-freq 2 --dropout 0.1 \
  --learning-rate 3e-4 --weight-decay 0.05 --seed 20260727 \
  --init-checkpoint /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/checkpoint_latest.pth \
  --output-dir /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_finetune_e1_seed20260727
```

The mandatory matched scratch control was then launched once on GPU1 with the
same effective data, seed, architecture, optimizer, and 157-round update
budget.  Its exact process command is:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_residual_runner.py \
  --mode supervised --sharing shared --sensors s2,l89,emit,s5p \
  --device cuda:1 --epochs 1 --batch-size 64 --num-workers 2 \
  --stats-json /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/normalization_stats.json \
  --seed 20260727 \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir /diniuvol/yuyao/methanefuse_research_20260727/manifests_protocol/four_sensor_global_val_purge_v1 \
  --max-train-rounds 0 --max-val-batches 0 \
  --log-interval-rounds 10 \
  --output-dir /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_scratch_globalpurge_e1_seed20260727
```

#### Completed global-purge pretraining and scratch control

The one-epoch pretraining run completed all 157 balanced rounds / 628 optimizer
steps with mask ratio `0.6`.  Its reconstruction validation was deliberately
bounded to 30 batches, or 1,920 samples, per sensor; it was not full
validation.  All losses were finite:

| Sensor | Mean train reconstruction loss | Validation reconstruction loss | Validation samples |
|---|---:|---:|---:|
| S2 | 0.313529 | 0.134204 | 1,920 |
| L89 | 0.302189 | 0.229541 | 1,920 |
| EMIT | 0.194625 | 0.120944 | 1,920 |
| S5P | 0.314750 | 0.260600 | 1,920 |
| Macro over sensors | — | **0.186322** | 7,680 |

Measured epoch wall time, including bounded reconstruction validation, was
`1,824.880 s` (`30.41 min`).  The completed checkpoint is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/checkpoint_latest.pth
SHA-256 89644dd3fac2310eb3f2eebe17b7757649424ee35c1b6ae7dec40d732257e718
```

The matched scratch run then completed the same 157-round supervised update
budget and evaluated every row of each recent-validation manifest:

| Sensor | Val rows | Positive rate | F1@0.5 | Best positive F1 | Best threshold | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 | 16,546 | 0.502055 | 0.668046 | 0.715232 | 0.278257 | 0.745224 | 0.745180 |
| L89 | 9,614 | 0.432702 | 0.609331 | 0.618006 | 0.438545 | 0.634397 | 0.675728 |
| EMIT | 5,201 | 0.395693 | 0.483565 | 0.585046 | 0.294215 | 0.533902 | 0.639065 |
| S5P | 3,805 | 0.520894 | 0.170022 | 0.684984 | **0.000000** | 0.570401 | 0.527725 |
| Macro over sensors | — | — | **0.482741** | **0.650817** | — | **0.620981** | **0.646924** |

Measured scratch epoch wall time, including full validation, was `2,793.006 s`
(`46.55 min`).  Its completed checkpoint is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_scratch_globalpurge_e1_seed20260727/checkpoint_latest.pth
SHA-256 373422676482663dfca44c96c948f1b82127864d9da1f5ff794c320739c0502a
```

The S5P best-F1 threshold is exactly zero.  Because sigmoid probabilities are
non-negative, that operating point predicts all 3,805 validation rows
positive (`TP=1982, FP=1823, FN=0, TN=0`); its positive F1 `0.684984` is
exactly the all-positive class-prevalence baseline, not evidence of useful
classification.  Consequently, the macro best-positive-F1 `0.650817` is also
optimistic.  S5P AP `0.570401`, AUROC `0.527725`, and F1@0.5 `0.170022`
better expose the weak discrimination.  This scratch result is validation
evidence only; no sealed test was read.

The predeclared one-epoch promotion gate is:

1. protocol overlap after purge is zero; pretrain/finetune/scratch manifest and
   normalization hashes match;
2. transfer coverage is 1.0; only decoder/legacy-head keys are excluded; the
   finetune head SHA is unchanged by transfer and equals scratch initialization;
3. all four train and validation reconstruction losses are finite;
4. finetune minus scratch macro AP is at least `+0.010`, at least three of four
   per-sensor AP deltas are non-negative, and no sensor AP delta is below
   `-0.010`;
5. macro best-F1 is no worse than scratch by more than `0.005`; F1@0.5,
   positive rate, and confusion behavior must also be inspected, especially
   for the known S5P all-positive failure mode.

Passing promotes this branch to a multi-seed, two-to-three-epoch confirmation;
it does not itself support a paper claim.  A non-positive macro-AP delta or
two sensor AP drops worse than `-0.010` stops this implementation after one
epoch.

#### Unmasked MAE gate result — credible negative, `do_not_promote`

The encoder-only finetune completed the same 157 balanced rounds / 628
supervised optimizer steps as scratch and evaluated every validation row.
The comparison below is finetune initialized from the one-epoch unmasked MAE
checkpoint versus the same-seed scratch control.  Each cell reports
`scratch -> finetune (finetune - scratch)`:

| Sensor | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| S2 | 0.745224 -> 0.717887 (**-0.027336**) | 0.745180 -> 0.737752 (-0.007427) | 0.715232 -> 0.726288 (+0.011055) | 0.668046 -> 0.661167 (-0.006878) |
| L89 | 0.634397 -> 0.606806 (**-0.027591**) | 0.675728 -> 0.662275 (-0.013453) | 0.618006 -> 0.608347 (-0.009659) | 0.609331 -> 0.544347 (-0.064985) |
| EMIT | 0.533902 -> 0.555246 (**+0.021344**) | 0.639065 -> 0.664518 (+0.025453) | 0.585046 -> 0.605784 (+0.020737) | 0.483565 -> 0.534458 (+0.050892) |
| S5P | 0.570401 -> 0.571657 (**+0.001256**) | 0.527725 -> 0.526291 (-0.001433) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.244162 (+0.074140) |
| **Macro over sensors** | **0.620981 -> 0.612899 (-0.008082)** | **0.646924 -> 0.647709 (+0.000785)** | **0.650817 -> 0.656351 (+0.005533)** | **0.482741 -> 0.496033 (+0.013292)** |

The artifact/transfer side of the comparison is fully valid: **21/21
machine checks passed**.  In particular:

- global train/validation overlap is zero for pretrain, finetune, and scratch;
- all three data signatures and encoder signatures are identical;
- supervised run signatures, model sizes, 157-round budgets, and 628 update
  counts match;
- the source checkpoint is the declared pretrain checkpoint with SHA-256
  `89644dd3fac2310eb3f2eebe17b7757649424ee35c1b6ae7dec40d732257e718`;
- transfer loaded all `23,316,480 / 23,316,480` encoder parameters (coverage
  `1.0`), explicitly excluded 61 decoder keys and zero legacy checkpoint-head
  keys, and did not copy or alter the classification head;
- the transfer-before, transfer-after, finetune-initial, and scratch-initial
  head hashes are all
  `d9295af89085ba120baff83ca381a68ce385f1db9c635961e7f584aa7ab07f6b`;
- every sensor's train and validation reconstruction loss is finite, and the
  logged macro metrics exactly match their recomputation.

The effect gate nevertheless **failed three of its four performance
criteria**.  Macro AP changed by `-0.008082`, rather than the required
`>= +0.010`; only EMIT and S5P had non-negative AP deltas (`2/4`, rather than
`>= 3/4`); and the worst AP delta was L89 at `-0.027591`, below the allowed
`-0.010`.  Macro best-F1 changed by `+0.005533` and passed its
`>= -0.005` guard, but that secondary guard cannot override the failed AP
criteria.

**Decision: `do_not_promote`.**  Do not add epochs or seeds to this unmasked
MAE implementation.  This is a credible negative result, not a
protocol-invalid diagnostic: under a leakage-safe, matched one-epoch screen,
the representation helped EMIT AP, was nearly neutral for S5P AP, and harmed
S2 and L89 AP enough to lower macro AP.

The current objective reconstructs normalized/clipped patches without the
available valid-pixel mask, so zero-filled TIFF no-data elements and empty
channels enter the loss; S5P also reconstructs an upsampled coarse field.
The gate does **not** prove that invalid-pixel weighting caused the regression
or that masking it will solve transfer.  It motivated the isolated,
now-completed fallback documented below: change only reconstruction weighting
to the predeclared validity-masked objective while preserving the global-purge
protocol, architecture, optimizer, update budget, and gate.  See
`VALIDITY_MASKED_MAE_PLAN.md` for its preregistration; do not expand this
failed unmasked branch with extra epochs.

Authoritative decision artifact:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/mae_vs_scratch_gate.json
status=fail
decision=do_not_promote
```

### Validity-masked MAE gate result — credible negative, `do_not_promote`

The controlled fallback changed only reconstruction validity semantics.  It
used native validity (`finite and nonzero` for TIFF sensors, `finite including
zero` for S5P), residual-validity intersections, area downsampling / nearest
upsampling for masks, and equal weighting over valid samples and non-empty
streams.  Data protocol, architecture, seed `20260727`, optimizer, mask ratio,
157 balanced rounds, and 628 updates remained matched to the same scratch
control.

The one-epoch validity-masked pretraining run completed in `2,019.064 s`
(`33.65 min`).  Reconstruction validation remained bounded to 30 batches /
1,920 samples per sensor.  Every stream had 1,920 valid samples and zero
empty or excluded samples:

| Sensor | Mean train reconstruction loss | Validation reconstruction loss | Current valid fraction | Recent valid fraction | Seasonal valid fraction | Validation samples |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.374103 | 0.163186 | 0.833304 | 0.833304 | 0.833239 | 1,920 |
| L89 | 0.333499 | 0.245650 | 0.899359 | 0.898995 | 0.899001 | 1,920 |
| EMIT | 0.195131 | 0.121261 | 0.998427 | 0.996712 | 0.996145 | 1,920 |
| S5P | 0.310674 | 0.233914 | 0.972383 | 0.947734 | 0.928457 | 1,920 |
| **Macro over sensors** | **0.303352** | **0.191003** | — | — | — | 7,680 |

The completed pretraining checkpoint is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_validitymasked_e1_seed20260727/checkpoint_latest.pth
SHA-256 84a13ffd1869f091be0295b2c7d96f9764b689143d799732e60b1d22001bbaee
```

Encoder-only finetuning then completed the matched one-epoch supervised budget
and full validation in `3,427.543 s` (`57.13 min`).  Each cell reports
`scratch -> validity-masked-MAE finetune (finetune - scratch)`:

| Sensor | Val rows | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| S2 | 16,546 | 0.745224 -> 0.719978 (**-0.025246**) | 0.745180 -> 0.722581 (-0.022599) | 0.715232 -> 0.708650 (-0.006582) | 0.668046 -> 0.582177 (-0.085869) |
| L89 | 9,614 | 0.634397 -> 0.601040 (**-0.033357**) | 0.675728 -> 0.663003 (-0.012725) | 0.618006 -> 0.620816 (+0.002809) | 0.609331 -> 0.594227 (-0.015105) |
| EMIT | 5,201 | 0.533902 -> 0.548353 (**+0.014451**) | 0.639065 -> 0.656264 (+0.017199) | 0.585046 -> 0.594397 (+0.009350) | 0.483565 -> 0.524719 (+0.041153) |
| S5P | 3,805 | 0.570401 -> 0.575338 (**+0.004937**) | 0.527725 -> 0.533810 (+0.006085) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.233139 (+0.063117) |
| **Macro over sensors** | 35,166 | **0.620981 -> 0.611177 (-0.009804)** | **0.646924 -> 0.643915 (-0.003010)** | **0.650817 -> 0.652211 (+0.001394)** | **0.482741 -> 0.483565 (+0.000824)** |

The finetune checkpoint is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_validitymasked_finetune_e1_seed20260727/checkpoint_latest.pth
SHA-256 d769b8d952616910d6f658f245ffcfd745219d3f83da9d6a0ac826a0212ff13b
```

All **42/42** artifact, protocol, objective, schema, optimization, transfer,
head-initialization, validity-diagnostic, full-validation, and metric
recomputation checks passed.  The masked adapter and objective contracts match
exactly between pretraining and finetuning; after projecting only the
registered reconstruction-only fields, their data, parameter-bearing encoder,
optimization, and supervised signatures match scratch.  Encoder transfer was
`23,316,480 / 23,316,480` parameters (`1.0` coverage), excluded 61 decoder
keys and zero checkpoint-head keys, and preserved the same scratch head SHA.

The preregistered gate outcome was:

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Artifact/protocol/transfer integrity | 42/42 | 42/42 | pass |
| Macro AP delta | >= +0.010 | **-0.009804** | **fail** |
| Sensors with non-negative AP delta | >= 3/4 | **2/4** (EMIT, S5P) | **fail** |
| Worst sensor AP delta | >= -0.010 | **L89 -0.033357** | **fail** |
| Macro best-F1 delta | >= -0.005 | +0.001394 | pass |

**Decision: `do_not_promote`.**  Validity masking did not reverse the
unmasked-MAE result.  EMIT and S5P AP improved slightly, but S2 and L89
regressed enough to lower macro AP.  Both controlled MAE variants have now
failed the same one-epoch promotion screen; neither should receive extra
epochs or seeds.  This is validation-only evidence and no sealed test was
opened.

Authoritative decision artifact:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/validitymasked_mae_vs_scratch_gate.json
status=fail
integrity=42/42
decision=do_not_promote
```

### Supervised BCE + RankNet fallback — positive ranking result, hard-stop `do_not_promote`

After both controlled MAE objectives failed, the final preregistered fallback
changed only the **supervised classification loss**.  It retained weighted
BCE and added an all-positive-negative-pairs RankNet term within each sensor's
current mini-batch:

```text
objective=sensorwise_weighted_bce_plus_all_pairs_ranknet_v1
rank_weight=0.5
rank_temperature=1.0
pair_scope=within_sensor_current_batch
pair_selection=all_positive_negative_pairs
one_class_policy=connected_zero_rank_loss
```

This is a supervised, ranking-aware classification fallback, **not a
pretraining method** and not evidence that self-supervised pretraining helps.
It used the same globally purged manifests and normalization statistics as
the BCE scratch control, zero global canonical-event train/validation
overlap, seed `20260727`, the same sensor-native residual architecture and
initial head SHA, no MAE initialization, batch size 64, one epoch, 157
balanced rounds, 628 optimizer steps, and full validation for all four
sensors.

The training plus its in-loop full-temporal validation completed in
`3,054.594 s` (`50.91 min`) on `cuda:0`.  Every one of the 628 sensor batches
contained both classes, so all batches contributed valid within-sensor
ranking pairs:

The formal process used runner source SHA-256
`4a2c54aebf2ec957a4ef0b1aa85368092b50e79ddbc13adff3e3b1292caa9fed`;
an exact launch-time source snapshot is stored beside the checkpoint.  The
working-tree runner is now SHA-256
`85b4a8a1366692396b4251e0bb57ad14574e650db977e6208b598658163ebbab`
after a legacy-call compatibility guard was changed to
`getattr(..., False)`.  That post-launch change does not alter the RankNet
objective or formal-run mathematics, and both the 15/15 RankNet and 9/9
validity-masked legacy CPU suites pass.

| Sensor | Batches | Paired batches | One-class batches | Paired fraction | Positive / negative samples | Pair count | Weighted BCE loss | RankNet loss | Total loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 157 | 157 | 0 | 1.000000 | 5,010 / 5,038 | 158,472 | 0.614425 | 0.525296 | 0.877073 |
| L89 | 157 | 157 | 0 | 1.000000 | 4,448 / 5,585 | 155,566 | 0.699178 | 0.546581 | 0.972601 |
| EMIT | 157 | 157 | 0 | 1.000000 | 4,125 / 5,923 | 153,113 | 0.792354 | 0.637662 | 1.111185 |
| S5P | 157 | 157 | 0 | 1.000000 | 5,186 / 4,862 | 158,320 | 0.682252 | 0.702607 | 1.033555 |
| **All sensors** | **628** | **628** | **0** | **1.000000** | **18,769 / 21,408** | **625,471** | — | — | — |

L89 contributed 10,033 training samples because its 157th loader batch was
the partial final batch; the other sensors each contributed 10,048 samples.
The loss columns above preserve the history's aggregation conventions:
total/BCE are sample-weighted, while RankNet is averaged over pair-bearing
batches.

Against the exact one-epoch BCE scratch control, every sensor improved AP.
Each cell reports `scratch -> BCE+RankNet (RankNet - scratch)`:

| Sensor | Val rows | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| S2 | 16,546 | 0.745224 -> 0.752423 (**+0.007199**) | 0.745180 -> 0.753565 (+0.008386) | 0.715232 -> 0.724228 (+0.008995) | 0.668046 -> 0.632706 (-0.035340) |
| L89 | 9,614 | 0.634397 -> 0.640748 (**+0.006351**) | 0.675728 -> 0.679762 (+0.004033) | 0.618006 -> 0.630248 (+0.012241) | 0.609331 -> 0.596926 (-0.012405) |
| EMIT | 5,201 | 0.533902 -> 0.560876 (**+0.026974**) | 0.639065 -> 0.666551 (+0.027486) | 0.585046 -> 0.598788 (+0.013741) | 0.483565 -> 0.578706 (+0.095140) |
| S5P | 3,805 | 0.570401 -> 0.579910 (**+0.009509**) | 0.527725 -> 0.540408 (+0.012684) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.522841 (+0.352819) |
| **Macro over sensors** | **35,166** | **0.620981 -> 0.633489 (+0.012508)** | **0.646924 -> 0.660072 (+0.013147)** | **0.650817 -> 0.659562 (+0.008745)** | **0.482741 -> 0.582795 (+0.100053)** |

The same frozen RankNet checkpoint was then evaluated with the current input
preserved and the normalized recent/seasonal streams zeroed at the model
entrance.  This is an inference-time **temporal-input ablation**, not a
separately trained current-only model.  It used the exact contract
`current_preserved_recent_seasonal_zeroed_after_normalization_v1`, processed
all 35,166 validation rows on `cuda:1`, and completed in `1,893.652 s`
(`31.56 min`).  Each cell reports
`current-only ablation -> full temporal (full - current)`:

| Sensor | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| S2 | 0.747140 -> 0.752423 (**+0.005282**) | 0.750442 -> 0.753565 (+0.003124) | 0.723116 -> 0.724228 (+0.001112) | 0.617488 -> 0.632706 (+0.015218) |
| L89 | 0.633311 -> 0.640748 (**+0.007437**) | 0.672574 -> 0.679762 (+0.007188) | 0.621593 -> 0.630248 (+0.008655) | 0.572965 -> 0.596926 (+0.023961) |
| EMIT | 0.549697 -> 0.560876 (**+0.011179**) | 0.645630 -> 0.666551 (+0.020921) | 0.591906 -> 0.598788 (+0.006882) | 0.042634 -> 0.578706 (+0.536072) |
| S5P | 0.576525 -> 0.579910 (**+0.003385**) | 0.535402 -> 0.540408 (+0.005007) | 0.685249 -> 0.684984 (-0.000266) | 0.431628 -> 0.522841 (+0.091213) |
| **Macro over sensors** | **0.626668 -> 0.633489 (+0.006821)** | **0.651012 -> 0.660072 (+0.009060)** | **0.655466 -> 0.659562 (+0.004096)** | **0.416179 -> 0.582795 (+0.166616)** |

Full temporal input improved AP on all four sensors and exceeded current-only
macro AP by `+0.006821`, passing the preregistered `>= +0.005` temporal-use
criterion.  This supports a validation-only statement that the trained
classifier uses the recent/seasonal inputs under this ablation; it does not
establish causal methane-transient localization or a pretraining benefit.

All **20/20** artifact, protocol, objective, matched-comparator, checkpoint,
full-validation, prediction-SHA, metric-recomputation, pair-diagnostic, and
same-checkpoint ablation integrity checks passed.  The promotion criteria
were:

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Artifact/protocol/metric/objective integrity | 20/20 | 20/20 | pass |
| Macro AP delta vs BCE scratch | >= +0.010 | **+0.012508** | pass |
| Sensors with non-negative AP delta | >= 3/4 | **4/4** | pass |
| Worst sensor AP delta | >= -0.010 | **L89 +0.006351** | pass |
| Macro best-F1 delta | >= -0.005 | **+0.008745** | pass |
| Macro F1@0.5 delta | >= -0.010 | **+0.100053** | pass |
| Pair-bearing training batches | >= 0.990 | **1.000000** | pass |
| Full-temporal macro AP advantage | >= +0.005 | **+0.006821** | pass |
| Sensors with positive temporal AP contribution | >= 2/4 | **4/4** | pass |
| S5P probability/ranking behavior | non-degenerate | **best-F1 predicts 3,805 / 3,805 positive** | **fail** |

The S5P ranking metrics were above chance-level references—AP `0.579910`
versus positive prevalence `0.520894`, and AUROC `0.540408`—and its
probabilities were not constant (`min=0.286169`, `max=0.769080`,
`std=0.070002`).  However, its best-F1 threshold was exactly `0.0`, so the
selected operating point predicted every one of the 3,805 validation rows
positive.  This was the **only failed promotion criterion**.  The AP/AUROC
improvements cannot override the preregistered requirement that S5P exhibit
non-degenerate best-F1 classification behavior.

**Decision: `do_not_promote`.**  Do not tune the ranking weight or
temperature, add epochs or seeds, introduce hard-negative mining, or start
another objective search.  The result is a credible validation-only finding:
the fixed supervised ranking loss improved macro AP and all four per-sensor
AP values, and full temporal input outperformed its same-checkpoint
current-only ablation, but S5P still collapsed to an all-positive best-F1
operating point.  No sealed test was read for this fallback.

Authoritative signed artifacts and SHA-256 values:

```text
gate:
/diniuvol/yuyao/methanefuse_research_20260727/results/ranknet_vs_bce_gate.json
SHA-256 c36e007a6e03025b4bf54195143d0876e96a9e9376c75e7b53634d29552cb6fc

BCE scratch history:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_scratch_globalpurge_e1_seed20260727/metrics_history.json
SHA-256 f278d859eee6c8c36c56f295ab34f1245700f60530ef11d86fc104c27d77c5db

RankNet history:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/metrics_history.json
SHA-256 0923eb5525d3db5d840a8666465dfd2159797183614bd080fbb923b7980bf1a9

RankNet checkpoint:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/checkpoint_latest.pth
SHA-256 ef070e6f47abaa2a8401203da2222365c51a913fe97b310d59784bb9c648f683

Exact launch-time runner source:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/multisensor_residual_runner.launch_4a2c54aebf2e.py
SHA-256 4a2c54aebf2ec957a4ef0b1aa85368092b50e79ddbc13adff3e3b1292caa9fed

Full-temporal evaluation JSON:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_full_temporal.json
SHA-256 ca3066b996ecef6cb17b37786624282f9e1b993069380f10cbf1f4637145c054

Full-temporal predictions CSV:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_full_temporal.csv
SHA-256 26508dc2da884e3835df0f1e55f4f6a4e3d6582bcd2ffc6f8913a29a154baa92

Current-only ablation evaluation JSON:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_current_only.json
SHA-256 e34ca0c047adac56c3b8b720369eb5229f8a54ba1f38c67416e3f0e1d1ada7bf

Current-only ablation predictions CSV:
/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_current_only.csv
SHA-256 bab1dd6ba196e931648944688e507db1e6cf1bc0eab40a613d1b228fc334b13b

status=fail
integrity=20/20
failed_criterion=s5p_non_degenerate_probability_and_ranking_audit
decision=do_not_promote
```

In the parallel unified-data track, production recropping remains blocked by
the manifest-v2 provenance gate.  The metadata pilot found unresolved aliases,
missing query-usability/per-channel-validity fields, and a known EMIT `prev3`
embedded-product conflict.  A stratified metadata/header pilot, deterministic
alias/quarantine decisions, a small query-coverage pilot, and materialized
readback checks must pass before mass crop execution; see
`MANIFEST_V2_PROVENANCE_PILOT.md`.

## Irregular-history current-query experiment — positive L89 result

The next experiment isolated temporal decision structure on frozen official
Panopticon CLS features.  It did not update the image backbone and is not a
pretraining result.  The leakage-safe L89 cache contains 10,033 train rows
from 723 canonical events and 9,614 validation rows from 124 canonical events,
with zero event overlap, no read errors, no invalid t0 rows, and explicit
acquisition-duplicate suppression.

All four arms used one parameter-identical, initialization-identical,
two-block current-query cross-attention head:

- `t0_masked`: current acquisition only;
- `role_only`: unique history plus discrete acquisition roles;
- `delta_time`: `role_only` plus the real `Δdays` Fourier encoding;
- `history_shuffle_train`: retain t0 but replace the full history with another
  canonical event's history during training.

Three formal seeds produced:

| Arm | AP mean ± SD | AUROC mean ± SD | macro-F1@0.5 mean ± SD |
|---|---:|---:|---:|
| `t0_masked` | 0.724271 ± 0.011171 | 0.767009 ± 0.010415 | 0.702226 ± 0.011013 |
| **`role_only`** | **0.765151 ± 0.009935** | **0.803680 ± 0.009246** | **0.732275 ± 0.010523** |
| `delta_time` | 0.761870 ± 0.011755 | 0.797899 ± 0.012407 | 0.722152 ± 0.016423 |
| `history_shuffle_train` | 0.718205 ± 0.004580 | 0.766092 ± 0.010021 | 0.699776 ± 0.005892 |

Thus `role_only - t0_masked` was `+0.040880 AP`, `+0.036671 AUROC`,
and `+0.030049 macro-F1`; `role_only - history_shuffle_train` was
`+0.046946 AP`.  Paired canonical-event bootstrap used 2,000 replicates per
seed.  The AP CIs for role minus t0 were:

- seed `20260727`: `+0.042770`, 95% CI `[0.027069, 0.061524]`;
- seed `20260728`: `+0.041047`, 95% CI `[0.025509, 0.058830]`;
- seed `20260729`: `+0.038824`, 95% CI `[0.021039, 0.058500]`.

The AP CIs for role minus shuffled history were also strictly positive in all
three seeds.  The preregistered candidate was `delta_time`; it failed its gate
because it did not beat `role_only`.  Therefore the valid conclusion is not
that continuous-time encoding helped.  The strong exploratory finding is that
matched, unique historical content plus discrete acquisition role improved
current-query classification, while cross-event history replacement removed
the gain.

A `model_dim=512` capacity check gave AP `0.714222 / 0.758823 / 0.683056`
for t0 / role / shuffle and did not exceed the 256-dimensional role result.
This makes head capacity an unlikely explanation of the gain.

Formal result directories:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260727
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260728
/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260729
```

## Native-grid S5P approximation — directional only

The same four-arm idea was screened on finite-mask adaptive averages of the
upsampled S5P product, yielding an approximate `3×3` grid.  It is not an exact
read of the native source product and therefore cannot support a final native
S5P claim.

Across three seeds:

| Arm | AP | AUROC | macro-F1 |
|---|---:|---:|---:|
| `t0_masked` | 0.565491 | 0.529042 | 0.522456 |
| `role_only` | **0.577494** | **0.539372** | 0.530489 |
| `delta_time` | 0.576504 | 0.538939 | **0.532025** |
| `history_shuffle_train` | 0.560186 | 0.524860 | 0.520816 |

`role_only - t0_masked` was `+0.012003 AP`; role minus shuffle was
`+0.017308 AP`.  This is a consistent direction but too close to chance and
too dependent on an approximate data path to enter the headline result.

## Cross-fitted L89 background/innovation pretraining — negative

A three-fold event-level cross-fitted predictor was trained only on historical
context to predict the current frozen Panopticon latent.  The larger
shape-matched predictor reached OOF MSE `0.443202`, OOF cosine `0.851583`,
validation MSE `0.466639`, and validation cosine `0.849615`.  The external
checkpoint is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_innovation_v1b/pretrain_dim256/predictor_final_all_train.pt
SHA-256 c28feca5619995060cce2ae72a2d2efd38788435190414228051bbf9401e4ad7
```

The completed two-seed downstream readout experiment gave:

| Readout | AP mean |
|---|---:|
| t0 only | 0.722215 |
| predicted background only | 0.665358 |
| innovation only | 0.668805 |
| t0 + innovation | 0.707034 |
| shuffled-history innovation | 0.688481 |

Innovation-only lost `0.053410 AP` and t0+innovation lost `0.015181 AP`
versus t0.  The predictor learned stable current content, but its residual was
not a sufficient methane representation.  **Decision: stop this residual
readout branch.**

## Label-free temporal correspondence pretraining — converged negative

The next pretext asked whether a t0 query and a whole history came from the
same canonical event.  It used a deterministic global bijection of cross-event
donors, excluded t0 from pretext keys/values, consumed no methane labels in
the pretext objective, and matched initial state, batch plan, examples, and
optimizer steps to a permuted-pretext-label control.

The pretext learned its objective:

| Arm | Epoch-1 accuracy | Epoch-2 | Epoch-3 |
|---|---:|---:|---:|
| correspondence | 0.6271 | 0.9214 | **0.9654** |
| permuted-label control | 0.4951 | 0.5116 | 0.5226 |

Nevertheless, coherent validation transfer was:

| Labeled events | Scratch AP | Correspondence AP | Permuted control AP | Past predictor AP |
|---|---:|---:|---:|---:|
| 10% | 0.660947 | **0.614345 (-0.046602)** | 0.632821 | 0.661818 |
| 100% | 0.749511 | **0.726655 (-0.022857)** | 0.761008 | 0.756975 |

The negative is not attributable to a failed or unlearned pretext.  It is
consistent with temporal correspondence learning persistent scene/event
identity that transfers poorly to a weak transient target.

The formal directory is:

```text
/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_correspondence_v1_seed20260727
run_status=complete
sealed_test_read=false
summary SHA-256 8f7d13b659b25be89e477edab6a717b482843abd5b9b3b5a1f77d95d745c02b9
launch source SHA-256 a9da431dc0d23fd636a221be6470818c79af0fee028a66a4602259d34d2e4137
```

An independent read-only audit passed 269 checks, including all 26 referenced
artifact hashes, checkpoint metadata/state hashes, 16 prediction files,
metric recomputation, global donor bijection, event separation, and matched
batch plans.  **Decision: do not promote or add replication seeds.**

## EMIT role-aware current-query replication — positive direction, weaker CI

The EMIT adapter used the exact globally purged train manifest and the existing
recent validation manifest.  The cache contains 11,688 train rows and 5,201
validation rows, zero event overlap, zero read errors, zero invalid t0 rows,
and explicit duplicate suppression.  The official Panopticon weights and
train-bound normalization hashes match between train and validation.

Across three seeds:

| Arm | AP mean ± SD | AUROC mean ± SD | macro-F1@0.5 mean ± SD |
|---|---:|---:|---:|
| `t0_masked` | 0.762661 ± 0.005438 | 0.814410 ± 0.004071 | 0.736839 ± 0.009048 |
| **`role_only`** | **0.785824 ± 0.003989** | **0.828723 ± 0.003965** | **0.747473 ± 0.001781** |
| `delta_time` | 0.780331 ± 0.001626 | 0.824684 ± 0.001737 | 0.740732 ± 0.011279 |
| `history_shuffle_train` | 0.758382 ± 0.001258 | 0.812943 ± 0.005198 | 0.719203 ± 0.023637 |

Mean role minus t0 was `+0.023163 AP`, `+0.014313 AUROC`, and
`+0.010634 macro-F1`.  Mean role minus shuffle was `+0.027442 AP`.
The three role-minus-t0 AP point estimates were `+0.019500`, `+0.017986`,
and `+0.032002`.  Only the third seed's event-cluster bootstrap CI was
strictly positive (`[0.010603, 0.052372]`); the first two crossed zero.
Role-minus-shuffle AP CIs were positive for seeds 20260727 and 20260729, while
seed 20260728 had a lower bound of `-0.000477`.

Every integrity check passed for all three gates.  The preregistered
`delta_time` arm failed because it did not beat `role_only` in any seed.
The correct wording is therefore “consistent cross-sensor replication with
weaker event-level uncertainty,” not three independently significant wins.

Formal directories:

```text
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260727
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260728
/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260729
```

The fail-closed evaluator supports only the original L89 schema and an EMIT
schema with exactly two additional provenance fields
(`cache_script_version`, `sensor=emit32`).  Evaluator SHA-256 is
`c2cb5ecc42b22f43cdd714227ce1e80375f6dd32a9d0431e3c369a088a4c0eb4`;
its CPU suite passed 10/10, and the EMIT adapter suite passed 4/4.

## Current research decision

The evidence now supports a stronger and narrower story than “six-time MAE”:

1. pixel reconstruction, background prediction/residual readout, and
   same-event correspondence all favor persistent content and all produced
   negative methane transfer in controlled experiments;
2. retaining current evidence as a query and using unique, role-tagged matched
   history as context improved L89 materially and replicated directionally on
   EMIT and S5P;
3. cross-event history permutation removed the benefit, while exact `Δdays`
   and increased head capacity were not load-bearing.

This is the basis of `JOURNAL_STORY_20260727.md`.  It remains
validation-only.  It is not a four-sensor SOTA or successful backbone
pretraining claim.

### S2 evidence exclusion

During an early read-only audit, one overly broad `rg` search displayed rows
from a different S2 all-rows source that included `test` markers.  The target
S2 temporal test manifest was not opened and no displayed row value or metric
was used.  Work stopped immediately.  Conservatively, S2 is excluded from all
current temporal model selection and headline results until it is rebuilt from
512 sources and split again.
