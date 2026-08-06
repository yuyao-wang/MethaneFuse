# AxisStats dev-only strong control

**Run date:** 2026-07-28 UTC  
**Authoritative run:** `axisstats_pilot16k_dev_results_v3/`  
**Claim scope:** strong simple legacy360 engineering control; not RCTP
novelty and not a sealed-test result.

## Protocol

- Train:
  `/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/features/pilot16k_train_core_universal_s2hybrid.pt`
  (`15,973` rows, SHA-256
  `38f0b968c33f8ca281791e1de6b558fda639a5eb2a19ee5ca8f9bd9cd2335a3c`).
- Development:
  `/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/features/dev_universal_s2hybrid.pt`
  (`12,621` rows, SHA-256
  `e7379ef5eb1b005e0bf655e7ea10dd53ed043008ff39a0eea175b645c343767a`).
- Train/dev overlap is zero for IDs, plume IDs, canonical event IDs, and
  `query360_index`.
- The CLI accepts only `train_core` and `dev` cache splits and rejects input
  path/manifest tokens `test` and `sealed`. No outer-test cache or metric was
  read.
- Features: 130 scalars containing universal/hybrid fused and per-sensor
  logits, observation/base validity, per-sensor t0–history cosine/L2/norm,
  and pairwise current-sensor cosine.
- Models: 13 small LogisticRegression/HistGradientBoosting candidates;
  `max_iter=100`, HGB early stopping enabled, seed 42, 8 CPU threads.
- Selection: positive-class F1 and threshold both selected on canonical dev.

## Result

| Control | Features | F1@0.5 | Dev-best F1 | AP | AUC | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Direct universal PTH reference | universal fused logit | 0.886986 | **0.897926** | 0.954828 | 0.954632 | strongest untrained reference |
| Locked LogisticRegression | universal fused logit | **0.897007** | **0.897926** | 0.954828 | 0.954632 | monotone train-only calibration; same ranking/F1 ceiling as reference |
| Best full AxisStats | all 130 statistics, HGB | 0.890491 | 0.892545 | 0.952671 | 0.953330 | `-0.005381` F1 vs direct reference |
| Best nontrivial fused LR | universal + hybrid fused logits | 0.897261 | 0.897920 | **0.954983** | **0.954853** | negligible AP/AUC increase; no F1 gain |

The locked LogisticRegression threshold is `0.4830034038`. At that global
threshold, sensor-containing dev strata are:

| Sensor-containing rows | Rows | Positive F1 | Macro F1 | AP | AUC |
|---|---:|---:|---:|---:|---:|
| S2 | 6,057 | 0.931865 | 0.928339 | 0.969446 | 0.975286 |
| L89 | 4,068 | 0.901488 | 0.897324 | 0.951370 | 0.953534 |
| EMIT | 3,644 | 0.876812 | 0.843698 | 0.948500 | 0.932550 |
| S5P | 1,558 | 0.875773 | 0.855239 | 0.946395 | 0.936552 |

These strata overlap when a row contains multiple sensors and therefore are
diagnostics, not four disjoint test sets.

## Verdict

The near-0.90 development result is real within this engineering protocol, but
AxisStats does **not** improve the strong universal PTH ranking. A one-feature
train-only calibration makes the fixed-0.5 operating point nearly equal to the
dev-optimal F1, while adding complete time×sensor summary statistics reduces
F1 by 0.54 point. This is evidence against presenting hand-designed axis
statistics as the method. It strengthens the baseline that learned two-axis or
RCTP methods must beat.

It is not a SOTA/outer-test result because:

- the 15,973-row train cache is a pilot subset, not full train_core;
- both legacy PTHs were historically selected with old test feedback;
- the present dev rows came from the PTH's old source-train pool;
- candidate and threshold selection used the same canonical dev.

## Locked artifacts

- Global fitted-control lock:
  `axisstats_pilot16k_dev_results_v3/selection_lock.json`, SHA-256
  `c2fc51bbec7ab50ca8eed33742b32f2dec660858e705e9c63e6db3ce524738c8`.
- Calibrated model:
  `axisstats_pilot16k_dev_results_v3/axisstats_best_model.joblib`, SHA-256
  `3864bdacea07023ff3b85658946d210a2a2d8587e8da03bf12c75a418fb29360`.
- Best complete-axis model:
  `axisstats_pilot16k_dev_results_v3/axisstats_best_full_model.joblib`,
  SHA-256
  `d8b32df1f36375e841d3e0c1ed38b3ccddc9c81eee91142f7f29b60d3867480a`.
- Candidate table and predictions:
  `axisstats_pilot16k_dev_results_v3/candidate_metrics.csv`,
  `axisstats_pilot16k_dev_results_v3/dev_predictions_all_candidates.csv`,
  and `axisstats_pilot16k_dev_results_v3/dev_predictions_best.csv`.

Independent replay of the locked joblib produced all 12,621 probabilities
with maximum absolute difference `1.11e-16`; both model hashes matched the
selection lock.

## Reproduction

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/legacy360_dual_axis/axisstats_control.py \
  --train-cache /diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/features/pilot16k_train_core_universal_s2hybrid.pt \
  --dev-cache /diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/features/dev_universal_s2hybrid.pt \
  --output-dir research/pretraining_20260727/legacy360_dual_axis/axisstats_pilot16k_dev_results_v3 \
  --grid small \
  --max-iter 100 \
  --threads 8 \
  --seed 42
```

The runner refuses a non-empty output directory, so an exact reproduction
must use a new immutable run directory.
