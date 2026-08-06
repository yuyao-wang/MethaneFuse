# RCTP L89 real-transfer failure audit

Date: 2026-07-28 UTC

Scope: the completed **train/dev-only** P0/P4/P5 comparison under
`rctp_l89_real_cls_v1/downstream_role_only_seed20260728`, its dev prediction
CSVs, the P4/P5 pretraining summaries, and the three frozen dev caches. No
test/sealed artifact was read, and no GPU was used.

## Verdict

The current RCTP continuation is a real-transfer **no-go**, not a positive
pretraining result. P5's small row-AP increase is produced by row-rich events
and does not survive event balancing. At equal event weight, P5 loses both
ranking quality and thresholded classification quality. The loss is amplified
by a three-level objective mismatch in the downstream protocol, but it remains
after fixed-threshold and post-hoc best-threshold diagnostics.

| Metric | P0 | P4 scrambled | P5 correct | P5 − P0 |
|---|---:|---:|---:|---:|
| Row AP | 0.750716 | 0.739443 | 0.755838 | **+0.005122** |
| Event-balanced AP | 0.741929 | 0.728275 | 0.737478 | **−0.004450** |
| Event-balanced AUC | 0.838673 | 0.820783 | 0.818863 | **−0.019810** |
| Event-balanced positive F1 | 0.709580 | 0.697661 | 0.686513 | **−0.023068** |
| Event-balanced macro F1 | 0.757392 | 0.749437 | 0.730239 | **−0.027152** |

The canonical-event cluster interval for P5 − P0 macro F1 is
`[-0.047795, -0.005652]`; it excludes zero on this already-used dev split.
The positive-F1 interval `[-0.048138, +0.003118]` does not exclude zero.

## Why row AP rose while event-balanced macro F1 fell

### 1. Row AP is dominated by a small number of row-rich events

The dev set has 9,614 rows but only 124 canonical events. Event size has median
32 and maximum 608 rows. The largest 10 events contain 31.79% of all rows, and
the largest 31 events (one quarter of events) contain 64.76% of all rows.

The P5 ranking change reverses with event size:

| Event stratum | Events / rows | P0 row AP | P5 row AP | Delta |
|---|---:|---:|---:|---:|
| At most 32 rows | 71 / 1,878 | 0.751651 | 0.726012 | **−0.025639** |
| More than 102 rows | 31 / 6,226 | 0.729658 | 0.745611 | **+0.015953** |

Thus the aggregate row AP rewards improvement in a few large events while
hiding degradation across many small events. With each event assigned total
weight one, AP falls from 0.741929 to 0.737478 instead of rising. At each
arm's selected threshold, P5 improves the fraction correct in 37 events,
ties in 21, and worsens it in 66; the mean per-event correctness change is
−0.02949, versus only −0.01425 when weighted by row count.

### 2. P5 buys almost no true-positive mass at the cost of much more
false-positive mass

At each arm's dev-selected threshold, the equal-event confusion masses are:

| Arm | TP | FP | FN | TN | Predicted-positive mass |
|---|---:|---:|---:|---:|---:|
| P0 | 35.3239 | 18.5301 | 10.3849 | 59.7610 | 43.43% |
| P5 | 35.6644 | 22.5270 | 10.0445 | 55.7642 | 46.93% |

Relative to P0, P5 gains only `+0.3405` TP mass while adding `+3.9969` FP
mass. The true event-balanced positive prevalence is 36.86%. Among the 27
all-negative events alone, FP mass increases from 5.3506 to 7.5727. Therefore
positive F1 falls by 0.02307 and negative-class F1 falls from 0.80520 to
0.77397; their mean is the observed macro-F1 loss.

AP is threshold-free and positive-focused, whereas macro F1 is thresholded and
also penalizes the damaged negative class. Their opposite movement is therefore
mathematically consistent.

### 3. The original downstream protocol optimizes the wrong unit three times

The audited code:

1. trains with row-mean BCE plus a global class `pos_weight`, but no inverse
   event-size weight;
2. promotes the checkpoint by ordinary row AP;
3. chooses the threshold that maximizes event-balanced **positive** F1, then
   reports macro F1.

This makes the large-event/positive-class failure mode unsurprising. For P5,
epoch 2 is promoted by row AP (0.755838) even though epoch 1 has higher
event-balanced macro F1 (0.739090 versus 0.730239).

The mismatch does not fully explain away the loss:

- at a fixed threshold of 0.5, P5 − P0 event-balanced macro F1 is
  `−0.028726`, with cluster interval `[-0.060003, -0.000729]`;
- a diagnostic, post-hoc macro-F1 threshold search gives P5 0.741426 at
  0.349195 versus P0 0.763362 at 0.578681. Even under this optimistic and
  non-promotable diagnostic, P5 remains 0.021936 behind.

## What the pretext run actually established

It did not establish sensor-response selectivity. On the matched synthetic dev
task, P4 scrambled response is easier than P5 correct response:

| Synthetic metric | P4 scrambled | P5 correct |
|---|---:|---:|
| AP | **0.971989** | 0.952442 |
| AUC | **0.981452** | 0.969470 |
| Selected F1 | **0.912451** | 0.873380 |
| Paired win rate | **0.972656** | 0.947266 |
| Strength log-MAE (lower is better) | 0.807522 | **0.689948** |

The network can detect and grade rendered perturbations, but the type objective
rewards the scrambled response more strongly. This is compatible with learning
generic injection magnitude/artifacts rather than a uniquely methane-relevant
response.

P4/P5 also update 14,178,816 parameters in the final two backbone blocks for
1,024 steps with the clean anchor disabled. On real clean dev images, P5's t0
CLS has mean cosine 0.86850 to P0 and mean relative L2 change 0.52359 (norm
ratio 1.04211). This is material unanchored drift, but not classic collapse:
the t0 variance participation ratio is 303.1 for P5 versus 298.5 for P0, and
the simple label-centroid gap is almost unchanged. The supported conclusion is
that synthetic specialization displaced a useful real decision geometry, not
that the representation collapsed globally.

## One pre-registered, falsifiable next representation change

Test exactly one change: **Sidecar-RCTP**, a representation-preserving bounded
residual adapter.

- Freeze the byte-identical P0 encoder.
- Attach one zero-initialized rank-8 response-conditioned sidecar to its final
  tokens. Preserve the original feature as a separate channel and expose
  `[z_base, 0.1 * residual]` to the temporal head; never overwrite `z_base`.
- Train only the sidecar and pretext probe for the same two epochs / 1,024
  steps. P4 and P5 retain identical draws, initialization, optimizer, and
  parameter count; only correct versus scrambled response changes. P0 uses the
  same downstream input shape with a zero residual channel.
- Freeze encoder and sidecar for real classification and train the identical
  temporal head. This makes forgetting impossible in the base channel and
  gives the head an explicit choice to use or ignore the synthetic response
  feature.

Do not tune this proposal on the current dev again. Before implementation,
freeze rank 8, residual scale 0.1, optimizer, epochs, and seed. Build a new
deterministic canonical-event hash holdout from the present training events;
freeze its manifest and SHA, retrain all P0/P4/P5 controls on the complement,
and evaluate each once. Fit any decision threshold from training-only
out-of-fold predictions, never from the new holdout.

Pre-register the go criterion as:

`P5 event-balanced macro F1 − max(P0, P4) >= +0.020`

with the paired event-cluster 95% lower bound above zero, plus no increase in
all-negative-event FP mass. Failure of either condition rejects Sidecar-RCTP
and prevents a sealed/test evaluation. This is deliberately a single
falsifiable run, not a grid adapted to the already-inspected dev split.

## Completed protocol-only diagnostic

A separate CPU-only run retrained all three existing frozen caches with:

- inverse-event-size BCE;
- checkpoint selection by event-balanced AP;
- threshold selection by event-balanced macro F1;
- the same initial-state SHA, batch-plan SHA, optimizer, seed, and three epochs.

| Arm | Best epoch | Event AP | Event AUC | Positive F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | **0.742812** | **0.834014** | **0.702039** | **0.764032** |
| P4 | 2 | 0.733988 | 0.820993 | 0.674370 | 0.742631 |
| P5 | 3 | 0.728613 | 0.819589 | 0.675485 | 0.739310 |

P5 − P0 macro F1 is `−0.024723`, with paired event-cluster interval
`[-0.046099, -0.004851]`. P5 − P4 is `−0.003321`
`[-0.021154, +0.015753]`. Aligning the downstream objective therefore does not
reverse the finding: P5 remains significantly below P0 on this dev-only
bootstrap and is indistinguishable from the scrambled control.

This run is explicitly post-hoc and cannot rescue or validate RCTP. Its value
is diagnostic: the objective mismatch accounts for only a small part of the
gap, while the transferred representation remains the main failure.

Artifacts:

- `rctp_l89_real_cls_v1/downstream_event_balanced_posthoc_seed20260728/COMPARISON.md`
- comparison SHA-256:
  `1ab631e2ed31aaed6c10368b121272142a70ccfe22ab0e22471f0008d817294a`
- run-config SHA-256:
  `c8f710b499920d5c912fe961ead642259f7d13cef91f2c3013d6ffe0ab3336bb`
