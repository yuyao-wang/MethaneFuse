# Availability-conditioned calibration: strict dev-only OOF audit

## Scope and interpretation

This is a **post-hoc calibration study on one frozen model's dev probabilities**.
It is not a dual-axis architecture gain, not a representation/pretraining gain,
and not a final test-set result. No model weights or logits were changed.
OOF splitting removes held-row leakage from *threshold fitting only*; it cannot
undo any optimism if the checkpoint, epoch, or model hyperparameters that produced
`dev_predictions_best.csv` were themselves selected using this same dev set.

Every row was held out while its decision threshold was fitted on the other four
folds. Folds are grouped by canonical `event_id`, which also makes plumes disjoint
because the manifest audit established that every plume maps to exactly one event.
No path containing a `test` or `sealed` token was accepted or read.

## OOF results

95% intervals use 2000 canonical-event cluster
bootstrap replicates with the already-generated OOF predictions held fixed.
Paired deltas use identical sampled events for each strategy and the global
calibrator.

| Strategy | Binary F1 (95% CI) | Macro F1 (95% CI) | Δ binary vs global (95% CI) | Δ macro vs global (95% CI) |
|---|---:|---:|---:|---:|
| global | 0.8969 [0.8810, 0.9111] | 0.8868 [0.8696, 0.9022] | reference | reference |
| availability_signature | 0.8986 [0.8837, 0.9122] | 0.8878 [0.8710, 0.9029] | +0.0017 [-0.0014, +0.0046] | +0.0010 [-0.0023, +0.0038] |
| sensor_count | 0.8963 [0.8812, 0.9097] | 0.8852 [0.8681, 0.9002] | -0.0005 [-0.0028, +0.0019] | -0.0016 [-0.0041, +0.0009] |
| primary_sensor | 0.8967 [0.8804, 0.9111] | 0.8863 [0.8682, 0.9017] | -0.0002 [-0.0023, +0.0020] | -0.0005 [-0.0028, +0.0017] |
| count_primary | 0.8984 [0.8834, 0.9120] | 0.8876 [0.8710, 0.9028] | +0.0015 [-0.0017, +0.0044] | +0.0008 [-0.0024, +0.0037] |

Highest observed dev-only OOF binary F1: `availability_signature`. Highest observed
dev-only OOF macro F1: `availability_signature`. These labels are descriptive, not an
independent model-selection claim.

## Calibrators

- `global`: one empirical binary-F1 threshold fitted on the other four folds.
- `availability_signature`: a local threshold for each exact sensor signature.
- `sensor_count`: a coarse local threshold based only on number of available sensors.
- `primary_sensor`: a coarse local threshold based on canonical `anchor_sensor`.
- `count_primary`: a coarse local threshold for `(sensor count, anchor sensor)`.

For every conditioned calibrator, a group must have at least
128 rows, 16 rows of each
class, and 8 events in the four calibration folds.
Otherwise it falls back to that fold's global threshold. Eligible local
thresholds are shrunk toward global with
`w = n / (n + 256.0)`. These rules were fixed before reading
the held fold.

## Identity and leakage audit

- Frozen prediction rows: 12,621
- Canonical manifest rows: 12,621
- Exact one-to-one `id` join: 12,621; ID sets identical
- Canonical events/plumes: 291 / 621
- Prediction metadata checked against manifest: label, plume ID, availability signature
- Event and plume overlap between calibration and held portions: zero in every fold
- Input SHA-256 hashes and all per-fold fitted thresholds are retained in the JSON
- Test/sealed inputs read: **none**

| Held fold | Calibration rows | Held rows | Held events | Held plumes | Held labels 0/1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 10677 | 1944 | 49 | 90 | 933/1011 |
| 1 | 9801 | 2820 | 60 | 140 | 1283/1537 |
| 2 | 9857 | 2764 | 69 | 148 | 1312/1452 |
| 3 | 9826 | 2795 | 53 | 115 | 1326/1469 |
| 4 | 10323 | 2298 | 60 | 128 | 1084/1214 |

## Correct research claim

This audit can support only the statement that availability metadata may (or may
not) improve **decision calibration** for a fixed classifier on development data.
It cannot support a claim that two-axis attention or invisible-band pretraining
improved representation quality. Any threshold policy selected from this study
must be frozen and evaluated once on a separately authorized final set before
making a generalization claim.
