# Query360 TransientQuery screening log

Date: 2026-07-27 UTC

Status: frozen-feature screen, matched end-to-end runs, and canonical-event
bootstrap complete.

## Question

Does a current-time query benefit from coherent temporal evidence on the
engg-leung 360 m classification data, and does the answer depend on
Panopticon pretraining?  A second question is whether all sensor histories
should be treated as spatially equivalent.

This is an inner-validation screening experiment, not an official held-out
test result or a SOTA claim.

## Leakage-safe data contract

- Only
  `/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_360m_cluster_split/manifest_time_train_360m_macroregion_by_plume.csv`
  was used. Its SHA-256 is
  `e2b8de67efb79a243d14c3522b23ffd0345176032a0a903929d418c6993069c1`.
- The external test manifest was not opened.
- One deterministic representative was retained per
  `(plume_id, label, availability_signature)` to keep this screen tractable:
  11,435 of 121,041 source rows.
- Macro-region-disjoint inner split: 9,729 train rows and 1,706 validation
  rows. Plume, canonical acquisition event, cluster, macro-region, and input
  path overlap are all zero.
- Inner validation has 923 negatives and 783 positives over 73 macro-regions.
- The 32,028 referenced TIFF/NPZ inputs (67.0 GiB) were copied once into the
  hashed `/diniuvol/yuyao` cache. Training then required local cached files
  and did not stat or read the remote source paths.
- Plume masks were neither read nor cached.

## Model and matched arms

Each valid sensor/time image is encoded independently with its native channel
group: S2 12 bands, L89 7, EMIT 16, and S5P 1. No zero-padded channels from a
different sensor enter Panopticon spectral attention.

For every sensor, the head uses the current token as the query. Temporal
evidence is aggregated within sensor first; available sensor evidence is then
combined by masked sensor attention.

- `current_only`: all historical roles are masked.
- `transient_query`: current query attends to all valid roles of that sensor.
- `scale_aware_transient_query`: S2/L89/EMIT use temporal evidence; S5P
  contributes only its current token because the stored S5P footprint is
  approximately 10.5 km rather than a spatially matched 360 m plume crop.
- `history_shuffle_train`: valid history is replaced during training by the
  same sensor/role from a different canonical acquisition event and plume;
  coherent history is restored at validation.

Within each encoder and seed, arms have byte-identical initialization,
architecture, parameter count, epoch row order, optimizer, and five-epoch
compute. The only intervention is the temporal evidence mask/content.

## Frozen-feature results

Means over seeds 17, 29, and 43. Checkpoint selection is by inner-validation
AP; all principal arms selected epoch 5. `best F1` tunes its threshold on the
same inner validation set and is diagnostic only.

| Encoder | Arm | AP | ROC-AUC | Macro-F1 @ 0.5 | Diagnostic best macro-F1 |
|---|---|---:|---:|---:|---:|
| Panopticon | current only | 0.6513 | 0.6932 | 0.5788 | 0.6392 |
| Panopticon | shuffled history | 0.6491 | 0.6905 | 0.5807 | 0.6398 |
| Panopticon | TransientQuery | 0.6787 | 0.7244 | 0.5835 | 0.6648 |
| Panopticon | scale-aware TransientQuery | **0.6884** | **0.7304** | **0.6427** | **0.6679** |
| Random frozen | current only | 0.5109 | 0.5432 | 0.4987 | 0.5357 |
| Random frozen | shuffled history | 0.4888 | 0.5182 | 0.4048 | 0.5221 |
| Random frozen | TransientQuery | 0.4902 | 0.5187 | 0.4041 | 0.5242 |
| Random frozen | scale-aware TransientQuery | 0.5183 | 0.5496 | 0.4081 | 0.5379 |

Panopticon paired deltas, scale-aware TransientQuery minus current-only:

- AP `+0.03712`; per-seed `+0.03916`, `+0.03584`, `+0.03635`.
- ROC-AUC `+0.03722`; per-seed `+0.04152`, `+0.03648`, `+0.03366`.
- Macro-F1 at 0.5 `+0.06397`; per-seed `+0.08343`, `+0.02946`,
  `+0.07903`.
- Diagnostic best macro-F1 `+0.02866`; per-seed `+0.03145`, `+0.03235`,
  `+0.02218`.

A 2,000-repeat paired cluster bootstrap resampled all rows from the same
canonical acquisition event together (441 validation events), used the same
event draw for every arm, and averaged metric deltas—not probabilities—over
the three seeds. For scale-aware TransientQuery minus current-only, the 95%
intervals are:

- AP `+0.03712 [0.01754, 0.05566]`.
- ROC-AUC `+0.03722 [0.01776, 0.05506]`.
- Macro-F1 at 0.5 `+0.06397 [0.03865, 0.08788]`.

Regular TransientQuery minus current-only is AP
`+0.02742 [0.00689, 0.04694]` and ROC-AUC
`+0.03120 [0.01131, 0.04895]`; its fixed-threshold F1 interval crosses zero.
The scale-aware pretraining-by-temporal interaction relative to the random
frozen representation is AP `+0.02972 [0.00697, 0.05214]` and ROC-AUC
`+0.03076 [0.00703, 0.05259]`.

Scale-aware minus unqualified TransientQuery is `+0.00969` AP and
`+0.00602` ROC-AUC. Regular TransientQuery minus current-only is `+0.02742`
AP and `+0.03120` ROC-AUC. In contrast, regular TransientQuery on the random
frozen representation is `-0.02069` AP and `-0.02449` ROC-AUC versus
current-only.

### Sensor/availability diagnosis

Single-sensor AP, averaged over three seeds:

| Available sensor | Current only | TransientQuery | Scale-aware TQ |
|---|---:|---:|---:|
| S2 (470 rows) | 0.5758 | 0.6527 | 0.6500 |
| L89 (297 rows) | 0.6972 | 0.7246 | 0.7283 |
| EMIT (543 rows) | 0.5942 | 0.6307 | 0.6338 |
| S5P (137 rows) | 0.7934 | 0.7554 | 0.7925 |

For all 259 multi-sensor validation rows, AP is 0.7046 current-only, 0.7318
TransientQuery, and 0.7305 scale-aware TransientQuery. Thus temporal evidence
helps multi-sensor rows, while the extra overall gain from scale awareness
comes primarily from preventing the coarse S5P history from behaving like
spatially matched plume evidence.

The inner-validation split contains two- and three-sensor combinations but no
row with all four sensors simultaneously available. The experiment therefore
tests one availability-aware model across all four sensor families; it does
not yet establish simultaneous four-sensor fusion.

## End-to-end result

The complete Panopticon backbone was genuinely updated for three short epochs.
Within each initialization condition, the three arms started from the same
backbone/head state, saw the same row order and all the same encoded frames,
and used the same number of optimizer steps. Panopticon used a `1e-5` backbone
learning rate; genuine scratch used `1e-4`.

| Initialization | Arm | Best epoch | AP | ROC-AUC | Macro-F1 @ 0.5 | Diagnostic best macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Panopticon | current only | 3 | 0.7000 | 0.7378 | 0.6258 | 0.6727 |
| Panopticon | TransientQuery | 3 | **0.7180** | 0.7513 | 0.6146 | 0.6809 |
| Panopticon | scale-aware TQ | 3 | 0.7139 | **0.7537** | **0.6272** | **0.6833** |
| Scratch | current only | 3 | **0.5940** | **0.6202** | **0.5793** | 0.5900 |
| Scratch | TransientQuery | 1 | 0.5859 | 0.6129 | 0.5141 | **0.5853** |
| Scratch | scale-aware TQ | 3 | 0.5616 | 0.5940 | 0.5157 | 0.5725 |

On the single online seed, Panopticon TransientQuery minus current-only is
`+0.01793` AP and `+0.01342` ROC-AUC; Panopticon scale-aware minus current is
`+0.01388` AP and `+0.01590` ROC-AUC. Their event-bootstrap intervals cross
zero, so this short online run is supporting evidence rather than a standalone
significance claim.

The direction reverses without pretraining. Scratch TransientQuery is
`-0.00810` AP versus scratch current-only; scratch scale-aware is
`-0.03236 [-0.05459, -0.01216]` AP and
`-0.02621 [-0.04354, -0.00964]` ROC-AUC. The scale-aware
pretraining-by-temporal interaction is therefore
`+0.04624 [0.01801, 0.07541]` AP,
`+0.04212 [0.02003, 0.06526]` ROC-AUC, and
`+0.06499 [0.03806, 0.09341]` fixed-threshold macro-F1.

### Full-finetuning failure mode on multi-sensor rows

| Panopticon arm | Single-sensor AP (1,447 rows) | Multi-sensor AP (259 rows) |
|---|---:|---:|
| current only | 0.6938 | **0.7366** |
| TransientQuery | **0.7255** | 0.6884 |
| scale-aware TQ | 0.7205 | 0.6729 |

Unlike the frozen screen, full supervised finetuning improves the dominant
single-sensor subset but degrades the sparse multi-sensor subset. This is not
evidence that the proposed fusion problem is solved. It indicates that a
single unconstrained finetuning pass can overwrite useful multi-sensor
alignment when only 259 validation rows—and similarly sparse training
combinations—contain multiple sensors.

As a post-hoc diagnostic only, routing single-sensor rows to the finetuned TQ
model and multi-sensor rows to the current-only model gives AP 0.7257 and
ROC-AUC 0.7536. Routing the scale-aware single-sensor model the same way gives
AP 0.7210, ROC-AUC 0.7586, fixed-0.5 macro-F1 0.6297, and diagnostic best
macro-F1 0.6883. This policy was discovered on the same validation set and is
not a reportable final result; it is evidence for a learnable
availability/scale-conditioned temporal gate.

## Current interpretation

The positive result is not evidence that arbitrary cross-attention is useful.
Coherent history beats both current-only and cross-event shuffled history in
the frozen Panopticon representation, while regular temporal querying harms a
random frozen encoder and short genuine scratch training. The unqualified
frozen temporal model fails on coarse S5P history, and a sensor-scale
constraint recovers that loss. Under complete supervised finetuning, however,
both temporal variants hurt the sparse multi-sensor subset, so the hard mask
is a diagnostic rather than the final method.

The supported prototype story is therefore:

> Methane event evidence is transient, but heterogeneous EO histories are not
> interchangeable. A current-query temporal encoder should form evidence
> within each sensor, condition historical reliability on acquisition scale
> and sensor availability, and preserve rather than overwrite multi-sensor
> alignment during downstream adaptation.

This remains a mechanism screen. It does not yet establish novelty, official
test SOTA, segmentation transfer, or the benefit of a new self-supervised
pretraining objective.
