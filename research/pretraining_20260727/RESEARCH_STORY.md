# MethaneFuse pretraining research story — v2

Last updated: 2026-07-27

## 0. Evidence status and source constraint

This document deliberately separates three kinds of statements:

- **[Literature fact]**: supported by a linked primary paper or official paper
  page.
- **[Project evidence]**: observed in the current repository, controlled
  experiment log, or an explicitly identified historical experiment.
- **[Hypothesis]**: a proposed explanation or method component that still
  requires a matched experiment.

The original macOS survey path,
`/Users/ouuyou/Documents/Codex/2026-07-13/ab/outputs/MethaneFuse_Journal_Pretraining_Survey.md`,
is not mounted on this Linux host.  However, the user supplied attached copies
of the complete survey summary, paper-search record, history/data audit, and
the revised RankMask delta.  Those copies were read and incorporated.  The
survey's conservative novelty verdict is retained: **Level 2 — High Overlap
(fragile)**.  Residual anomaly detection, anomaly/tail ranking, pairwise
nonconformity, heterogeneous EO masked transfer, irregular acquisition-aware
latents, and near-time cross-sensor prediction are occupied families.  The
remaining possible contribution is the experimentally validated combination
of leakage-safe irregular histories, sensor/acquisition-conditioned handling,
weak-transient-preserving training, and transfer to classification plus
segmentation.  Primary-source URLs are used below for manuscript-facing
claims; the full bibliography should still be cross-checked at submission.

## 1. The minimal research story

> Generic Earth-observation foundation models make heterogeneous sensors
> technically compatible, while methane-specific models learn plume-sensitive
> spectral or physical cues. Neither capability alone establishes robust
> classification of short-lived methane events from any one of four highly
> heterogeneous sensors and their temporal context. We study whether
> **sensor-native, transient-aware pretraining** can close that gap.

The smallest convincing paper is not a claim that one model can already fuse
all four sensors. It is a controlled study with the following progression:

1. establish leakage-safe, event-held-out baselines independently on S2, L8/9,
   EMIT, and S5P;
2. show that an off-the-shelf generic frozen representation is not sufficient
   for methane transience on at least the sensor where it can be evaluated
   cleanly;
3. after enforcing zero global canonical-event overlap, pretrain one shared
   backbone over sensor-native temporal inputs, using sensor-specific input
   adapters but a common latent space;
4. test whether this raises per-sensor AP/F1 and the macro/worst-sensor result
   relative to scratch and generic-EO initialization;
5. only after rebuilding a globally aligned dataset, test missing-sensor fusion,
   sensor dropout, and full-to-subset distillation.

The defensible contribution is consequently about the intersection of
**heterogeneous sensor support**, **methane-specific transient learning**, and
**deployment-aligned evaluation**. It is not simply another universal EO
backbone, and it is not merely a new single-sensor plume detector.

The completed screen does **not** establish a positive backbone-pretraining
contribution.  It establishes a sharper diagnosis and a positive temporal
decision rule.  Both leakage-safe MAE variants reduced macro AP after
supervised transfer; a cross-fitted background predictor reconstructed the
current latent well but its innovation readout lost `0.0534 AP`; and a
label-free temporal-correspondence pretext reached `96.5%` training accuracy
yet degraded downstream AP by `0.0229` with all labels and `0.0466` with 10%
labels.  In contrast, a current-conditioned role-aware comparator on frozen
Panopticon CLS features improved three-seed L89 AP by `0.0409` and EMIT AP by
`0.0232` relative to current-only, with cross-event history shuffling removing
the gain.  The present story is therefore: persistent-scene objectives can
erase weak transients; matched historical context helps when it conditions,
rather than replaces or reconstructs, the current evidence.  The full result,
claim boundary, abstract, and next gates are in
`JOURNAL_STORY_20260727.md`.

## 2. Related-work map

The table records only the aspect needed to position this project. “Not the
central target” is intentionally weaker than claiming a paper cannot support a
setting.

| Work | Main task | Sensor scope | Temporal treatment | Missing-sensor / any-subset inference | Limitation relative to this project |
|---|---|---|---|---|---|
| [SatMAE](https://arxiv.org/abs/2207.08051) | Generic satellite masked pretraining | Multispectral satellite imagery | Models spatial/spectral/temporal structure in its pretraining setting | Not the central evaluation target | Generic EO representation learning; no methane-transient objective or four-sensor methane benchmark |
| [AnySat](https://arxiv.org/abs/2412.14123) | Resolution- and modality-flexible EO representation learning | Multiple EO modalities/sensors | Supports spatiotemporal EO inputs | Broad sensor flexibility is central | Input universality does not itself establish methane-plume discriminability |
| [Panopticon](https://arxiv.org/abs/2503.10845) | Generic multi-sensor EO representation learning | Broad surface-observation sensor scope | Supports heterogeneous EO observations | Broad sensor applicability is central | Generic EO objective; methane-specific temporal sensitivity must be measured downstream |
| [UniverSat](https://arxiv.org/abs/2606.23503) | Universal EO foundation model | Arbitrary spatial, temporal, and spectral resolutions through a Universal Patch Encoder | Cross-modal contrastive and latent masked modeling | Designed for unseen/configuration-varying sensors | In our frozen S2 probe, universality did not translate into the best methane ranking performance |
| [MethaneMapper](https://openaccess.thecvf.com/content/CVPR2023/html/Kumar_MethaneMapper_Spectral_Absorption_Aware_Hyperspectral_Transformer_for_Methane_Detection_CVPR_2023_paper.html) | Methane detection/localization | Hyperspectral imagery | Not a universal six-date interface | Not the central target | Strong methane-specific spectral modeling, but not heterogeneous S2/L8/9/EMIT/S5P classification |
| [Autonomous methane detection with Sentinel-2](https://arxiv.org/abs/2308.11003) | Operational methane detection | Sentinel-2 | Uses the sensor-specific observational setting | No | Demonstrates methane detection from S2, not one sensor-native framework spanning all four project sensors |
| [PRISMA cross-sensor transfer](https://arxiv.org/abs/2211.15429) | Methane detection with cross-sensor transfer | Imaging-spectrometer transfer involving PRISMA | Not the project's common six-date protocol | No | Cross-sensor transfer among related imaging-spectroscopy domains is different from inference across multispectral, hyperspectral, and coarse atmospheric products |
| [MethaneSAT cross-sensor transfer](https://arxiv.org/abs/2605.24273) | Methane retrieval/detection transfer | Methane-observing spectral instruments | Sensor-specific temporal setup | Not the central target | Valuable evidence for transfer, but not proof of one S2/L8/9/EMIT/S5P classifier with sensor-native temporal inputs |
| [MAPL-EMIT](https://arxiv.org/abs/2604.10094) | Methane plume learning/detection | EMIT hyperspectral imagery | EMIT-specific setup | No | Methane-specific but sensor-specific |
| [NormWear](https://doi.org/10.1145/3803808) ([preprint](https://arxiv.org/abs/2412.09758)) | Foundation model for heterogeneous wearable signals | Multiple physiological sensing channels | Masked reconstruction of transformed time series | Heterogeneous channels, but not EO missing-sensor deployment | Methodological analogy only: independent channel encoding plus periodic fusion; its CWT preprocessing is not directly appropriate for satellite imagery |

Two families are therefore adjacent but incomplete for this question:

- **[Literature fact] Generic any-sensor EO foundation models** emphasize
  flexible spatial/spectral/temporal input and transferable representations.
- **[Literature fact] Methane-specific models** emphasize absorption-aware,
  hyperspectral, retrieval, or sensor-specific plume cues.
- **[Interpretation, to be tested]** The open intersection is a methane-specific
  temporal objective that remains usable when only one of several heterogeneous
  sensors is present.

This is a cautious gap statement, not an absolute “first” claim. The final
paper should say that the surveyed literature provides limited direct evidence
at this exact intersection, rather than claiming no prior work exists.

## 3. What the project already demonstrates

### 3.1 Controlled evidence from the 2026-07-27 protocol

The independently trained sensor baselines used event-disjoint
train/recent-validation manifests within each sensor. After those decisions
and thresholds were frozen, exactly one sealed evaluation was run for each
selected S2, L89, EMIT, and S5P model. Rows below are validation evidence
unless explicitly marked as sealed-test evidence.

#### Joint split audit and evidence boundary

- **[Project evidence — protocol audit]** A later cross-sensor audit found **124
  canonical events** in
  `union(all sensor training events) ∩ union(all sensor validation events)`.
  Thus, per-sensor disjointness was not sufficient once an encoder could learn
  from several sensors: an event present in one sensor's training set could
  reappear through another sensor's validation set.
- **[Project evidence — exclusion rule]** Every old four-sensor
  shared-residual diagnostic, including any artifact described as shared
  Panopticon, is marked `invalid_diagnostic_only` and excluded from the result
  registry; so is its joint independent comparator. None of their accuracy,
  F1, AP, or AUROC values is project evidence, a model-selection signal, or a
  pretraining result.
- **[Project evidence — protocol audit]** The replacement
  `four_sensor_global_val_purge_v1` protocol freezes the union of validation
  events and removes those events from every sensor's training manifest. Its
  recorded global train/validation overlap is **124 before purge and 0
  after purge**, with canonicalization rules, manifest hashes, and a protocol
  fingerprint stored for audit.

This is also a benchmark lesson and a contribution independent of model
accuracy: the leakage unit must follow the model's information-sharing graph.
A collection of legal single-sensor splits is not automatically a legal
shared-encoder benchmark. Cross-sensor union-overlap auditing and a
machine-verifiable zero-overlap gate should therefore be prerequisites for
joint training, checkpoint comparison, and transfer. The globally purged
unmasked and validity-masked MAE screens below meet that evidence boundary and
are therefore valid shared-pretraining experiments, but both are **negative
promotion results**, not evidence that shared pretraining improves
classification. The supervised RankNet screen that follows uses the same
globally purged comparator, but it is deliberately reported separately: it
changes the downstream classification loss and performs no pretraining.

#### Globally purged unmasked MAE — credible negative result

**[Project evidence]** One epoch of four-sensor shared unmasked MAE
pretraining was compared with a matched scratch model after one full
supervised epoch.  Pretrain, finetune, and scratch used the same globally
purged data protocol; finetune and scratch used the same seed, architecture,
optimizer, 157 balanced rounds, and 628 supervised updates.  Each cell reports
`scratch -> MAE finetune (finetune - scratch)`:

| Sensor | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| S2 | 0.745224 -> 0.717887 (**-0.027336**) | 0.745180 -> 0.737752 (-0.007427) | 0.715232 -> 0.726288 (+0.011055) | 0.668046 -> 0.661167 (-0.006878) |
| L89 | 0.634397 -> 0.606806 (**-0.027591**) | 0.675728 -> 0.662275 (-0.013453) | 0.618006 -> 0.608347 (-0.009659) | 0.609331 -> 0.544347 (-0.064985) |
| EMIT | 0.533902 -> 0.555246 (**+0.021344**) | 0.639065 -> 0.664518 (+0.025453) | 0.585046 -> 0.605784 (+0.020737) | 0.483565 -> 0.534458 (+0.050892) |
| S5P | 0.570401 -> 0.571657 (**+0.001256**) | 0.527725 -> 0.526291 (-0.001433) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.244162 (+0.074140) |
| **Macro over sensors** | **0.620981 -> 0.612899 (-0.008082)** | **0.646924 -> 0.647709 (+0.000785)** | **0.650817 -> 0.656351 (+0.005533)** | **0.482741 -> 0.496033 (+0.013292)** |

All **21/21** machine-verifiable artifact, leakage, signature, transfer,
initial-head, update-budget, model-size, reconstruction-finiteness, and metric
recomputation checks passed.  Encoder coverage was
`23,316,480 / 23,316,480` parameters (`1.0`), and the transfer-before,
transfer-after, finetune-initial, and scratch-initial head hashes were
identical.  Thus the negative result cannot be dismissed as a partial load,
random-head transfer, data mismatch, unequal supervised budget, or known
cross-sensor leakage.

The preregistered promotion gate failed: macro AP was `-0.008082` rather than
the required `>= +0.010`; only `2/4` sensor AP deltas were non-negative rather
than at least three; and L89 was the worst sensor at `-0.027591`, below the
`-0.010` floor.  Macro best-F1 passed its secondary guard with `+0.005533`,
but that does not offset the failed AP criteria.  The authoritative decision
is **`do_not_promote`**: do not add epochs or seeds to this unmasked objective.

The sensor pattern is informative but not causal proof: EMIT AP improved,
S5P AP was nearly unchanged, and S2/L89 AP regressed.  The unmasked loss
includes zero-filled invalid TIFF elements and empty channels because the
validity mask is discarded before reconstruction; S5P is also a coarse field
resized to the common input size.  This motivated the isolated
validity-masked-reconstruction fallback below, with protocol, architecture,
optimizer, update budget, and gate held fixed.

Decision artifact:
`/diniuvol/yuyao/methanefuse_research_20260727/results/mae_vs_scratch_gate.json`.

#### Globally purged validity-masked MAE — second credible negative

**[Project evidence]** The matched fallback used native per-channel validity,
residual-validity intersections, mask-safe resizing, and valid-element
normalization while preserving the same seed, globally purged data,
architecture, 157 balanced rounds, and 628 updates.  All **42/42**
machine-verifiable checks passed.  Each cell reports
`scratch -> masked-MAE finetune (finetune - scratch)`:

| Sensor | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| S2 | 0.745224 -> 0.719978 (**-0.025246**) | 0.745180 -> 0.722581 (-0.022599) | 0.715232 -> 0.708650 (-0.006582) | 0.668046 -> 0.582177 (-0.085869) |
| L89 | 0.634397 -> 0.601040 (**-0.033357**) | 0.675728 -> 0.663003 (-0.012725) | 0.618006 -> 0.620816 (+0.002809) | 0.609331 -> 0.594227 (-0.015105) |
| EMIT | 0.533902 -> 0.548353 (**+0.014451**) | 0.639065 -> 0.656264 (+0.017199) | 0.585046 -> 0.594397 (+0.009350) | 0.483565 -> 0.524719 (+0.041153) |
| S5P | 0.570401 -> 0.575338 (**+0.004937**) | 0.527725 -> 0.533810 (+0.006085) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.233139 (+0.063117) |
| **Macro over sensors** | **0.620981 -> 0.611177 (-0.009804)** | **0.646924 -> 0.643915 (-0.003010)** | **0.650817 -> 0.652211 (+0.001394)** | **0.482741 -> 0.483565 (+0.000824)** |

The promotion gate again failed three of four performance criteria: macro AP
was `-0.009804` instead of `>= +0.010`, only EMIT and S5P were non-decreasing
in AP (`2/4` instead of `>= 3/4`), and L89 was below the worst-sensor floor at
`-0.033357`.  Macro best-F1 passed its secondary guard at `+0.001394`.
Validity masking therefore did **not** reverse the transfer pattern: the two
weaker sensors gained modest AP, but S2 and L89 regressed enough to reduce the
four-sensor macro.  The decision is **`do_not_promote`**; no extra epochs or
seeds are warranted for either tested MAE objective.

Decision artifact:
`/diniuvol/yuyao/methanefuse_research_20260727/results/validitymasked_mae_vs_scratch_gate.json`.

#### Supervised BCE + RankNet — broad ranking gain, failed S5P promotion gate

**[Project evidence — supervised partial success]** After both MAE objectives
failed, the single preregistered fallback kept the same shared
sensor-native architecture, data, seed, initialization, optimizer, 157
balanced rounds, 628 updates, and complete validation sets, but changed the
supervised loss from weighted BCE to weighted BCE plus within-sensor,
all-positive/negative-pairs RankNet (`weight=0.5`, `temperature=1.0`). It
started from scratch and used no MAE checkpoint and no sealed-test data.
All **20/20** protocol, objective, artifact, metric-recomputation, and
same-checkpoint temporal-ablation checks passed; every one of the 628 sensor
batches contained both classes.

Each cell reports `matched BCE scratch -> BCE+RankNet (RankNet - scratch)`:

| Sensor | AP | AUROC | Best positive F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| S2 | 0.745224 -> 0.752423 (**+0.007199**) | 0.745180 -> 0.753565 (**+0.008386**) | 0.715232 -> 0.724228 (+0.008995) | 0.668046 -> 0.632706 (-0.035340) |
| L89 | 0.634397 -> 0.640748 (**+0.006351**) | 0.675728 -> 0.679762 (**+0.004033**) | 0.618006 -> 0.630248 (+0.012241) | 0.609331 -> 0.596926 (-0.012405) |
| EMIT | 0.533902 -> 0.560876 (**+0.026974**) | 0.639065 -> 0.666551 (**+0.027486**) | 0.585046 -> 0.598788 (+0.013741) | 0.483565 -> 0.578706 (+0.095140) |
| S5P | 0.570401 -> 0.579910 (**+0.009509**) | 0.527725 -> 0.540408 (**+0.012684**) | 0.684984 -> 0.684984 (+0.000000) | 0.170022 -> 0.522841 (+0.352819) |
| **Macro over sensors** | **0.620981 -> 0.633489 (+0.012508)** | **0.646924 -> 0.660072 (+0.013147)** | **0.650817 -> 0.659562 (+0.008745)** | **0.482741 -> 0.582795 (+0.100053)** |

The AP screen itself was consistently positive: all four sensor AP deltas and
all four AUROC deltas were positive, the worst AP delta was `+0.006351`, and
macro AP exceeded the preregistered `+0.010` requirement. The same
checkpoint's temporal-input ablation also favored full
current/recent/seasonal input over zeroed recent/seasonal input on every
sensor:

| Sensor | Current-information-only AP | Full temporal AP | Full - current |
|---|---:|---:|---:|
| S2 | 0.747140 | 0.752423 | **+0.005282** |
| L89 | 0.633311 | 0.640748 | **+0.007437** |
| EMIT | 0.549697 | 0.560876 | **+0.011179** |
| S5P | 0.576525 | 0.579910 | **+0.003385** |
| **Macro over sensors** | **0.626668** | **0.633489** | **+0.006821** |

This is evidence that the trained classifier uses the temporal inputs for
ranking under the declared zero-input ablation. It is not evidence that
RankNet is a pretraining method, and the ablation does not structurally remove
the temporal branches.

The formal decision is nevertheless **`do_not_promote`**. S5P probabilities
were nonconstant, its AP exceeded prevalence (`0.579910` versus `0.520894`),
and AUROC exceeded chance (`0.540408`), but its best-F1 threshold was `0.0`
and predicted `3,805/3,805` rows positive. Thus the ranking gain did not solve
the preregistered balanced-decision requirement. No weight/temperature sweep,
extra epoch, or extra seed is authorized from this screen. The defensible
interpretation is **supervised ranking-aware partial success with unresolved
S5P operating-point behavior**, not a promoted model and not a pretraining
win.

Authoritative artifacts and SHA-256:

| Artifact | SHA-256 |
|---|---|
| `/diniuvol/yuyao/methanefuse_research_20260727/results/ranknet_vs_bce_gate.json` | `c36e007a6e03025b4bf54195143d0876e96a9e9376c75e7b53634d29552cb6fc` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/metrics_history.json` | `0923eb5525d3db5d840a8666465dfd2159797183614bd080fbb923b7980bf1a9` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/checkpoint_latest.pth` | `ef070e6f47abaa2a8401203da2222365c51a913fe97b310d59784bb9c648f683` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_full_temporal.json` | `ca3066b996ecef6cb17b37786624282f9e1b993069380f10cbf1f4637145c054` |
| `/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_ranknet_globalpurge_e1_seed20260727/validation_current_only.json` | `e34ca0c047adac56c3b8b720369eb5229f8a54ba1f38c67416e3f0e1d1ada7bf` |

#### Sealed-test summary

| Sensor / selected model | Rows | Fixed val threshold | Positive F1 | Macro-F1 | AP | AUROC | Recall | FPR | R@FPR 5% | Confusion (`TP/FP/FN/TN`) | Development-event overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 current-only, val-AP epoch 2 | 23,413 | 0.520996 | 0.738 | 0.702 | 0.815 | 0.805 | 0.822 | 0.411 | 0.383 | 9,678 / 4,781 / 2,093 / 6,861 | **0** (retrospective manifest-only audit) |
| L89 raw-three-time, val-AP epoch 1 | 7,942 | 0.314209 | 0.737 | 0.736 | 0.775 | 0.816 | 0.836 | 0.344 | 0.347 | 2,942 / 1,521 / 578 / 2,901 | **0** |
| EMIT residual-three-time, val-AP epoch 1 | 6,138 | 0.228882 | 0.611 | 0.591 | 0.666 | 0.691 | 0.742 | 0.522 | 0.232 | 1,963 / 1,823 / 681 / 1,671 | **0** |
| S5P raw-six-time, val-AP epoch 3 | 3,213 | 0.118713 | 0.687 | 0.343 | 0.657 | 0.638 | 1.000 | 1.000 | 0.154 | 1,680 / 1,533 / 0 / 0 | **0** |

**[Project evidence — derived sealed summary]** The unweighted arithmetic
macro over these four independently selected sensor models is:

| Aggregate | Positive F1 | Macro-F1 | AP | AUROC | R@FPR 5% |
|---|---:|---:|---:|---:|---:|
| Four-sensor sealed macro | 0.693 | 0.593 | 0.728 | 0.738 | 0.279 |

This row is not the result of a shared model and does not pool predictions
across sensors. Its positive-F1 average is especially easy to misread because
S5P contributes positive F1 `0.687` while predicting every sealed row
positive. The four-sensor macro-F1 `0.593`, S5P macro-F1 `0.343`, S5P FPR
`1.0`, and macro R@FPR 5% `0.279` are the appropriate counterweights; AP and
AUROC summarize ranking rather than fixed-threshold calibration.

The exact L89 threshold was `0.314208984375`; its positive F1 was `0.737066`,
macro-F1 `0.735702`, AP `0.774746`, AUROC `0.816375`, recall `0.835795`,
FPR `0.343962`, and recall at FPR 5% `0.347443`.  The exact EMIT threshold
was `0.2288818359375`; its positive F1 was `0.610575`, macro-F1 `0.591124`,
AP `0.666042`, AUROC `0.691294`, recall `0.742436`, FPR `0.521752`, and
recall at FPR 5% `0.231846`.  EMIT's FPR makes the distinction between
ranking evidence and an operationally useful threshold especially important.

The L89 and EMIT artifacts both report zero canonical-event overlap with
development data.  The older S2 artifact did not store this field, so a
metadata-only audit compared its 1,376 development event IDs with the 210
sealed-test event IDs and also found zero overlap; it did not rerun inference.
Checkpoints were selected by validation AP and thresholds were fixed on
validation before sealed inference.  These results must **not**
be compared directly with the historical L89 `0.754` and EMIT `0.567` F1
figures: those older numbers use different splits, data/input protocols, and
test-conditioned/test-selected model-selection procedures.

| Observation | Status | Interpretation |
|---|---|---|
| S2 current-only Panopticon reaches validation AP `0.863` at epoch 2; the finished run reaches F1@0.5 `0.787`, best-threshold F1 `0.788`, and AUROC `0.859` | **[Project evidence]** | A short, task-trained S2 baseline is already strong under the new split |
| S2 raw-six-time Panopticon at epoch 1 reaches AP `0.811`, best F1 `0.769`, AUROC `0.823` | **[Project evidence]** | Temporal input contains signal, but the current raw-fusion implementation does not yet beat current-only |
| Frozen UniverSat S2 probes reach AP `0.731` for current and `0.735` for raw six-time input | **[Project evidence]** | On this dataset and frozen-probe setup, a generic universal representation is well below the task-trained Panopticon baseline; this is not a claim about full fine-tuning or other datasets |
| Naive S2 current+residual channel concatenation gives AP `0.680`, AUROC `0.736`, and F1@0.5 `0.392` in the pilot | **[Project evidence]** | Reflectance and normalized differences should not be forced through one undifferentiated stem |
| A separate temporal-token residual pilot improves over naive concatenation but remains below current/raw input (AP `0.731`, best F1 `0.726`, AUROC `0.753`) | **[Project evidence]** | Architectural separation helps, but current residual encoding is not yet the main result |
| AP-selected L89 raw-three-time Panopticon reaches sealed AP `0.775`, AUROC `0.816`, positive F1 `0.737`, and macro-F1 `0.736` with development-event overlap zero | **[Project evidence]** | L89 provides the strongest balanced sealed classification among the newly evaluated non-S2 sensors under this protocol |
| AP-selected EMIT residual-three-time Panopticon reaches sealed AP `0.666` and AUROC `0.691`, but fixed-threshold macro-F1 is `0.591` and FPR is `0.522` | **[Project evidence]** | EMIT retains ranking signal, while its operating-point discrimination remains a clear target for improvement |
| AP-selected S5P raw-six-time Panopticon reaches validation AP `0.644` and AUROC `0.622`; on the one authorized sealed evaluation it reaches AP `0.657` and AUROC `0.638` | **[Project evidence]** | Coarse CH4 context has modest ranking signal, but scaling the S2 recipe directly is not justified |
| The frozen S5P validation threshold `0.118713` predicts 3,804/3,805 validation rows positive and all 3,213 sealed rows positive; sealed positive F1 is `0.687`, but macro-F1 is `0.343` and FPR is `1.0` | **[Project evidence]** | Positive-class F1 is the all-positive baseline here, not useful classification evidence; calibration and balanced discrimination remain unsolved |
| S5P raw-six-time ResNet18 reaches AP `0.568`; current+residual ResNet18 reaches AP `0.559` | **[Project evidence]** | Raw history currently ranks slightly better than explicit residual input, but neither solves S5P |
| Globally purged unmasked MAE finetuning changes macro AP by `-0.008082` versus matched scratch; S2/L89/EMIT/S5P AP deltas are `-0.027336/-0.027591/+0.021344/+0.001256`, with all 21/21 integrity checks passing | **[Project evidence — credible negative]** | The implementation is `do_not_promote`; it motivated the isolated validity-masked fallback whose completed negative result appears next |
| Globally purged validity-masked MAE finetuning changes macro AP by `-0.009804`; S2/L89/EMIT/S5P AP deltas are `-0.025246/-0.033357/+0.014451/+0.004937`, with all 42/42 integrity checks passing | **[Project evidence — credible negative]** | Masking invalid reconstruction elements did not rescue transfer; both MAE objectives are `do_not_promote`, so the masked branch is no longer pending |
| Supervised weighted-BCE + RankNet changes macro AP by `+0.012508` and macro AUROC by `+0.013147`; AP/AUROC rise on all four sensors, and full temporal input adds AP on all four sensors versus the same checkpoint's current-information-only ablation | **[Project evidence — supervised partial success]** | This is not pretraining. The formal decision remains `do_not_promote` because S5P's best-F1 threshold predicts all 3,805 validation rows positive; ranking improved, but useful S5P decision behavior was not established |

The immediate negative result is scientifically useful:

> **[Project evidence]** In the present frozen S2 probe, generic
> sensor-universal spatial features do not equal transient methane
> discriminability.

That sentence is supportable. “UniverSat is worse for methane” without the
dataset, frozen-probe, and downstream-protocol qualifiers is not.

The S5P result sets an equally important reporting boundary.  Its sealed
positive-class F1 must never be presented as a gain over the historical S5P
F1: the fixed validation-derived threshold classifies every sealed example as
positive.  AP/AUROC show only modest ranking skill, while macro-F1, FPR, and
the confusion matrix reveal failed binary calibration.  A credible shared
pretraining result must improve the weakest sensor under balanced metrics and
a validation-fixed operating point, not merely preserve an all-positive
positive-class F1.

### 3.2 Historical evidence that must remain contextual

Historical single-sensor F1 values are S2 `0.764`, L8/9 `0.754`, EMIT `0.567`,
and S5P `0.638`, but their split and checkpoint-selection procedures differ.
They must not be compared numerically against the new recent-validation runs.

The older NormWear-derived methane experiments found:

- residual-only MAE: AUROC `0.90870`, AP `0.90404`, F1 `0.84621`;
- matched residual-only scratch: AUROC `0.9037`, AP `0.9001`, F1 `0.8342`.

This is only a very small matched gain (`+0.00326` AP), and the older
pretraining pool included unlabeled train and test manifests. Therefore:

- **[Project evidence]** normalized temporal residuals can be a strong input;
- **[Project evidence]** residual MAE beat the other tested MAE targets in that
  historical setup;
- **not established:** residual MAE materially outperforms a matched scratch
  model under a sealed-test, train-only pretraining protocol.

The new paper must not use the historical table as its central pretraining
claim.

## 4. Research questions and hypotheses

### RQ1 — Is generic EO pretraining sufficient?

- **H1 [Hypothesis]:** Generic EO initialization improves sample efficiency,
  but frozen generic features under-represent the transient signal required for
  methane classification.
- **Current support:** the frozen UniverSat S2 gap.
- **Required test:** matched scratch, frozen probe, partial fine-tune, and full
  fine-tune with the same head, data, seed, and checkpoint rule.

### RQ2 — Does methane-temporal pretraining help across sensors?

- **H2 [Hypothesis]:** A shared backbone trained on sensor-native temporal
  sequences improves macro AP/F1 and the weakest sensor relative to four
  independently trained scratch models.
- **Required test:** per-sensor versus mixed-sensor pretraining under matched
  model capacity and update budget, using the fingerprinted zero-overlap
  global-purge protocol.
- **Current result:** both tested shared-MAE variants failed their matched
  macro-AP promotion gates. The supervised RankNet gain does not answer this
  pretraining question.

### RQ3 — What representation of time is useful?

- **H3a [Hypothesis]:** Raw six-date tokens are a safer default than naive
  residual concatenation.
- **H3b [Hypothesis]:** A separately normalized residual stream may add value
  when it has its own adapter and objective.
- **Current support:** H3a is consistent with the S2/S5P pilots. In the
  supervised RankNet checkpoint, full temporal input improves AP over the
  zeroed-recent/seasonal ablation on all four sensors, which supports temporal
  input utility but does not isolate one residual stream or establish a
  pretraining benefit.

### RQ4 — Can a model tolerate naturally missing sensors?

- **H4 [Hypothesis]:** Sensor/time dropout and full-to-subset distillation
  improve subset robustness without sacrificing the strongest single-sensor
  result.
- **Required prerequisite:** one globally aligned and globally split
  multi-sensor dataset. Independent sensor CSVs cannot test this claim.

### RQ5 — Does AP-aligned supervised ranking help?

- **Current support:** weighted-BCE + within-sensor RankNet raises macro AP by
  `0.012508`, raises AP and AUROC on all four sensors, and raises macro
  best-F1 and F1@0.5 under the matched one-epoch screen.
- **Boundary:** the S5P best-F1 operating point remains all-positive, so the
  result is `do_not_promote`. It motivates studying ranking/calibration
  jointly, but does not justify a sweep or a pretraining claim.

## 5. Proposed method: sensor-native transient-aware pretraining

“MethaneFuse-PT” is a working name, not a finalized novelty claim.

### Module A — sensor-native input adapters

For each sensor:

1. compute band statistics from training events only;
2. preserve its native band count, wavelength/resolution metadata, and crop
   geometry;
3. map native spatial/spectral patches into a common latent width with a small
   sensor adapter;
4. add acquisition-time and sensor identity embeddings.

This follows the useful abstraction from universal EO models—shared latent
processing without pretending the raw channels are interchangeable.

### Module B — explicit temporal representation

Use six acquisition tokens where available:

`t0, prev1, prev2, prev3, seasonal, year`.

The default mainline is raw temporal tokens. A residual option computes
normalized differences only after train-only per-band normalization and sends
them through a distinct residual adapter. Reflectance and difference tensors
must not share one first-layer distribution by simple channel concatenation.

### Module C — shared backbone with periodic cross-stream fusion

The NormWear implementation provides a useful design pattern:

- encode each variable/channel independently;
- periodically fuse summary tokens across channels;
- keep the bulk of the representation learner shared.

For EO, the analogous unit is a sensor/time stream rather than a wearable
channel. We should reuse the pattern, not the CWT transform:

- within-sensor spatial/spectral encoding;
- temporal fusion among available dates;
- shared latent blocks;
- optional cross-sensor fusion only for genuinely aligned samples.

### Module D — pretraining objectives

The minimum objective set should be ablated rather than bundled:

1. **masked current reconstruction/prediction** — generic scene content
   baseline;
2. **masked temporal-token prediction** — recover a held-out date from the
   remaining dates;
3. **transient/residual prediction** — predict the normalized change between
   current and reference observations using a separate residual decoder;
4. **optional aligned cross-sensor agreement** — only on samples proven to
   share event, location, crop footprint, and split.

The primary method claim should be assigned only after a matched objective
ablation. Historical residual MAE is motivation, not proof.

### Module E — supervised adaptation

- one sensor-specific classification head per sensor for the first mainline;
- compare frozen, adapter-only, partial, and full backbone updates;
- use a shared head only if label semantics and calibration are shown to be
  compatible;
- select checkpoints without test access.

The completed weighted-BCE + RankNet screen belongs in this supervised module,
not in Module D. It provides positive evidence for AP-aligned downstream
optimization, but its failed S5P non-degeneracy gate prevents promotion and
shows that ranking and operating-point quality must be audited separately.

### Module F — missing-sensor robustness, deferred

After joint data rebuilding:

- train with sensor dropout and time dropout;
- distill from the full available sensor set to every observed subset;
- report performance by availability pattern, not only a pooled score.

This is a second-stage contribution and must not delay the independent-sensor
pretraining result.

## 6. Unified-data prerequisite

The old methane pipeline contains the necessary geometry logic:

- outer-join sensor records by `plume_id`;
- coalesce event latitude, longitude, and time;
- select the finest available sensor as the physical anchor;
- convert one anchor offset from pixels to metres and then to each sensor's
  native pixel offset;
- crop from each sensor's 512-source imagery;
- retain real sensor availability rather than imputing nonexistent sensors.

For the new benchmark this logic must be tightened:

1. define a canonical `event_group_id` before sample construction;
2. assign every event group once to train/validation/test using event time;
3. only then generate all positive/negative crops for every sensor;
4. use the same physical location, label definition, plume ID, and partition
   for every sensor view;
5. audit both `event_group_id` and `plume_id` overlap;
6. preserve S5P's coarse native support rather than treating an upsampled
   3×3 grid as fine-resolution texture.

The older `plume_id` hash split is not sufficient for the new temporal claim.
The upgraded event-group temporal split is the protocol to retain.

The metadata-only manifest-v2 provenance pilot is complete, but production
crop remains blocked.  It found candidate temporal aliases, missing
query-usability/per-channel-validity and verified-observation fields, and a
known EMIT `prev3` embedded-product conflict.  Before any mass materialization,
the project must freeze v2 execution fields, repair or quarantine provenance
conflicts, deterministically resolve aliases, pass a split/label/sensor-pattern
stratified metadata/header pilot, and then pass small query-coverage and
materialized-readback gates.  See
`MANIFEST_V2_PROVENANCE_PILOT.md`.

## 7. Evaluation protocol

### 7.1 Fixed split policy

- Seal the four original test CSVs until the method and ablations are frozen.
- Build validation from the most recent event groups within the original
  training partition.
- Group all derived crops by canonical event, never by row.
- For joint experiments, use one global event assignment across sensors.
- Before joint training, require
  `union(all sensor train events) ∩ union(all sensor validation events) = ∅`;
  within-sensor overlap checks alone are insufficient.
- Compute normalization statistics from training events only.

### 7.2 Metrics and checkpoint selection

Per sensor, report:

- AP;
- positive-class F1 at a validation-selected threshold;
- F1 at the fixed threshold `0.5`;
- AUROC;
- recall at FPR 5% and 10%;
- precision, recall, and the confusion matrix;
- calibration error if probabilities are intended for deployment.

Across sensors, report:

- macro average;
- sample-weighted average;
- worst-sensor score;
- each sensor separately, so S2 cannot hide S5P failure.

Select checkpoints by validation AP, with positive-class F1 as the declared
tie-breaker. Tune the operating threshold on validation only and transfer it
unchanged to test. Recall@FPR should also use a validation-derived operating
point when reporting final test behavior.

### 7.3 Uncertainty

- Run at least three seeds for promoted configurations.
- Bootstrap confidence intervals by event group, not crop row.
- Keep compute, number of updates, labeled rows, and model capacity matched in
  pretraining-versus-scratch comparisons.
- Include label-fraction experiments (for example 1%, 10%, 50%, 100%) because
  the strongest justification for pretraining may be sample efficiency rather
  than full-data asymptotic F1.

## 8. Minimum ablation matrix

| Axis | Required settings | Question answered |
|---|---|---|
| Initialization | scratch; generic EO frozen probe; generic EO fine-tune; methane-temporal pretrain | Is improvement from pretraining or merely architecture? |
| Pretraining data | same-sensor only; mixed unaligned sensors; aligned sensor subset | Does heterogeneous pretraining transfer, and does true pairing add value? |
| Input | current; raw six-time; residual-only; current + separate residual adapter | Is transience useful, and is separation necessary? |
| Objective | current MAE; temporal masked prediction; residual prediction; combined | Which objective causes the gain? |
| Sharing | four independent models; shared backbone + sensor adapters; shared backbone without adapters | Is sharing beneficial without erasing sensor physics? |
| Adaptation | frozen; adapter-only; partial; full | Is the representation reusable or only a good initialization? |
| Temporal metadata | acquisition days; shuffled days; no dates | Does the model use time rather than extra images? |
| Missingness | no dropout; time dropout; sensor dropout; subset distillation | Does robustness extend to actual missing inputs? |
| Supervision | 1%, 10%, 50%, 100% labels | Where does pretraining provide practical value? |

Sanity checks are mandatory:

- random/all-positive classifier;
- metadata/missingness-only classifier;
- label shuffle;
- history-only model;
- small-subset overfit;
- event-overlap audit;
- a control that shuffles historical images across events while preserving
  their marginal distribution.

## 9. Decision gates for the ten-hour mainline

1. **Promote a generic checkpoint** only if it beats scratch or clearly improves
   label efficiency under a matched protocol. Frozen-probe failure alone is
   not enough to discard fine-tuning.
2. **Promote raw temporal input** if it improves validation AP or F1 on at least
   two sensors without a material worst-sensor regression.
3. **Promote the residual branch** only if the separate-adapter version beats
   raw temporal input; do not scale naive concatenation.
4. **Claim shared pretraining benefit** only if shared pretraining beats
   same-capacity independent pretraining on macro or worst-sensor performance,
   with no test-based model selection.  Both the unmasked and validity-masked
   MAE screens failed this gate (`do_not_promote`) despite complete integrity
   checks; do not add epochs or seeds to either objective.  Any next
   pretraining objective requires a separate preregistered comparison.
5. **Promote the supervised RankNet fallback** only if every registered
   ranking, F1, temporal-use, and S5P non-degeneracy criterion passes. Macro AP
   improved by `+0.012508`, and every sensor gained AP/AUROC, but the S5P
   best-F1 prediction was all-positive. The formal decision is
   `do_not_promote`; do not tune its weight/temperature or add epochs/seeds.
6. **Claim missing-sensor robustness** only after global recropping/splitting and
   availability-pattern evaluation.

## 10. Claim boundary

### Claims currently supportable

- The four sensors form substantially different spatial/spectral regimes and
  require sensor-aware inputs.
- Under the current event-held-out S2 protocol, task-trained Panopticon
  outperforms the tested frozen UniverSat probes in AP.
- Raw temporal input is a stronger current default than naive residual
  concatenation in the tested S2/S5P pilots.
- Historical experiments motivate residual modeling but do not prove a large,
  leakage-safe pretraining advantage over matched scratch.
- A shared-encoder benchmark requires a global cross-sensor event audit; the
  project found 124 leaked event IDs under the old per-sensor composition and
  established a fingerprinted purge protocol with zero remaining overlap.
- Under that valid global-purge protocol, the one-epoch unmasked-MAE transfer
  is a credible negative result: 21/21 integrity checks passed, but macro AP
  fell by `0.008082`, with material AP regressions on S2 and L89; the decision
  is `do_not_promote`.
- The matched validity-masked fallback is also a credible negative result:
  42/42 integrity checks passed, but macro AP fell by `0.009804`.  EMIT and S5P
  AP rose slightly while S2 and L89 fell, so validity masking did not reverse
  the cross-sensor transfer pattern and is also `do_not_promote`.
- A separately preregistered **supervised** weighted-BCE + RankNet screen
  raises macro AP by `0.012508`, macro AUROC by `0.013147`, macro best-F1 by
  `0.008745`, and macro F1@0.5 by `0.100053` versus matched BCE scratch.
  AP/AUROC rise on all four sensors, and the same-checkpoint temporal-input
  ablation gives positive AP contributions on all four sensors.
- That RankNet result is only a partial success: S5P's best-F1 threshold is
  `0.0` and predicts all 3,805 validation rows positive. The gate is
  `do_not_promote`, and useful S5P classification remains unsolved.
- The selected S5P baseline has modest sealed ranking skill but a degenerate
  validation-selected operating threshold; useful calibrated S5P
  classification remains an open problem.

### Claims that require new evidence

- one shared pretrained backbone improves all four sensor classifiers;
- transient-aware pretraining improves full-data F1;
- a different, separately preregistered objective improves macro or
  worst-sensor classification over matched scratch;
- ranking-aware supervision yields non-degenerate, deployable S5P decisions or
  generalizes beyond this one validation seed;
- cross-sensor aligned pretraining is better than a mixed but unpaired sensor
  pool;
- sensor dropout provides robust any-subset inference;
- the proposed method is the first of its kind.

### Claims to avoid

- comparing historical test-selected metrics directly with the new validation
  metrics;
- using the S5P sealed positive-class F1 `0.687` as evidence of useful
  classification without stating that all sealed rows were predicted positive;
- calling upsampled S5P a fine-resolution image;
- treating frozen-probe results as full fine-tuning results;
- describing the old residual-MAE table as a clean train-only pretraining win;
- claiming validity masking rescued MAE transfer or remains an unevaluated
  fallback;
- calling the supervised RankNet experiment pretraining, calling it promoted,
  or claiming that its AP gain solved S5P classification;
- claiming cross-sensor fusion from four independently split datasets;
- citing any metric from the invalid old four-sensor shared-residual/shared
  Panopticon diagnostics as evidence for sharing, pretraining, or model
  selection;
- asserting an absolute literature “first” before the inaccessible survey is
  restored and the search is updated.

## 11. Candidate paper framing

Working title:

> **Transient-aware sensor-native pretraining for methane plume
> classification across heterogeneous satellite sensors**

That title remains conditional on a future promoted pretraining result. An
evidence-aligned framing for the completed screen is:

> **When reconstruction does not transfer: leakage-safe temporal ranking for
> methane classification across heterogeneous satellite sensors**

Candidate contributions, conditional on results:

1. a leakage-controlled four-sensor methane classification benchmark with
   event-level temporal evaluation and a cross-sensor zero-overlap audit;
2. a sensor-native temporal pretraining framework with shared latent modeling
   and sensor-specific adapters;
3. a controlled comparison between generic EO pretraining, methane-temporal
   pretraining, and scratch across S2, L8/9, EMIT, and S5P;
4. an analysis of current imagery, raw history, explicit residuals, and
   missing-sensor robustness.

The most attractive result would be a consistent macro/worst-sensor gain and
better low-label performance. A smaller but still publishable result is a
careful objective-alignment study: two integrity-clean MAE transfer failures
contrasted with broad supervised ranking gains, alongside an explicit S5P
non-degeneracy failure that prevents promotion. Such a story must keep
pretraining, ranking, and calibrated decision quality as separate claims.

## 12. Primary sources used in this v1

- SatMAE: <https://arxiv.org/abs/2207.08051>
- AnySat: <https://arxiv.org/abs/2412.14123>
- Panopticon: <https://arxiv.org/abs/2503.10845>
- UniverSat: <https://arxiv.org/abs/2606.23503>
- MethaneMapper, CVPR 2023:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Kumar_MethaneMapper_Spectral_Absorption_Aware_Hyperspectral_Transformer_for_Methane_Detection_CVPR_2023_paper.html>
- Autonomous methane detection with Sentinel-2:
  <https://arxiv.org/abs/2308.11003>
- PRISMA cross-sensor transfer: <https://arxiv.org/abs/2211.15429>
- MethaneSAT cross-sensor transfer: <https://arxiv.org/abs/2605.24273>
- MAPL-EMIT: <https://arxiv.org/abs/2604.10094>
- NormWear DOI: <https://doi.org/10.1145/3803808>
- NormWear preprint: <https://arxiv.org/abs/2412.09758>

Project evidence is drawn from:

- `research/pretraining_20260727/EXPERIMENT_LOG.md`;
- `/home/yuyao/NormWear/README.md`;
- `/home/yuyao/NormWear/methane_pipeline/README.md`;
- `/home/yuyao/NormWear/modules/normwear.py`;
- `/home/yuyao/methane_train/README.md`;
- `/home/yuyao/methane_train/preprocess_dataset_multisensor/merge_master_csv.py`;
- `/home/yuyao/methane_train/preprocess_dataset_multisensor/crop.py`;
- the controlled UniverSat probe results reported for this research run.
