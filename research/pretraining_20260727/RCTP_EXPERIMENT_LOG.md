# RCTP experiment log

## 2026-07-28 — L89 frozen-encoder mechanism screen

Purpose: test whether a small response-conditioned temporal adapter on the
existing Panopticon representation can distinguish a methane-shaped SWIR
counterfactual from equal-energy non-methane changes before spending compute
on continue-pretraining.

Protocol:

- train/dev only; no test input or test evaluation path;
- 4,096 train controls and 2,048 canonical-event-disjoint dev controls;
- six visits, with one deterministic injected visit per reference row;
- frozen Panopticon ViT-B/14 PTH;
- same-scene clean, methane, achromatic, and wavelength-shuffled renders;
- equal L2 energy for the two nuisance deltas;
- three epochs, dev average precision selects the checkpoint;
- explicit L89 screen response `[0, 0, 0, 0, 0, 0.15, 1.0]`;
- this response is a SWIR approximation, not yet a versioned SRF integral.

Artifacts:

- summary:
  `/diniuvol/yuyao/methanefuse_research_20260727/results/rctp_l89_screen_seed20260728/summary.json`
- checkpoint:
  `/diniuvol/yuyao/methanefuse_research_20260727/results/rctp_l89_screen_seed20260728/checkpoint_best_dev_ap.pt`
- checkpoint SHA-256:
  `0c4759202c3d426f47ddc7f3f9bfe2c25b3877c9321c819aa9fcc8e1153c6761`
- test evaluations: `0`

Best result (epoch 3):

| panel | AP | ROC-AUC | paired win |
|---|---:|---:|---:|
| all strengths | 0.7476 | 0.8382 | 0.7607 |
| weak, <=2% | 0.5472 | 0.7076 | 0.6029 |
| mid, 2–4% | 0.7618 | 0.8448 | 0.7814 |
| strong, >4% | 0.8845 | 0.9321 | 0.9024 |
| history injections | 0.7400 | 0.8331 | 0.7521 |
| t0 injections | 0.7868 | 0.8632 | 0.8041 |

Matched nuisance panels:

- methane versus equal-energy achromatic: AP `0.8321`, AUC `0.8194`;
- methane versus equal-energy wavelength shuffle: AP `0.8543`,
  AUC `0.8569`.

Interpretation:

- Chance AP and matched-group paired-win reference are both `1/3`.
- The screen is positive even in the <=2% and historical-visit panels, so the
  response-shaped counterfactual is not detected only through large generic
  perturbation energy.
- This supports proceeding to RCTP continue-pretraining; it is not downstream
  F1 evidence because the backbone was frozen and the task was synthetic.
- Single-sensor L89 cannot validate multi-sensor response conditioning because
  its response vector is constant. The formal experiment must render the same
  latent column field through multiple versioned SRF/PSF/product operators and
  include response-shuffled controls.

## 2026-07-28 — L89 real-classification continue-pretraining closure

Purpose: test whether response-conditioned continue-pretraining transfers to
real, event-disjoint L89 classification, and whether the correct-response arm
(P5) beats the response-scrambled matched control (P4).

Protocol:

- train/dev only; `test_or_sealed_read=false`;
- P0 is the unchanged public Panopticon PTH;
- P4 and P5 start from the same PTH, train the last two ViT blocks for two
  epochs, use the same deterministic rows, renderer draws, optimizer steps,
  probe initialization, and compute;
- the intended P4/P5 difference is only scrambled versus correct response
  ordering/metadata;
- 10,033 real train rows and 9,614 dev rows; train/dev event overlap is zero;
- identical `role_only` downstream heads, three epochs, AP-selected checkpoint;
- 2,000 canonical-event cluster bootstrap replicates, with each arm's
  dev-selected threshold held fixed;
- cached CLS is not used as temporal context or probe input; clean-anchor
  weight is zero.

Execution note:

- the initial single-GPU wrapper was modified by the coordinating task while a
  running Bash process was reading it, so it exited after P4 train-cache
  completion with a parse error;
- no model/data step failed and no artifact was overwritten;
- the continuation bound the completed P4 checkpoint, summary, train cache, and
  cache audit to exact SHA-256 values, then ran only P4 val cache, P5, heads,
  and audit;
- all cache builds completed with zero invalid t0 rows and zero read errors.

Synthetic capability at the selected pretraining checkpoint:

| arm | AP | ROC-AUC | paired win | selected F1 |
|---|---:|---:|---:|---:|
| P4 response-scrambled | 0.971989 | 0.981452 | 0.972656 | 0.912451 |
| P5 correct-response | 0.952442 | 0.969470 | 0.947266 | 0.873380 |

Real L89 event-balanced dev result:

| arm | best epoch | AP | ROC-AUC | positive F1 | macro F1 |
|---|---:|---:|---:|---:|---:|
| P0 base PTH | 3 | 0.741929 | 0.838673 | 0.709580 | 0.757392 |
| P4 response-scrambled | 1 | 0.728275 | 0.820783 | 0.697661 | 0.749437 |
| P5 correct-response | 2 | 0.737478 | 0.818863 | 0.686513 | 0.730239 |

Paired canonical-event bootstrap deltas:

- P5 minus P0 macro F1: `-0.027152`,
  95% CI `[-0.047795, -0.005652]`;
- P5 minus P0 positive F1: `-0.023068`,
  95% CI `[-0.048138, +0.003118]`;
- P5 minus P4 macro F1: `-0.019198`,
  95% CI `[-0.040161, +0.000470]`;
- P4 minus P0 macro F1: `-0.007955`,
  95% CI `[-0.031898, +0.017217]`.

The row-level AP values are P0/P4/P5 =
`0.750716/0.739443/0.755838`; the apparent P5 gain over P0 is only
`+0.005122` and does not translate to event-balanced F1.

Verdict:

- the default RCTP continuation does **not** improve real event-balanced L89
  classification and P5 does not beat P4 on synthetic capability;
- the pre-registered clean-anchor fallback condition (P5 synthetic capability
  beats P4 while real transfer is weak) is therefore false, so that registered
  fallback was not launched at this gate; the later final-patch and
  preserve-base Sidecar v2 runs below are explicitly exploratory diagnostics,
  not continuations promoted by this result;
- this is a useful falsifier: the current online objective can solve the
  injected task while moving the encoder away from transferable real-event
  evidence. It cannot support an RCTP performance claim or a Journal result.

Artifacts:

- comparison:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_real_cls_v1/downstream_role_only_seed20260728/comparison.json`
  (SHA-256
  `1eeea4013f10ac5cb42631d9393b3cae382b7293c243c5e18fcc555439f851d6`);
- readable comparison:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_real_cls_v1/downstream_role_only_seed20260728/COMPARISON.md`;
- P4/P5 checkpoints:
  `5fba988efc8de5ee1af185603a4ec353969d7a84898b8eda63ea621f28c4a16a`
  and
  `9ca14845ba9cf3b5963faf48729c58f8a74fa6f50e32ea5219fb519c9944e0ed`;
- audited continuation:
  `research/pretraining_20260727/resume_rctp_l89_after_p4_train_cache_gpu0.sh`.

## 2026-07-28 — Frozen final-patch local follow-up closure

Purpose: test whether the in-place P4/P5 continuation learned useful
same-position local transient evidence in final patch tokens that the global
CLS downstream head had discarded.

Frozen protocol:

- train/development only; `test_or_sealed_read=false`, no holdout read;
- P0/P4/P5 encoders and each arm's selected role-only base logit are frozen;
- local evidence is final-token
  `t0 − mean(valid unique history)` at the same patch position;
- local head is zero-initialized, bias-free, and receives no static CLS;
- epoch 0 is bit-exact to the frozen base logit;
- identical projection, local-head initialization, batches, optimizer and
  three-epoch cap across arms;
- checkpoint selection by event-balanced AP.

Authoritative results:

| Arm | Selected epoch | Event AP base→best | Event macro F1 base→best | Row AP base→best |
|---|---:|---:|---:|---:|
| P0 | 0 | 0.741929→0.741929 | 0.757392→0.757392 | 0.750716→0.750716 |
| P4 response-scrambled | 1 | 0.728275→0.735145 | 0.749437→0.756928 | 0.739443→0.737900 |
| P5 correct-response | 0 | 0.737478→0.737478 | 0.730239→0.730239 | 0.755838→0.755838 |

Paired 2,000-replicate canonical-event AP bootstrap:

- P4 local-best minus own base:
  `+0.006870 [−0.000446, +0.015446]`;
- P4 local-best minus P0:
  `−0.006783 [−0.055320, +0.049051]`;
- P5 best minus P0:
  `−0.004450 [−0.039888, +0.036818]`.

Verdict:

- P5 selected epoch 0, so its selected patch-local change is exactly zero;
- P4 has a small within-arm recovery, but its two-sided interval crosses zero
  and it remains below P0;
- frozen final-patch local readout does not rescue correct-response RCTP and is
  a no-go.

Artifacts:

- comparison:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/comparison.json`  
  SHA-256
  `d72fc6525d8ae06f114c704f4832b40bfa93b45e38f75d50e6a71aa6ab9f3131`
- paired bootstrap JSON:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/paired_event_bootstrap_seed20260728.json`  
  SHA-256
  `6afbf1d50637b6934e73076da9c7fbcf47d52343c0856d8087d462b623152f45`
- readable bootstrap:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/PAIRED_EVENT_BOOTSTRAP.md`  
  SHA-256
  `42b687436d1bd7144e8618cd053fab7db6065bcb75bdef0be972dd0863217a53`

## 2026-07-28 — Sidecar-RCTP v2 closure

Purpose: test a single representation-preserving change after in-place encoder
updates failed—keep the public Panopticon base byte-identical and expose a
small response-conditioned residual as an optional downstream side channel.
This is exploratory because the current recent-dev result motivated the
design.

Frozen protocol:

- Panopticon base fully frozen; transferable sidecar has 12,416 parameters;
- three bias-free matrices form a zero-initialized rank-8 residual;
- disposable synthetic probe sees residual only, never `z_base`;
- downstream representation is `[z_base, 0.1 × residual]`; P0 receives
  `[z_base, exact_zero]` with the same 1,536-D head;
- P4/P5 use matched rows, draws, initialization, parameter count, optimizer
  and 1,024 updates; only scrambled versus correct response differs;
- clean residual anchor weight `0.10`;
- downstream heads use inverse-event-size BCE, event-AP checkpoint selection,
  event-macro-F1 threshold selection and 2,000 paired event bootstraps;
- no test, sealed, or holdout artifact was read.

Synthetic selected-checkpoint AP:

| P4 response-scrambled | P5 correct-response |
|---:|---:|
| **0.593971** | 0.387171 |

Thus the sidecar synthetic task itself does not establish response
specificity.

Real L89 recent-dev results:

| Arm | Event AP | Event AUC | Positive F1 | Macro F1 |
|---|---:|---:|---:|---:|
| P0 | 0.749149 | 0.842789 | 0.691468 | 0.761262 |
| P4 scrambled | 0.754381 | 0.842398 | 0.701309 | 0.771155 |
| P5 correct | **0.765148** | **0.848388** | **0.713650** | **0.772047** |

Paired canonical-event deltas:

- P5 − P0 event AP:
  `+0.015998 [+0.002681, +0.030047]`;
- P5 − P4 event AP:
  `+0.010767 [+0.000714, +0.020273]`;
- P5 − P0 positive F1:
  `+0.022182 [+0.002006, +0.044495]`;
- P5 − P0 macro F1:
  `+0.010785 [−0.003346, +0.026284]`;
- P5 − P4 macro F1:
  `+0.000892 [−0.012323, +0.015677]`.

All-negative-event FP audit:

- definition: 27 canonical events whose maximum row label is zero; at each
  arm's already-locked threshold, FP mass is the sum of within-event hard-FP
  rates, giving each event total weight one;
- P0/P4/P5 FP mass:
  `2.742803/2.531088/3.871023`;
- P5 increases FP mass by `+1.128220` versus P0 and `+1.339935` versus P4.

Locked decision rule and verdict:

- required:
  `P5 macro-F1 − max(P0,P4) >= +0.020`, paired lower CI > 0, and no
  all-negative-event FP increase;
- observed P5 − P4 macro F1 is only `+0.000892` with lower CI `−0.012323`,
  and P5 FP mass increases;
- promotion is **no-go** and no test/sealed evaluation is permitted;
- the positive paired AP signal and positive-F1 delta versus P0 are a useful
  representation-preserving partial signal, not evidence that response
  specificity is established. They also do not support an anomaly-ranking
  novelty claim.

Artifacts:

- comparison:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/comparison.json`  
  SHA-256
  `7fe0fff947c195ea3b9096e2d9b499d55b3302ae2ece9402d5ee459827a34daf`
- readable comparison:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/COMPARISON.md`  
  SHA-256
  `1beb227164425056b99e0de2d270ba063e53bc24102e3e4a28fe7258ac923310`
- artifact manifest:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/ARTIFACT_SHA256SUMS.txt`  
  SHA-256
  `0c5426791b9015510bedba692d59756ab92adbb171965c7cfa1abd50b44b9987`
- AP/FP read-only audit:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/sidecar_all_negative_event_fp_audit.json`  
  SHA-256
  `19f58f091915415ea73084142c1be37c26055279ccef2ed5b31d86b25e2937fd`
- readable AP/FP audit:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/sidecar_all_negative_event_fp_audit.md`  
  SHA-256
  `36ed323b5b0370ac6e35aa450c970a68a23861dc904ab561bcbfee76b3dc2692`

## 2026-07-28 — Train-only event-null calibration audit closure

Purpose: test whether the Sidecar P5 ranking partial signal could be converted
into a cleaner decision rule using only train-derived acquisition/availability
null offsets.

Scope and protocol:

- CPU-only reconstruction from frozen train/dev caches and downstream
  checkpoints; no model training or feature extraction;
- train/recent-dev only; no test, sealed, or holdout artifact read;
- inherited heads were already dev-AP-selected, so this is explicitly
  post-hoc exploratory evidence;
- 723 train canonical events split into deterministic five-fold event groups
  (`seed=20260728`), with no event crossing folds;
- fold calibrators use only all-negative events from the other four folds;
- candidates: identity or shrunken q75/q90 null offsets by six-visit
  availability pattern, acquisition quarter, or their cross-product;
- candidate and threshold selected only from train OOF event-balanced metrics
  and all-negative-event FP mass; applied once to dev.

Locked train-only selection and dev result:

| Arm | Selected calibrator | Dev event AP | Dev macro F1 | Train-OOF threshold | All-negative FP mass |
|---|---|---:|---:|---:|---:|
| P0 | identity | 0.7491 | 0.7342 | 0.6951 | **1.8331** |
| P4 scrambled | identity | 0.7544 | **0.7571** | 0.8532 | 2.3188 |
| P5 correct | identity | **0.7651** | 0.7560 | 0.5475 | 2.7922 |

Paired 2,000-replicate canonical-event macro-F1 bootstrap, with train-OOF
threshold fixed:

- P5 − P0:
  `+0.0218 [+0.0056, +0.0403]`;
- P5 − P4:
  `−0.0011 [−0.0117, +0.0092]`.

Verdict:

- all three arms selected identity, so train-only selection did not justify
  any static null correction;
- P5 fails the `+0.020` margin over the strongest control, paired lower-bound
  and no-FP-increase clauses;
- the calibration audit is **no-go** and does not permit sealed/test
  evaluation;
- static availability/quarter offsets are insufficient. A future framework
  needs train-time event-null selectivity in the representation/objective;
- L89 is a single-sensor cohort and cannot validate sensor-conditioned
  calibration.

Audit:

- [EVENT_NULL_CALIBRATION_AUDIT.md](/home/yuyao/panopticon/research/pretraining_20260727/EVENT_NULL_CALIBRATION_AUDIT.md)
- SHA-256:
  `9a969abf16f45ea7c30c72bf27def84a306a15c7f3769df3c3b04ec52519f4c1`
- `test_or_sealed_read=false`; `holdout_read=false`.
