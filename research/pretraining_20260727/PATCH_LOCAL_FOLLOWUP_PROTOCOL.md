# Frozen final-patch RCTP follow-up

## Why this diagnostic exists

The matched real-L89 experiment found that the correct-response RCTP arm P5
reduced event-balanced macro-F1 relative to the unmodified P0 encoder by
0.0272, with a 95% event-bootstrap interval of `[-0.0478, -0.0057]`. The next
question is not whether a larger global head can recover the result. It is
whether RCTP changed spatial patch evidence in a useful way that global CLS
pooling discarded.

## Audit of the older patch-axial implementation

The older `legacy360_patch_cache.py` and `legacy360_patch_axial.py` are useful
engineering prototypes, and all 11 existing CPU contracts pass. They are not a
clean RCTP diagnostic for four reasons:

1. RCTP updated only Panopticon blocks 10 and 11. The older cache defaults to
   `--patch_layer early` after block 2, so that representation cannot observe
   the RCTP update.
2. Its cached fused residual receives
   `base_features + temporal_delta`. It can therefore improve by fitting the
   static projected CLS feature even if patch-temporal evidence contributes
   nothing.
3. Its temporal attention includes t0 among its own keys/values and ranks
   patches by absolute attended energy. It does not enforce a
   current-minus-background transient.
4. It supports the legacy three-role Query360 schema, whereas the clean L89
   experiment uses six irregular visits and event-disjoint train/development
   manifests.

No completed patch cache or patch-axial result existed at audit time.

## Frozen protocol

The follow-up is implemented in `l89_patch_local_rctp_followup.py`.

- Arms: P0, P4 response-scrambled, and P5 correct-response.
- Rows: the existing event-disjoint L89 six-visit train/development manifests,
  10,033/9,614 rows and 723/124 canonical events.
- Encoders: the declared P0/P4/P5 PTH files in evaluation mode, with zero
  trainable backbone parameters.
- Tokens: final normalized patch tokens only. This is necessary because RCTP
  changed the final two transformer blocks.
- Local evidence at patch `p`:

  `delta[p] = token(t0,p) - mean(token(valid unique history,p))`.

- Spatial selection: fixed top 10% by RMS delta energy in the original 768-D
  token space.
- Cached evidence: a common seeded 768-to-64 orthogonal projection of
  top-k signed delta, top-k absolute delta, and global signed delta. The local
  head sees 192 values and no static CLS feature.
- Base: each arm's already-selected, frozen role-only CLS logit.
- Local head: a 192-to-1 linear map without bias, initialized to exact zero,
  followed by a capped (`1.5`) tanh logit residual.
- Invariant: epoch 0 is bit-exact to the frozen base logit.
- Optimization: three epochs, AdamW, learning rate `3e-3`, no weight decay,
  identical batches and zero initial state for all arms. Training weights
  equalize canonical-event mass and positive/negative class mass.
- Selection: development event-balanced AP. Reports also include row AP/AUC
  and row/event-balanced F1 at fixed and development-selected thresholds.
- Scope: train/development only; paths containing test, sealed, or holdout are
  rejected.

The local caches do not persist raw final tokens. They deterministically derive
the registered evidence while the final tokens are resident on the GPU. This
reduces approximately 20+ GiB of raw-token storage per arm to roughly 10--20
MiB per arm without changing the registered local readout.

## Resource contract

- Raw source I/O: none. All 60,198 train and 57,684 development observations
  were checked under `/diniuvol/yuyao/methanefuse_research_20260727/cache`.
- GPU: one physical GPU, one arm/split at a time.
- Loader: batch 12, four workers, prefetch factor one.
- Expected CUDA allocation: 5--8 GiB; operational measured-allocation cap:
  12 GiB.
- Launch guard: at least 30 GiB free before loading the encoder.
- Expected time: 16--22 GPU minutes per arm for train plus development;
  48--66 minutes for P0/P4/P5. The three CPU heads should finish in under
  three minutes.
- Storage: approximately 10--20 MiB per arm plus logs/checkpoints.

The existing P5 role-only head was replayed on all 9,614 development rows on
CPU; its probabilities matched the saved prediction file with maximum absolute
error `2.98e-08`. All 16 relevant CPU contracts pass.

## Decision rule and story interpretation

- If P5 local residual improves over its own epoch-zero base and closes or
  reverses the P5-versus-P0 gap while P4 does not, the result supports a narrow
  mechanism: correct sensor response produced useful local transient features,
  but CLS transfer destroyed them. The next method should therefore put the
  response-conditioned counterfactual objective on patch tokens and retain a
  local temporal readout.
- If P4 and P5 improve equally, the gain belongs to generic patch-temporal
  residualization, not response conditioning.
- If P5 remains below P0 and does not improve its own base, the present RCTP
  formulation is a no-go. Patch aggregation cannot rescue an encoder that did
  not learn the correct local signal.

This diagnostic does not itself establish novelty or SOTA. Its value is as a
falsifiable bridge between the negative global-CLS result and a credible
patch-level counterfactual pretraining claim.

## Completed result and verdict (appended after frozen protocol)

The frozen protocol above was executed without changing its registered design.
This was train/development-only; `test_or_sealed_read=false`, and no holdout
artifact was read.

| Arm | Selected epoch | Event AP base→best | Event macro F1 base→best | Row AP base→best |
|---|---:|---:|---:|---:|
| P0 | 0 | 0.741929→0.741929 | 0.757392→0.757392 | 0.750716→0.750716 |
| P4 response-scrambled | 1 | 0.728275→0.735145 | 0.749437→0.756928 | 0.739443→0.737900 |
| P5 correct-response | 0 | 0.737478→0.737478 | 0.730239→0.730239 | 0.755838→0.755838 |

Paired 2,000-replicate canonical-event AP bootstrap:

- P4 local-best − P4 base:
  `+0.006870 [−0.000446, +0.015446]`;
- P4 local-best − P0:
  `−0.006783 [−0.055320, +0.049051]`;
- P5 best − P0:
  `−0.004450 [−0.039888, +0.036818]`.

P5 selected epoch 0, which is exactly its frozen base logit; the selected
local change is zero. P4's small recovery is not conclusive and remains below
P0. The frozen final-patch hypothesis is therefore **no-go**.

Artifacts:

- comparison JSON:
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
