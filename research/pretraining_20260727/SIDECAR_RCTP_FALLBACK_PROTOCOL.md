# Sidecar-RCTP fallback protocol

Status: **completed** on train/recent-dev only; promotion **no-go**. The
pre-launch protocol is preserved below and the executed results are appended
at the end.

## Falsifiable question

The previous L89 continuation updated 14,178,816 parameters in the final two
Panopticon blocks.  It solved the synthetic objective but reduced
event-balanced dev macro-F1 from `0.757392` (P0) to `0.730239` (P5).  This
fallback tests one representation change only:

> Can a small response-selective residual improve real transfer when the
> original Panopticon representation is made impossible to forget?

This run is exploratory because the current recent-dev split already motivated
the design.  It cannot be presented as confirmatory evidence.

## Frozen design

- Panopticon encoder: byte-identical public PTH, fully frozen, eval mode.
- Sidecar: three bias-free matrices
  `768→8`, `response_metadata→8`, and `8→768`.
- Trainable parameters are `12,416` in the transferable sidecar and `900,707`
  in the disposable residual-only probe (`913,123` total); P4 and P5 have
  identical shapes and counts.
- The last matrix is initialized to exact zero.
- Synthetic pretext probe input: the unscaled sidecar residual only.  Frozen
  `z_base` is not concatenated or otherwise exposed to the disposable probe.
  Therefore a zero sidecar produces all-zero clean and delta features, and the
  synthetic task cannot be solved through a direct frozen-feature shortcut.
- Downstream representation:
  `[z_base, 0.1 × residual]`.
- P0 uses `[z_base, exact_zero]` with the same `1536`-dimensional downstream
  head shape.
- Clean anchor: `0.10 × mean(residual(clean)^2)` over valid unique visits.
- P4/P5: identical seed, initialization, rows, plume draws, optimizer,
  parameter count, and 1,024 updates.  The only intended difference is
  scrambled versus correct response ordering and metadata.
- Two pretraining epochs, 2,048 negative train references per epoch, fixed
  512-row development panel.
- Downstream: the existing inverse-event-size/class-balanced head, selection
  by event-balanced AP, threshold by event-balanced macro-F1, three epochs,
  and paired canonical-event bootstrap.
- Every path containing `test`, `sealed`, or `holdout` is rejected.  No outer
  test or newly constructed holdout is read.

The cache path is cheap because the frozen base CLS tensors already exist.
After pretraining, the learned sidecar is applied on CPU to those exact
train/dev tensors; the 768-dimensional base half is checked byte-for-byte.

## Locked decision rule

For this already-used development split, a promising signal would be:

`P5 macro-F1 − max(P0, P4) >= +0.020`,

with the paired event-cluster lower confidence limit above zero and no increase
in all-negative-event false-positive mass.  Anything weaker is a no-go and
must not trigger a sealed/test evaluation.

## Resource estimate and safety

Measured earlier full-backbone continuation took roughly 2.5 minutes per
2,048-row epoch while encoding nine images per row and backpropagating through
two ViT blocks.  This sidecar encodes four images per row with a frozen
backbone, so the conservative estimate is:

- `3–6` GPU-minutes per arm, `6–12` GPU-minutes total when run sequentially;
- hard timeout `15` minutes per arm;
- expected CUDA allocation `4–8 GiB`;
- enforced allocation cap `12 GiB`;
- start requires at least `30 GiB` free and runtime requires at least `25 GiB`
  free on the selected physical GPU.

The launcher is dry-run by default.  On `--run`, it snapshots the runner,
launcher, evaluator, tests, and this protocol into the output provenance
directory, re-executes the frozen launcher, and writes source/artifact SHA-256
manifests.

```bash
bash research/pretraining_20260727/run_rctp_l89_sidecar_fallback.sh
```

GPU execution requires explicit authorization:

```bash
SIDECAR_GPU=1 \
bash research/pretraining_20260727/run_rctp_l89_sidecar_fallback.sh --run
```

## Completed result and verdict (appended after frozen protocol)

The frozen Sidecar v2 protocol above was executed without changing its
registered design. Pretraining, cache construction, downstream fitting and
read-only audit were train/recent-dev-only:
`test_or_sealed_read=false`, `holdout_read=false`.

Synthetic selected-checkpoint AP:

| P4 response-scrambled | P5 correct-response |
|---:|---:|
| **0.593971** | 0.387171 |

Real L89 recent-dev metrics:

| Arm | Event AP | Event AUC | Positive F1 | Macro F1 |
|---|---:|---:|---:|---:|
| P0 | 0.749149 | 0.842789 | 0.691468 | 0.761262 |
| P4 scrambled | 0.754381 | 0.842398 | 0.701309 | 0.771155 |
| P5 correct | **0.765148** | **0.848388** | **0.713650** | **0.772047** |

Paired 2,000-replicate canonical-event deltas:

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

At the already-locked arm-specific thresholds, all-negative-event FP mass
(27 canonical events, equal total mass per event) was
P0/P4/P5 = `2.742803/2.531088/3.871023`.

The locked rule required
`P5 macro-F1 − max(P0,P4) >= +0.020`, a paired lower bound above zero, and no
increase in all-negative-event FP mass. P5 exceeds P4 macro F1 by only
`+0.000892`, its lower bound is `−0.012323`, and its FP mass increases.
Promotion is therefore **no-go**, and no test/sealed evaluation is permitted.

The paired AP signal is a useful exploratory partial positive for preserving
the base representation in a sidecar design. It is not response-specific
success: P4 wins the synthetic control and P5 does not separate from P4 on
macro F1. It also cannot support an anomaly-ranking novelty claim.

Artifacts:

- comparison JSON:
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
