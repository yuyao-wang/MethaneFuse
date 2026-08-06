# TEMPO legacy360 global results

## Outcome

The selected video-inspired architecture is **R4 motion excitation**, not the
hard-coded background-trend operator and not the NormWear-style sensor
liaison. A fourth predeclared seed confirms a mean improvement over P0, while
also showing that R4's advantage over the simpler raw-delta R1 is small.

R4 freezes the promoted scale-aware classifier and adds a zero-initialized
global residual. For every sensor/history pair:

```text
current appearance = A(t0)
signed motion      = M(t0 - ht)
motion magnitude   = U(|t0 - ht|)
excitation         = sigmoid(G(M, U))
evidence           = A × excitation + M
```

Available history and sensor evidence is then mask-averaged and added to the
frozen P0 logit. The complete matched head declares 134,210 parameters; R4
activates 123,314 of them and epoch zero is exactly P0.

## Final development numbers

Data boundary: complete legacy `train_core` (113,843 rows) and development
(12,621 rows), four sensors, three temporal roles, frozen 768-D universal
Panopticon features. Test/sealed artifacts read: **none**.

| Model | Binary F1 | Macro F1 | AP | AUC | FP events | FP rows | Positive-event detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.90558 | 0.89507 | 0.96003 | 0.96038 | 12/27 | 29 | 257/264 |
| R1 three-seed logit ensemble | 0.91017 | 0.90203 | 0.96146 | 0.96234 | 11/27 | 28 | 258/264 |
| R4 four-seed logit ensemble | **0.91024** | 0.90077 | **0.96169** | **0.96253** | 12/27 | 31 | **260/264** |
| R4 P0-constrained threshold | 0.91007 | **0.90176** | **0.96169** | **0.96253** | **11/27** | **28** | 259/264 |

The fourth seed was run with the already frozen configuration. It peaks at
epoch 2 with F1 `0.90712`, then regresses at epoch 3 and stops. Including it
reduces the earlier exploratory three-seed ensemble F1 from `0.91104` to
`0.91024`; the four-seed value is the final stability estimate and is only
`0.00007` above R1's three-seed ensemble. No hyperparameter was changed after
seeing seed 101.

### Four-seed R4 stability

| Metric | Seed 17 | Seed 42 | Seed 73 | Seed 101 | Mean ± sample std | Mean Δ vs P0 |
|---|---:|---:|---:|---:|---:|---:|
| Binary F1 | 0.91129 | 0.90945 | 0.91080 | 0.90712 | 0.90966 ± 0.00186 | +0.00408 |
| Macro F1 | 0.90255 | 0.90179 | 0.90199 | 0.89714 | 0.90087 ± 0.00251 | +0.00579 |
| AP | 0.96180 | 0.96075 | 0.96038 | 0.96060 | 0.96088 ± 0.00063 | +0.00085 |
| AUC | 0.96248 | 0.96208 | 0.96133 | 0.96114 | 0.96176 ± 0.00063 | +0.00138 |

### Sensor strata

Metrics below use each sensor stratum's development F1 threshold and are
diagnostic, not independently locked test operating points.

| Sensor | P0 F1 | R4 mean F1 | ΔF1 | P0 AP | R4 mean AP | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.93319 | 0.93513 | +0.00194 | 0.97159 | 0.97241 | +0.00083 |
| L89 | 0.90487 | 0.91282 | +0.00795 | 0.95911 | 0.95931 | +0.00020 |
| EMIT | 0.89372 | 0.89259 | -0.00113 | 0.95217 | 0.95264 | +0.00046 |
| S5P | 0.90105 | 0.90769 | +0.00664 | 0.95469 | 0.95974 | +0.00505 |

R4 improves ranking AP for every sensor; EMIT thresholded F1 is effectively
flat and is the one sensor without a mean F1 gain.

### Availability and sensor-history mechanism audit

The four-seed ensemble and P0 were re-audited at their already frozen global
thresholds. No subgroup threshold or ensemble weight was refit. Overall, R4
corrects 84 P0 false negatives and 128 P0 false positives, while losing 77
true positives and adding 63 false positives: net `+7` TP and `-65` FP rows.
The 27 canonical all-negative events remain 12 event alarms, although hard-FP
rows increase from 29 to 31.

The gain is concentrated in naturally sparse observation patterns:

| Availability | Rows | ΔF1 | ΔAP | ΔFP rows |
|---|---:|---:|---:|---:|
| One available sensor | 10,219 | +0.00661 | +0.00164 | -56 |
| Two available sensors | 2,130 | -0.00201 | +0.00253 | -11 |
| Three available sensors | 240 | -0.00640 | -0.00362 | +2 |
| S2 absent | 6,564 | +0.00705 | +0.00212 | -57 |
| S2 present | 6,057 | +0.00201 | +0.00150 | -8 |
| L89 only | 3,032 | +0.01349 | +0.00112 | -5 |
| S5P only | 273 | +0.01859 | +0.03993 | +2 |

The small three/four-sensor groups cover only seven/one events, so they are
diagnostic rather than evidence of a multi-sensor regression. The defensible
mechanism is that sensor-native temporal evidence is most useful when a row
has only one sensor, especially without S2.

One sensor's history was then shuffled at a time while preserving t0, masks,
the other three sensors, every history marginal, all four checkpoints, equal
ensemble weights, and the original R4 threshold:

| Shuffled history | Eligible rows | ΔF1 | Δmacro F1 | ΔAP | ΔAUC | Mean \|Δp\| |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 6,057 | -0.001381 | -0.001228 | -0.001017 | -0.000876 | 0.01596 |
| L89 | 4,068 | -0.000801 | -0.000493 | -0.000987 | -0.001082 | 0.02577 |
| EMIT | 3,644 | -0.001157 | -0.001473 | -0.000366 | -0.000163 | 0.03400 |
| S5P | 769 | -0.000145 | -0.000160 | -0.000003 | -0.000067 | 0.01248 |

S2 history supplies the largest F1/AP contribution, L89 the largest ranking
AUC contribution, and EMIT changes scores most but mostly without improving
ranking. S5P history is nearly inert because only 769 rows have any and its
spatial scale is mismatched.

An exact counterfactual sensor-drop stress is not identifiable from this
cache: P0's shortcut uses a fusion head over the elementwise maximum of
legacy-concatenated sensor features, but only the already fused base logit was
cached. Masking a sensor only in the P0/R4 adapters would leave that sensor in
the base shortcut. The audit records this limitation instead of reporting a
confounded adapter-only drop as end-to-end robustness.

### Frozen availability deployment rule

The audit motivated one zero-training binary rule declared before evaluation:
use the frozen four-seed R4 ensemble only for exactly-one-sensor rows and
otherwise fall back to P0. Each branch retains its own locked global
threshold; no continuous weight or new threshold is fitted.

| Frozen model/rule | F1 | Macro F1 | AP | AUC | FP rows | all-neg events/rows |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.90542 | 0.89492 | 0.96003 | 0.96038 | 915 | 12/29 |
| Global R4 | 0.91024 | 0.90077 | **0.96169** | **0.96253** | **850** | 12/31 |
| Single-sensor R4, else P0 | **0.91075** | **0.90113** | 0.96119 | 0.96207 | 859 | 12/31 |

At 10,000 paired event resamples, the rule minus P0 F1 is `+0.005275`,
95% CI `[+0.002109,+0.008604]`; versus global R4 it is only `+0.000512`,
CI `[-0.000696,+0.001895]`. Its AP is `0.000505` below global R4, also with a
CI crossing zero. Thus the binary rule is the higher-F1 frozen deployment
policy, while global R4 remains the better continuous ranking score.

### Direct t0-query cross-attention control

A direct answer to “is this only cross-attention?” uses t0 as the query and
the two same-sensor histories as keys/values, followed by masked-mean sensor
fusion. It has exactly the same graph-active parameter count as R4
(`123,314`), the same seed-42 steps/optimizer, and an exact P0 epoch-zero
fallback.

| Seed-42 arm | Best epoch | F1 | Macro F1 | AP | AUC | FP events/rows |
|---|---:|---:|---:|---:|---:|---:|
| R4 motion excitation | 2 | **0.90945** | **0.90179** | **0.96075** | **0.96208** | 11/26 |
| t0-query history-K/V attention | 1 | 0.90908 | 0.89939 | 0.96016 | 0.96153 | 12/29 |

Attention history shuffle still lowers F1 by `0.002457` and AP by `0.000692`,
so it genuinely uses history. It nevertheless loses to R4 on F1, macro F1,
AP, and AUC and was stopped after seed 42. The supported difference is the
explicit decomposition into current appearance plus signed/magnitude motion,
not merely replacing Panopticon fusion with another attention block.

### Paired event bootstrap

Ten thousand canonical-event resamples, paired against P0. Every R4 seed keeps
its own already-selected checkpoint and threshold; no bootstrap replicate
refits either.

| R4 − P0 | Mean delta | 95% CI | Win probability |
|---|---:|---:|---:|
| Binary F1 | +0.004024 | [+0.000581,+0.007410] | 0.9894 |
| Macro F1 | +0.005744 | [+0.002189,+0.009302] | 0.9993 |
| AP | +0.000863 | [-0.000610,+0.002379] | 0.8767 |

This bootstrap is post-selection diagnostic evidence, not a confirmatory
confidence interval.

### Strict active-capacity control

R4-add invokes exactly the same learned transforms as R4 but replaces the
multiplicative gate with additive fusion:

```text
R4:     evidence = appearance × sigmoid(excitation) + signed_motion
R4-add: evidence = appearance + signed_motion + tanh(excitation)
```

Both seed-42 runs have the same initial-state digest, optimizer, batch order,
active parameter set (`123,314`), active-name digest
`c547428eb3dbb74ac83faf8a3f104fffcfbe383aa04afde73753b8650ffe8dbc`,
and learned-linear compute (`924,144` MACs per row). The full declared
signature remains `134,210` parameters in both. The fixed pointwise fusion is
the only treatment difference.

| Model | Best epoch | Binary F1 | Macro F1 | AP | AUC | FP events | FP rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4 seed 42 | 2 | **0.90945** | **0.90179** | **0.96075** | **0.96208** | **11/27** | **26** |
| R4-add seed 42 | 1 | 0.90766 | 0.89749 | 0.96009 | 0.96124 | 12/27 | 30 |

For 10,000 fixed-threshold paired canonical-event resamples, R4-add minus R4
has binary-F1 delta `-0.001778`, 95% CI
`[-0.005098,+0.001587]`, and macro-F1 delta `-0.004313`, 95% CI
`[-0.007905,-0.000869]`. Thus R4 has the better point estimate and materially
better class balance, but the primary positive-class F1 interval crosses
zero. The defensible claim is **dual-stream current appearance plus
signed/magnitude motion**. Multiplicative excitation is the best tested fusion
variant, not yet an independently established source of the primary F1 gain.

## What the ablations establish

| Arm | Seed-42 best F1 | AP | Interpretation |
|---|---:|---:|---|
| P0 | 0.90558 | 0.96003 | Frozen promoted base |
| P1 plain delta, 2 epochs | 0.90934 | 0.96030 | Explicit temporal high-pass helps |
| P2 hard linear normality | 0.90779 | 0.95924 | Hard trend subtraction hurts ranking |
| Q2 learned normality | 0.90708 | 0.95966 | Normality mixer still unnecessary |
| R1 plain delta, early-stopped | **0.91122** | **0.96133** | Strong scoring control |
| R2 learned history/sensor gates | 0.91066 | 0.96122 | Extra scalar gates add little |
| R3 event-null loss | 0.90792 | 0.96057 | Direct null penalty trades away F1 |
| R4 motion excitation | 0.90945 | 0.96075 | Best active-capacity-matched fusion variant |
| R4-add capacity control | 0.90766 | 0.96009 | Appearance+motion helps; multiplication is not isolated by primary F1 |
| R5 CLS liaison set attention | 0.90959 | 0.96053 | NormWear liaison does not beat simple fusion |
| t0-query history-K/V attention | 0.90908 | 0.96016 | Uses history but does not explain R4 |

R4 history shuffle lowers the four-seed ensemble F1 by `0.003638`, macro F1
by `0.003607`, AP by `0.002098`, and AUC by `0.002204`. The model is therefore using matched
history rather than merely adding head capacity.

## Final non-promoted controls

The exactly-one-current-sensor specialist trains the same R4 head only on
the `91,751/113,843` training rows with one valid t0 sensor. At inference it
replaces only that branch and leaves the multi-sensor branch at frozen P0.
Seed 42 peaks at epoch 3:

| Model | Binary F1 | Macro F1 | AP | AUC | FP rows | all-negative FP events/rows |
|---|---:|---:|---:|---:|---:|---:|
| frozen v7 availability gate | **0.910745** | 0.901134 | **0.961190** | **0.962070** | 859 | **12/31** |
| single-sensor specialist gate | 0.910571 | **0.901752** | 0.960990 | 0.962049 | **809** | 14/33 |

The specialist removes 50 row-level false positives but loses 44 true
positives and spreads false alarms across two additional all-negative events.
Its F1 is `-0.000174` below v7 and its AP/AUC also regress. In 10,000
canonical-event resamples, specialist minus v7 binary-F1 has mean
`-0.000140`, 95% CI `[-0.003644,+0.003079]`. History shuffle at its frozen
specialist threshold changes F1 by `-0.005330` and AP by `-0.003412`.
It fails the predeclared requirement of at least `+0.001` F1 with no AP/AUC
regression, so no additional specialist seeds are run.

The final zero-training ensemble control combines final R1 seeds
`17/42/73` and final R4 seeds `17/42/73/101` by a fixed `0.5/0.5`
logit average. No weight or subgroup threshold is searched:

| Model | Binary F1 | Macro F1 | AP | AUC | FP rows | all-negative FP events/rows |
|---|---:|---:|---:|---:|---:|---:|
| frozen v7 availability gate | **0.910745** | 0.901134 | 0.961190 | 0.962070 | 859 | 12/31 |
| R1/R4 fixed blend, global | 0.910663 | **0.902775** | **0.961754** | **0.962604** | **749** | **11/27** |
| one-sensor blend else P0 | 0.910472 | 0.902153 | 0.961271 | 0.962213 | 777 | 11/28 |

The global blend is a useful ranking/class-balance diagnostic, but its primary
F1 is `-0.000082` below v7; the one-sensor blend gate is `-0.000273` below
v7. Their paired event-bootstrap F1 intervals versus v7 both cross zero:
`[-0.002601,+0.002359]` and `[-0.002405,+0.001701]`. The blend therefore
also fails the `+0.001` promotion rule. The immutable v9 evaluator remains
unchanged: it retains global four-seed R4 and the frozen R4/P0 availability
gate.

## Technical conclusion

The supported mechanism is narrower and cleaner than the original proposal:

1. global Panopticon features already contain useful appearance;
2. plain temporal differences add consistent signal;
3. imposing a linear “normal background trajectory” is harmful on these
   sparse 90/360-day observations;
4. current appearance and signed/magnitude motion should be modeled as
   explicit complementary streams; multiplicative excitation gives the best
   tested class balance, but is not isolated by one-seed primary F1;
5. a generic sensor-attention liaison is not needed at this global stage.

This gives a concrete video-derived method story:

> Treat sparse methane observations as appearance plus motion excitation, not
> as a regular video and not as a learned sensor-fusion problem.

## Artifact map

- Final R4 aggregate:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/final_r4_4seed_v5/final_aggregate.json`
- Final R4 report:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/final_r4_4seed_v5/FINAL_RESULTS.md`
- Final R1 aggregate:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/final_r1_3seed_v3/final_aggregate.json`
- Fixed ensemble comparison:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/model_comparison_v4/fixed_equal_logit_comparison.json`
- Screen/follow-up log:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/video_mechanism_screen_v4/e4_followup/run.log`
- Seed-101 frozen run:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/final_r4_4seed_v5/r4_seed101/result.json`
- Strict capacity-control result:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/capacity_control_v5/comparison/capacity_control_comparison.json`
- Strict capacity-control report:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/capacity_control_v5/comparison/CAPACITY_CONTROL_COMPARISON.md`
- Availability/mechanism audit:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/mechanism_availability_audit_v6/r4_mechanism_availability_audit.json`
- Availability/mechanism audit report:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/mechanism_availability_audit_v6/R4_MECHANISM_AVAILABILITY_AUDIT.md`
- Frozen availability-gate control:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/availability_gate_control_v7/availability_gate_control.json`
- t0-query attention control:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/attention_control_v8/r6attn_seed42/result.json`
- Fair single-sensor specialist result:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/single_sensor_specialist_v10_fair/result.json`
- Final R1/R4 fixed equal-logit audit:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/final_r1_r4_blend_gate_v11/result.json`
- Immutable evaluator lock:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/LOCK_MANIFEST.json`
- Development dry-run receipt:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/dev_dry_run/RECEIPT.json`

The locked evaluator manifest SHA-256 is
`62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92`.
Its CPU development dry-run exactly reproduced the recorded P0, four-seed
equal-logit R4, and frozen availability-gate metrics. The sealed/test entry
point has not been invoked and remains disabled pending explicit authorization
of that exact lock digest.

## Claim boundary

This is an old-legacy, development-only engineering experiment. The warm
start and legacy split have known historical-selection/identity limitations.
No test/sealed artifact was read. The results support a method hypothesis and
a strong downstream control, not a clean SOTA claim.
