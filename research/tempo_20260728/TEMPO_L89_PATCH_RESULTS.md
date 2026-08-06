# TEMPO L89 projected-patch results

## Current outcome

The 128-D, readout-zero TEMPO head establishes a useful patch-local temporal
mechanism, but it does **not** pass the preregistered three-seed promotion gate:

```text
t0 patch
  -> soft correspondence to each history in a 3x3 neighbourhood
  -> signed/current/history-normalized onset features
  -> time-gap and observation-quality gating
  -> top-10% patch MIL
  -> bounded residual on the frozen P0 logit
```

The backbone and P0 logit are frozen. In the stable initialization, the patch
readout weight and bias are zero, while the residual gate is fixed to one.
Epoch zero is therefore bit-exact to P0, but the first optimization step learns
the supervised readout directly instead of learning the sign of a scalar gate
through a random readout.

All numbers below are development-only on 9,614 rows and 124 canonical events.
“Event-balanced row” means that metrics use row labels and row scores while
assigning every canonical event equal total weight; rows are not first
aggregated into a single event prediction.
The train cache contains 10,033 rows. Train/development canonical-event overlap
is zero. No test, sealed, or holdout artifact was read.

## Mechanism and capacity ablations

| Configuration | Event-balanced row AP | Event-balanced row macro F1 | Interpretation |
|---|---:|---:|---|
| Frozen P0 | 0.749149 | 0.761262 | Exact epoch-zero base |
| Frozen P5 | 0.765148 | 0.772047 | Strong clean global reference |
| 64-D, no history normality | 0.752558 | 0.766168 | Local onset alone |
| 64-D, history normality | 0.755654 | 0.767678 | Normality adds 0.00310 AP |
| 64-D, same-pixel matching | 0.749149 | 0.761262 | No learned gain |
| 64-D, 3x3 matching | 0.755654 | 0.767678 | Best balanced local search |
| 64-D, 5x5 matching | 0.755972 | 0.765709 | Tiny AP gain, worse balance |
| 128-D scalar-zero, seed 20260728 | 0.770474 | 0.782304 | Best scalar single seed |
| 128-D readout-zero, four-seed ensemble | 0.770319 | 0.780399 | Stable patch expert |

The 64-to-128 change raises same-seed scalar-zero AP by 0.014820 and macro F1
by 0.014626. The small head grows only from 19,187 to 23,411 parameters
(1.220x), while projected-cache storage grows about 1.978x. Thus projection
width changes both retained frozen-token information and a modest amount of
head capacity; the result must not be attributed to information width alone.

The independent Gaussian-QR 256-D random subspace reaches AP 0.780817,
macro F1 0.781912 and AUC 0.858378 at epoch 3 for seed 20260728. This is
+0.008409 AP and +0.002012 macro F1 over the matched 128-D seed and passes the
single-seed numerical screen. Its unified macro-selected all-negative
false-positive mass is 3.809118 (19/27 negative events, 76 rows), still above
P0's 2.742803; the older positive-F1-selected threshold yields 5.396618. The
256-D head has 31,859 total and 31,858 active parameters. Because the 256-D
matrix does not contain the 128-D projection as a prefix, this is only
random-subspace sensitivity evidence and does not trigger more seeds.

Cache cost is explicit. The 128-D train/validation cache directories occupy
7,814,853,797 bytes combined; the independent 256-D pair occupies
15,540,370,022 bytes (1.989x by directory bytes). Independent-256 extraction
took 1,433.60 seconds for train and 1,337.49 seconds for validation; strict
nested-256 took 1,434.20 and 1,346.59 seconds. Train and validation were
extracted concurrently, so each 256-D pair completed in about 24 minutes of
wall time. Peak CUDA allocation reported by either 256-D extractor was
2,718,745,088 bytes. Raw 768-D patch tokens were never persisted.

For the strict nested control, the original 128 columns are bit-exact
(maximum absolute prefix error 0), the full basis has maximum
`|Q^T Q - I|` of 2.98e-7, and the audited projection hashes are:

- base 128-D: `6d7690a58e78248a491e7102d7f9ecc33ed47755d4abbc319f77357273339c53`;
- orthogonal extension: `d4db53cf2b646f79fd966cb4f1531c089c91083ea96e9689e77cb50a0fbfa03`;
- full nested 256-D: `65fcc5970845f017cf0a296373a47c0819935edeb1a65e0c7b22fe612307f21d`.

The strict nested result does not reproduce the independent-subspace gain.
At seed 20260728 it reaches AP 0.766638, macro F1 0.774334 and AUC 0.844087
at epoch 3. Its unified macro-selected all-negative false-positive mass is
3.846618 (18/27 negative events, 81 rows). Relative to the bit-compatible
128-D seed, this is -0.005770 AP and -0.005566 macro F1, far below the
0.774408/0.779900 gate. Training stops at this single seed. Therefore the
0.780817 independent-subspace score is evidence of sensitivity to the chosen
random projection (and associated selection risk), not evidence that adding
128 projected dimensions improves the model.

The spatial result is equally concrete: same-pixel matching falls back to P0,
whereas a 3x3 search learns a useful residual. A 5x5 window does not improve
the balanced result. The supported explanation is compensation for small
sub-patch georegistration changes, not wide-window trajectory tracking.

## Initialization stability

The original scalar-zero form showed a sign pathology. The random local
readout is present at initialization, but only the zero scalar residual gate
receives a first-step gradient. Across four seeds its learned gate became
positive or negative, producing high variance:

| Scalar-zero seed | Event-balanced row AP | Event-balanced row macro F1 |
|---|---:|---:|
| 17 | 0.765878 | 0.772583 |
| 42 | 0.756933 | 0.769381 |
| 73 | 0.762931 | 0.777439 |
| 20260728 | 0.770474 | 0.782304 |
| Equal-logit ensemble | 0.767654 | 0.777001 |

Zeroing the readout and fixing the gate to one removes that random-sign
bottleneck:

| Readout-zero seed | Best epoch | Event-balanced row AP | Event-balanced row macro F1 |
|---|---:|---:|---:|
| 17 | 2 | 0.765950 | 0.782335 |
| 42 | 3 | 0.769429 | 0.779000 |
| 73 | 2 | 0.764075 | 0.774576 |
| 20260728 | 3 | 0.772408 | 0.779900 |
| Equal-logit ensemble | — | 0.770319 | 0.780399 |

The stable ensemble gains 0.00260 macro F1 over the scalar ensemble without an
AP regression. A direct-P5 readout-zero residual still selects epoch zero:
P5 AP is 0.765148, while epochs one and two are 0.764964 and 0.764961.
Therefore patch onset is useful as an independent P0 residual/expert, not as
another residual stacked directly onto the already strong P5 head.

At 128 dimensions both variants declare 23,411 parameters. Scalar-zero trains
all 23,411; readout-zero fixes one scalar and trains 23,410. Across the four
readout-zero seeds, a complete train-plus-development epoch takes
31.08 +/- 3.63 seconds (sample standard deviation), and the three-epoch run
takes 93.24 seconds on average. The corresponding scalar-zero measurements
are 32.61 +/- 5.04 seconds per epoch and 97.83 seconds per three-epoch run.
The temporal head is therefore a roughly 23K-parameter, sub-two-minute
fine-tuning stage after one frozen cache extraction.

The four-seed table above uses seeds 17/42/73/20260728 and is an additional
stability diagnostic, not the original Gate-B protocol. The preregistered
Gate-B seeds are:

| Readout-zero Gate-B seed | Event-balanced row AP | Event-balanced row macro F1 |
|---|---:|---:|
| 20260728 | 0.772408 | 0.779900 |
| 20260729 | 0.772789 | 0.778731 |
| 20260730 | 0.758851 | 0.777253 |
| Across-seed mean | 0.768016 | 0.778628 |

The mean macro-F1 gain over frozen P0 is +0.017366, below the declared +0.020
Gate-B requirement. Therefore the 128-D patch candidate formally fails
promotion; a favorable ensemble point estimate cannot replace the across-seed
mean gate. For completeness, the fixed equal-logit Gate-B ensemble reaches AP
0.770202 and macro F1 0.778641, but its all-negative false-positive mass is
3.740368 versus frozen P0's 2.742803 at each model's development-selected
macro-F1 threshold. It therefore also fails the false-positive safeguard.

The only protocol-declared null-loss control also fails promotion. At
readout-zero seed 20260728, A2 (`null_weight=0.1`) reaches AP 0.766228 and
macro F1 0.777292, versus A1's 0.772408 and 0.779900. Under the legacy
positive-F1-selected threshold, both have all-negative-event FP mass
4.861201. Under the unified macro-F1 threshold, A2 reduces FP mass from
4.136201 to 3.868939, alarmed negative events from 20 to 17, and FP rows from
85 to 78, but loses 0.006180 AP and 0.002608 macro F1. It is rejected without
expanding seeds or searching another null weight.

## Fixed global/local fusion

D1 is the three-seed global temporal-delta ensemble. No ensemble weight was
searched.

| Fixed model | Event-balanced row AP | Macro F1 | Positive F1 | AUC |
|---|---:|---:|---:|---:|
| P5 | 0.765148 | 0.772047 | 0.713650 | 0.848388 |
| D1, three seeds | 0.771320 | 0.779493 | 0.729455 | 0.853668 |
| P5 + D1, equal logit | 0.772943 | 0.777219 | 0.724457 | 0.853505 |
| Patch, four readout-zero seeds | 0.770319 | 0.780399 | 0.734171 | 0.850673 |
| P5 + D1 + patch, equal thirds | **0.773405** | 0.778265 | 0.729028 | 0.853292 |
| Global/local equal scales: .25 P5 + .25 D1 + .50 patch (post-hoc exploratory) | 0.772921 | **0.780566** | **0.731970** | 0.852818 |

At 10,000 paired bootstrap replicates clustered by canonical event:

- Equal-thirds minus P5: AP +0.008257, 95% interval
  [+0.001679, +0.015613]; positive F1 +0.015378,
  [+0.003720, +0.028671].
- Equal-thirds minus P5+D1: AP +0.000462,
  [-0.003339, +0.004923]; macro F1 +0.001046,
  [-0.004412, +0.006389].
- Two-scale minus P5+D1: macro F1 +0.003346,
  [-0.002627, +0.009405]; positive F1 +0.007513,
  [+0.000176, +0.015441].

The defensible conclusion is complementarity, not a confirmed improvement over
P5+D1: the point estimate improves, but its AP and macro-F1 intervals cross
zero. The equal-thirds score is the ranking-oriented fixed rule; the
two-scale rule is a post-hoc exploratory class-balance diagnostic and is not a
preregistered confirmatory result.

## History-use intervention

The clean history intervention freezes the trained models, original decision
threshold, current t0 tokens, P0 base logit, target time gaps, target quality
and target availability mask. It replaces only historical patch tokens with
deterministic donors from a different canonical event **within the identical
complete six-role `unique_mask` pattern**. All 9,614 rows are eligible across
four mask strata; there are zero exclusions, zero mask mismatches and zero
target-valid-to-donor-invalid slots.

For the four-seed readout-zero ensemble, the fixed-threshold intervention
changes event-balanced row AP from 0.770319 to 0.751296
(delta -0.019023), macro F1 from 0.780399 to 0.763523
(delta -0.016876), positive F1 by -0.020117, and AUC by -0.010643.
Mean absolute probability change is 0.037930. This supports the narrow
mechanism claim that the patch branch actually uses historical patch content.
It does **not** override the failed 128-D Gate-B promotion decision, prove that
the temporal mechanism is sufficient by itself, or establish a multisensor
result.

An earlier unmatched cross-event shuffle is retained only as a harsh
diagnostic. It replaced 3,070 of 44,521 target-valid historical slots with
donor-invalid zero tokens (6.896%) and is therefore not used as causal
mechanism evidence.

## Technical story supported so far

Sparse methane sequences should be decomposed into three complementary
signals:

1. **current appearance**, represented by the frozen P5 classifier;
2. **global temporal motion**, represented by D1/R4-style feature changes;
3. **patch-local onset**, represented by locally aligned current-to-history
   deviations normalized by history-to-history variation.

This is a two-axis video adaptation, not ordinary cross-attention. The temporal
axis separates current appearance from signed/magnitude change and background
variation. The spatial axis searches only a small correspondence neighbourhood
and uses sparse MIL so a plume need not dominate the whole field of view.
Global and local experts make different errors, which is why fixed fusion
improves the point estimate even though directly stacking the patch residual
on P5 does not. The clean token intervention verifies history use, while the
formal seed gate shows that the current 128-D implementation is not yet stable
enough to promote.

The current evidence is L89 development evidence. It supports the architecture
and ablation logic, but not a multisensor SOTA claim. Cross-sensor evidence
comes from the separate global TEMPO/R4 experiment; this patch branch has not
yet been validated on all four sensors.

## Audited artifacts

- 64-D head summary:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_v1/audits/patch_head_summary_64d_v1.json`
- 64/128 capacity audit:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/projection_capacity_64_vs_128_v1.json`
- Readout-zero fixed ensemble/bootstrap:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_multiseed_fixed_v1/fixed_multiseed_audit.json`
- Preregistered Gate-B three-seed fixed ensemble/bootstrap:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_gateb3_fixed_v1/fixed_multiseed_audit.json`
- Gate-B fixed prediction artifact SHA-256:
  `39da1fad8a1efe1e1aade6369e0fb718550629c693eb72d2ff08677539163f96`
- Readout-zero audit SHA-256:
  `a285bdbe8b9c6077f658623c621c6190782f70684c9062a12cb4fd042cf24a54`
- Fixed prediction artifact SHA-256:
  `c115503e253d76e8f0e467e3d57ab98714bbda8330c98601e36ab2bac8e78033`
- Availability-pattern-matched history intervention:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_history_shuffle_maskmatched_v1/history_shuffle_audit.json`
- Clean-shuffle prediction artifact SHA-256:
  `4c763d96c9c89a9683899307bc0e693011d9c2d033a9ada706213f1fe3b34ee1`
- Independent-subspace 256-D result:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim256_v1/heads/p0_a1_readout_seed20260728/result.json`
- Independent-subspace result SHA-256:
  `de74ed3b8e927db0ed4230564fe7c293da224abec4f9ac3bf05384df4c658241`
- Strict nested 128-to-256-D result:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim256_nested128_v1/heads/p0_a1_readout_nested_seed20260728/result.json`
- Strict nested result/checkpoint/prediction SHA-256:
  `438b070a16f9dfcf9b83bb016666579e6df5afa198c8e8d4a9643430ba0821d2`,
  `5d45570f624e12dff01d7554f0b8e9666fc5dc022973027b1cb020248bdb81f3`,
  `e8c3160e9da72eb80d9febdf2aa898eb59a6a01bc2fde549a8261038f1f484aa`
- Unified 128/independent-256/strict-nested-256 control audit:
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim256_nested128_v1/audits/projection_128_256_control_v1.json`
- Unified projection-control audit SHA-256:
  `1ce7755335b081ea1385214af72361cd97168a2cb0ad530554d71b69fa02e792`

## Bounded-control verdicts

- The independent Gaussian-QR 256-D projection is not nested across
  requested widths. With seed 36064, the 128-D matrix is not equal to the
  first 128 columns of the 256-D matrix (maximum absolute difference
  0.211971; mean absolute difference 0.040579). A 256-D result is therefore a
  wider same-algorithm random-subspace sensitivity control, not a pure
  add-dimensions intervention; it is not expanded to more seeds.
- A separate strict nested projection reuses the 128-D columns bit-exactly and
  appends a deterministic orthogonal 128-D complement. It fails the
  preregistered single-seed gate and is not expanded to more seeds.
- The sole protocol-declared readout-zero A2 control uses null weight 0.1. No
  alternative null weight is searched.
