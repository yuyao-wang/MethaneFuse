# TEMPO L89 global six-visit experiment log

Date: 2026-07-28 UTC

## Bottom line

The global experiment supports a **two-path inference architecture**, not a
new pretraining framework:

1. a frozen methane-response / appearance expert (the existing P5 head);
2. an independently trained, video-style transient expert (D1) that compares
   the current visit with five irregular historical visits;
3. fixed equal-logit late fusion.

The fixed 0.5 P5+D1 fusion is the strongest global result under the current
development protocol. Across three independently trained D1 seeds it has mean
event-balanced AP `0.769946`, AUC `0.852054`, and selected macro-F1 `0.776842`,
versus P5 AP `0.765148`, AUC `0.848388`, and macro-F1 `0.772047`.

For the three-seed logit ensemble, canonical-event paired bootstrap gives:

- AP delta versus P5: `+0.007795`, 95% CI
  `[+0.001459, +0.015550]`;
- AUC delta: `+0.005117`, 95% CI `[+0.000609, +0.009796]`;
- positive-class F1 delta: `+0.010807`, 95% CI
  `[+0.001963, +0.020793]`;
- macro-F1 delta: `+0.005172`, 95% CI
  `[-0.001493, +0.012264]`.

Thus AP, AUC, and positive-class F1 have positive canonical-event-cluster
bootstrap intervals; the macro-F1 interval still crosses zero. This is
promising development evidence, not a locked-test or SOTA claim.

The stricter single-model test—training D1 directly above an exact frozen P5
base—failed its promotion gate and was stopped after epoch 2. It confirms that
the useful effect is **independent evidence diversity at late fusion**, rather
than merely adding another residual block to P5.

A subsequent fully independent transient-only D6 control also failed. It uses
history strongly, but its methane discrimination is too weak and noisy when
trained without any appearance-logit context. The supported middle ground is
therefore not “temporal evidence alone”; it is a separately trained
P0-conditioned transient expert, fused late with the stronger P5 expert.

## Data and leakage boundary

Only event-disjoint train and development caches were read:

- original train:
  `/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_ragged_cls_v1/train.pt`
- original development:
  `/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_ragged_cls_v1/val.pt`
- P5 train:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/cache/p5/train.pt`
- P5 development:
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/cache/p5/val.pt`

The original cache contains `10,033` usable training rows from `723`
canonical events and `9,614` development rows from `124` canonical events.
Train/development canonical-event overlap is zero. Each row has six roles:
`t0`, `prev1`, `prev2`, `prev3`, `seasonal`, and `year`.

For the direct-P5 experiment, the runner enforced and recorded:

- identical ordered ID, plume ID, canonical event ID, and label;
- bit-exact equality between the original 768-D feature and the first 768
  dimensions of the P5 cache;
- exact unique mask, time gap, and quality;
- SHA-256 for both cache families and the P5 checkpoint.

No path containing `test`, `sealed`, or `holdout` was read. All output
manifests contain `test_or_sealed_or_holdout_read: false`.

## Architecture actually tested

For visit \(t\), let \(x_t\) be its frozen Panopticon feature. D1 computes:

1. `z_t = Linear(LayerNorm(x_t))`, with width 192;
2. for each valid history visit \(h\),
   `d_h = MLP([z_0 - z_h, |z_0 - z_h|])`;
3. a visit gate from:
   - discrete visit-role embedding;
   - signed-log and Fourier encoding of the real acquisition gap;
   - valid-pixel fraction;
   - current/history cosine similarity;
   - current/history Euclidean distance;
4. masked softmax over valid historical visits;
5. `transient = sum_h gate_h * d_h`;
6. a bias-free residual classifier whose final layer is initialized to zero.

For standalone D1:

`final_logit = frozen_no_sidecar_P0_logit + transient_logit`.

For the supported two-path result:

`final_logit = 0.5 * P5_logit + 0.5 * D1_logit`.

The fusion coefficient is fixed at 0.5 for every seed; it is not refit per
seed. The two experts are independently trained. This is analogous to the
technical separation used by two-stream video models: one path preserves
appearance/response evidence, while the other models motion-like temporal
change. It also follows the useful part of the NormWear design principle:
form evidence in each branch first, then communicate evidence late instead of
mixing raw heterogeneous inputs early.

This experiment does **not** establish novelty for generic two-stream fusion,
temporal differencing, attention, or residual learning. The methane-specific
claim would need to be the tested combination of irregular six-visit change,
acquisition-aware gating, response/appearance preservation, and
event-balanced row evaluation with canonical-event-cluster uncertainty.

## Main results

All values below are event-balanced development metrics. “Macro-F1” uses the
development threshold that maximizes event-balanced macro-F1; AP and AUC are
threshold-free. Mean and sample SD are over D1 initialization/batch seeds
`20260727`, `20260728`, and `20260729`. P0 and P5 are fixed checkpoints.

| System | AP | AUC | Macro-F1 | Positive F1 | all-negative FP mass |
|---|---:|---:|---:|---:|---:|
| Exact no-sidecar P0 | 0.749149 | 0.842789 | 0.761262 | 0.691468 | 2.742803 |
| Fixed P5 | 0.765148 | 0.848388 | 0.772047 | 0.713650 | 3.871023 |
| D1 gated delta, mean | 0.761963 ± 0.006198 | 0.848658 ± 0.004054 | 0.777726 ± 0.005622 | 0.723346 ± 0.005708 | 3.700965 ± 0.508706 |
| D3 gated delta + null loss, mean | 0.761934 ± 0.006004 | 0.848592 ± 0.003539 | 0.776221 ± 0.004731 | 0.723059 ± 0.004844 | 3.914592 ± 0.483322 |
| Fixed P5+D1 equal-logit, mean | 0.769946 ± 0.006155 | 0.852054 ± 0.002614 | 0.776842 ± 0.002740 | 0.720921 ± 0.003759 | 3.769390 ± 0.573986 |
| Fixed P5+D3 equal-logit, mean | 0.769671 ± 0.005912 | 0.851902 ± 0.002389 | 0.776382 ± 0.003423 | 0.724135 ± 0.003472 | 4.173692 ± 0.798234 |

The D1 standalone mean versus exact P0 changes AP by `+0.012813`,
macro-F1 by `+0.016464`, and positive F1 by `+0.031879`, but increases
all-negative FP mass by `+0.958162`. Against P5, D1 has higher mean macro-F1
and positive F1 but slightly lower AP. The fixed equal-logit fusion is the only
global configuration that raises mean AP, AUC, and macro-F1 over P5 while also
slightly reducing mean all-negative FP mass.

Do not cherry-pick the best D1 seed. The table and bootstrap retain all three
predeclared seeds.

## Controls and falsified variants

### History intervention

The D1 history-shuffle intervention replaces historical visits with visits
from different canonical events while holding the base logit fixed. Across
three seeds, shuffling reduces AP by `0.014475` and macro-F1 by `0.012268`,
and raises all-negative FP mass by `0.287689`. The transient path therefore
uses historical content rather than behaving only as a larger current-image
classifier.

### Hard onset normalization failed

The initially proposed onset:

`current-to-history difference - mean history-to-history difference`

was worse than raw current/history difference at global-CLS resolution.
In the matched initial screen:

- raw delta P1: AP `0.744213`, macro-F1 `0.762133`;
- hard onset P2: AP `0.724554`, macro-F1 `0.749704`;
- gated onset P3: AP `0.728625`, macro-F1 `0.756192`;
- gated onset plus null loss P4: AP `0.729656`, macro-F1 `0.752070`.

The likely technical reason is that subtracting a global history-normality
vector removes weak methane evidence together with nuisance variation. It
should not be a headline component without a successful patch-local version.

### All-negative null loss failed

D3 adds a top-k penalty on all-negative canonical events. Across seeds, it does
not improve AP or macro-F1 over D1 and has worse all-negative FP mass.
Therefore the current null loss is rejected, not promoted.

### Direct P5-base residual failed

The direct single-model run used the exact 1536-D P5 cache and exact P5
checkpoint:

- epoch 0 P5 replay: AP `0.765148`, AUC `0.848388`,
  macro-F1 `0.772047`, FP mass `3.871023`;
- epoch 1 D1, selected by AP: AP `0.763889`, AUC `0.847155`,
  macro-F1 `0.774672`, FP mass `4.067451`;
- epoch 2: AP fell to `0.753889`, so patience-1 stopped training.

The selected direct residual has history-shuffle AP delta only `-0.000329`,
far smaller than independent D1. It failed the predeclared gate
(`AP > 0.7729` or clearly better macro-F1), so no three-seed run was made.

This negative result is technically informative: end-to-end residual
correction above P5 collapses toward P5’s existing decision surface, while
independent late fusion retains complementary errors.

### Fully independent D6 was genuinely temporal but too weak

D6 keeps the D1 gated current/history difference branch but fixes its base
logit to zero during training. Its classifier is therefore optimized with BCE
as a transient-only expert. On seed `20260728`, AP rose from `0.564519` at
epoch 1 to `0.688079` at epoch 3; the three-epoch limit then stopped training.
The selected D6 metrics are AP `0.688079`, AUC `0.779818`, macro-F1
`0.717530`, and FP mass `4.785985`.

Cross-event history shuffle reduces D6 AP by `0.202406`, AUC by `0.139188`,
and macro-F1 by `0.125313`. D6 therefore genuinely uses historical change,
but the change signal alone is not sufficiently methane-specific.

Fixed 0.5 P5+D6 fusion yields AP `0.759702`, AUC `0.843626`, macro-F1
`0.776522`, and FP mass `3.767045`. Its AP is below P5 and below the
predeclared `0.772943` promotion threshold, so it was not repeated with three
seeds. This rejects the strongest possible “independent temporal evidence is
enough” interpretation.

### SparseSlowFast isolated history strongly but failed the AP gate

A bounded video-inspired control partitioned history by acquisition rate:

```text
Fast = prev1/prev2/prev3 signed+absolute deltas
Slow = seasonal/year signed+absolute deltas
```

The branches share the delta encoder but have independent
role/gap/quality/relevance gates. Their transient vectors use a fixed
availability-normalized sum before the zero-initialized P0 residual. There is
no learned fast/slow liaison or branch-weight search.

This is a matched-capacity control: D1 at width 192 has 412,801 trainable
parameters; SparseSlowFast at width 174 has 413,744 (`+943`, `+0.228%`).
Both selected epoch 2 in essentially the same wall time (`5.269 s` versus
D1's `5.317 s`) under the same seed, batches, optimizer, loss, and update
budget.

| model | event-balanced row AP | event-balanced row macro-F1 | null FP mass |
|---|---:|---:|---:|
| same-seed D1 | 0.764133 | 0.780371 | 3.205114 |
| SparseSlowFast epoch 1 | 0.759725 | 0.771300 | 4.190368 |
| SparseSlowFast epoch 2 selected | 0.761416 | 0.781642 | 3.063474 |
| SparseSlowFast epoch 3 | 0.744371 | 0.762463 | 3.754951 |

SparseSlowFast genuinely uses matched history: cross-event shuffle reduces AP
by `0.033508` and macro-F1 by `0.030897`. For matched D1 the corresponding
deltas are `0.038221/0.021667`. Hard rate isolation therefore preserves strong
history dependence but does not improve ranking.

Only the predeclared 0.5 P5+D7 logit blend was evaluated. It gives AP
`0.772062`, macro-F1 `0.783478`, and null FP mass `4.146429` with
`21/27` all-negative events alarmed. The same-seed P5+D1 reference is
`0.775794/0.779884/3.350758`; P5 alone alarms `16/27` events with FP mass
`3.871023`.

The promotion gate required AP at least `0.777794` with macro-F1 no lower than
`0.779884`. Macro-F1 passes, but AP is `0.003732` below matched P5+D1 and
`0.005732` below the required threshold. D7 is formally rejected; no weight
search or additional seed is allowed.

The technical reading is useful for the video-method story: ordinary videos
have a stable frame rate, so a hard fast/slow partition is meaningful.
Irregular sparse EO revisits do not. A nominal `prev3` gap can overlap a
`seasonal` gap, visits can be duplicated or absent, and acquisition quality
varies. D1's single continuous-gap gate lets all usable observations compete
on their actual lag and quality; that is more stable than imposing two hard
rate bins.

### Post-hoc equal-thirds three-expert ceiling is not promotable

After D7 had already failed its predeclared gate, one final CPU-only ceiling
audit evaluated exactly one additional combination:

```text
reference = 0.5 * logit(P5)
          + 0.5 * mean_seed(logit(D1))

ceiling   = 1/3 * logit(P5)
          + 1/3 * mean_seed(logit(D1))
          + 1/3 * logit(D7_seed20260728)
```

There was no weight, checkpoint, seed, feature, or threshold-family search.
Each complete prediction received only the standard event-balanced row
macro-F1-maximizing threshold from the existing metric implementation; those
two thresholds were then held fixed through 5,000 paired canonical-event
cluster bootstrap replicates.

| system | event-balanced row AP | AUC | macro-F1 | positive-F1 | null FP mass |
|---|---:|---:|---:|---:|---:|
| fixed P5+D1 three-seed reference | 0.772943 | 0.853505 | 0.777219 | 0.724457 | 3.931439 |
| post-hoc P5+D1+D7 equal thirds | **0.774619** | **0.855701** | **0.782989** | **0.725292** | **3.285985** |

The post-hoc ceiling minus the P5+D1 reference has:

- AP `+0.001676`, 95% CI `[-0.003918,+0.007595]`;
- AUC `+0.002196`, 95% CI `[-0.001143,+0.005849]`;
- macro-F1 `+0.005769`, 95% CI `[-0.001691,+0.013382]`;
- positive-F1 `+0.000836`, 95% CI `[-0.008739,+0.010342]`;
- null FP mass `-0.645455`, 95% CI `[-1.243239,-0.187500]`
  (lower is better).

The point estimates pass the deliberately minimal ceiling screen because both
AP and macro-F1 are strictly higher. They do not establish a classification
gain: all four classification-metric intervals cross zero. The FP-mass
reduction is the only interval separated from zero.

The final decision is nevertheless **reject / not promotable**, regardless of
these point estimates. D7 had already failed its predeclared AP promotion gate;
adding it after observing D1 and D7 development behavior is post-hoc and
cannot reverse that decision. This combination is an upper-bound diagnostic,
not a new default, a lock candidate, or permission to search weights or add
D7 seeds. The supported default remains fixed P5+D1.

## Statistical audit

Bootstrap resamples the 124 canonical development events with replacement and
assigns equal total weight to each sampled event. Checkpoint and threshold are
held fixed within each replicate; thresholds are never refit in bootstrap.
There are 2,000 paired replicates.

The fixed P5+D1 three-seed logit ensemble has:

- AP `0.772943`;
- AUC `0.853505`;
- macro-F1 `0.777219`;
- positive F1 `0.724457`.

Relative to P5, AP, AUC, and positive F1 have 95% intervals above zero.
Macro-F1 does not. Moreover, checkpoints and thresholds were selected using
this same development set, and P5 has only one fixed seed. A fresh locked
event split is still required for a confirmatory claim.

## Technical decision

Promote for the next stage:

- raw current/history difference rather than hard onset subtraction;
- acquisition/role/quality-aware masked gating;
- independently trained response/appearance and transient experts;
- fixed late evidence fusion;
- event-balanced row metrics and paired canonical-event-cluster bootstrap.

Reject for the current global implementation:

- large foundation-model pretraining;
- hard global history-normality subtraction;
- the current all-negative top-k loss;
- direct D1 residual training above P5;
- a fully appearance-logit-free transient classifier;
- SparseSlowFast D7, including the post-hoc equal-thirds ceiling;
- per-seed or same-development fusion-weight tuning.

The technically defensible research story is:

> Methane classification from irregular EO revisit sequences is not ordinary
> video recognition: the event is weak, sparse, and often invisible in a
> single pooled representation. Early temporal fusion and a single residual
> correction entangle stable response/appearance with nuisance change. TEMPO
> instead learns an acquisition-aware transient expert independently from the
> response expert and communicates only decision evidence at the end. This
> preserves methane-sensitive appearance while adding motion-like evidence
> from six irregular visits.

The current evidence supports that mechanism on L89 development data. A final
paper claim still requires a locked event-disjoint test and confirmation on
the other sensors under the same compute and selection protocol.

## Post-hoc fixed-bin mechanism audit

A CPU-only audit used canonical seed `20260728`, fixed acquisition-semantic
bins, and each system's already selected global threshold. No subgroup
threshold or cut point was fitted from performance.

The fixed P5+D1 fusion corrects 240 P5 row errors and introduces 160:

- 77 false negatives and 163 false positives are corrected;
- 99 false negatives and 61 false positives are introduced;
- net effect: 102 fewer false-positive rows and 22 fewer true-positive rows.

Thus the gain is principally precision and ranking, not broader event recall.
D1 alone corrects 403 P5 errors but breaks 360 P5-correct rows. Late fusion
retains 236 of D1's useful corrections while filtering 205 of its harmful
overrides. This is direct evidence for late evidence consensus.

The largest interpretable AP deltas are:

- four unique history visits: `+0.025078`, versus `+0.007920` with all five;
- nearest usable history at 17–24 days: `+0.023679`;
- nominal 351–365-day coverage: `+0.009965`, versus `-0.015922` when the
  longest history is at most 350 days;
- degraded historical quality: `+0.058778` over 372 rows;
- missing `prev2`: `+0.026813`, versus `+0.000221` when `prev3` is missing.

These results support explicit acquisition role, lag, validity, and quality
gating. They also identify a limitation: all-negative FP mass falls by
`0.520265`, but no P5 false-positive event is fully cleared and one event gains
a new sparse FP. A patch-local follow-up should target whole-event
false-positive elimination.

The development CSV has no plume-size, concentration, enhancement, flux, or
emission-rate column, so no physical-magnitude subgroup was fabricated.

## Five-fold OOF reliability liaison

A stricter CPU-only experiment tested whether the mechanism bins can support a
learned consensus gate. Canonical events were assigned to five stratified
folds. Imputation, standardization, logistic coefficients, and the operating
threshold were fitted on four folds and applied to the held-out events.

- G0: L2 logistic regression on the P5 and D1 logits.
- G1: G0 plus absolute disagreement, unique-history count, role-missing flags,
  nearest gap, year-gap deviation from 365 days, history quality, and a fixed
  D1-logit × reliability interaction.
- Both use fixed `C=1`; there is no feature, fold, or regularization search.

Held-out OOF results:

| model | AP | AUC | macro-F1 | null FP mass |
|---|---:|---:|---:|---:|
| fixed equal-0.5 | 0.775794 | 0.853309 | 0.775602 | 4.063258 |
| G0 | 0.771820 | 0.849148 | 0.770086 | 4.443939 |
| G1 | 0.769782 | 0.851152 | 0.775948 | 3.558198 |

G1 improves macro-F1 and null FP mass over G0, but loses AP and AUC to the
fixed equal fusion. Its AP delta versus equal fusion is `-0.006012`, with
paired canonical-event 95% CI `[-0.027128, +0.015544]`. G0 is significantly
worse than equal fusion in AP, AUC, macro-F1, and null FP mass under this OOF
protocol.

The intended mechanism is visible but insufficient: the
D1-logit × reliability coefficient is positive in all five folds and the
absolute-disagreement coefficient is negative in all five. Nevertheless, G1
fails the predeclared promotion gate and is rejected. The fixed equal fusion
remains the global choice; reliability bins should guide patch/representation
design rather than another learned calibration layer on only 124 development
events.

## Clean train-only replicate readiness

A later train-only audit examined
`L89_temporal_train_full_event_balanced.csv` without reading, listing, or
stating any held-out path. It contains 23,359 rows and 848 canonical events.
Applying the same recent-complete-event policy gives:

- inner train: 18,682 rows, 672 events;
- inner development: 4,677 rows, 176 events;
- canonical-event overlap: zero.

The old P5 and D1 heads cannot be replayed as clean comparators on that new
inner development split: their training lineage overlaps it. P5 and every D1
seed would have to be retrained from the new inner train.

The immediate blocker is local imagery coverage. Exact local-only checks found
all six roles for only 10,122/23,359 rows (`43.332%`); 13,237 rows are
incomplete. A bounded feasibility audit then touched only 512 deterministically
sampled missing source files and copied only 64 as a resumable prefix:

- sampled source existence: 512/512;
- unique missing files: 79,401;
- estimated total staging: 148.56 GiB;
- measured 16-worker copy rate: 30.01 MiB/s;
- estimated staging time: 84.43 minutes;
- projected `/diniuvol` free space after completion: 269.68 GiB.

Although those feasibility gates pass, full staging and GPU extraction were
not started in this campaign. The current inner mechanism data also should not
be described as directly test-error-filtered merely because its directory
name contains `hard_event_filtered`: a separate train-only lineage audit found
the source-train payload byte-identical to the corresponding non-filtered
train variants. By contrast, the associated hard-filtered outer cohort is
model-conditioned and can only serve as an engineering external check, not a
clean confirmatory/SOTA benchmark.

Readiness artifacts:

- inner split and local-coverage audit:
  `/home/yuyao/panopticon/research/tempo_20260728/l89_clean_inner_v1/READINESS_AUDIT.json`
- bounded staging feasibility audit:
  `/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_full_local_v1/FEASIBILITY_AUDIT.json`
- feasibility audit contract SHA-256:
  `787e2feb126d05f81a5e74c3091c1ba8fdfedec16de4b5b14ebce7b74a8ccfb0`

## Artifacts

- implementation:
  `/home/yuyao/panopticon/research/tempo_20260728/tempo_l89_global.py`
- CPU tests:
  `/home/yuyao/panopticon/research/tempo_20260728/test_tempo_l89_global_cpu.py`
- D1 three-seed aggregate:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/eventbase_d1_multiseed_fixed/aggregate.json`
- D3 three-seed aggregate:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/eventbase_d3_multiseed_fixed/aggregate.json`
- fixed P5+D1 equal-logit audit:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d1_sidecar_p5_equal_logit_05_audit.json`
- fixed P5+D1 paired event bootstrap:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d1_sidecar_p5_equal_logit_05_paired_event_bootstrap_2000.json`
- fixed P5+D3 paired event bootstrap:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d3_sidecar_p5_equal_logit_05_paired_event_bootstrap_2000.json`
- direct P5-base D1 negative experiment:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/p5base_d1_seed20260728_direct`
- independent D6 negative experiment:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d6_independent_seed20260728`
- fixed P5+D6 negative fusion audit:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d6_sidecar_p5_equal_logit_05_seed20260728_audit.json`
- post-hoc P5+D1+D7 equal-thirds ceiling audit:
  `/home/yuyao/panopticon/research/tempo_20260728/l89_posthoc_three_expert_ceiling_v1.json`
  (SHA-256
  `027323a2535f7bf5f4702a75e6bcd86f9a174afc59182c33be3e99ff17df53e7`)
- mechanism audit JSON and Markdown:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/mechanism_seed20260728/`
- five-fold OOF reliability audit, Markdown, and held-out predictions:
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/oof_reliability_seed20260728/`

Verification at handoff: Python compilation passes and all eight CPU unit
tests for the main runner pass. The mechanism audit has three additional
CPU-only unit tests, and the OOF liaison has three.
