# Legacy 360 m dual-axis head experiment log

Status: **16k development pilot complete; full train cache in progress**  
Protocol date: 2026-07-27 UTC

## Engineering side control: six-frame channel concatenation

This control is not one of the formal legacy360 head arms. It fine-tunes
Panopticon on the separate S2 GEE six-visit temporal-cutoff split by
concatenating six 12-band images into a 72-channel tensor. The loader carries
no visit identity or explicit temporal axis, and the script searches the test
threshold every epoch, so the result is diagnostic only.

The run was deliberately stopped after the first completed epoch because it
was well below the historical 360 m target and further epochs would repeatedly
reuse test labels:

| Epoch | Train F1 | Test F1 @ 0.5 | Test macro F1 | Test-best positive F1 | Threshold | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7651 | 0.7714 | 0.7838 | 0.7955 | 0.3135 | 0.8677 | 0.8646 |

Artifacts:

- metrics:
  `/diniuvol/yuyao/checkpoints/s2_gee_legacy_notebook_6time/s2_gee_legacy_notebook_6time_all12_concat_gate3/metrics_history.json`
- checkpoints: the same directory (`ckpt_latest.pth`,
  `ckpt_best_test.pth`, `ckpt_best_train.pth`, `ckpt_best_val_ap.pth`)
- console log:
  `Upgraded_dataset/s2_gee_legacy_notebook_6time/train_gate3.log`

Interpretation: extra visits alone are insufficient. A channel-set
concatenation without explicit visit identity reaches only 0.7955 even after
optimizing its threshold on test; it does not support the claim that
six-visit input solves the temporal problem.

## Feature-extraction recovery record

The first monolithic two-GPU extraction attempt exposed two independent
engineering faults before any formal head was trained:

1. the historical base path incorrectly reused the strict temporal
   finite-fraction gate, so S5P-only rows with valid t0 but all-NaN history
   appeared to have no three-role legacy input; and
2. after restoring the historical declared-role/global-padding payload,
   two 12-worker extractors prefetched too much host memory and the OS
   terminated the first process at 11,520/62,250 rows.

The PTH itself was not missing `row_fusion_head`: it contains the expected
max-fusion LayerNorm/Linear state. The repaired loader now keeps strict
dual-axis tokens separate from the declared three-role checkpoint path,
including historical S5P `nan_to_num`-before-normalization semantics.
Synthetic tests and a live offending batch agree bit-for-bit with the old
collate payload.

Recovery protocol:

- both train shards are divided into 15 complete event/plume components
  (8 chunks on GPU0 and 7 on GPU1; 113,843 rows total);
- each chunk is atomically committed and is independently restartable;
- two workers and prefetch factor one are used per GPU;
- completed chunks are skipped by the launcher;
- no sealed test manifest is involved.

Chunk audit:
`manifests/chunks/legacy360_train_chunk_audit.json`.

This is an engineering comparison on one merged `train_core` feature cache
and one event-/plume-disjoint development cache. The sealed test cache is not
an input to training, epoch selection, threshold selection, or this sweep
launcher. After one configuration is locked, a separate `evaluate-locked`
command may read sealed test exactly once.

The target of approaching 0.90 positive-class F1 is a promotion target, not a
claimed result. A development-selected threshold and F1 at the conventional
fixed threshold 0.5 are always reported separately.

## Fixed protocol

- Data: sanitized 360 m legacy manifests from `sanitized_min1024/`.
- Warm starts encoded in each feature cache:
  - universal Panopticon 360 m checkpoint;
  - hybrid base using the verified independently trained S2 checkpoint when
    S2 is present and the exact universal row fusion otherwise.
- Encoder inputs: the same cached `[row, sensor, time-role, 768]` tensors for
  every matched head comparison; no image I/O during head training.
- Time roles: `t0`, approximately `t-90`, and approximately `t-360`.
- Sensor axis: S2, L89, EMIT, and S5P with explicit missingness masks.
- Scale-aware arm: S5P history is masked because its stored context is about
  10.5 km rather than a plume-local 360 m crop.
- Epochs: 3.
- Seed: 42.
- Dropout: 0.1.
- Learning rates: `1e-4` and `3e-4`.
- Selection: maximum development `best_binary_f1`; the corresponding
  development threshold is locked in `selection_lock.json`.
- Default concurrency: two small head processes per GPU on GPUs 0 and 1.
- Caveat: the warm-start 360 m checkpoint saw the old source-train pool from
  which development was carved. This is an engineering selection protocol,
  not a leakage-free foundation-model claim.

## Planned matched sweep

| Run directory | Base | Arm | LR | Status |
|---|---|---|---:|---|
| `hybrid_current_only_lr1e4` | hybrid | current only | 1e-4 | pending |
| `hybrid_current_only_lr3e4` | hybrid | current only | 3e-4 | pending |
| `hybrid_two_axis_lr1e4` | hybrid | full two axis | 1e-4 | pending |
| `hybrid_two_axis_lr3e4` | hybrid | full two axis | 3e-4 | pending |
| `hybrid_scale_aware_lr1e4` | hybrid | scale-aware two axis | 1e-4 | pending |
| `hybrid_scale_aware_lr3e4` | hybrid | scale-aware two axis | 3e-4 | pending |
| `universal_current_only_lr1e4` | universal | current only | 1e-4 | pending |
| `universal_current_only_lr3e4` | universal | current only | 3e-4 | pending |
| `universal_scale_aware_lr1e4` | universal | scale-aware two axis | 1e-4 | pending |
| `universal_scale_aware_lr3e4` | universal | scale-aware two axis | 3e-4 | pending |

The `current_only` arms preserve sensor fusion but mask historical roles. They
are the matched control for whether temporal evidence helps; they are not a
single-sensor baseline.

## Result fields to transcribe after completion

For every run, `metrics_history.json` contains epoch 0 (the exact
zero-residual checkpoint base) and epochs 1–3. Record all of the following
before deciding which run, if any, deserves sealed evaluation:

| Run | Best epoch | F1@0.5 | Dev-best F1 | Dev threshold | Macro F1@0.5 | AP | AUC | S2 F1@0.5 | L89 F1@0.5 | EMIT F1@0.5 | S5P F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pending | — | — | — | — | — | — | — | — | — | — | — |

Also retain the complete `availability` strata and `by_sensor` dictionaries;
the table above is only a compact view. The required artifacts per run are:

- `stdout_stderr.log`: exact command plus live epoch metrics;
- `launcher_status.json`: shell-level completion/failure and assigned GPU;
- `run_status.json`: trainer-level status with `sealed_test_read=false`;
- `metrics_history.json`: epoch-0 base and every trained epoch, including
  fixed-0.5, development-optimized threshold, AP/AUC, exact availability
  signatures, and each sensor-containing stratum;
- `summary.json`: best development record and complete history;
- `checkpoint_best.pth`: development-selected head;
- `selection_lock.json`: checkpoint SHA, best epoch, base/arm, and locked
  development threshold;
- `dev_predictions_best.csv`: development predictions only.

## Launch

The launcher is deliberately dry-run by default:

```bash
research/pretraining_20260727/legacy360_dual_axis/run_two_axis_head_sweep.sh
```

After the merged train and development caches exist, start it explicitly:

```bash
research/pretraining_20260727/legacy360_dual_axis/run_two_axis_head_sweep.sh --run
```

If extraction used different cache names, set only development-safe paths:

```bash
TRAIN_CACHE=/absolute/path/train_core_merged.pt \
DEV_CACHE=/absolute/path/dev.pt \
HEADS_ROOT=/absolute/path/formal_v2/heads \
research/pretraining_20260727/legacy360_dual_axis/run_two_axis_head_sweep.sh --run
```

There is intentionally no `TEST_CACHE` option in this launcher.

The current formal defaults are:

```text
/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2/features/train_core_universal_s2hybrid.pt
/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2/features/dev_universal_s2hybrid.pt
/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2/heads
```

## 2026-07-28 development pilot

Inputs:

- event-safe pilot train cache: 15,973 rows from the first completed chunk on
  each GPU;
- canonical sanitized dev cache: 12,621 rows, manifest SHA-256
  `4f6fa26292461feb3def0213d97782c4968c4ad096394b5d94448ab23e7ed924`;
- no test read or evaluation.

Exact checkpoint-preserving dev bases:

| Base | F1 @ 0.5 | Dev-best F1 | Threshold | Macro F1 @ 0.5 | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|
| universal | 0.88699 | 0.89793 | 0.36221 | 0.88274 | 0.95483 | 0.95463 |
| hybrid | 0.88183 | 0.89031 | — | 0.87777 | 0.94975 | — |

Main 256-d axial findings:

- best hybrid two-axis result was epoch 1, LR `3e-4`: dev-best F1
  `0.89268`, F1@0.5 `0.88697`, AP `0.95079`;
- the matched hybrid current-only epoch-1 result was dev-best F1 `0.89163`,
  F1@0.5 `0.88671`, AP `0.95008`;
- the incremental two-axis advantage over current-only is therefore only
  `+0.00105` dev-best F1 and `+0.00072` AP in this pilot;
- every universal axial run selected epoch 0. Universal scale-aware at
  LR `1e-4` calibrated F1@0.5 to `0.89492` at epoch 1, but its dev-best F1
  decreased from `0.89793` to `0.89758` and AP decreased slightly.

Low-capacity gated-delta findings:

- best run: universal, LR `1e-4`, cap `1.5` or `4.0`, epoch 2;
- dev-best F1 `0.89810`, threshold `0.38074`, F1@0.5 `0.88830`,
  macro F1@0.5 `0.88349`, AP `0.95483`, AUC `0.95467`;
- the improvement over the exact epoch-0 universal base is only `+0.00017`
  dev-best F1. This is not a material model gain.

Compact follow-up:

- six 64-d, depth-1, dropout-0 heads tested both factorization modes at
  `1e-5`, `3e-5`, and `1e-4` for two epochs;
- best dev-best F1 was `0.89803`; no compact run beat gated-delta or separated
  materially from epoch 0.

Decision:

- the pipeline and existing PTH have recovered a genuine near-0.90 dev
  baseline, so the earlier 0.77–0.80 six-channel-concat result was not a
  representative ceiling;
- attention capacity and learning rate are not the primary bottleneck;
- full-cache training remains justified because the 15,973-row pilot is only
  14% of train, but a `0.0002` gain is not a research result;
- retain universal gated-delta and one compact scale-aware head for full-cache
  retraining, add a strong low-dimensional two-axis statistics control, and
  evaluate availability-conditioned calibration by grouped OOF only;
- do not unlock sealed test until the full-cache winner/configuration and dev
  threshold are frozen.

### S2-only diagnostic guardrail

The exact verified S2 concat-PTH subset contains 11,159 pilot-train and 6,057
canonical-dev rows with all three visits. It is not the fused four-sensor
primary endpoint.

| Arm | Best epoch | F1 @ 0.5 | Dev-best F1 | AP | AUC |
|---|---:|---:|---:|---:|---:|
| current-only, 64-d, LR 3e-5 | 2 | 0.91505 | 0.91812 | 0.96060 | 0.96778 |
| two-axis, 64-d, LR 3e-5 | 2 | 0.91505 | 0.91812 | 0.96060 | 0.96778 |

Epoch-0 F1@0.5 was already `0.91464`. Thus the S2 guardrail is comfortably
above 0.90, but the two temporal axes add no measurable benefit on these
frozen CLS tokens. This result prevents calling the full-model shortfall a
general implementation failure; it also prevents claiming that temporal
attention is the source of the strong S2 result.

### Availability-conditioned calibration falsifier

A five-fold, canonical-event-grouped OOF analysis fitted every threshold on
the other four folds. All 12,621 dev rows joined exactly to 291 events and 621
plumes; train/validation event and plume overlap was zero in every fold.

| Calibration | OOF binary F1 | OOF macro F1 |
|---|---:|---:|
| global threshold | 0.89688 | 0.88684 |
| exact availability with shrinkage | 0.89862 | 0.88779 |
| count + primary sensor | 0.89840 | 0.88765 |

For exact availability, the event-cluster bootstrap paired binary-F1 change
was `+0.00174`, 95% CI `[-0.00143, +0.00458]`; macro-F1 change was
`+0.00096`, CI `[-0.00229, +0.00383]`. The intervals cross zero. Therefore
availability calibration does not provide a material or confirmed gain and
will not be presented as a dual-axis/pretraining result.

### AxisStats strong simple control

The 130-d AxisStats control uses universal/hybrid fused and per-sensor logits,
validity masks, per-sensor t0–history cosine/L2/norm statistics, and
cross-sensor current-token cosine statistics. Thirteen logistic/HGB candidates
were trained on the same 15,973-row pilot train cache; no test input exists.

- A train-fitted monotonic calibration of the universal fused logit exactly
  tied the direct PTH dev-best F1 at `0.897926`; it changed F1@0.5 to
  `0.897007` but did not improve ranking (`AP 0.954828`, `AUC 0.954632`).
- The best model that actually consumed all 130 time×sensor statistics reached
  only `0.892545`, a `-0.005381` change from the direct PTH reference.
- Under the selected global dev threshold, overlapping sensor-containing
  strata were S2 `0.9319`, L89 `0.9015`, EMIT `0.8768`, and S5P `0.8758`.

This negative control agrees with the neural pilots: temporal and sensor-axis
statistics derived from frozen global CLS features do not add useful ranking
information. It strengthens the decision to move the scientific contribution
upstream into response-sensitive/local pretraining rather than present a new
fusion head.

## 2026-07-28 full-cache development promotion

The complete sanitized train-core cache was finished before any sealed-test
feature was extracted:

- 113,843 rows in 15 atomic parts, with tensor shape
  `[113843, 4 sensors, 3 roles, 768 dims]`;
- cache SHA-256
  `f3b9061d804c6e3f337a704040988237f85a1e84ccb745fe82855d19a6868f56`;
- shard/merge audit SHA-256
  `d295cf37aa99c65b918bcc39fd7db384aa662968f7a01b960d932242aa234e9f`;
- exact unique `query_index` coverage 0--113842, with no plume overlap
  between train-core and development;
- every candidate used the same frozen encoder fingerprint
  `c18304506b8d2d27755a29dc72303fd7012afcc7ca868128051acf349f496f36`.

Only the four predeclared short-run candidates were trained on this complete
cache. Selection used the 12,621-row development cache only:

| Candidate | Best epoch | F1 @ 0.5 | Dev-best F1 | Dev threshold | Macro F1 @ 0.5 | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact scale-aware time×sensor, d64, LR 1e-4 | 3 | 0.90313 | **0.90558** | 0.39103 | **0.89622** | 0.96003 | 0.96038 |
| gated delta, LR 3e-4 | 3 | 0.90036 | 0.90430 | — | 0.89427 | **0.96042** | **0.96066** |
| gated delta, LR 1e-4 | 3 | 0.89891 | 0.90304 | — | 0.89312 | 0.95850 | — |
| compact plain time×sensor, d64, LR 3e-5 | 3 | 0.89616 | 0.89952 | — | 0.88982 | 0.95580 | — |

The compact scale-aware model was frozen as the sole promoted winner. Its
checkpoint SHA-256 is
`9ef9f19c0a68d9e92691cd1661032cb778eace0225895507de57491d425d9ae2`;
the master development-selection receipt SHA-256 is
`d4e3f7be423d8d20ac342204c478b9c1a4c7f7b08df9dbe8ca2bd327e70fe7c2`.
The plain axial arm reaching only `0.89952`, versus `0.90558` for the
scale-aware arm, is evidence that acquisition/scale conditioning matters in
this engineering head. It does not establish a new attention mechanism.

## 2026-07-28 exactly-once locked evaluation

After the winner, epoch, and threshold were frozen, the sealed test cache was
extracted in two shards and merged exactly once:

- shard 0: 15,896 rows, SHA-256
  `08b09b80b7b1f2eb3d7d210b33f4cbca05446522613b1f69c5396a66e8285c4b`;
- shard 1: 15,668 rows, SHA-256
  `d061b5c65157686372d1d39937ad157a265224e3913c0cd04338ea2d6d988c98`;
- merged cache: 31,564 rows, source/manifest SHA-256
  `6d7f5b401709da6878bf83aefc2b6c06d3f759c431a0b154447dacc9d1ef608b`;
- locked evaluation count: 1; threshold search on test: false;
- evaluation receipt SHA-256
  `554f22ac7fa8bfb59fb7609c5fdce684cb7d75d1c4a371571c9cf65aec18471d`.

At the development-locked threshold `0.3910266161`, the complete old-legacy
test metrics are:

| Scope | Binary F1 | Macro F1 | AP | AUC |
|---|---:|---:|---:|---:|
| overall | **0.86485** | **0.85063** | **0.93313** | **0.93042** |
| rows containing S2 | **0.90616** | 0.90140 | 0.95842 | 0.96368 |
| rows containing L89 | 0.83623 | 0.83273 | 0.91275 | 0.91435 |
| rows containing EMIT | 0.85082 | 0.80467 | 0.92817 | 0.90326 |
| rows containing S5P | 0.86739 | 0.83363 | 0.94298 | 0.92622 |

The fixed-0.5 overall result was binary F1 `0.86076` and macro F1 `0.85282`.
Exact single-sensor availability strata at the locked threshold were S2
`0.90389`, L89 `0.81609`, EMIT `0.83675`, and S5P `0.75131` binary F1.
Thus the requested 0.90 guardrail is met for S2, but not for the four-sensor
overall endpoint. Relative to the audited historical universal full-test F1
`0.82832`, the new engineering model improves binary F1 by `+0.03653`
(3.65 points); relative to the previous paper's 360 m F1 `0.8377`, it is
`+0.02715` (2.71 points). These are historical engineering comparisons, not
matched clean-split SOTA claims.

The immutable old split contains 706 canonical events in both source train
and source test. No test label, prediction, or metric was used for candidate,
epoch, or threshold selection, but this overlap prevents presenting the
locked number as a clean journal benchmark. It is retained as a faithful
legacy-protocol comparison and as evidence that the downstream implementation
is competitive enough to test upstream pretraining.

### Forensic input recovery and resource incident

One source TIFF (`group_00000448/s2_90.tif`) consisted of a 272-byte
zero-IFD header followed by an otherwise complete 12×224×224 float32 payload.
The source file was never modified. A pixel-exact reconstruction was written
only to the content-addressed local raw cache, with recovered destination
SHA-256
`ef9426963950a5f87bc578ac5cc333b7962169c97f1898bc9fe8752cbe2a46f07`
and payload SHA-256
`89fedb212115e931d524160b3eeebf6a5c2d910d4895138561782df2e87949af`.
All payload values were finite; an adjacent JSON audit records the recovery.

The first simultaneous two-shard extraction was stopped before an atomic
shard-0 output existed when host available memory fell to roughly 18 GiB.
Shard 1 was allowed to finish, then shard 0 was rerun alone. This avoided a
host OOM and produced the exact row counts and hashes above. No other task's
GPU process was stopped or modified.

### Scientific interpretation

The full-data run materially improves the legacy engineering endpoint, but it
does not rescue a novelty claim for TransientQuery or factorized attention.
The head only learns how to aggregate already-computed features; it never
defines which weak spectral change is physically consistent with methane.
The journal contribution must therefore remain upstream: matched-compute,
sensor-response-conditioned counterfactual pretraining, with the two-axis
model retained as a strong downstream control.
