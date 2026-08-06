# TEMPO L89 patch-local experiment protocol

## Status

Implementation and CPU contracts are complete. **No GPU extraction or model
training has been launched by this task.** A GPU run requires an explicit
resource hand-off from the root task.

Executable:

`research/tempo_20260728/tempo_l89_patch.py`

CPU contracts:

`research/tempo_20260728/test_tempo_l89_patch_cpu.py`

## Technical hypothesis

The existing frozen classifier answers whether the current scene resembles a
methane-positive scene. TEMPO adds a different quantity:

> Does a small current region fail to match this location's historical
> background by more than the history normally fails to match itself?

This is not another CLS cross-attention arrangement. It operates before global
pooling, retrieves local background patches, explicitly estimates
history-to-history variation, and adds only a bounded onset residual to the
existing logit.

## Frozen architecture

For every row, the frozen Panopticon encoder produces final normalized patch
tokens with shape `[T,P,768]`. Extraction immediately applies one common,
seeded orthogonal projection:

```text
[T,P,768] @ Q[768,64] -> [T,P,64]
```

Only float16 64-D projected tokens are persisted. The raw 768-D tensor exists
only inside the inference batch and is never written.

For a current patch `p`, the head computes low-rank query/key similarity
against every valid history visit within either a 3x3 (`radius=1`) or 5x5
(`radius=2`) window:

```text
A(t,p,q) = softmax_q(cos(Wq z0[p], Wk zt[q]) / temperature)
h(t,p)   = sum_q A(t,p,q) Wv zt[q]
```

History visits are gated using five trainable inputs:

```text
signed log(|delta_days|)
log(|delta_days|)
sin(annual phase)
cos(annual phase)
valid-pixel fraction
```

Duplicate, invalid and non-finite-time visits are masked before the softmax.
The gate cannot make an invalid visit usable.

Current-to-history change and normal history variation are then distinct:

```text
current_absolute[p] =
    sum_t w_t |Wv z0[p] - h(t,p)|

history_normality[p] =
    weighted_mean_{i<j} |h(i,p) - h(j,p)|

excess[p] =
    ReLU(current_absolute[p]
         - beta * history_normality[p])
```

The onset MLP receives signed change, current absolute change, explicit
pairwise history normality, positive excess and a clipped change/normality
ratio. It emits one patch score. The highest-scoring 10% of patches are
averaged as a MIL logit.

The final classifier is:

```text
final_logit =
    frozen_base_logit
    + cap * tanh(alpha * MIL_logit / cap)
```

`alpha` is initialized to exact zero. Consequently epoch zero is bit-exact to
the existing frozen classifier even though the onset branch itself has
nonzero features.

For every canonical event whose complete training group is negative, the
top-k patch scores receive an additional softplus null penalty. Mixed events
are not incorrectly treated as all-negative:

```text
loss =
    event-and-class-balanced BCE(final_logit, label)
    + lambda_null * mean softplus(topk(negative-event patch scores))
```

## Cache and leakage contract

- Accepted split names are only `train` and `val`; validation is development,
  not a locked outer split.
- Every input and output path containing `test`, `sealed` or `holdout` is
  rejected before data/model loading.
- Train and validation manifests must have zero canonical-event overlap.
- Backbone PTH, frozen base head, projection, t0 definition, grid and
  comparable preprocessing contracts must match between train and validation.
- Every cache shard is written atomically with file SHA-256 and tensor SHA-256.
- A resume skips a shard only after its file SHA, tensor SHA, configuration SHA
  and row range pass. A different batch/projection/source/configuration fails
  instead of silently mixing shards.
- Shards cover canonical row indices contiguously with no substitution or
  repeated row.
- The base classifier is replayed from its frozen checkpoint; it is never an
  input to the local patch feature path and is never optimized.
- No command in this protocol reads a test/sealed/holdout artifact.

### Base-logit overlays

Projected patches and the frozen base logit are deliberately separable. The
cache retains its original role-only logit as a safe default, while the
CPU-only `rebase` command can write a small `[N]` float32 overlay without
running Panopticon or changing any patch shard.

Two sources are supported:

1. exact replay of a declared CLS cache and frozen temporal-head checkpoint;
2. a prediction CSV with mandatory `id`, `label`, `event_id` and probability
   columns.

Checkpoint replay infers tensor dimensions from the state and requires the
original head configuration when it was not embedded in the checkpoint. CSV
conversion parses float64 probabilities, clips only at a declared epsilon,
stores float32 logits and refuses a sigmoid roundtrip error above `5e-7`.

Every overlay binds:

- patch-manifest content and identity SHA;
- exact ordered IDs, labels and canonical-event IDs;
- source feature-cache/checkpoint/config or prediction-CSV SHA;
- float32 logit tensor SHA;
- a base-family contract independent of split.

Training accepts overlays only as a train/validation pair. Different family,
checkpoint, feature provenance or conversion contracts are rejected. Epoch
zero must then be bit-exact to the overlay, not merely to the role-only logit
embedded during patch extraction.

At the expected 16x16 Panopticon patch grid, storage is approximately:

```text
train: 10,033 * 6 * 256 * 64 * 2 bytes ~= 1.84 GiB
val:    9,614 * 6 * 256 * 64 * 2 bytes ~= 1.76 GiB
```

The cache is placed on `/diniuvol/yuyao`, not a remote source mount.

## Registered first run

The first run uses only the unmodified P0 Panopticon PTH and its already
audited role-only base head. It does not reuse P4/P5 pretraining.

Paths:

```text
data root:
  /diniuvol/yuyao/methanefuse_research_20260727

CSV:
  .../manifests_staged/l89_6time/{train,val}.csv

audited CLS cache:
  .../cache/l89_ragged_cls_v1/{train,val}.pt

frozen Panopticon:
  /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth

frozen base head:
  .../rctp_l89_real_cls_v1/downstream_role_only_seed20260728/
      p0/role_only/checkpoint_best_ap.pt

projected patch cache:
  /diniuvol/yuyao/methanefuse_research_20260728/
      tempo_l89_patch_p0_v1/cache/{train,val}
```

Template extraction command, to be used only after GPU authorization:

```bash
python_bin=/home/yuyao/miniconda3/envs/panopticon/bin/python
script=/home/yuyao/panopticon/research/tempo_20260728/tempo_l89_patch.py
old_root=/diniuvol/yuyao/methanefuse_research_20260727
new_root=/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_v1
base_root=$old_root/rctp_l89_real_cls_v1/downstream_role_only_seed20260728

"$python_bin" "$script" extract \
  --split train \
  --csv "$old_root/manifests_staged/l89_6time/train.csv" \
  --cls-cache "$old_root/cache/l89_ragged_cls_v1/train.pt" \
  --weights /home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth \
  --base-head-checkpoint "$base_root/p0/role_only/checkpoint_best_ap.pt" \
  --output-dir "$new_root/cache/train" \
  --required-local-root "$old_root/cache" \
  --device cuda:1 \
  --batch-size 8 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --shard-rows 96 \
  --projection-dim 64 \
  --projection-seed 36064 \
  --resume
```

The validation command changes `train` to `val` in the split, CSV, CLS cache
and output directory. It does not change the projection seed or PTH.

Resource ceiling for extraction:

- one authorized physical GPU;
- batch 8, four loader workers, prefetch one;
- launch only with at least 12 GiB free;
- measured allocation hard cap 16 GiB;
- one shard committed every 96 rows, so interruption loses at most one shard;
- source images must already be below the required local cache root.

## Head-search schedule

All heads consume the same completed projected cache. No repeated image I/O or
Panopticon inference is needed.

### Gate A: cheap discrimination

Run three configurations for at most three epochs:

| ID | Window | Normality inputs | Beta | Null weight | Purpose |
|---|---:|---:|---:|---:|---|
| A0 | 3x3 | disabled | n/a | 0 | locally aligned current-history residual |
| A1 | 3x3 | enabled | 1 | 0 | effect of history-history normality |
| A2 | 3x3 | enabled | 1 | 0.1 | effect of negative-event selectivity |

`A0` must pass `--no-use-normality-features`. It zeroes the normality and
ratio channels and defines effective excess as `current_absolute`; merely
setting beta to zero is not a valid ablation because raw normality and ratio
would otherwise remain visible. `A1/A2` pass `--use-normality-features`.

Common settings:

```text
match rank=16
value dim=32
hidden dim=64
temperature=0.10
top-k fraction=0.10
lr=3e-4
weight decay=1e-4
batch=16
selection=event-balanced AP
patience=1
```

Example:

```bash
"$python_bin" "$script" train \
  --train-manifest "$new_root/cache/train/manifest.json" \
  --val-manifest "$new_root/cache/val/manifest.json" \
  --output-dir "$new_root/heads/A2_r1_beta1_null0p1_seed20260728" \
  --device cuda:1 \
  --radius 1 \
  --normality-scale 1 \
  --null-weight 0.1 \
  --epochs 3 \
  --patience 1 \
  --seed 20260728
```

Stop the configuration immediately when dev AP and macro-F1 both deteriorate
for two consecutive epochs or all-negative-event false-positive mass rises
materially without an AP gain.

### Gate B: spatial scale and robustness

Only the best Gate-A mechanism advances:

- 3x3 versus 5x5;
- top-k fractions 0.05 and 0.10;
- seeds 20260728, 20260729 and 20260730;
- optional normality beta 0.5 versus 1.0 if beta=1 is too aggressive.

Do not search more temperatures/ranks unless matching entropy is degenerate.
This prevents a broad hyperparameter search from becoming the story.

### Gate C: causal checks

Before promotion, the selected configuration must satisfy:

1. real history beats history-shuffled input using the same checkpoint;
2. local 3x3/5x5 matching beats the matched same-coordinate residual;
3. explicit normality improves either macro-F1 or negative-event FP mass;
4. null loss reduces negative-event FP mass without erasing AP;
5. the selected epoch remains above the exact epoch-zero P0 base.

## Decision rule

This L89 development experiment is promoted to a larger multisensor
implementation only if all of the following hold across three seeds:

- mean event-balanced macro-F1 is at least 2.0 points above the frozen P0 base;
- the paired canonical-event bootstrap lower bound is above zero;
- all-negative-event false-positive mass does not increase;
- history shuffle removes the gain;
- no seed finishes below the previous article's matched-protocol baseline.

AP-only improvement is not sufficient. A 5x5 window is not promoted merely
because it is larger. Test data remain unopened until the architecture,
checkpoint rule and threshold rule have been frozen from development only.

## Completed CPU validation

The CPU-only synthetic dry-run validates:

- exact epoch-zero frozen-base identity;
- finite gradient at the first optimization step;
- irregular temporal/quality gating;
- all-negative-event selection;
- output and top-k shapes.

The unit suite additionally validates:

- correct 3x3 one-patch retrieval;
- correct 5x5 two-patch retrieval;
- 3x3 cannot retrieve a two-patch displacement;
- border windows never wrap around;
- explicit history-pair normality subtraction;
- exact no-history fallback even after the residual gate is manually opened;
- held-out path rejection;
- canonical-event-overlap rejection;
- successful exact-config shard resume;
- file-SHA tamper rejection;
- configuration-SHA mismatch rejection.
- base-overlay float32 probability/logit roundtrip;
- paired-overlay family mismatch rejection;
- one complete streamed train/evaluate command with exact epoch-zero fallback.

No GPU was used by these validations.
