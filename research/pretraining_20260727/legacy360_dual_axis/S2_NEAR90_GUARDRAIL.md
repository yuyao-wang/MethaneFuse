# Legacy360 S2 near-90 guardrail

Audit date: 2026-07-27 UTC. This is an engineering fallback for the old
legacy360 protocol, not evidence that a four-sensor model reaches 0.90 F1.

## Audited historical reference

- Run: `wandb/run-20260422_185210-7c09ty1p`.
- Program at recorded commit
  `cb8beae5e73f949f0de8fe6941f57a92ece7aa27`:
  `universal_models_fusion/dino_clssifier_head_s2_temportal_one_block.py`.
- Checkpoint:
  `/transferdiniu2/yuyao/checkpoints/360m_single4_retrain_20260422_185153/s2/ckpt_best_test.pth`.
- Checkpoint SHA-256:
  `ade1f2331757171c00ffc66c0b33aac2b9c11c661dfeb469b81e3c8c76575afe`.
- Checkpoint metadata, read with `map_location=meta`: epoch 2,
  global step 3728, `best_test_acc=0.8934064464357363`,
  `best_train_acc=0.8855537855286325`.
- Initialization: `weights/panopticon_vitb14_teacher.pth`.
- Input: the three S2 roles `(t0,t-90,t-360)` concatenated along the channel
  axis, producing 36 channels before Panopticon. The 12 S2 channel IDs are
  repeated three times.
- Optimization: full-backbone fine-tuning; batch 32; head LR `1e-3`;
  backbone LR `1e-4`; Adam; weight decay `5e-4`; label smoothing `0.05`;
  Noam schedule with 4000 warm-up steps; no frozen epochs.
- Old source-train S2 rows: 59,635. The sanitized inner protocol retains
  53,575 train_core and 6,057 dev S2 rows. Three malformed negative
  train_core rows were removed; dev is unchanged.
- Logged held-out operating point at epoch 2:
  accuracy `0.893406`, recall `0.903678`, FPR `0.117090`, AUROC `0.955772`.
  The corresponding confusion counts are TP 6830, FN 728, FP 866, TN 6530,
  giving positive-class F1 `0.895503` and macro-F1 `0.893363`.

The historical checkpoint was selected by repeatedly evaluating old test
accuracy, not by an inner development set. The old source train/test split is
ID- and plume-disjoint but has 706 canonical-event overlaps. Therefore
`0.895503` is a useful engineering reference only, not a leakage-free SOTA
number.

## Is the current full feature cache sufficient?

Yes, for a strict S2 engineering adapter:

1. `features_hybrid[:,0,:,:]` contains three separate-visit CLS tokens from
   the fine-tuned S2 checkpoint backbone.
2. `base_sensor_logits_hybrid[:,0]` contains the verified historical
   three-visit concatenated-band S2-head logit.
3. On every complete S2 row, `base_hybrid_logits` chooses that exact S2-head
   logit instead of universal multisensor fusion.
4. The zero-initialized residual head therefore begins at the old strong S2
   decision boundary. The trainer keeps epoch 0 as a valid candidate, so a
   poor adapter cannot erase the fallback.

Use
`make_s2_only_feature_cache.py` to retain only rows with all three S2 tokens
and the exact S2 concat logit, reduce the sensor axis from four to one, and
remove universal-base fields that could accidentally retain other-sensor
information. The utility rejects test caches by default and has five passing
CPU tests.

Two semantic caveats matter:

- With one sensor, the sensor axis degenerates. This S2 experiment tests
  temporal residual attention, not cross-sensor fusion. The full four-sensor
  run is still required to validate both axes.
- `current_only` means “old three-visit S2 base plus a new t0-only residual.”
  `two_axis_query` means “the identical old base plus a new three-visit
  residual.” It is a matched residual ablation, not a true t0-only baseline.

The extractor currently uses FP16 autocast, whereas the recorded historical
training script has no AMP block. Architecture and weights are exact, but
logits are not guaranteed bit-identical to the old FP32/TF32 execution. If
the epoch-0 replay differs materially, re-extract the S2 arm with
`--amp-dtype float32` before interpreting the difference.

## Train_core/dev commands (no test access)

Run these only after the merged full train cache and dev cache exist:

```bash
PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
CODE=/home/yuyao/panopticon/research/pretraining_20260727
FULL=/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2/features
S2=/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2/s2_near90

"$PY" "$CODE/legacy360_dual_axis/make_s2_only_feature_cache.py" \
  --input-cache "$FULL/train_core_universal_s2hybrid.pt" \
  --output-cache "$S2/train_core_s2_exact.pt" \
  --expected-split train_core

"$PY" "$CODE/legacy360_dual_axis/make_s2_only_feature_cache.py" \
  --input-cache "$FULL/dev_universal_s2hybrid.pt" \
  --output-cache "$S2/dev_s2_exact.pt" \
  --expected-split dev
```

Run a small dev-only sweep. Each invocation evaluates and preserves epoch 0,
trains only on train_core, selects epoch and threshold only on dev, and writes
its own immutable `selection_lock.json`.

```bash
for arm in current_only two_axis_query; do
  for lr in 1e-4 3e-4; do
    out="$S2/${arm}_lr${lr}_seed42"
    "$PY" "$CODE/query360_two_axis_full_legacy.py" train \
      --train-cache "$S2/train_core_s2_exact.pt" \
      --dev-cache "$S2/dev_s2_exact.pt" \
      --output-dir "$out" \
      --arm "$arm" \
      --base-mode hybrid \
      --epochs 3 \
      --seed 42 \
      --learning-rate "$lr" \
      --selection-metric best_binary_f1 \
      --model-dim 256 \
      --num-heads 8 \
      --temporal-depth 2 \
      --dropout 0.1 \
      --device cuda:0
  done
done
```

The four heads are small and may be distributed across the two GPUs after
feature extraction. Do not inspect or build an S2 test cache until one run is
chosen by dev `best_binary_f1` (with fixed-0.5 F1, AP, AUC, and epoch-0 base
also recorded as diagnostics).

## One post-lock reproduction

After choosing exactly one run from dev, first record its checkpoint and
`selection_lock.json`. Only then extract the full sealed test cache. Convert
that cache to S2-only with the selected lock; the converter validates the
lock and checkpoint SHA before opening the test-cache path and withholds label
counts:

```bash
WINNER="$S2/<chosen-dev-run>"
TEST_FULL="$FULL/test_universal_s2hybrid.pt"
TEST_S2="$S2/test_s2_exact_locked.pt"

"$PY" "$CODE/legacy360_dual_axis/make_s2_only_feature_cache.py" \
  --input-cache "$TEST_FULL" \
  --output-cache "$TEST_S2" \
  --expected-split test \
  --selection-lock "$WINNER/selection_lock.json" \
  --sealed-test

"$PY" "$CODE/query360_two_axis_full_legacy.py" evaluate-locked \
  --checkpoint "$WINNER/checkpoint_best.pth" \
  --selection-lock "$WINNER/selection_lock.json" \
  --test-cache "$TEST_S2" \
  --output-dir "$WINNER/locked_s2_test_once" \
  --sealed-test \
  --device cuda:0
```

`evaluate-locked` fails if output artifacts already exist, so the selected
checkpoint/threshold is evaluated once. Report the new locked-threshold F1
and fixed-0.5 F1 beside historical `0.895503`; do not describe them as
strictly equivalent because the old checkpoint saw the present dev rows
during its original supervised training and was itself selected on test.

For a publication-grade comparison, retrain the exact 36-channel S2
checkpoint on train_core only, use dev for epoch/threshold selection, and
evaluate the old test once. That clean rerun—not this warm-start adapter—is
the valid near-90 reference.
