# GatedDelta: low-capacity two-axis fallback

`query360_gated_delta_model.py` is the fast fallback if the larger axial
attention head overfits or fails to improve within the first few epochs. It
uses both axes explicitly:

- **time axis:** for every available sensor, masked `t0 - history` features
  pass through a sample-dependent temporal gate;
- **sensor axis:** a second masked gate fuses only sensors with a valid current
  observation;
- **PTH anchor:** a bounded, zero-initialized residual is added to the cached
  existing-PTH logit. Epoch 0 is therefore exactly the old checkpoint, not a
  random replacement.

With 768-dimensional CLS features and a 32-dimensional bottleneck, the head
has **51,288 parameters**. S5P history can be disabled with the recommended
`scale_aware_gated_delta` arm because its stored support is not a local 360 m
crop.

The trainer is development-only. It accepts `train_core` and `dev` cache
splits, refuses path names containing standalone `test`/`sealed` tokens, and
has no test-cache or sealed-evaluation CLI option. The best checkpoint may be
epoch 0, so a weak adapter can never displace the existing PTH under the
configured development metric.

## Recommended first launch

Run two learning rates per base mode after the merged caches are complete.
The heads are small enough for two concurrent processes per GPU.

```bash
PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
ROOT=/home/yuyao/panopticon/research/pretraining_20260727
FORMAL=/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v2
TRAIN="$FORMAL/features/train_core_universal_s2hybrid.pt"
DEV="$FORMAL/features/dev_universal_s2hybrid.pt"

CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/query360_gated_delta_runner.py" train \
  --train-cache "$TRAIN" --dev-cache "$DEV" \
  --output-dir "$FORMAL/heads/gdelta_hybrid_lr1e4" \
  --base-mode hybrid --arm scale_aware_gated_delta \
  --epochs 3 --early-stop-patience 2 --learning-rate 1e-4 --device cuda:0 &

CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/query360_gated_delta_runner.py" train \
  --train-cache "$TRAIN" --dev-cache "$DEV" \
  --output-dir "$FORMAL/heads/gdelta_hybrid_lr3e4" \
  --base-mode hybrid --arm scale_aware_gated_delta \
  --epochs 3 --early-stop-patience 2 --learning-rate 3e-4 --device cuda:0 &

CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/query360_gated_delta_runner.py" train \
  --train-cache "$TRAIN" --dev-cache "$DEV" \
  --output-dir "$FORMAL/heads/gdelta_universal_lr1e4" \
  --base-mode universal --arm scale_aware_gated_delta \
  --epochs 3 --early-stop-patience 2 --learning-rate 1e-4 --device cuda:0 &

CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/query360_gated_delta_runner.py" train \
  --train-cache "$TRAIN" --dev-cache "$DEV" \
  --output-dir "$FORMAL/heads/gdelta_universal_lr3e4" \
  --base-mode universal --arm scale_aware_gated_delta \
  --epochs 3 --early-stop-patience 2 --learning-rate 3e-4 --device cuda:0 &

wait
```

Selection remains by development positive-class F1 and its development-only
threshold. Compare `summary.json` across these four runs and the axial sweep;
perform no locked test read until one configuration is selected.

## CPU verification

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python -m unittest -v \
  research/pretraining_20260727/test_query360_gated_delta_cpu.py
```
