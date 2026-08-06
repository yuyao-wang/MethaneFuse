# Validity-masked MAE fallback plan

> **Contingency only — not currently authorized for implementation.**
>
> This plan may be implemented only after the main research task explicitly
> declares the current MAE gate failed. Until then, do not modify either
> runner/model, do not launch the command below, and do not treat this document
> as authorization for another experiment. The audit that produced this plan
> was read-only and used no GPU.

## Scope

This fallback is deliberately limited to:

- `research/pretraining_20260727/multisensor_residual_runner.py`;
- `/home/yuyao/NormWear/modules/methane_residual_mae.py`;
- the existing globally event-purged four-sensor manifests, normalization
  statistics, architecture, optimizer, and one-epoch gate.

It does not introduce another repository, a new encoder, validity-aware token
sampling, sensor-specific loss coefficients, a different temporal
representation, or a new spatial product.

## Read-only audit result

### The valid mask exists but is discarded

`load_three_frames` already returns native data and a same-shaped validity
mask with shape `(3,C,H,W)`:

- TIFF sensors use `isfinite(raw) & (raw != 0)`;
- S5P uses `isfinite(raw)`, because a finite zero must not be reinterpreted
  as TIFF-style no-data.

Normalization statistics correctly exclude invalid elements.

`TemporalResidualDataset.__getitem__` also initially handles validity
correctly:

- invalid normalized elements are replaced with zero;
- current validity is `valid[0]`;
- recent residual validity is `valid[0] & valid[1]`;
- seasonal residual validity is `valid[0] & valid[2]`.

The dataset then returns only `streams, label`. The per-pixel/per-channel
validity masks are discarded. `move_streams` resizes only the zero-filled
values.

### Current MAE loss treats filled zeros as observations

`MethaneResidualMAE.forward_loss` currently:

1. computes MSE over every element of a patch;
2. averages over the complete `C * patch_height * patch_width` target;
3. multiplies only by the random MAE patch mask and whole-stream presence;
4. averages over masked patches.

The runner never supplies `sensor_present`, so every provided stream is
considered present. Consequently, zero-filled no-data pixels and entirely
empty channels are easy reconstruction targets rather than excluded
elements. Random masking is independent of validity.

### Evidence from the effective normalization record

The existing globally purged MAE normalization statistics show:

- S2 bands 8 and 9 have zero valid elements;
- L89 band 8 has zero valid elements;
- S2 valid bands have approximately
  `393,144 / (128 * 3 * 42 * 42) ≈ 0.58` native valid coverage.

The current loss nevertheless includes the empty S2/L89 channels and
zero-filled regions in the reconstruction denominator.

An eight-sample S5P inspection found:

- stored arrays have shape `(6,224,224)`;
- selected current/history validity or intersection coverage ranges roughly
  from `0.78` to `1.00`;
- missing support is represented by non-finite values;
- no finite zero happened to occur in those eight samples, but finite zero
  remains valid by definition.

## Minimal fallback patch contract

### 1. Dataset return value

Add an explicit validity mapping with exactly the same keys as `streams`.
A minimally disruptive batch contract is:

```python
({"values": streams, "validity": validity_by_stream}, label)
```

For a sensor with native validity tensor `valid`:

```python
current_valid = valid[0]
recent_valid = valid[0] & valid[1]
seasonal_valid = valid[0] & valid[2]
```

The masks must remain per-channel `(C,H,W)`. Do not reduce them to one
spatial channel, and do not infer them from whether a normalized value equals
zero. A valid observation can normalize to exactly zero.

Clipping does not modify validity. The two random augmentation flips must use
the same sampled decisions for values and masks.

### 2. Resize semantics

Keep value resizing unchanged in the first fallback, so validity masking is
the only experimental difference.

Cast masks to floating point before resizing:

- downsample, such as `224 -> 112`: use `area`, producing fractional coverage
  in `[0,1]`;
- upsample, such as `42 -> 112`: use nearest-neighbour interpolation;
- unchanged resolution: pass through;
- clamp the result to `[0,1]`.

Apply the same policy whether resizing occurs in the dataset or in
`move_streams` on the device. A later validity-normalized image interpolation
could address boundary contamination, but it is intentionally outside this
minimal fallback.

### 3. Model interface

For pretraining, pass both mappings:

```python
backbone(streams, validity_by_stream=validity_by_stream)
```

`forward_encoder_pretrain` should patchify the validity tensor using the same
layout as its reconstruction target:

```python
target_validity[stream] = patchify(validity_by_stream[stream].float())
```

This gives `(B,L,C*patch_height*patch_width)` and aligns exactly with
`targets[stream]` and `preds[stream]`.

Whole-stream presence can be derived per sample as:

```python
validity_by_stream[stream].flatten(1).any(dim=1)
```

Do not overload the current spatial mask into the existing
`sensor_present` indexing interface; that interface represents one Boolean
per complete stream, not spatial/channel support.

Supervised classification may ignore the spatial mask. The dataset and data
signature should still retain identical validity semantics in pretraining and
finetuning.

### 4. Masked reconstruction loss and normalization

For stream `s`, sample `b`, patch `p`, and flattened patch element `j`, define:

```text
E[b,p,j] = (prediction[b,p,j] - target[b,p,j])²
V[b,p,j] = resized and patchified validity in [0,1]
M[b,p]   = random MAE mask, where 1 means reconstruct this patch
```

Compute per-sample numerator and denominator:

```text
N[b,s] = Σ(p,j) E[b,p,j] · V[b,p,j] · M[b,p]
D[b,s] = Σ(p,j) V[b,p,j] · M[b,p]
L[b,s] = N[b,s] / max(D[b,s], 1)
```

Then:

1. average `L[b,s]` equally over samples for which `D[b,s] > 0`;
2. average equally over streams that contain at least one valid sample.

This is an effective-valid-element mean within each sample and stream. A
partially valid patch is weighted by its valid element count. Empty channels
and empty pixels contribute neither numerator nor denominator.

When every element is valid, this is numerically equivalent to the current
patch-MSE followed by masked-patch averaging. Therefore it changes only the
no-data semantics rather than the scale of fully observed examples.

If one stream has no valid masked element, exclude that stream and record it.
If every stream in the batch has `D == 0`, raise a descriptive exception.
Never silently return zero loss.

For auditability, return or log per stream:

- valid fraction;
- valid masked element count;
- number of samples with a non-zero reconstruction denominator;
- validity-masked reconstruction loss.

### 5. Random MAE masking

Keep random token masking unchanged in the first fallback. The loss will
select the intersection of:

- randomly masked patches;
- valid target elements;
- present samples/streams.

Do not simultaneously introduce validity-aware token selection. That would
change the encoder's visible-token distribution and confound the isolated
validity-loss test.

The all-empty fail-fast rule and the logged valid masked element counts catch
pathological batches.

### 6. Reproducibility and checkpoint contract

Add an explicit CLI option:

```text
--validity-masked-reconstruction
```

The fallback command below includes it deliberately. Before the patch exists,
the command must fail with an unknown argument rather than silently launch the
old objective.

Bump the data/signature schema and record a fixed semantic identifier such
as:

```text
per_channel_native_validity;
tiff_finite_nonzero;
s5p_finite;
residual_validity_intersection;
resize_area_down_nearest_up;
masked_element_per_sample_stream_v1
```

The NormWear model source hash will also change. Old unmasked MAE checkpoints
must therefore be rejected for resume or transfer. A validity-masked
pretraining checkpoint and its supervised finetuning command must use the
same new data/signature semantics even though the classifier does not consume
the reconstruction loss.

## S5P coarse-field policy

The minimal fallback must not pretend that the gridded S5P field provides
independent high-resolution observations.

Use only these rules:

- a finite S5P value, including finite zero, is valid;
- a non-finite value is invalid;
- recent/seasonal residual validity is the intersection with current;
- preserve the existing `224 -> 112` value preprocessing;
- resize validity with area coverage;
- exclude fully missing patch elements and proportionally weight partial
  coverage through `V`;
- normalize reconstruction within each sample and stream before averaging.

Do not:

- apply the TIFF `value != 0` rule;
- synthesize a morphology or high-frequency validity mask;
- give repeated/interpolated pixels extra sample weight;
- claim independent 10 km source-cell weighting without original
  geolocation/support metadata;
- drop S5P or change its reconstruction target in this first fallback.

Because every sensor reaches the same `112 x 112` model grid and loss is
normalized per sample/stream over valid elements, the S5P swath footprint
cannot gain weight merely by containing more valid output pixels. This patch
removes no-data cheating; it does not turn S5P into a high-resolution sensor.

## Required CPU tests

All nine tests are mandatory before any GPU launch.

1. **Native validity semantics.** With synthetic TIFF arrays, verify that
   zero, NaN, and infinity are invalid. With a synthetic S5P NPZ, verify that
   finite zero is valid and NaN is invalid. Check current, recent, and
   seasonal validity intersections element by element.

2. **Geometry alignment.** Use a spatially asymmetric value/mask pattern and
   verify that both random flips transform values and masks identically.
   Verify both resize paths, including `[0,1]` fractional area coverage for
   `224 -> 112`.

3. **Manual loss and gradient.** Put an arbitrarily large prediction error
   only at invalid elements and require exactly zero contribution and zero
   gradient there. Put an error at one valid element and compare the loss to
   a hand-computed `N/D`.

4. **All-valid regression.** With fixed predictions, targets, and random MAE
   masks, compare the new all-valid loss with the old `patch_loss.mean` /
   masked-patch mean. Require `atol <= 1e-7`.

5. **Entirely invalid channels and partial patches.** Make complete S2-like
   bands 8/9 and an L89-like band 8 invalid. Verify that they do not dilute
   loss from valid channels. Verify that a partially valid patch is weighted
   by its valid element count.

6. **Empty-stream handling.** Verify that one empty stream is excluded and
   audited while the other streams train normally. Verify that an all-empty
   batch raises a descriptive exception.

7. **S5P coarse-field semantics.** Construct a 224-grid field containing
   finite zero and a NaN swath block. Verify finite-zero weight 1, missing
   weight 0, and correct fractional downsample coverage. For a constant
   prediction error, require equal per-sample normalized loss across
   different valid footprint sizes.

8. **Tiny end-to-end MAE backward.** Run a tiny CPU S2+S5P pretraining
   forward/backward. Require finite loss/gradients and zero decoder-target
   gradient contribution at invalid positions.

9. **Signature and transfer contract.** Verify that an old unmasked
   checkpoint is rejected. Verify validity-masked pretrain-to-supervised
   transfer has encoder coverage `1.0`, excludes decoder state, and leaves
   the matched scratch/finetune classification-head SHA unchanged.

## One-epoch fallback command

Do not run this command unless the current MAE gate has explicitly failed and
the main task authorizes the fallback implementation and launch.

It deliberately matches the current globally purged MAE gate and changes only
the new validity-masked reconstruction semantics. Validation remains the same
deterministic 30-batch screen.

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_residual_runner.py \
  --mode pretrain \
  --sharing shared \
  --sensors s2,l89,emit,s5p \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/manifests_protocol/four_sensor_global_val_purge_v1 \
  --stats-json \
    /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_globalpurge_e1_seed20260727/normalization_stats.json \
  --validity-masked-reconstruction \
  --stats-samples 128 \
  --stats-workers 2 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --device cuda:0 \
  --image-size 112 \
  --patch-size 14 \
  --embed-dim 256 \
  --depth 6 \
  --num-heads 8 \
  --mlp-ratio 4 \
  --fuse-freq 2 \
  --dropout 0.1 \
  --mask-ratio 0.6 \
  --decoder-embed-dim 128 \
  --decoder-depth 2 \
  --decoder-num-heads 4 \
  --learning-rate 3e-4 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --max-val-batches 30 \
  --seed 20260727 \
  --output-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/shared_residual/four_sensor_shared_mae_validitymasked_e1_seed20260727
```

## Fallback acceptance requirements

Implementation is complete only if:

- all nine CPU tests pass;
- the command fails before launch if the new flag/semantics are absent;
- the effective global event overlap remains zero;
- the same purged manifests and normalization hashes are retained;
- the old unmasked checkpoint is not resumed or transferred;
- all four train and validation reconstruction losses are finite;
- validity coverage diagnostics are non-zero and plausible for every sensor;
- no additional objective, model, data, or optimizer change is bundled into
  the one-epoch comparison.

