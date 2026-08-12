# Six-time multi-sensor two-axis model

This experiment implements two independent Panopticon axes:

- Axis A encodes only the current observation and produces an appearance logit.
- Axis B owns a separate encoder. Its six roles share that one B encoder, and
  historical deltas produce a zero-initialized residual over the frozen A logit.
- The final logit is fixed at `0.5 * A + 0.5 * B`.
- Available sensor logits are combined by a non-learned masked mean.

## Input contract

Each supplied sensor uses `SensorSixTimeInput`:

```python
inputs = {
    "s2_cdse": SensorSixTimeInput(
        images=s2_images,             # [B, 6, 10, H, W]
        channel_ids=s2_wavelengths,   # [10] or [B,6,10]
        time_valid=s2_valid,           # bool [B,6]
        time_delta_days=s2_day_gaps,  # [B,6]
        quality=s2_quality,            # [B,6,Q], if quality_dim > 0
    ),
    "emit": SensorSixTimeInput(
        images=emit_images,            # [B, 6, 32, H, W]
        channel_ids=emit_wavelengths,
        time_valid=emit_valid,
        time_delta_days=emit_day_gaps,
        quality=emit_quality,
    ),
}
output = model(inputs)
```

A sensor may be absent from the mapping, and an individual row may be missing a
sensor or historical role. Every row must have at least one valid current
sensor. Invalid frames are never passed to either encoder.

`spectral_mask=True` means that Panopticon must ignore that spectral channel;
this is intentionally different from `time_valid=True`, which means that a
temporal image exists.

## Output contract

- `fused_logits`: fixed final binary logits, `[B]`.
- `sensor_logits`: fixed final logits, `[B,4]`.
- `sensor_valid`: current-sensor validity, `[B,4]`.
- `appearance_*`: Axis-A logits.
- `transient_*`: Axis-B logits (`A.detach() + residual`).
- `residual_sensor_logits`: zero-initialized residual, `[B,4]`.
- `history_attention`: masked temporal weights, `[B,4,5]`.
- `effective_time_valid`: `[B,4,6]`.

At initialization, the implementation guarantees exactly:

```text
transient_sensor_logits == appearance_sensor_logits
sensor_logits == appearance_sensor_logits
fused_logits == appearance_fused_logits
```

## Independent training stages

```python
model.configure_training_stage("appearance", train_axis_encoder=False)
output = model(inputs, compute_transient=False)
# optimize output.appearance_fused_logits; Encoder B is not executed

model.configure_training_stage("transient", train_axis_encoder=False)
# Axis A is frozen; optimize output.transient_fused_logits

model.configure_training_stage("inference")
# consume output.fused_logits
```

Passing `train_axis_encoder=True` allows the selected axis encoder to be
fine-tuned. Encoder A and Encoder B are still distinct modules with no shared
parameter storage.

## Train-only normalization

The model can register per-sensor normalization buffers through
`normalization_stats`. Use `load_train_normalization_stats()` for a JSON with
`source_split: "train"`; the loader rejects any other split declaration. The
actual statistics file must be produced after the event-disjoint training split
is frozen.

## CPU tests

```bash
python -m unittest \
  experiments_tryout.six_time_multisensorfusion.test_model
```

To exercise the actual ViT-B/14 and checkpoint interface with a tiny image:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  experiments_tryout/six_time_multisensorfusion/smoke_real_panopticon.py \
  --weights weights/panopticon_vitb14_teacher.pth
```
