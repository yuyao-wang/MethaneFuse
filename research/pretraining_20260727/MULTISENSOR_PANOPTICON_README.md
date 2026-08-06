# Shared Panopticon multi-sensor runner

`multisensor_panopticon_runner.py` is the stronger shared-backbone prototype
for the four methane classification tasks. It is intentionally separate from
`multisensor_residual_runner.py`: the latter is a compact scratch model for
testing a residual representation, while this runner retains the released
Panopticon ViT-B/14 representation and its native wavelength-aware,
variable-band patch embedding.

The first 30-round/full-validation gate was stopped during L89 validation
when a global audit found cross-sensor event leakage. Its completed S2 metric
is diagnostic-only and the output directory contains `PROTOCOL_INVALID.json`;
that run must never be resumed or used for model selection. The runner now
audits the global event protocol before normalization, image I/O, checkpoint
loading, or GPU initialization.

## Method

There is exactly one shared Panopticon backbone. Each sensor owns only:

1. a zero-initialized projection from the inherited 384-dimensional calendar
   embedding to the 768-dimensional token space;
2. an ordered temporal-slot embedding;
3. a `LayerNorm(768) -> Linear(768, 2)` classification head.

The shared wavelength-aware patch embed accepts different band counts in
different calls. A training round processes one batch from every enabled
sensor sequentially, accumulates the four losses divided by the number of
sensors, and performs one optimizer step. This keeps peak memory close to one
sensor batch and defines an equal-sensor objective without channel padding.
The sensor order rotates across rounds. By default, an epoch has as many
rounds as the shortest loader, so no sensor is oversampled; an explicit
`--rounds-per-epoch` can opt into cycling.

Inference is sensor-local:

```python
logits = model("s2", s2_batch)
```

No aligned sample from another sensor is required. A four-sensor checkpoint
can be loaded into a model containing only the requested sensor adapter/head.

## Fixed input plan

| Sensor | Source data read | Model tokens | Normalization |
|---|---|---:|---|
| S2 | `path_t0` | 1 current frame | Train rows, t0, non-zero pixels, 12 per-band values |
| L89 | `path_t0,path_prev1,path_seasonal` | 3 raw frames | Train rows, the same 3 frames, non-zero pixels, 7 per-band values |
| S5P | existing `image_path` NPZ | 6 raw frames | Train NPZ rows, all finite pixels, one shared scalar |
| EMIT32 raw | `path_t0,path_prev1,path_seasonal` | 3 raw frames | Train rows, the same 3 frames, finite non-zero pixels, 32 per-band values |
| EMIT32 residual | same three source TIFFs | `t0-prev1,t0-seasonal` | Same train-only source normalization; subtraction occurs after normalization |

The EMIT choice is controlled by:

```text
--emit-input-mode raw
--emit-input-mode residual
```

It should be frozen only after the matched full raw/residual comparison is
complete. The verified constant in the current EMIT script can be requested
with `--emit-stats-source verified-script-constant`, but the default is to
compute statistics from the selected training manifest and three selected
frames.

## Data and I/O policy

On this machine the defaults prefer the already staged inner
train/validation manifests:

```text
/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/s2/
/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/l89_3time/
/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/s5p/
/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/emit_3time/
```

These are event-held-out inner validation splits, not the sealed original
test manifests. However, their event assignment was made separately per
sensor. Although every sensor has zero within-sensor train/validation
overlap, the shared model's `union(train)` intersects `union(validation)` by
124 canonical physical events. The canonical rule uses S2
`event_group_id`; for L89, S5P, and EMIT it removes the final derived-sample
suffix from `plume_id`.

The CLI therefore defaults to fail-closed behavior:

```text
--cross-sensor-event-policy error
```

A valid shared run must explicitly use:

```text
--cross-sensor-event-policy purge
--protocol-manifest-dir /one/common/write-once/directory
```

`purge` removes the union of all validation events from every training
manifest before statistics or training, materializes deterministic write-once
CSVs, rereads them, and requires `global_train_val_overlap_after=0`. The
resulting train rows are S2 18,107, L89 10,033, S5P 18,271, and EMIT 11,688.
The protocol audit and manifest fingerprint are embedded in
`event_protocol_audit.json`, `run_config.json`, and every checkpoint. An old
checkpoint with no zero-overlap audit is refused for resume, transfer, or
evaluation.

If the staged files are absent, the CLI falls back to the canonical
repository CSVs and the paths can always be overridden explicitly; the same
global audit still applies.

The local cache defaults to:

```text
/diniuvol/yuyao/methanefuse_research_20260727/cache/multisensor_panopticon/
```

Files already under `/diniuvol/yuyao` bypass the cache copier, preventing an
accidental second local copy. Remote paths use the inherited lock-protected
cache in synchronous or asynchronous mode. Per-worker prefetch and DataLoader
worker counts are exposed separately.

Normalization records include the train CSV path/size/mtime, selected path
columns, seed, sample count, semantics, and a fingerprint. A stale record is
not silently reused. The values are also embedded in every checkpoint.

## Bad-sample policy

The four inherited temporal datasets have a retry path that can replace an
unreadable index with a random row. This runner never calls that path.
`StrictIndexedDataset` invokes `_get_one(original_index)` once. A missing or
corrupt file raises `SampleLoadError` with sensor, original index, sample ID,
plume ID, and path context; `run_status.json` records the full traceback.

The staged manifests should be prepared/audited before training. If a source
must be dropped, create a deterministic filtered manifest and record those
rows during staging. Do not enable random in-loader replacement: it changes
the effective label/event distribution and makes metric denominators
unreliable.

## Parameter counts

The official backbone contains `98,976,000` parameters.

| Configuration | Total | Frozen-run trainable | Adapter total | Head total |
|---|---:|---:|---:|---:|
| EMIT raw, T=3 | 100,177,928 | 1,201,928 | 1,189,632 | 12,296 |
| EMIT residual, T=2 | 100,177,160 | 1,201,160 | 1,188,864 | 12,296 |

For the raw configuration, private counts are:

| Sensor | Temporal adapter | Head |
|---|---:|---:|
| S2 | 295,680 | 3,074 |
| L89 | 297,216 | 3,074 |
| S5P | 299,520 | 3,074 |
| EMIT raw | 297,216 | 3,074 |

Thus the frozen screen updates about 1.2 M parameters while using a single
98.98 M-parameter pretrained backbone, rather than four backbone copies.

## Reproducible checks

Manifest-only plan. It performs no image/checkpoint/statistics I/O, but it
does read all manifests, writes deterministic purged copies under `/tmp`, and
asserts global overlap `124 -> 0`:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_panopticon_runner.py \
  --plan-only \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir /tmp/panopticon_event_protocol_audit \
  --emit-input-mode raw
```

Synthetic CPU integration test. It covers default fail-closed behavior,
deterministic event purge and post-purge zero overlap, balanced training,
per-sensor evaluation, four-sensor-to-one-sensor checkpoint loading, and the
no-random-replacement assertion:

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_panopticon_runner.py \
  --self-test \
  --output-dir /tmp/multisensor_panopticon_selftest
```

The implementation-time official checkpoint test used all four channel/time
schemas at batch size one and 14 × 14. All four outputs were `(1, 2)`, one
joint backward/optimizer step succeeded, and peak allocated CUDA memory was
`458,986,496` bytes. The checkpoint SHA-256 was:

```text
55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e
```

## Frozen 30-round real-data smoke

`--smoke-30` forces a frozen backbone, one epoch, at most 30 balanced rounds,
and at most two validation batches per sensor. The validation numbers from
this mode are pipeline diagnostics only and must not be reported as full-set
metrics.

Use small statistics only for a plumbing smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_panopticon_runner.py \
  --smoke-30 \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/multisensor_panopticon/global_event_v1/protocol_manifests \
  --emit-input-mode raw \
  --stats-samples 32 \
  --stats-workers 8 \
  --num-workers 4 \
  --s2-batch-size 4 \
  --l89-batch-size 2 \
  --s5p-batch-size 2 \
  --emit-batch-size 1 \
  --output-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/multisensor_panopticon/smoke_raw
```

Before comparing F1/AP, compute and freeze the intended train-only
normalization with the same `--stats-samples`, `--stats-seed`, and manifest
for every matched run.

## Short full comparison

Only after a replacement 30-round gate passes with audit overlap zero should
a longer comparison be considered. Batch sizes are per sensor because the
temporal token length and channel-fusion cost differ substantially. The
following command is a template, not authorization to launch it:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_panopticon_runner.py \
  --backbone-mode frozen \
  --epochs 3 \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/multisensor_panopticon/global_event_v1/protocol_manifests \
  --emit-input-mode raw \
  --stats-samples 200 \
  --stats-workers 8 \
  --num-workers 8 \
  --s2-batch-size 8 \
  --l89-batch-size 4 \
  --s5p-batch-size 4 \
  --emit-batch-size 2 \
  --output-dir \
    /diniuvol/yuyao/methanefuse_research_20260727/multisensor_panopticon/frozen_raw_e3
```

Run the residual variant only by changing `--emit-input-mode` and the output
directory. Preserve all other arguments and the seed. The runner reports
per-sensor and macro validation loss, accuracy at 0.5, positive F1 at 0.5,
validation-optimized F1/threshold, AP, and AUROC. Model selection is by macro
validation AP.

Full-backbone fine-tuning is opt-in:

```text
--backbone-mode finetune --backbone-lr 1e-5
```

It should be attempted only if the frozen shared run clears its gate, because
the current independent L89 evidence showed degradation under continued full
fine-tuning.

## Resume and single-sensor inference

Every completed epoch atomically writes:

```text
run_config.json
run_status.json
event_protocol_audit.json
metrics.json
checkpoint_latest.pth
checkpoint_best_macro_ap.pth
train_only_normalization.json
protocol_manifests/*.csv
```

Resume exact training state:

```bash
.../multisensor_panopticon_runner.py \
  --resume /path/to/checkpoint_latest.pth \
  --epochs 3 \
  --cross-sensor-event-policy purge \
  --protocol-manifest-dir /the/same/protocol_manifests \
  [the same sensor/order/model arguments]
```

Exact resume and training-time initialization require the same protocol
fingerprint. Checkpoints created before the global audit was added are
diagnostic-only and fail closed.

Evaluate one sensor from a four-sensor checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/multisensor_panopticon_runner.py \
  --sensors s2 \
  --eval-only \
  --eval-sensor s2 \
  --resume /path/to/checkpoint_best_macro_ap.pth \
  --output-dir /path/to/s2_eval
```

This loads the shared backbone plus only the S2 adapter/head and constructs
only the S2 validation loader. It does not require L89, S5P, or EMIT input,
but the source checkpoint must contain a zero-overlap global audit and the S2
validation manifest hash must match that audit.

## Current blockers and interpretation limits

1. S5P still uses the inherited constant channel ID `0.0`; this is a proxy,
   not a measured Sentinel-5P methane averaging-kernel/spectral-response
   wavelength. The weak independent S5P Panopticon result may partly reflect
   this interface mismatch.
2. The sensors use different spatial support and are not sample-aligned in
   this runner. Shared weights test cross-sensor regularization, not
   cross-sensor fusion.
3. Macro AP gives each sensor equal selection weight, while the balanced
   objective gives each sensor one batch per round. This is deliberate and
   must be stated in any comparison with sample-weighted training.
4. Optimized validation F1 is descriptive. A sealed-test F1 threshold must be
   frozen from the complete inner validation set using the existing two-phase
   evaluation gate.
5. The four sensor loaders reuse current parser/normalization semantics, but
   this is still a new joint optimizer. It must be compared against the
   existing independent Panopticon checkpoints with identical manifests,
   inputs, seeds, and validation protocol.
