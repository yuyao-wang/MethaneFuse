# Partial-sensor and missing-time policy for unified crops

Status: proposed contract required before a production crop executor is
authorized.

This policy extends, rather than changes, the fixed global event split and
canonical query geometry in `unified_crop_orchestrator.py`.  It distinguishes
four facts that must not be collapsed into one `path exists` flag:

1. the sensor is represented in the source index;
2. a source artifact exists for a requested temporal slot;
3. that artifact actually observes the canonical query footprint;
4. the observation has enough native valid support to be usable by the
   mainline model.

No missing sensor, missing temporal observation, or invalid spatial element is
imputed into a real observation.

## 1. What `010011` means

The fixed temporal order is:

```text
t0 | prev1 | prev2 | prev3 | seasonal | year
```

For task `9b2242faf6b8b9e9502fbee9`, plume
`emi20230405t190335p13004-A`, all six EMIT 512 TIFF paths exist.  The task
therefore has a **source-path mask** of `111111`.  Reading the same 360 m
window from those six arrays gives:

```text
t0       0 / 1,152 finite elements
prev1    1,152 / 1,152
prev2    0 / 1,152
prev3    0 / 1,152
seasonal 1,152 / 1,152
year     1,152 / 1,152
```

Thus `010011` is a **query-level observation mask**, not a missing-file mask
and not a missing-sensor mask.  A one-band read of the complete 512 arrays
confirmed that `t0`, `prev2`, and `prev3` contain finite values elsewhere
(`59,105`, `18,810`, and `18,810` pixels respectively).  Their canonical
query window is outside the valid EMIT swath/support.  The correct semantic is
“EMIT exists for this event, but those three temporal slots do not observe
this query.”

The old EMIT preprocessing code marks a temporal output present when the TIFF
was successfully written.  Its reprojection initializes the common 512 grid
with NaN and fills only pixels close enough to the source swath.  It does not
require the eventual query window to contain valid pixels.  Consequently,
`has_all6_512` cannot be used as a training observation mask.

### Provenance/alias finding on the same example

Missingness is not the only temporal integrity issue in this row:

- `emit_prev2_512.tif` and `emit_prev3_512.tif` both declare the same
  `source_npz` path in their TIFF tags.
- That NPZ contains granule
  `EMIT_L2A_RFL_001_20230328T221308_2308715_003`.
- The manifest assigns that product to `prev2`, but assigns
  `EMIT_L2A_RFL_001_20220828T174353_2224012_006` to `prev3`.
- `seasonal` and `year` are both genuinely derived from the latter 2022-08-28
  granule.

Therefore the executor must also distinguish:

- a real acquisition intentionally selected into more than one target slot
  (**temporal alias**); and
- an artifact whose embedded provenance disagrees with its manifest time or
  product metadata (**provenance conflict**).

Aliases are retained for audit but counted once by the model.  Provenance
conflicts are quarantined and are never converted into ordinary missingness.

## 2. Fixed units and orders

- One task row is one canonical geographic query.
- `task_id`, query centre, metre offsets, Carbon Mapper label, `plume_id`,
  `event_group_id`, and `split` are invariant across all sensor views.
- Sensor order is `s2|l89|emit|s5p`.
- Time order is `t0|prev1|prev2|prev3|seasonal|year`.
- `event_group_id` receives exactly one global train/validation/test
  assignment before crop generation.  Crop success, missingness, and sensor
  availability must never trigger a new split.
- The Carbon Mapper footprint is the only label authority.  Sensor masks are
  geometry/coverage diagnostics and do not vote on the label.

Rows are not deleted merely because one sensor or time is unavailable.  A
task may be ineligible for a particular objective while remaining in the
canonical manifest and its original split.

## 3. Manifest v2 contract

The current v1 field `sensor_presence_mask` means that source records were
crop-ready, and each present sensor's
`<sensor>_source_time_presence_mask` is hard-coded to `111111`.  The
query-level `010011` result currently exists only in the smoke-validation
sidecar.  A production executor must not overload those fields.  It must
write a v2 execution manifest with the following explicit layers.

### 3.1 Task-level columns

Retain all current identity, split, label, query-geometry, and grid columns,
then add:

| Column | Meaning |
|---|---|
| `source_sensor_presence_mask` | Four bits: sensor is represented and passes source-level prerequisites |
| `query_usable_anytime_sensor_mask` | Four bits: at least one unique usable query observation exists |
| `query_current_sensor_mask` | Four bits: `t0` is a unique usable query observation |
| `eligible_fused_current_classification` | At least one bit in `query_current_sensor_mask` is one |
| `execution_status` | `ok`, `warning`, or `fail`; read/corruption failures cannot be hidden as absence |
| `missingness_policy_version` | Immutable policy/config identifier |
| `coverage_config_sha256` | Hash of reference-band and coverage thresholds |

The existing `sensor_presence_mask` may be retained as a v1 compatibility
alias, but training code must consume the explicitly named v2 masks.

### 3.2 Per-sensor columns

For every sensor `<s>`:

| Column | Meaning |
|---|---|
| `<s>_source_path_time_mask` | Six bits; exact source artifact exists |
| `<s>_query_any_valid_time_mask` | Six bits; at least one native valid element intersects the query |
| `<s>_query_usable_time_mask` | Six bits; predeclared native coverage criterion passes |
| `<s>_query_unique_time_mask` | Usable bits after temporal aliases are collapsed |
| `<s>_num_usable_times` | Popcount of usable mask |
| `<s>_num_unique_usable_times` | Popcount of unique mask |
| `<s>_current_usable` | The unique `t0` bit |
| `<s>_ssl_current_eligible` | `t0` is usable |
| `<s>_ssl_temporal_eligible` | At least two unique usable observations |

`query_unique_time_mask` is always a subset of
`query_usable_time_mask`, which is always a subset of
`query_any_valid_time_mask`, which is always a subset of the
source-path mask.

### 3.3 Per-sensor/per-time columns

For every sensor `<s>` and time `<t>`:

| Column | Meaning |
|---|---|
| `<s>_<t>_source_path` | Original source path, retained even when the query has no coverage |
| `<s>_<t>_source_exists` | Exact, case-sensitive source check |
| `<s>_<t>_acquisition_time` | Actual acquisition time, not the target lag |
| `<s>_<t>_product_id` | Source product/granule identity |
| `<s>_<t>_observation_key` | Verified acquisition identity used for de-duplication |
| `<s>_<t>_alias_of` | Empty for the retained unique slot, otherwise its canonical time slot |
| `<s>_<t>_crop_status` | Status enum below |
| `<s>_<t>_valid_reference_pixels` | Valid pixels in the frozen coverage reference |
| `<s>_<t>_total_reference_pixels` | Native query pixels in that reference |
| `<s>_<t>_valid_reference_fraction` | Their ratio |
| `<s>_<t>_valid_elements_by_channel` | JSON integer vector for diagnostics |
| `<s>_<t>_crop_path` | Materialized native value crop; empty unless usable |
| `<s>_<t>_validity_path` | Per-channel/native validity mask matching the crop shape |
| `<s>_<t>_crop_sha256` | Readback hash |
| `<s>_<t>_validity_sha256` | Readback hash |

Allowed `crop_status` values are:

```text
sensor_absent
source_missing
read_error
provenance_conflict
out_of_bounds
all_invalid_at_query
below_min_coverage
usable_partial
usable_full
```

`read_error` and `provenance_conflict` are execution failures/quarantine
states, not natural missing observations.  They must be counted separately in
the audit and cannot silently set a mask bit to zero in an otherwise `ok`
row.

### 3.4 Verified observation identity

De-duplication must use the artifact actually read:

- EMIT: compare the TIFF `source_npz` tag, the NPZ scalar `granule_id` and
  `source_nc`, and manifest product/time metadata.
- S5P: use the product ID plus actual acquisition time from the NetCDF/file
  metadata; mapping indices are query-time diagnostics, not product identity.
- S2/L89: use the source product/overpass ID and acquisition time, with the
  exact source path as a fallback.

If two slots resolve to one verified acquisition, retain the slot closest to
its intended temporal target (stable time-order tie break), set the other
slots' `alias_of`, and clear their unique-mask bits.  Never expose duplicate
copies as independent temporal tokens.

If embedded identity conflicts with manifest identity, set
`provenance_conflict`.  Do not guess the acquisition time and do not train on
that slot until it is repaired or explicitly quarantined.

## 4. Native validity and usability

Values and validity remain sensor-native:

- S2/L89: finite and non-nodata/nonzero validity per channel.
- EMIT: finite and not declared nodata per channel; NaN is invalid.
- S5P: finite CH4 is valid, including a legitimate numeric zero.  Keep the
  native 3×3 support and do not manufacture a 224×224 spatial observation.

Every materialized value crop has a same-shape per-channel `uint8` validity
sidecar.  A channel that is empty in the source remains all-zero in its
validity mask; it is not made valid because another channel has support.

The first production version should freeze the old pipeline's coverage
criteria as a named configuration, while retaining exact fractions so later
threshold ablations do not require recropping:

| Sensor | Reference | Usable criterion |
|---|---|---|
| S2 | band index 11 | finite nonzero fraction `> 0.80` |
| L89 | band index 0 | finite nonzero fraction `>= 0.75` |
| EMIT | band index 0 | finite/non-nodata fraction `>= 0.75` |
| S5P | native 3×3 CH4 | finite fraction `>= 0.50` |

These are quality gates, not value imputation.  The executor records
`any_valid` separately from `usable`, so a below-threshold real observation
is not mislabeled as a nonexistent file.

## 5. Crop write policy

| Condition | Source path | Crop/value path | Validity path | Training bits |
|---|---|---|---|---|
| Sensor absent | empty | empty | empty | all zero |
| Source missing | retained if declared | empty | empty | source bit zero; task warning/fail per cause |
| Read/corruption error | retained | empty | empty | quarantine; never natural missing |
| Query all-invalid/outside swath | retained | empty | empty | source bit one; query bits zero |
| Some valid but below coverage | retained | empty in mainline executor | empty | `any_valid=1`, `usable=0` |
| Usable partial/full | retained | native crop | matching native validity crop | usable bit one |
| Temporal alias | retained | may share canonical artifact | canonical validity artifact | usable one, unique zero |

Do not write one zero/NaN placeholder file per missing slot.  The loader
creates a zero tensor only at runtime, after normalization, and supplies an
attention mask of zero.  This saves I/O and ensures file presence cannot be
mistaken for observation presence.  Original paths and measured coverage
remain in the manifest for reproducibility.

Writes must be atomic and read back before the manifest row becomes `ok`.

## 6. Training-mask semantics

The loader returns:

```text
values:            sensor-native tensors
time_mask:         [B, S, T] unique usable observations
spatial_validity:  [B, S, T, C, H, W] (ragged/native before batching)
source_sensor_mask:[B, S] provenance/audit only
current_mask:      [B, S] usable t0 branches
```

- Invalid values are set to zero **after train-only normalization**.
- Attention and losses use `time_mask`; reconstruction/pooling additionally
  use `spatial_validity`.
- An all-zero value tensor with a one mask is forbidden.
- Missing sensors have all time bits zero.  A sensor with `010011` remains a
  known sensor with partial temporal coverage; it is not relabeled as an
  absent sensor.
- Sensor/time dropout may only change observed one bits to zero.  It may never
  change a real zero bit to one.
- Mainline current-event classification fuses only sensor branches whose
  `current_mask` is one.  Historical-only EMIT data can be used for
  self-supervised pretraining, but it is not treated as an EMIT current
  observation for a positive/negative F1 denominator.
- A fused supervised task is retained when at least one sensor has a usable
  current observation.  Per-sensor classification metrics use only tasks
  whose corresponding current bit is one.
- Normalization statistics use training events and valid observed elements
  only.
- Report metrics by current-sensor pattern and temporal-coverage bucket as
  well as pooled metrics, so a gain cannot come only from an easier
  availability subset.

For the audited all-four positive example, the source sensor mask remains
`1111`.  Its EMIT query mask is `010011`; EMIT is excluded from current-event
classification because its `t0` bit is zero, while S2/L89 and independently
validated S5P may still make the fused task eligible.  The duplicate
`seasonal/year` EMIT acquisition is counted once after provenance
verification.

## 7. Acceptance gates before full crop

A production crop launch is blocked until all of the following pass.

### A. Synthetic/unit tests

1. Six-bit order is exactly the declared temporal order; four-bit order is
   exactly the declared sensor order.
2. Missing sensor, missing path, all-invalid query, below-threshold partial
   coverage, usable partial coverage, and read error produce distinct status
   and mask combinations.
3. A masked tensor can be changed arbitrarily without changing model output
   or loss (`atol <= 1e-7`); an unmasked tensor must affect it.
4. S5P finite zero remains valid; S5P NaN remains invalid.
5. A per-channel empty raster band remains masked even if other bands are
   valid.
6. Duplicate acquisition IDs produce one unique bit and deterministic
   `alias_of`; conflicting embedded/manifest provenance is quarantined.
7. Executor output is deterministic across worker counts and resume.

### B. Manifest and split tests

1. `task_id` is unique and deterministically derived.
2. Every derived query keeps its original `plume_id`, `event_group_id`, and
   global split.
3. Pairwise event overlap among train/validation/test is zero before and
   after eligibility filtering.
4. Every sensor view of a task has the same WGS84 query centre, physical
   footprint, Carbon Mapper label, and split.
5. No row changes split because of a missing sensor/time.
6. Every usable crop has non-empty value/validity paths, matching shapes,
   matching hashes, and at least one valid element; every unusable slot has
   an empty crop path.
7. Zero unresolved provenance conflicts are marked usable.

### C. Stratified no-write/readback pilot

Before production materialization, validate both labels from every observed
sensor combination and every split, plus a stable sample large enough to
include:

- complete time coverage;
- missing sensor;
- all-invalid query coverage;
- partial coverage immediately below and above each threshold;
- temporal aliases;
- any provenance conflicts.

The existing smoke must be upgraded to read positive **and negative** tasks.
Its audited regression expectations include:

```text
emi20230405t190335p13004-A, positive query:
  source sensor mask                 1111
  EMIT source-path time mask         111111
  EMIT any-valid query time mask     010011
  EMIT current usable                0
  no EMIT t0/prev2/prev3 crop paths
```

The pilot audit must publish status counts, all four mask-pattern
distributions, valid-fraction quantiles by sensor/time/split/label, alias and
provenance-conflict counts, objective eligibility counts, and event-overlap
checks.  Label-conditioned missingness should be reported rather than hidden
by dropping rows.

### D. Small materialization/readback pilot

Materialize only a small deterministic pilot to the local cache:

1. atomic write, immediate readback, and SHA verification pass;
2. native dimensions and georeferencing/support are correct;
3. value and validity arrays align after the exact same crop operation;
4. no placeholder files exist for missing times;
5. rerun is idempotent and resume does not duplicate rows/files;
6. a two-worker and higher-worker run produce identical manifest hashes.

Only after A–D pass should a full crop be launched.  The full executor should
stage/cache source files under `/diniuvol/yuyao`, process bounded shards, emit
per-shard audits/checkpoints, and merge only after the same global gates pass.

## 8. Mainline decision

The recommended mainline is partial-sensor **and** partial-time training with
explicit attention/spatial-validity masks, not a complete-case-only dataset.
Complete four-sensor rows are too rare, and requiring six valid times would
discard real current observations.  Conversely, file-presence padding would
fabricate observations.

The canonical event manifest remains the superset.  Each objective selects
eligibility through explicit masks while preserving the same event split and
evaluation denominator definition.
