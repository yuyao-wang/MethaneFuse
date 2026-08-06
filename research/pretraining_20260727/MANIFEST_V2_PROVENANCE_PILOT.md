# Manifest-v2 provenance metadata pilot

Status: metadata-only pilot complete; production crop remains blocked.

This pilot implements the read-only audit requested by
`MISSING_SENSOR_TIME_POLICY.md`.  It does not create a v2 execution manifest,
modify the old pipeline, read raster pixels, read NetCDF arrays, materialize
crops, or use a GPU.

## Reproducible command

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/pretraining_20260727/audit_manifest_v2_provenance.py \
  --wide-manifest \
    /diniuvol/yuyao/methanefuse_research_20260727/manifests/multisensor_6time_512_wide.csv \
  --task-manifest \
    /diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.csv \
  --validation-sidecar \
    /diniuvol/yuyao/methanefuse_research_20260727/manifests/unified_crop_tasks_smoke_partial.validation.csv \
  --emit-probe-plume-id emi20230405t190335p13004-A \
  --output-json \
    research/pretraining_20260727/manifest_v2_provenance_pilot.audit.json \
  --output-probe-csv \
    research/pretraining_20260727/manifest_v2_provenance_pilot.emit_probe.csv
```

The script's `--self-test` and Python bytecode compilation pass.  Repeating
the command without changing an input produces identical JSON and CSV hashes.

Frozen identities for this run:

| Artifact | SHA-256 |
|---|---|
| Audit script | `283f5ee0ecd3acab48363d0e228d7f86bccf0b415104c638e4cc1638411d261c` |
| Wide manifest | `2ada7e41e397689710440d242a1975271e3bb0c5bd2cc260122170da12bf0703` |
| Audit JSON | `4fc9dc8f0a952857ea20832ccc76710e6895ba6f0253a2a55ba38b44e1373e6a` |
| EMIT probe CSV | `83a4e307cc09e77b2379bd78c1bfdade44f72474d81b8190c9b238b95b0a1f59` |

## I/O boundary

The full 19,834-row wide CSV is a sequential metadata read from the local
`/diniuvol` cache.  Query-valid masks are reused from an existing two-row
smoke sidecar.

For the one explicitly named EMIT plume, the pilot opens six GeoTIFF headers
and their default tags.  It never calls `dataset.read`.  From each tagged NPZ,
it reads only `granule_id.npy` and `source_nc.npy`, with a 1 MiB per-member
hard limit.  The 12 scalar members total 6,192 uncompressed bytes.  Raster
pixel, large NPZ array, and NetCDF array read counts are all zero.

## Full-manifest source findings

The wide manifest contains 7,108 event groups: 4,517 train, 695 validation,
and 1,896 test.  Pairwise event-group overlap remains zero.

“Represented” below means that a sensor row exists or at least one exact
source path is declared.  It is deliberately broader than complete,
crop-ready six-time coverage.

| Sensor | Represented rows | Rows with a manifest-identity alias candidate | Alias groups | Extra aliased slots |
|---|---:|---:|---:|---:|
| S2 | 4,614 | 435 | 455 | 455 |
| L89 | 3,458 | 27 | 27 | 27 |
| EMIT | 3,297 | 2,492 | 3,319 | 3,670 |
| S5P | 14,757 | 0 | 0 | 0 |

These are candidate aliases from manifest product/overpass/time metadata,
not full embedded-provenance verification.  Examples include the same S2
product in `t0/prev1`, the same L89 overpass and acquisition in
`prev2/seasonal`, and one EMIT product in multiple historical target slots.
They demonstrate that six populated paths cannot be interpreted as six
independent temporal observations.

No exact output-TIFF path is repeated with two manifest identities in the
wide CSV.  This does not prove provenance consistency: two distinct TIFF
paths can still point to one tagged NPZ, as the embedded probe below shows.

The required identity metadata is also incomplete:

- S2 has 4,614 represented rows and six declared path strings per row, but
  only 4,097 `prev1/prev2/prev3` paths exist.  Product identity is absent for
  492 `t0` rows and 530 `seasonal/year` rows.
- L89 has acquisition time for all 3,458 represented rows, but only 1,074
  rows have product/overpass identity for `t0`, `prev1`, `seasonal`, and
  `year`; 3,457 have it for `prev2/prev3`.  Exact existing-path counts range
  from 3,253 (`prev2`) to 3,420 (`prev1`).
- EMIT has populated path, existence, acquisition-time, and product fields
  for all 3,297 represented rows and six slots.  This is still only manifest
  metadata, not embedded verification.
- S5P has populated path, existence, acquisition-time, and product fields for
  all 14,757 represented rows and six slots.

The represented fields contain no literal `NaN`/`None`/`null` sentinel
strings.  Natural absence is encoded as empty, while declared-but-missing
paths appear as `exists=false`; the audit preserves those states separately.

The audit also finds 303 L89 and 56 EMIT groups where one primary identity is
paired with multiple recorded acquisition times.  For L89, the sampled cause
is an under-specific overpass key such as `LANDSAT_8|42|35` reused across
dates.  For EMIT, samples include a product paired once with its acquisition
time and once with an event-like compact timestamp.  These are provenance
completeness/consistency warnings; they are not silently converted into
aliases.

An independent EMIT product-timestamp check parses the acquisition time
encoded in each granule ID.  With a 300-second tolerance, 78 slots in 72 rows
disagree with their manifest acquisition time: 8 `t0`, 6 `prev2`, and 64
`prev3`.  This catches conflicts even when that product appears in only one
slot and therefore is not found by duplicate/alias analysis.

## Embedded EMIT regression

For `emi20230405t190335p13004-A`, five of six time slots verify against the
TIFF `source_npz` tag and the NPZ `granule_id/source_nc` scalars.  `prev3` is
a provenance conflict:

| Slot | Manifest product | Embedded product | Result |
|---|---|---|---|
| `prev2` | `...20230328T221308_2308715_003` | same | verified |
| `prev3` | `...20220828T174353_2224012_006` | `...20230328T221308_2308715_003` | conflict |
| `seasonal` | `...20220828T174353_2224012_006` | same | verified |
| `year` | `...20220828T174353_2224012_006` | same | verified alias candidate |

`prev2` and `prev3` TIFFs point to the exact same tagged NPZ.  The `prev3`
manifest time differs from the embedded granule time by 18,332,955 seconds,
so that slot must be repaired or quarantined.  `seasonal` and `year` use
different NPZ paths but resolve to the same verified granule; they are a real
temporal alias and must contribute one unique token.

The tagged NPZ directory token for verified `prev1` is `prev2`, and for
verified `prev2` it is `prev3`.  Directory-slot disagreement is retained as a
warning, not declared a conflict when embedded product and time identity
agree.  This is why de-duplication and quarantine must use the artifact
identity rather than directory names.

## Query missingness and schema readiness

The existing validation sidecar reproduces the known query-level result:

```text
S2   111111
L89  111111
EMIT 010011
```

Thus EMIT has six source paths but no finite query support at `t0`, `prev2`,
or `prev3`.  The smoke-validation sidecar has only two positive records; the
task CSV has four train tasks (two positive, two negative) across only two
sensor patterns.  It is not the policy's required
label/split/sensor-combination stratified pilot.  It also exposes only
aggregate finite counts, not per-channel validity, frozen usable-threshold
decisions, or S5P as a six-bit query-valid mask.

Across the wide and task CSV headers, the policy defines 410 required v2
fields:

| State | Fields |
|---|---:|
| Directly present | 26 |
| Available only as a v1 compatibility alias | 83 |
| Missing | 301 |

The missing fields include explicit query-usability/unique-time masks,
verified observation keys, `alias_of`, quarantine-aware crop statuses,
per-channel validity diagnostics, materialized value/validity paths, and
readback hashes.  A v1 source-presence mask must not substitute for them.

## Remaining blockers

Production crop is not authorized.  Before a bounded materialization pilot:

1. Materialize the v2 execution/query-coverage columns without overloading v1
   source-presence fields.
2. Repair or explicitly quarantine embedded/manifest provenance conflicts;
   the known EMIT `prev3` regression must be a fixed test.
3. Freeze intended target-time metadata for every sensor/time slot, then make
   `alias_of` selection deterministic by target distance plus stable time
   order.
4. Fill or explicitly mark unavailable source product identities, especially
   the under-specified L89 slots and missing S2 identities.
5. Run a deterministic metadata/header pilot covering both labels, all
   train/validation/test splits, observed sensor combinations, aliases, and
   conflict cases.
6. Only then run the policy's small query-coverage pilot to produce
   per-channel validity and frozen usable masks.  Do not launch a mass crop
   until those gates and small readback/materialization tests pass.
