# Legacy-360 canonical manifest selection audit

Audit time: 2026-07-28 UTC

## Decision

The formal train/dev/test manifest family is the payload-audited
`sanitized_min1024/` family. In particular:

- extract development features from
  `sanitized_min1024/legacy360_dev_sanitized.csv`;
- the two train shards already come from
  `sanitized_min1024/legacy360_train_core_sanitized.csv`; and
- after model/epoch/threshold selection is locked, the only allowed sealed
  test manifest is
  `sanitized_min1024/legacy360_test_sanitized.csv`.

Do **not** mix `manifests/legacy360_dev.csv` with the sanitized train shards.
Its rows are semantically the same, but its old `query360_index` namespace is
not: 11,146 of its indices collide numerically with the sanitized train
namespace.

This audit opened CSV and JSON metadata only. It did not open any TIFF/NPZ
payload, read or create a test feature cache, or start a GPU process.

## Source versus sanitized manifests

| split | manifest | rows | file SHA-256 | `query360_index` |
|---|---|---:|---|---|
| dev source | `manifests/legacy360_dev.csv` | 12,621 | `d1c8084914e1f6a94047e296c5f6da6a12394e72f35c169b250f36483315597f` | 12,621 unique; min 0, max 126,440; non-contiguous |
| dev canonical | `sanitized_min1024/legacy360_dev_sanitized.csv` | 12,621 | `4f6fa26292461feb3def0213d97782c4968c4ad096394b5d94448ab23e7ed924` | contiguous 113,843–126,463 |
| test source | `manifests/legacy360_test.csv` | 31,564 | `a547e6d2f76db884b47f6fbaca3df6a1fb4c164ab3dcf332b861d40778117920` | contiguous 126,467–158,030 |
| test canonical | `sanitized_min1024/legacy360_test_sanitized.csv` | 31,564 | `6d7f5b401709da6878bf83aefc2b6c06d3f759c431a0b154447dacc9d1ef608b` | contiguous 126,464–158,027 |

The different SHA values are expected. Sanitization adds
`source_query360_index` and assigns a new global, contiguous
`query360_index` after the three unusable train rows have been removed.

Detailed identity checks:

- Dev has no row removal or sensor-arm removal. Its source and sanitized
  `query360_index` sets are not equal: intersection 1,475, source-only 11,146,
  sanitized-only 11,146. Nevertheless,
  `sanitized.source_query360_index` equals the source
  `query360_index` row-for-row, and every original column is value-identical
  after restoring that source index.
- Test has no row removal or sensor-arm removal. Its source and canonical
  index sets are not equal: intersection 31,561, source-only 3,
  sanitized-only 3. Its canonical index is exactly
  `source_query360_index - 3` for all 31,564 rows; every other original column
  is value-identical row-for-row.
- Canonical train, dev, and test indices are pairwise disjoint and their union
  is exactly the 158,028 integers from 0 through 158,027.
- Using the source dev file would create 11,146 numeric index overlaps with the
  canonical train file. Using the canonical dev file creates zero.

## Bad-payload exclusion

The authoritative payload audit is
`sanitized_min1024/legacy360_payload_audit.json`, SHA-256
`977296cd8617876392f6e8128c22720158f46e351a389a206530a4a57a34203a`.
It used a 1,024-byte minimum for optical TIFFs, a 128-byte minimum plus ZIP
magic for S5P NPZs, and did not decode raster/array contents.

No dev or test row was dropped, and no dev or test sensor arm was cleared.
Only three negative, S2-only train rows were removed because their sole sensor
arm contained unavailable/undersized payload references:

| source query index | id | plume/event |
|---:|---|---|
| 339 | 410 | `GAO20191020t163829p0000-D` / `GAO20191020t163829p0000` |
| 344 | 450 | `GAO20191020t163829p0000-D` / `GAO20191020t163829p0000` |
| 407 | 449 | `GAO20191020t165717p0000-D` / `GAO20191020t165717p0000` |

Thus train changes from 113,846 rows
(`99b27f20ea7d2fd865a8936929328f2520dcc3842d3c014b5e504b9f98e36d40`)
to 113,843 canonical rows
(`bdfec2f6dcc1d775c03c93ff2c276fd4974c45669deb1b725b608c80e935ddb9`).

## Train-shard provenance

`manifests/shards/legacy360_train_core_shard_audit.json` (SHA-256
`0a60df85e996046854486f935b66b14bfbdd162d1f5619bf81fda4329dc938a9`)
names the sanitized 113,843-row train file above as its input. Direct CSV
comparison confirms:

- GPU 0 shard: 62,250 rows, SHA-256
  `2d50f5f279447bf3de3cb65a3a03cc8e111c34ca74f2fda409b3e4b8c97991e5`;
- GPU 1 shard: 51,593 rows, SHA-256
  `df4179d655d14302523293a93f9aa300f8174be234954130b3b9e4d0c26fbebd`;
- the shard query-index sets are disjoint;
- their union equals the canonical sanitized train query-index set; and
- after sorting by numeric `query360_index`, the union is row-for-row equal to
  the canonical sanitized train CSV across every column.

The resumable chunk manifests are one more deterministic partition of those
two shards. Their audit
`manifests/chunks/legacy360_train_chunk_audit.json` has SHA-256
`a8583b2b32a664096fff8721cccbd471ee84a15ac8a18b011ed3c59956b8cc0c`
and records the exact two shard paths and SHAs above.

## Safe dev extraction

Run this only when a GPU is available. It pins the canonical dev SHA, refuses
an existing output, uses exactly the same universal and S2 PTHs and FP16
configuration as the train chunks, and has no test/sealed-test argument.
The populated 69 GiB audited cache is used in read-only-fallback mode; valid
cache entries are reused and cache misses are streamed without modifying the
cache.

```bash
set -euo pipefail

PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
CODE=/home/yuyao/panopticon/research/pretraining_20260727
DUAL="$CODE/legacy360_dual_axis"
FORMAL=/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5
MANIFEST="$DUAL/sanitized_min1024/legacy360_dev_sanitized.csv"
EXPECTED_SHA=4f6fa26292461feb3def0213d97782c4968c4ad096394b5d94448ab23e7ed924
OUT="$FORMAL/features/dev_universal_s2hybrid.pt"
LOG="$FORMAL/logs/dev_extract.log"
RAW_CACHE=/diniuvol/yuyao/methanefuse_research_20260727/legacy360_dual_axis/raw_cache
UNIVERSAL="/transferdiniu2/yuyao/checkpoints/universal_360m/query dataset 360m/ckpt_best_test.pth"
S2=/transferdiniu2/yuyao/checkpoints/360m_single4_retrain_20260422_185153/s2/ckpt_best_test.pth

observed_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
test "$observed_sha" = "$EXPECTED_SHA"
test ! -e "$OUT"
test ! -e "$OUT.audit.json"
test ! -e "$LOG"
mkdir -p "$FORMAL/features" "$FORMAL/logs"

"$PY" "$CODE/query360_two_axis_full_legacy.py" extract \
  --manifest "$MANIFEST" \
  --split dev \
  --output-cache "$OUT" \
  --weights "$UNIVERSAL" \
  --sensor-weights "s2=$S2" \
  --raw-cache-dir "$RAW_CACHE" \
  --raw-cache-readonly-fallback \
  --device cuda:1 \
  --amp-dtype float16 \
  --row-batch-size 64 \
  --encoder-microbatch 256 \
  --num-workers 2 \
  --prefetch-factor 1 \
  --log-interval 10 \
  2>&1 | tee "$LOG"
```

Do not use the current launcher default
`/diniuvol/yuyao/query360_raw_cache` without an explicit override: at audit
time it was effectively empty, whereas the cache named above occupied about
69 GiB.

## Reserved sealed-test manifest

The future sealed manifest is:

```text
/home/yuyao/panopticon/research/pretraining_20260727/legacy360_dual_axis/sanitized_min1024/legacy360_test_sanitized.csv
SHA-256: 6d7f5b401709da6878bf83aefc2b6c06d3f759c431a0b154447dacc9d1ef608b
rows: 31,564
query360_index: 126,464–158,027
```

It must remain untouched until a single winner, checkpoint, development
threshold, arm, base mode, model config, and encoder provenance have been
written to a selection lock. Only then may it be extracted once with
`--split test --sealed-test`, the same encoder PTHs and FP16 settings, a new
nonexistent output path, and no `--overwrite`. No test manifest or cache may
participate in head selection or hyperparameter iteration.
