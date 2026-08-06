# TEMPO exactly-once locked evaluator

Status: **locked; waiting for explicit sealed/test authorization**.

No sealed/test cache was read while constructing or validating this protocol.

## Frozen decision

The evaluator reports all three predeclared outputs from one shared cache:

1. promoted P0 at its frozen global threshold;
2. R4 seeds `17/42/73/101`, combined by equal-logit averaging, at its frozen
   global threshold;
3. exactly-one-current-sensor rows from R4 and all remaining rows from P0,
   retaining the corresponding branch threshold.

The manifest fixes checkpoint hashes, thresholds, ensemble weights, branch
logic, metrics, and prohibitions against any test-time selection.

Manifest:
`/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/LOCK_MANIFEST.json`

Manifest SHA-256:
`62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92`

Protocol SHA-256:
`284d179e0952fe17bbfd2f6aa9c5b597aeb0806ad74db4f6599fb55eaa618566`

## Guards

The sealed entry point rejects execution unless all conditions hold:

- the literal confirmation token is supplied;
- the explicitly authorized lock SHA matches the manifest file;
- the input path is explicitly marked `sealed` or `test`;
- the output directory does not exist;
- every locked checkpoint and source-development artifact still matches its
  recorded SHA-256.

After committing an exclusive intent record, the evaluator performs one
sequential source-file pass into local staging and computes its digest during
that pass. It loads only the staged copy once; all three outputs share that
single in-memory payload. The receipt records input bytes, digest, lock digest,
runtime, metrics/prediction hashes, and that no selection occurred.

## Development validation

The development dry-run is at:
`/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/dev_dry_run`

It exactly reproduced:

| Output | F1 | Macro F1 | AP | AUC |
|---|---:|---:|---:|---:|
| P0 | 0.905424 | 0.894916 | 0.960035 | 0.960381 |
| global R4 | 0.910235 | 0.900769 | 0.961690 | 0.962527 |
| one-sensor R4 else P0 | 0.910745 | 0.901134 | 0.961191 | 0.962071 |

The relevant CPU unit/regression suite passed `20/20`.

## Authorization boundary

Do not run `evaluate-once` merely because the program and lock exist. The
experiment owner must explicitly authorize the exact manifest SHA above.
Building the lock and running the development dry-run confer no sealed/test
authorization.

After that authorization, the exact command template is:

```bash
env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
/home/yuyao/miniconda3/envs/panopticon/bin/python \
/home/yuyao/panopticon/research/tempo_20260728/tempo_locked_evaluator.py \
evaluate-once \
--lock-manifest /diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/LOCK_MANIFEST.json \
--sealed-cache <EXPLICITLY_AUTHORIZED_ABSOLUTE_SEALED_OR_TEST_CACHE_PATH> \
--staging-dir /diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/sealed_stage_once \
--output-dir /diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/sealed_evaluation_once \
--authorized-lock-sha256 62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92 \
--confirm RUN_EXACTLY_ONCE_LOCKED_SEALED_EVALUATION \
--batch-size 512 \
--torch-threads 16
```

The placeholder must not be guessed: it must be replaced by the exact cache
path explicitly authorized by the owner. The source must contain:

- `features_universal`: tensor `[N,4,3,D]`;
- `valid_mask`: boolean tensor `[N,4,3]`, with at least one valid t0 sensor
  per row;
- `base_universal_logits`: tensor `[N]`;
- `base_sensor_logits_universal`: tensor `[N,4]`;
- `labels`: tensor `[N]`;
- `ids`, `event_ids`, and `availability_signatures`: length-`N` sequences.

The staged copy requires at least the source cache's full byte size. The
445 MiB development rehearsal produced only 1.3 MiB of outputs. At protocol
freeze, `/diniuvol` had 440 GiB free, so a conservative reservation of
`source size + 2 GiB` is sufficient for staging, predictions, and atomic
writes. The sealed source's actual size/path has deliberately not been
inspected before authorization.
