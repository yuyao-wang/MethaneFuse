# Legacy-360 sealed test：两卡只提特征、单次合并

本流程只能在四候选的 development-only master selection receipt 已经写出，
且唯一 winner 的 `selection_lock.json` 已冻结后执行。feature stage 不训练、
不推理、不计算 F1/概率，也不搜索阈值。
当前实现和 CPU 测试没有读取真实 test CSV、TIFF 或 cache，也没有启动 GPU。

## 不变量

- 两个入口都必须显式给出 `--sealed-test`。
- shard 与 merge 都必须给出同一个 SHA-sidecar-authenticated
  `--master-selection-receipt`；family-local loser 的 lock/checkpoint 不能
  单独 claim sealed manifest。
- 在接触 sealed manifest/cache 前，先验证 `summary.json` 与
  `run_status.json` 已完成且从未读 test，再验证 checkpoint 的路径、
  SHA256、epoch、arm、base、config、threshold/selection score 与 encoder
  provenance。
- Gated-delta checkpoint 自带 encoder，必须与 lock 精确相同。旧版
  two-axis checkpoint 没有 encoder 字段，因此只能记录为
  `sha_locked_legacy_checkpoint_plus_selection_lock`，不能声称 checkpoint
  自身携带 encoder provenance。
- manifest 按完整 `event_id` 分组；没有 `event_id` 时按 `plume_id`。
  同一个 event/plume 不能跨 GPU。
- 第一次 manifest claim、merge claim 及每个 part-cache use 都用
  `O_EXCL` 创建。没有 `--overwrite`，失败也不会静默重试。
- merge 只接受 plan 预先指定的两个 `split=test` cache；两者必须有
  `sealed_test_read=true`、`extraction.sealed_test_authorized=true`、相同
  encoder/base/tensor schema，并且组成原 manifest 的逐行完整 union。
- 输出恢复原 source row order，产生唯一 merged sealed-test feature cache。
  audit/receipt 明确写入 `metrics_computed=false`、
  `threshold_search_performed=false`、`sealed_test_evaluations=0`。
- 最终评估只能通过 `evaluate_promoted_locked.py`。它在读取 merged cache
  前创建 master-adjacent 全局 `O_EXCL` eval-use claim，固定 family、
  checkpoint/SHA、threshold、encoder 和 merge cache/SHA；更换输出目录也
  不能规避一次性限制。

## 获得 winner 后的精确执行顺序

以下四个路径必须先替换成最终 winner 和 sealed manifest 的实际绝对路径；
在此之前不要执行。

```bash
PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
ROOT=/home/yuyao/panopticon/research/pretraining_20260727
MASTER=/ABSOLUTE/CAMPAIGN/master_dev_selection_receipt.json
LOCK=/ABSOLUTE/WINNER_DIR/selection_lock.json
CKPT=/ABSOLUTE/WINNER_DIR/checkpoint_best.pth
TEST_MANIFEST=/ABSOLUTE/SEALED/manifest_time_test.csv
OUT=/diniuvol/yuyao/legacy360_sealed_feature_once
RAW_CACHE=/diniuvol/yuyao/methanefuse_research_20260727/legacy360_dual_axis/raw_cache
```

1. 唯一一次 claim 并切成两个完整 event/plume shard：

```bash
"$PY" "$ROOT/legacy360_dual_axis/sealed_test_feature_shards.py" \
  shard-manifest \
  --manifest "$TEST_MANIFEST" \
  --selection-lock "$LOCK" \
  --checkpoint "$CKPT" \
  --master-selection-receipt "$MASTER" \
  --output-dir "$OUT" \
  --prefix legacy360_sealed_test \
  --extractor "$ROOT/query360_two_axis_full_legacy.py" \
  --python "$PY" \
  --raw-cache-dir "$RAW_CACHE" \
  --raw-cache-readonly-fallback \
  --row-batch-size 64 \
  --encoder-microbatch 256 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --sealed-test
```

该命令生成
`$OUT/legacy360_sealed_test_plan.json`。其中两个
`shards[*].extract_argv` 是从冻结 lock 的 encoder 权重配置生成的精确
argv；必须原样运行，不能追加 `--overwrite`。等价的两卡命令形状如下，
实际 `--weights/--sensor-weights` 以 plan 中的 argv 为准：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/query360_two_axis_full_legacy.py" extract \
  --manifest "$OUT/legacy360_sealed_test_gpu0.csv" \
  --split test \
  --sealed-test \
  --output-cache "$OUT/legacy360_sealed_test_gpu0.features.pt" \
  --device cuda:0 \
  --raw-cache-dir "$RAW_CACHE" \
  --raw-cache-readonly-fallback \
  --row-batch-size 64 \
  --encoder-microbatch 256 \
  --num-workers 4 \
  --prefetch-factor 1
```

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/query360_two_axis_full_legacy.py" extract \
  --manifest "$OUT/legacy360_sealed_test_gpu1.csv" \
  --split test \
  --sealed-test \
  --output-cache "$OUT/legacy360_sealed_test_gpu1.features.pt" \
  --device cuda:0 \
  --raw-cache-dir "$RAW_CACHE" \
  --raw-cache-readonly-fallback \
  --row-batch-size 64 \
  --encoder-microbatch 256 \
  --num-workers 4 \
  --prefetch-factor 1
```

`CUDA_VISIBLE_DEVICES=1` 的进程内唯一可见卡编号仍是 `cuda:0`。可并行启动
这两条命令，但必须分别轮询 exit code、cache audit 和进程状态。

2. 两块 extraction 都成功后，做唯一一次 feature-only merge：

```bash
"$PY" "$ROOT/legacy360_dual_axis/sealed_test_feature_shards.py" \
  merge-cache \
  --input-cache "$OUT/legacy360_sealed_test_gpu0.features.pt" \
  --input-cache "$OUT/legacy360_sealed_test_gpu1.features.pt" \
  --plan "$OUT/legacy360_sealed_test_plan.json" \
  --selection-lock "$LOCK" \
  --checkpoint "$CKPT" \
  --master-selection-receipt "$MASTER" \
  --output-cache "$OUT/legacy360_sealed_test_merged.features.pt" \
  --sealed-test
```

只有 merge receipt 为 `status=complete` 后，才可以做一次固定 epoch、
固定 threshold 的最终评估：

```bash
MERGED="$OUT/legacy360_sealed_test_merged.features.pt"
EVAL_OUT=/diniuvol/yuyao/legacy360_locked_eval_once

"$PY" "$ROOT/legacy360_dual_axis/evaluate_promoted_locked.py" \
  --master-selection-receipt "$MASTER" \
  --merge-receipt "$MERGED.receipt.json" \
  --test-cache "$MERGED" \
  --output-dir "$EVAL_OUT" \
  --sealed-test \
  --eval-batch-size 4096 \
  --device cuda:0
```

## CPU-only 合成回归

```bash
cd /home/yuyao/panopticon/research/pretraining_20260727/legacy360_dual_axis
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  test_sealed_test_feature_shards_cpu.py
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  test_evaluate_promoted_locked_cpu.py
```

测试只在临时目录生成合成 manifests/checkpoints/caches。
