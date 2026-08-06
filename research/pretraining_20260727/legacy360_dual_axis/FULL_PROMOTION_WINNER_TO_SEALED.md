# Full-cache promotion → one dev winner → sealed evaluation

本链路现在同时支持：

- `gated_delta` winner：
  `query360_gated_delta_runner.py evaluate-locked`
- `compact_axial` winner：
  `query360_two_axis_full_legacy.py evaluate-locked`

跨 family 的唯一选择只读取四个 development run。选择 receipt 冻结原始
candidate 的 lock 路径/SHA、checkpoint 路径/SHA、epoch、threshold、
encoder 和 evaluator family；它不会生成新模型或重新拟合 threshold。

确定性排序规则依次为：

1. development `best_binary_f1` 较高；
2. AP 较高；
3. ROC-AUC 较高；
4. development-selected epoch 较早；
5. trainable head parameters 较少；
6. 四候选显式表中的 index 较早。

此外 winner 的 development `best_binary_f1` 必须至少为 `0.895`；否则
selector 输出 no-go，不会生成允许 sealed 流程继续的 receipt。

以下命令只可在四个 full-cache candidates 全部完成后执行。当前开发阶段
不要替换或打开 `SEALED_MANIFEST`。

## 1. 生成唯一 master dev selection receipt

```bash
PY=/home/yuyao/miniconda3/envs/panopticon/bin/python
ROOT=/home/yuyao/panopticon/research/pretraining_20260727
CAMPAIGN=/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/heads/full_cache_dev_promotion_v1
RECEIPT="$CAMPAIGN/master_dev_selection_receipt.json"

"$PY" "$ROOT/legacy360_dual_axis/select_full_cache_dev_winner.py" \
  --campaign-root "$CAMPAIGN"
```

selector 会拒绝 incomplete run、任何 held-out-use marker、非四个预注册配置、
checkpoint/lock SHA 或 config 不一致、不同 encoder/manifests、不同 epoch-0
base，以及已有 receipt。

## 2. 在 sealed claim 前重新验证 receipt、lock 与 checkpoint

```bash
(
  cd "$(dirname "$RECEIPT")"
  sha256sum -c "$(basename "$RECEIPT").sha256"
)

eval "$(
  "$PY" - "$RECEIPT" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
winner = receipt["winner"]
dispatch = receipt["locked_dispatch"]
values = {
    "WINNER": winner["run"],
    "FAMILY": winner["family"],
    "LOCK": winner["selection_lock"]["path"],
    "LOCK_SHA": winner["selection_lock"]["sha256"],
    "CKPT": winner["checkpoint"]["path"],
    "CKPT_SHA": winner["checkpoint"]["sha256"],
    "EVALUATOR": dispatch["evaluator"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

echo "$LOCK_SHA  $LOCK" | sha256sum -c -
echo "$CKPT_SHA  $CKPT" | sha256sum -c -
printf 'Locked winner: %s (%s)\n' "$WINNER" "$FAMILY"
```

## 3. 一次性切 sealed manifest

先人工填写以下三个绝对路径：

```bash
SEALED_MANIFEST=/ABSOLUTE/SEALED/manifest_time_test.csv
FEATURE_DIR=/diniuvol/yuyao/legacy360_sealed_feature_once
RAW_CACHE=/diniuvol/yuyao/methanefuse_research_20260727/legacy360_dual_axis/raw_cache
PLAN="$FEATURE_DIR/legacy360_sealed_test_plan.json"
```

然后执行：

```bash
"$PY" "$ROOT/legacy360_dual_axis/sealed_test_feature_shards.py" \
  shard-manifest \
  --manifest "$SEALED_MANIFEST" \
  --selection-lock "$LOCK" \
  --checkpoint "$CKPT" \
  --master-selection-receipt "$RECEIPT" \
  --output-dir "$FEATURE_DIR" \
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

此时 selection lock/checkpoint 已在 manifest read 前重新验证。命令使用
`O_EXCL` claim，不能覆盖或二次 claim。

## 4. 两卡并行执行 plan 中精确 extractor argv

plan 中的 argv 已从获胜 lock 的 encoder provenance 生成，必须原样执行：

```bash
"$PY" - "$PLAN" 0 <<'PY' \
  >"$FEATURE_DIR/gpu0_extract.log" 2>&1 &
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    plan = json.load(stream)
argv = plan["shards"][int(sys.argv[2])]["extract_argv"]
if "--sealed-test" not in argv or argv[argv.index("--split") + 1] != "test":
    raise SystemExit("plan lost sealed extraction authorization")
os.execv(argv[0], argv)
PY
PID0=$!

"$PY" - "$PLAN" 1 <<'PY' \
  >"$FEATURE_DIR/gpu1_extract.log" 2>&1 &
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    plan = json.load(stream)
argv = plan["shards"][int(sys.argv[2])]["extract_argv"]
if "--sealed-test" not in argv or argv[argv.index("--split") + 1] != "test":
    raise SystemExit("plan lost sealed extraction authorization")
os.execv(argv[0], argv)
PY
PID1=$!

FAILED=0
wait "$PID0" || FAILED=1
wait "$PID1" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "A sealed feature extractor failed; do not merge." >&2
  exit 1
fi
```

这里只提特征，不计算 probability、F1 或 threshold。

## 5. 唯一一次 feature-only merge

```bash
MERGED="$FEATURE_DIR/legacy360_sealed_test_merged.features.pt"

"$PY" "$ROOT/legacy360_dual_axis/sealed_test_feature_shards.py" \
  merge-cache \
  --input-cache "$FEATURE_DIR/legacy360_sealed_test_gpu0.features.pt" \
  --input-cache "$FEATURE_DIR/legacy360_sealed_test_gpu1.features.pt" \
  --plan "$PLAN" \
  --selection-lock "$LOCK" \
  --checkpoint "$CKPT" \
  --master-selection-receipt "$RECEIPT" \
  --output-cache "$MERGED" \
  --sealed-test
```

merge 只接受两个 plan-bound、`split=test`、authorized cache，要求相同
encoder/base/tensor contract 和原 manifest 的 exact full-row union，并拒绝
overwrite/duplicate use。

## 6. family-aware、固定 threshold 的唯一评估

只能通过 cross-family wrapper 调用 family evaluator。wrapper 重新验证
master receipt、winner lock/checkpoint/model/threshold/encoder、merge receipt
和 cache SHA，随后在 master receipt 旁以 `O_EXCL` 创建全局 eval-use claim。
因此即使换一个 output directory，也不能进行第二次评估。

```bash
EVAL_DIR=/diniuvol/yuyao/legacy360_locked_eval_once
MERGE_RECEIPT="$MERGED.receipt.json"

"$PY" "$ROOT/legacy360_dual_axis/evaluate_promoted_locked.py" \
  --master-selection-receipt "$RECEIPT" \
  --merge-receipt "$MERGE_RECEIPT" \
  --test-cache "$MERGED" \
  --output-dir "$EVAL_DIR" \
  --sealed-test \
  --eval-batch-size 4096 \
  --device cuda:0
```

Gated evaluator 额外要求 checkpoint、lock 与 cache 的 encoder/base contract
三方一致；compact axial 的旧 checkpoint 不内嵌 encoder，因此由 checkpoint
SHA-bound selection lock 的 encoder 与 cache 精确匹配。wrapper 只 dispatch
receipt 指定的 family，并验证 `sealed_test_evaluations=1` 与
`test_threshold_search_performed=false` 后写出 SHA-bound promotion receipt。
