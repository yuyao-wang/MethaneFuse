# L89 外层评估 exactly-once 协议

日期：2026-07-28 UTC  
当前状态：**只有不可授权的 development template；没有 final lock；没有
运行外层数据**

## 结论先行

默认候选冻结为：

```text
P5 methane-response expert
          0.5 logit
              \
               fixed sum -> sigmoid -> frozen threshold 0.2426486375
              /
D1 acquisition-aware transient expert
          0.5 logit
```

D1 是 seed `20260727/20260728/20260729` 三个 checkpoint 的等权 logit
ensemble。默认 lock 不包含 patch。readout-zero patch 虽有互补 point
estimate，但相对 P5+D1 的 AP 和 macro-F1 cluster-bootstrap 区间均跨
零，只保留 optional component 接口。

所有 `event_balanced_*` 数值都应称为 **event-balanced row metric**：
仍在 row label/probability 上计算，只是每个 canonical event 的总 sample
weight 相等。它们不是“每个 event 一个预测”的 event-aggregated metric。
paired bootstrap 才是以 canonical event 为 cluster 的重采样。

## 已冻结但不可执行的 template

- manifest:
  `research/tempo_20260728/l89_lock_template_v4/LOCK_MANIFEST.json`
- manifest SHA-256:
  `45cc89c982f882dbefe0803f724b3f773ba9c252073d9533b98dc12a33bd2cd5`
- protocol SHA-256:
  `c2c0e37df3a9acd1f2e1ba1dc9aac6a59d7953ae37b7bf12ddc9f708a4cf729a`
- status:
  `development_template_not_authorizable`
- producer spec:
  `research/tempo_20260728/l89_producer_spec_v1.json`
- producer spec SHA-256:
  `64c713f8892adfd4edcd63c610d927e0748c4ea8f5d39e1a74bc6e181022f58d`
- producer protocol SHA-256:
  `78a6abd3811b0534ebf18f588316c3217a47a4b3f5d2d97516d876017ed57b22`
- locked target-literal digest:
  `277e61432437cb60dd7a9d5635f85b7efd604994dce35b3bcc10bf45d312bce7`

template 已锁定：

- evaluator、global model runner、cache/head support 源码 SHA；
- producer、per-time CLS extractor、P5 sidecar builder 源码 SHA；
- Panopticon 权重与 P5 sidecar checkpoint SHA；
- P0/P5 head config 与 checkpoint SHA；
- 三个 D1 checkpoint、seed 顺序与 run config SHA；
- 6 个 role/path/time columns；
- 7 个 band、channel ID、训练集 normalization；
- image size、validity、duplicate-time 和 zero-invalid 规则；
- fixed fusion weight、threshold 和 metric policy；
- producer 输出 cache 的 identity/alignment 合同。

它明确不能授权 producer 或 evaluator。即使提供运行 token，
`validate_lock(require_final=True)` 也会在触碰任何外层路径之前拒绝。

## 安全 development replay

template 在安全的 event-disjoint development cache 上完成 CPU replay：

| 指标 | replay |
|---|---:|
| event-balanced row AP | 0.772942804 |
| event-balanced row AUC | 0.853505065 |
| event-balanced row macro-F1 | 0.777160447 |
| event-balanced row positive-F1 | 0.724372566 |

AP/AUC 与开发时记录的绝对误差分别为 `3.55e-7/6.37e-8`。固定阈值附近
少量 float32 CPU/GPU ulp 差异令 F1 误差最大为 `8.43e-5`，低于预先写入
runner 的 `1e-4` replay tolerance。receipt 位于：

`research/tempo_20260728/l89_lock_template_v4/dev_replay_v1/RECEIPT.json`

## Producer 的严格输入与输出合同

producer 只在 final lock 和显式 one-time token 均验证后运行：

1. 验证 producer token；
2. 验证 manifest 是 final status，并重算所有 locked artifact SHA；
3. 验证调用方给出的 exact manifest SHA；
4. 验证 producer spec 是 manifest 中绑定的同一个文件和 SHA；
5. 验证 `--outer-csv` 的普通字符串与 spec 中锁定 literal 完全相同；
6. 验证输出目录名显式带 `outer/test/sealed/holdout` 标记；
7. 到此才第一次把 CSV literal 构造成 `Path` 并访问文件系统。

producer 随后：

- 对源 CSV 做一次顺序 pass，同时 SHA-256 并 stage 到本地；
- 后续 pandas 只读 staged CSV；
- 用冻结 Panopticon 一次性提取 `[N,6,768]` base CLS；
- P5 sidecar 只从 base CLS 派生 `[N,6,1536]`，不再读图像；
- 校验 ordered ID/plume/event/label、mask/gap/quality 完全一致；
- 校验 P5 前 768 维与 base byte-exact；
- 把两个 cache SHA 和 feature SHA 写入 producer receipt；
- 使用 `O_EXCL` intent，禁止覆盖和同一 output intent 重跑。

正式 producer token 为：

`PRODUCE_EXACTLY_ONCE_LOCKED_L89_OUTER_CACHES`

## Evaluator 的 exactly-once 合同

evaluator 不接受 raw CSV，只接受 producer 形成的两个显式 outer cache。
它同样先验证 token、final manifest、locked artifact SHA 和调用方给出的
manifest SHA，之后才解析或 `stat` cache 路径。

执行时每个 source cache 只顺序 stage 一次并同时计算 SHA；后续
`torch.load` 只读本地 staging。模型、seed、fusion weight 和 threshold
全部来自 manifest，禁止：

- outer threshold fitting；
- outer checkpoint/epoch selection；
- outer fusion-weight fitting；
- calibration、subgroup selection 或重复运行。

正式 evaluator token 为：

`RUN_EXACTLY_ONCE_LOCKED_L89_OUTER_EVALUATION`

## 状态机

| 阶段 | 当前状态 | 是否触碰外层数据 |
|---|---|---:|
| producer spec from safe dev | 完成 | 否 |
| non-authorizable lock template | 完成 | 否 |
| safe development replay | 完成 | 否 |
| 决定 final candidate | 默认 P5+D1；尚未 final-lock | 否 |
| 构建 final manifest | **未执行** | 否 |
| produce outer caches once | **未执行** | 否 |
| evaluate outer caches once | **未执行** | 否 |

截至本文写入，没有读取、列目录、`stat` 或运行任何外层 CSV/cache。

## 命令合同

以下三条是已经运行过的安全命令：

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_cache_producer.py build-spec

/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_locked_evaluator.py build-lock \
  --outer-producer-spec \
  research/tempo_20260728/l89_producer_spec_v1.json

/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_locked_evaluator.py dry-run \
  --lock-manifest \
  research/tempo_20260728/l89_lock_template_v4/LOCK_MANIFEST.json \
  --output-dir \
  research/tempo_20260728/l89_lock_template_v4/dev_replay_v1
```

下面是未来 finalization 的合同示例，**本轮没有执行**：

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_locked_evaluator.py build-lock \
  --outer-producer-spec \
  research/tempo_20260728/l89_producer_spec_v1.json \
  --lock-status final \
  --finalize-token FREEZE_FINAL_L89_OUTER_CANDIDATE \
  --output-dir research/tempo_20260728/l89_final_lock_v1
```

只有上述 final manifest 已人工复核且 SHA 被单独授权后，才能各执行一次
下列 producer/evaluator；尖括号参数必须替换为 manifest/spec 中锁定的
精确值，不能凭文档猜测：

```bash
# DO NOT RUN without a separately approved final-manifest SHA.
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_cache_producer.py produce-once \
  --lock-manifest research/tempo_20260728/l89_final_lock_v1/LOCK_MANIFEST.json \
  --authorized-lock-sha256 <exact-final-manifest-sha256> \
  --confirm PRODUCE_EXACTLY_ONCE_LOCKED_L89_OUTER_CACHES \
  --producer-spec research/tempo_20260728/l89_producer_spec_v1.json \
  --outer-csv <exact-literal-bound-in-producer-spec> \
  --output-dir <explicitly-outer-marked-output-directory> \
  --local-image-cache-dir <local-diniuvol-image-cache> \
  --device cuda:0

# DO NOT RUN without the same separately approved final-manifest SHA.
/home/yuyao/miniconda3/envs/panopticon/bin/python \
  research/tempo_20260728/tempo_l89_outer_locked_evaluator.py evaluate-once \
  --lock-manifest research/tempo_20260728/l89_final_lock_v1/LOCK_MANIFEST.json \
  --authorized-lock-sha256 <exact-final-manifest-sha256> \
  --confirm RUN_EXACTLY_ONCE_LOCKED_L89_OUTER_EVALUATION \
  --base-cache <producer-outer-base-cache> \
  --p5-cache <producer-outer-p5-cache> \
  --staging-dir <explicitly-outer-marked-staging-directory> \
  --output-dir <explicitly-outer-marked-result-directory>
```

## Benchmark qualification

即使将来按该协议 one-shot 运行，当前绑定的 hard-filtered outer cohort
仍然是 model-conditioned cohort：其构造受旧模型错误影响。因此结果只能
作为严格锁定的工程外测，不能升级为 clean confirmatory test 或 SOTA
证据。

一个更干净的 full-balanced train-only replicate 已完成 readiness
审计：新 inner split event overlap 为零，但现有六角色本地覆盖只有
`43.332%`。512-file stat/64-file copy 的 bounded feasibility audit
估计全量 staging 为 `148.56 GiB/84.43 min`；本轮没有启动全量 staging
或 GPU extraction。

## CPU contracts

```bash
/home/yuyao/miniconda3/envs/panopticon/bin/python -m unittest \
  research.tempo_20260728.test_tempo_l89_outer_locked_evaluator_cpu \
  research.tempo_20260728.test_tempo_l89_outer_cache_producer_cpu -v
```

当前通过 `13/13`。覆盖：

- wrong token 在任何 outer path access 前拒绝；
- wrong lock SHA 在任何 outer path access 前拒绝；
- target literal mismatch 在任何 outer path access 前拒绝；
- template lock 不能授权；
- build-spec 不解析 target literal；
- six-role extraction namespace 完整冻结；
- fixed logit fusion、cache identity/alignment、single-pass staging。
