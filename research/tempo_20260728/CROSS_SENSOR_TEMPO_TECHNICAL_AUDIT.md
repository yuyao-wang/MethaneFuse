# TEMPO 技术架构、跨传感器证据与数据就绪审计

日期：2026-07-28 UTC  
范围：开发集实验；禁止读取 `test`、`sealed`、`holdout`

## 一句话结论

当前实验不支持“大规模不可见光预训练”，也不支持“把原来的
cross-attention 换一种 attention”作为论文主线。实验真正支持的是一个
**两尺度、两流、晚期证据融合**的监督学习架构：

1. 保留已经能识别甲烷光谱/外观的 response expert；
2. 另训一个显式比较当前时刻与历史时刻的 global transient expert；
3. 在有可靠 patch cache 的传感器上，再增加 local onset expert；
4. 各分支先独立形成分类证据，最后固定权重融合 logits；
5. 多传感器场景先在每个 sensor 内形成 appearance-motion evidence，再按
   可用性融合或路由，避免把不同比例尺的原始 token 过早混在一起。

这不是一个新的 attention 原理。技术增量应表述为：

> 在稀疏、不规则、缺测的甲烷 EO 序列上，把单时刻甲烷响应、全局瞬态变化
> 和局部 onset 分开建模，再进行 availability-aware late evidence
> consensus。

目前最干净的支持来自 L89；legacy360 提供了超过 0.90 F1 的工程结果，
但不是干净的 SOTA 证据；EMIT 只支持三时刻且强 role-only baseline
仍然更好；S5P 近似实验否决；S2 现有 cache 无法支持同协议。

## 架构到底是什么

### 1. Response / appearance 分支

输入当前及可用历史的 frozen Panopticon 特征，使用现有的强分类头产生
`l_response`。现有 role-only head 本身就是两层
`t0-query / all-valid-role-KV` cross-attention；因此论文不能声称
“第一次用 t0 做 query”。

这个分支的任务是保留已有模型已经学到的甲烷相关光谱响应和稳定外观，
而不是承担显式变化检测。

### 2. Global transient 分支

对每个有效历史时刻 \(h\)，从 frozen CLS 构造：

```text
signed motion    = z(t0) - z(h)
motion magnitude = |z(t0) - z(h)|
```

L89 的 D1 使用 role、真实时间差、历史质量、当前/历史相似度和距离做
masked temporal gating，得到全局 transient token：

```text
d_h       = MLP([signed_motion_h, motion_magnitude_h])
w_h       = masked_softmax(g(role_h, delta_days_h, quality_h,
                             cosine_h, distance_h))
transient = Σ_h w_h d_h
l_global  = frozen_P0_logit + bounded_residual(transient)
```

`bounded_residual` 最后一层为零初始化，因此 epoch 0 精确等于 frozen P0。
该分支独立训练，然后与更强的 response expert 做固定 equal-logit
融合；直接把同一 residual 训练在 P5 上的实验反而失败。

### 3. Patch-local onset 分支

这个分支不是 CLS attention。当前 patch 在每个历史时刻的 3×3
邻域内寻找低秩 soft match：

```text
h(t,p) = Σ_q softmax_q(cos(Wq z0[p], Wk zt[q]) / temperature) Wv zt[q]
```

然后分别计算：

```text
current-history change = |Wv z0[p] - h(t,p)|
history normality      = mean_{i<j} |h(i,p) - h(j,p)|
local onset            = learned function(change, normality,
                                          positive excess, ratio)
```

最高 10% patch onset 分数经 MIL pooling 形成 `l_patch`，再作为
zero-init bounded residual 加在 exact P0 上。该分支解决的是：

- 羽流只占少量 patch，global CLS 容易稀释；
- 地理配准存在小偏移，不能只做同坐标相减；
- 普通背景变化要由 history-history normality 显式描述。

### 4. 最终分类证据

当前有实验支持的形式是固定 late fusion，而不是小开发集上再学习一个
liaison：

```text
l_final = mean_logit(l_response, l_global [, l_patch])
p_final = sigmoid(l_final)
```

L89 五折 OOF reliability liaison 的 AP 低于固定 equal fusion；
legacy360 的 NormWear-style masked set-attention liaison 也没有超过简单
融合。因此 liaison 只能作为被否决的控制，不能放入最终架构。

### 5. 多传感器如何接入

对于 sensor \(s\)，先在 sensor 内形成：

```text
appearance_s = A_s(t0)
motion_s     = M_s(t0 - h)
magnitude_s  = U_s(|t0 - h|)
evidence_s   = appearance_s * sigmoid(G_s(motion_s, magnitude_s))
               + motion_s
```

缺失历史在 sensor 内 mask；缺失 sensor 在 sensor 间 mask。legacy360
实验表明，直接用 learned sensor liaison 没有优势，当前更稳妥的是：

- 全局 R4：对可用 sensor evidence 做 masked mean；
- 冻结路由：恰好一个当前 sensor 时用 R4，否则回退 P0。

这说明 TransientQuery 本身并没有“解决多 sensor 融合”。它解决的是
每个 sensor 内的 temporal evidence；跨 sensor 的技术问题应由
availability-aware evidence routing 处理。

## 四类模块不要混为一谈

| 名称 | 输入 | 解决的问题 | 当前结论 |
|---|---|---|---|
| role-only 强基线 | per-time CLS；t0 query、role KV | 常规时序聚合和分类 | EMIT 上仍强于 R4；必须保留为主基线 |
| L89 global D1 | 6 时刻 CLS delta + acquisition gate | irregular global transient | 有显著 history 信号；与 P5 固定晚融合有效 |
| L89 patch motion | 6 时刻 16×16 patch token | 稀疏羽流、局部错位、背景变化 | 单 seed 很强，但 scalar 四 seed 未超过 P5+D1；不进入当前 global candidate |
| legacy R4 | 4 sensor × 3 role CLS | per-sensor appearance-motion + availability | F1 超过 0.90；但属于旧 legacy 工程证据 |

## 实验结果

### A. L89：当前最强的干净六时刻证据

数据为 event-disjoint train/dev，六个角色为
`t0/prev1/prev2/prev3/seasonal/year`。没有读取任何外部测试数据。

| 系统 | Event-balanced row AP | Event-balanced row AUC | Event-balanced row macro-F1 | Event-balanced row positive-F1 |
|---|---:|---:|---:|---:|
| exact P0 | 0.749149 | 0.842789 | 0.761262 | 0.691468 |
| fixed P5 | 0.765148 | 0.848388 | 0.772047 | 0.713650 |
| D1 三 seed mean | 0.761963 | 0.848658 | 0.777726 | 0.723346 |
| fixed P5+D1 三 seed mean | 0.769946 | 0.852054 | 0.776842 | 0.720921 |
| fixed P5+D1 logit ensemble | **0.772943** | **0.853505** | **0.777219** | **0.724457** |

P5+D1 ensemble 相对 P5 的 canonical-event paired bootstrap：

- AP `+0.007795`，95% CI `[+0.001459,+0.015550]`；
- AUC `+0.005117`，95% CI `[+0.000609,+0.009796]`；
- positive-F1 `+0.010807`，95% CI `[+0.001963,+0.020793]`；
- macro-F1 `+0.005172`，CI 仍跨 0。

History shuffle 使 D1 的 AP 平均下降 `0.014475`、macro-F1 下降
`0.012268`，说明 temporal branch 确实使用历史。

直接在 P5 上训练同一 residual 没有通过 gate：epoch 1 AP
`0.763889 < 0.765148`，epoch 2 继续下降。支持的是独立 evidence
diversity 和 late fusion，不是给 P5 再叠一层。

Patch-local 最终采用稳定的 128-D readout-zero 初始化。四 seed
equal-logit ensemble 为 AP `0.770319`、AUC `0.850673`、macro-F1
`0.780399`、positive-F1 `0.734171`。固定 equal-thirds
P5+D1+patch 为 AP `0.773405`、macro-F1 `0.778265`；相对 P5+D1 的
AP delta 仅 `+0.000462`，95% CI
`[-0.003339,+0.004923]`，macro-F1 CI 也跨 0。

看到结果后提出的 two-scale `.25/.25/.50` 规则为 AP `0.772921`、
macro-F1 `0.780566`；它相对 P5+D1 的 macro-F1 delta 为
`+0.003346`，CI `[-0.002627,+0.009405]`。因此 patch 支持局部互补性，
但没有确认优于 P5+D1；默认外层候选仍应锁 P5+D1，patch 只保留
optional component 接口。

### B. Legacy360：超过 0.90 F1 的工程证据

输入为 `113,843` train-core 和 `12,621` dev 行、
`4 sensors × 3 roles × 768` frozen features；train/dev event 和 plume
均零重叠。旧 encoder/checkpoint 有历史选择和 identity 限制，因此不能
作为 clean SOTA。

| 系统 | Binary F1 | Macro-F1 | AP | AUC |
|---|---:|---:|---:|---:|
| promoted P0 | 0.905424 | 0.894916 | 0.960035 | 0.960381 |
| R4 四 seed equal-logit | 0.910235 | 0.900769 | **0.961690** | **0.962527** |
| one-sensor R4, otherwise P0 | **0.910745** | **0.901134** | 0.961191 | 0.962071 |

冻结 availability rule 相对 P0 的 10k event bootstrap F1 delta 为
`+0.005275`，95% CI `[+0.002109,+0.008604]`。R4 history shuffle
使 F1 下降 `0.003638`、AP 下降 `0.002098`。

严格 active-capacity 控制中 R4-add 低于 R4；匹配参数量的
`t0-query/history-KV` attention 也低于 R4。但 R4 对 plain raw-delta
R1 的优势极小，因此可以主张 appearance+motion decomposition，不能把
multiplicative excitation 单独夸成已证实的新原理。

### C. EMIT：历史有信号，但 R4 不能替代强 role-only

EMIT cache 是 event-disjoint train/dev frozen CLS，但只有
`t0/prev1/seasonal` 三个角色，不是六时刻实验。

| 系统 | Event-balanced row AP | Event-balanced row AUC | Event-balanced row macro-F1 | Event-balanced row positive-F1 |
|---|---:|---:|---:|---:|
| exact t0 P0 | 0.750368 | 0.835329 | 0.757995 | 0.685169 |
| R4 四 seed ensemble | 0.765439 | 0.847151 | 0.769199 | 0.693404 |
| role-only 三 seed ensemble | **0.784186** | **0.858284** | 0.777879 | 0.703098 |
| fixed role-only+R4 | 0.784821 | **0.860149** | **0.781784** | **0.709314** |

R4 相对弱 P0 的 AP/AUC bootstrap 区间为正，并且 history shuffle
使 AP/AUC 分别下降 `0.015808/0.012188`。但是 R4 明显低于强
role-only。

固定 0.5 融合相对 role-only 的 point delta 为：

- AP `+0.000635`；
- AUC `+0.001865`；
- macro-F1 `+0.003905`；
- positive-F1 `+0.006216`。

四项 canonical-event bootstrap CI 全部跨 0。结论是：R4 是一个可能
互补但不稳定的 expert，不能宣称 EMIT 上优于 TransientQuery。

### D. S5P：否决

S5P train/dev 是 event-disjoint 六角色，但每个 `3×3` frame 是对已经
上采样到 `224×224` 的产品做 finite-mask adaptive average；它既不是真正
native field，也不是 learned CLS。

| 系统 | Unweighted row AP | Unweighted row AUC | Event-balanced row AP | Event-balanced row AUC |
|---|---:|---:|---:|---:|
| exact P0 | 0.576047 | 0.529720 | 0.577098 | 0.524090 |
| raw delta, seed 20260728 | **0.584436** | **0.538068** | **0.581257** | **0.526015** |
| R4, seed 20260728 | 0.581062 | 0.535119 | 0.578915 | 0.525756 |

旧 role-only 三 seed row mean 为 AP `≈0.577494`、AUC `≈0.539372`。
raw delta 和 R4 都没有同时超过 AP/AUC gate；绝对 AUC 接近随机，R4
也低于 raw delta。因此停止更多 seed，不推广 S5P 结果。

另一个限制是 dev 的 679 个 canonical event 全都至少含一个正样本，
所以这里无法评估 all-negative-event false alarms。

## Safe train/dev 与历史 cache 审计

| 数据 | Safe train/dev | Event overlap | 历史表示 | 可做同协议六时刻？ | 证据等级 |
|---|---|---:|---|---|---|
| L89 | 是；10,033/9,614 行，723/124 events | 0 | `[N,6,768]` frozen CLS；另有 `[N,6,256,128]` projected patches | **是** | A |
| EMIT | 是；11,688/5,201 行，900/129 events | 0 | `[N,3,768]` frozen CLS | 否，只有 3 role | B |
| S5P | 是；18,772/3,805 行，3,759/679 events | 0 | `[N,6,3,3]` 近似 pooled grid | 仅近似，不可作为统一 CLS 协议 | C |
| S2 | 只有可审计的 safe train；缺少可用 inner-dev per-time cache | 未建立同协议 pair | 现有 UniverSat shard 是 `[N,1536]`，六时刻已折成一个向量 | **否** | Blocked |
| legacy360 四传感器 | 是；113,843/12,621 行，2,554/291 events | 0 | `[N,4,3,768]` | 三时刻工程协议；非新四数据集统一协议 | B- |

### L89 cache

- train SHA:
  `00a7523bab76db415624cb0a40f198605611bbb42b6d30205d07193fd52fca90`
- dev SHA:
  `a47611f25e3670af31199b9b99a07e5ca1e0b9ba67b98133411018769d8993ca`
- 有效 role visits：train `56,059`，dev `54,135`
- role：`t0/prev1/prev2/prev3/seasonal/year`

### EMIT cache

- train SHA:
  `ebcfa12803d4c91827a2578321c61c28b236c99406f8e8d11037622d9fa681ab`
- dev SHA:
  `d04302719c0ddb7fc1b610a7fe1eb6d36d8e7c712b1486825be5b10f75225eb2`
- 有效 role visits：train `31,648`，dev `13,816`
- role：`t0/prev1/seasonal`

### S5P cache

- train SHA:
  `e57abc70cab5b40a84a58453b2fd9cc85e719942a3f7aede0127bcda18de7483`
- dev SHA:
  `ceb933b2ea8fc0761d0629d21f61d4489b04bd4940cfa5f8cf724f6fab19f284`
- 有效 role visits：train `112,632`，dev `22,830`
- role：`t0/prev1/prev2/prev3/seasonal/year`

### S2 blocker

原始 safe train CSV 有 `109,212` 行、`1,376` events、`3,467` plumes
和 `655,272` 个互不重复的 role paths。实际 UniverSat cache 使用的
staged train 只有 `18,747` 行、`1,180` events、`2,695` plumes。

该 cache manifest 虽声明六个 active path/time columns，但每个 shard
只保存：

```text
features [N,1536], labels, sample_ids, source_indices
```

没有 `[N,6,D]`，所以无法恢复每个时刻，也无法构造
`t0-history` delta、history shuffle 或 R4。现有旧 evaluation artifact
的指标被隔离，不纳入任何结论。要做真正同协议 S2，必须从 safe
train/inner-dev 的六时刻图像重新提取 per-time features；在建立新的
event-disjoint inner-dev 之前不应启动训练。

## 最终技术判断

### 可以作为主架构保留

1. frozen methane-response / appearance expert；
2. independent irregular-time global transient expert；
3. patch-local matching + learned history normality + top-k MIL；
4. zero-init bounded residual，保证不劣化的 epoch-0 fallback；
5. fixed late-logit consensus；
6. sensor-native evidence 和 availability-aware routing；
7. event-balanced row metrics、history shuffle 和 paired
   canonical-event-cluster bootstrap。

### 已被实验否决或证据不足

- 大规模 multisensor pretraining；
- 把 t0-query cross-attention 当作新意；
- hard linear background-trend subtraction；
- 直接把 transient residual 接在强 P5 上联合训练；
- appearance-free transient-only classifier；
- learned NormWear liaison；
- 在小 dev 上学习 fusion weight/reliability gate；
- 把近似 S5P 或 collapsed S2 当作统一六时刻证据。

### 论文还缺什么

当前最需要的不是继续搜 attention，而是：

1. 完成 readout-zero patch 控制；只有它稳定超过 P5+D1 才能进入 lock；
2. 为 S2/EMIT/S5P 重建同一数据定义的 per-time feature cache；
3. 每个 sensor 都使用 event-disjoint inner-dev 完成方法冻结；
4. 最后只运行一次锁定的 outer evaluation。

如果 patch 多 seed 不稳定，论文主线应退回已经有统计支持的
`response expert + acquisition-aware global transient expert +
fixed late fusion`，而不是用单 seed 结果包装更复杂的架构。

## 主要产物

- L89 global：
  `research/tempo_20260728/TEMPO_L89_GLOBAL_EXPERIMENT_LOG.md`
- L89 patch 协议：
  `research/tempo_20260728/PATCH_PROTOCOL.md`
- legacy360：
  `research/tempo_20260728/TEMPO_LEGACY360_GLOBAL_RESULTS.md`
- EMIT 四 seed：
  `/diniuvol/yuyao/methanefuse_tempo_20260728/cross_sensor_r4_v1/emit_3time_4seed_ensemble/aggregate.json`
- EMIT 强基线融合审计：
  `/diniuvol/yuyao/methanefuse_tempo_20260728/cross_sensor_r4_v1/emit_role_r4_fixed_fusion_v1/audit.json`
- S5P 否决实验：
  `/diniuvol/yuyao/methanefuse_tempo_20260728/cross_sensor_r4_v1/s5p_approx_native_seed20260728_v2/aggregate.json`

本轮新增 runner 均拒绝 `test/sealed/holdout` 路径；EMIT/S5P
zero-init、P0 replay 和 history-shuffle 的 CPU contracts 已通过。
