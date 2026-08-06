# TEMPO 12 小时实验迭代：最终技术与结果报告

日期：2026-07-28 UTC  
状态：开发集架构已冻结；未读取新的 `test/sealed/holdout`

## 1. 最终结论

这轮实验没有支持“大规模不可见光预训练”，也没有支持“换一种
cross-attention”作为下一篇论文的核心。当前最强且最可复现的技术方案是：

> **冻结已有甲烷 response/appearance expert；另训一个显式比较
> `t0-history` 的 acquisition-aware transient expert；两者独立形成证据，
> 最后固定 late-logit consensus。多传感器时先在每个 sensor 内形成
> appearance/transient evidence，再按 availability mask 或冻结 router
> 融合。**

这个方案暂命名为 **TEMPO（Temporal Evidence Modulation for Partial
Observations）**。它借用视频 two-stream / motion-excitation 的思想，但不把
90/360 天、不规则、缺测的 EO 重访伪装成普通视频帧。

当前默认论文候选只保留：

```text
Frozen Panopticon per-visit encoder
├── frozen P5 response / appearance expert
└── D1 acquisition-aware global transient expert
    ├── signed t0-history delta
    ├── absolute delta magnitude
    ├── role + real gap + quality + relevance gate
    └── zero-init residual around exact P0

fixed 0.5/0.5 late-logit consensus
→ per-sensor evidence
→ availability-aware sensor fusion / frozen router
→ classification
```

Patch-local onset 有真实历史信号，但没有通过预声明的多 seed 和 false-positive
晋级门槛，因此不进入默认模型。硬分 Fast/Slow 的视频对照也没有通过 primary
AP 门槛。

## 2. 与上一篇 MethaneFuse 的技术关系

上一篇工作的主问题是 **sensor availability**：

```text
不同 sensor 缺失
→ sensor-native encoding
→ masked cross-sensor fusion
```

TEMPO 新处理的是 **同一 sensor 内的 irregular transient evidence**：

```text
同一地点的六次不规则观测
→ 保留当前 response/appearance
→ 显式计算 t0-history 方向与幅度
→ 用真实 gap、质量、缺失/重复 mask 选择历史证据
→ decision-level late consensus
```

因此它不是把旧 attention 换一个公式。两个轴的职责是：

- time axis：把单时刻 response 与跨时刻 transient 分开；
- sensor axis：在各 sensor 已形成证据后处理 availability。

当前已验证的是 L89 的 `1 sensor × 6 visits`，以及 legacy360 的
`4 sensors × 3 roles`。尚未完成共同 crop、共同事件切分下的
`4 sensors × 6 visits`，所以不能宣称统一四传感器六时刻模型已经成立。

## 3. 架构为何不是普通 cross-attention

现有 P5 / role-only head 已经使用 `t0-query / role-KV` cross-attention，
所以这不是 novelty。D1 的关键区别是先显式构造：

\[
z_t=W_z\operatorname{LN}(x_t),
\qquad
\delta_h=z_0-z_h,
\qquad
a_h=|\delta_h|,
\]

\[
d_h=f_\Delta([\delta_h,a_h]).
\]

随后根据 role、真实 acquisition gap、有效像素质量、当前/历史 cosine
和距离得到 masked gate：

\[
\alpha_h=\operatorname{MaskedSoftmax}
\left(f_g(e_h^{role},e_h^{gap},q_h,\cos_h,\|z_0-z_h\|_2)\right),
\]

\[
v_T=\sum_h\alpha_h d_h.
\]

最后的 residual classifier 为零初始化：

\[
\ell_{D1}=\ell_{P0}+r_T,\qquad
\ell_{D1}^{epoch\,0}=\ell_{P0}.
\]

D1 不直接接在强 P5 上联合训练。训练完成后才做：

\[
\ell_{final}
=0.5\,\ell_{P5}
+0.5\,\operatorname{mean}_k(\ell_{D1,k}).
\]

直接在 P5 上训练相同 residual 的实验失败，并且 history-shuffle 效应接近
消失；这说明有效点是**独立错误的晚期共识**，不是给旧分类器再叠一层。

## 4. 最佳结果

### 4.1 Legacy360：达到 90 F1 的工程结果

协议：113,843 train-core 行、12,621 development 行，
`4 sensors × 3 roles × 768` frozen features；train/dev event 和 plume
均无重叠。

| 模型 | Binary F1 | Macro-F1 | AP | AUC |
|---|---:|---:|---:|---:|
| promoted P0 | 0.905424 | 0.894916 | 0.960035 | 0.960381 |
| R4 四 seed equal-logit | 0.910235 | 0.900769 | **0.961690** | **0.962527** |
| one-sensor R4，其他回退 P0 | **0.910745** | **0.901134** | 0.961190 | 0.962070 |

冻结 availability rule 相对 P0：

- binary F1 `+0.005275`；
- 10,000 次 canonical-event paired bootstrap 95% CI
  `[+0.002109,+0.008604]`；
- R4 history shuffle 后 F1 `-0.003638`、AP `-0.002098`。

收益主要集中在恰好一个当前 sensor 可用的样本：同一 sensor 的历史在这里
相当于“虚拟补充观测”。这就是 time-axis transient 如何帮助
missing-sensor 场景的实验答案。

该 `0.910745` 是旧 legacy development 工程结果，不是 clean test SOTA。
其 warm start 和开发选择历史与新确认协议不同，不能与上一篇论文
360 m test F1 `0.8377` 直接相减。

同一旧 360 m test cohort 上，另一条在本轮开始前已锁定的
compact scale-aware two-axis checkpoint 已执行一次 locked evaluation：

| Rows | Binary F1 | Macro-F1 | AP | AUC |
|---:|---:|---:|---:|---:|
| 31,564 | **0.864846** | 0.850630 | 0.933134 | 0.930424 |

其 threshold `0.3910266` 来自 development，test 未搜索 threshold，
`evaluation_count=1`，train/dev 与 test 的 plume overlap 为 0。相对上一篇
同一 360 m 工程基准 F1 `0.8377`，point difference 为 **`+0.0271`**。
但是旧 split 有 706 个 canonical events 与 source train 重合，因此这只能
写成“旧协议下不弱于上一篇并提高 2.71 F1 points”，不能写成 leakage-free
clean SOTA。

### 4.2 L89：六时刻机制证据

#### 4.2.1 早期独立 sidecar split

协议：event-disjoint train/dev，10,033/9,614 行、723/124 canonical
events；角色为 `t0/prev1/prev2/prev3/seasonal/year`。

| 模型 | Event-balanced row AP | AUC | Macro-F1 | Positive-F1 |
|---|---:|---:|---:|---:|
| exact P0 | 0.749149 | 0.842789 | 0.761262 | 0.691468 |
| frozen P5 | 0.765148 | 0.848388 | 0.772047 | 0.713650 |
| D1 三 seed logit ensemble | 0.771320 | **0.853668** | **0.779493** | **0.729455** |
| fixed P5+D1 | **0.772943** | 0.853505 | 0.777219 | 0.724457 |

P5+D1 相对 P5 的 canonical-event-cluster paired bootstrap：

- AP `+0.007795`，95% CI `[+0.001459,+0.015550]`；
- AUC `+0.005117`，95% CI `[+0.000609,+0.009796]`；
- positive-F1 `+0.010807`，95% CI `[+0.001963,+0.020793]`；
- macro-F1 `+0.005172`，95% CI `[-0.001493,+0.012264]`。

D1 branch-isolated history shuffle 在冻结 base logit 后使：

- AP `-0.014475`；
- macro-F1 `-0.012268`。

因此历史内容确实参与了预测。这里的指标是“每个 event 总权重相同的 row
metric”，不是把每个 event 聚合成一个 prediction。

#### 4.2.2 Fresh clean replicate

为排除复用旧 head/split 的影响，又从完整 L89 source-train 建了新的
event-disjoint inner split：

- train：18,682 rows / 672 events；
- development：4,677 rows / 176 events；
- event overlap：0；
- public Panopticon PTH 不变；
- fresh cache 的 read error 和 invalid `t0` 都是 0。

P0 在 epoch 2 达峰，epoch 3 已从 AP `0.752329` 回落到 `0.743421`，
所以没有继续硬跑。默认 D1 做三 seed 后先在 logit 空间平均，再与 P0 固定
0.5/0.5 late consensus：

| 模型 | Event-balanced AP | AUC | Macro-F1 | Positive-F1 | Negative-event FP mass |
|---|---:|---:|---:|---:|---:|
| fresh P0 | 0.752329 | 0.841389 | 0.765197 | 0.686078 | **5.196** |
| D1 三 seed logit mean | 0.748303 | 0.841200 | 0.770990 | **0.702631** | 6.621 |
| fixed P0+D1 | **0.752429** | **0.842090** | **0.772695** | 0.699453 | 5.571 |

相对 fresh P0 的 2,000 次 canonical-event paired bootstrap：

- AP `+0.000100`，95% CI `[-0.004696,+0.004955]`；
- AUC `+0.000701`，95% CI `[-0.001426,+0.002914]`；
- macro-F1 **`+0.007498`**，95% CI
  **`[+0.001151,+0.014078]`**；
- positive-F1 **`+0.013375`**，95% CI
  **`[+0.004366,+0.023155]`**；
- FP mass `+0.375`，95% CI `[-0.125,+0.875]`。

这组 fresh 结果把 claim 进一步收窄：当前可靠提升是 classification
operating point/F1，不是 ranking/AP。三 seed 平均后误报代价也从单 seed
的 `+1.2125` 降到 `+0.375`。

六个 D1 小配置、三个 null-event penalty 强度和 21 点 post-hoc fusion
curve 都没有证明需要 learned liaison 或精确拟合融合权重；固定 0.5 位于
平坦高值区。完整结果、SHA 和负对照见
`L89_CLEAN_FRESH_ITERATION_LOG.md`。

### 4.3 EMIT、S5P、S2

| Sensor | 当前证据 | 判定 |
|---|---|---|
| EMIT | role-only AP 0.784186；role-only+R4 AP 0.784821、macro-F1 0.781784 | point estimate 互补，但四项 bootstrap CI 均跨 0；不晋级 |
| S5P | approximate six-role R4 event-balanced AP 0.578915、AUC 0.525756 | 接近随机且低于 raw delta；否决 |
| S2 | 现有 cache 已把六时刻折成 `[N,1536]` | 无法恢复 per-time delta；同协议实验 blocked |

所以目前不能说“四个 sensor 的 F1 都提高”，更不能说跨 sensor SOTA。

## 5. 关键正负对照

| 对照 | 结果 | 技术含义 |
|---|---|---|
| t0-query/history-KV matched attention | 低于 R4 | 不是换 attention 就得到提升 |
| plain raw delta R1 | 与 R4 非常接近 | 可主张 appearance+motion；不能夸大 excitation 本身 |
| R4-add matched capacity | class balance 低于 R4 | multiplicative gate 是较好实现，但 primary F1 独立贡献仍有限 |
| NormWear-style learned liaison | 低于简单 masked/fixed fusion | 小数据上先形成 evidence、晚沟通更稳 |
| OOF learned reliability gate | AP 低于 fixed equal fusion | 不继续学习融合权重 |
| direct-P5 D1 residual | AP 下降 | 强 appearance surface 会吞掉 transient diversity |
| appearance-free D6 | history signal 很强但 AP 仅 0.688079 | transient 不能替代 methane appearance |
| hard global normality subtraction | 明显下降 | global CLS 上会把弱 methane signal 一起减掉 |
| SparseSlowFast D7 | P5+D7 AP 0.772062，未过 0.777794 gate；epoch 3 过拟合 | 不规则 EO 不适合固定 fast/slow role 分箱 |

SparseSlowFast 的参数量为 413,744，D1 为 412,801，只差 0.228%；
optimizer、batch、loss 和 seed 匹配。D7 history shuffle 使 AP
`-0.033508`，说明它确实使用历史，但 primary ranking 和 false alarms
没有胜过连续 acquisition gate。

## 6. Patch-local 路线为何没有进入最终模型

Patch head 使用：

- 16×16 frozen patch tokens；
- 当前 patch 对历史 3×3 邻域的 soft correspondence；
- current-history change；
- history-history normality；
- top-10% sparse MIL；
- zero-init readout。

Clean exact-mask-pattern history shuffle 保持 t0、P0、target gap、quality 和
availability 不变，只换跨 event 历史 token；9,614/9,614 行可匹配、
0 mask mismatch。四 seed ensemble 的：

- AP `-0.019023`；
- macro-F1 `-0.016876`。

因此局部分支确实使用历史。但是预声明 Gate-B 三 seed：

| Seed | AP | Macro-F1 |
|---:|---:|---:|
| 20260728 | 0.772408 | 0.779900 |
| 20260729 | 0.772789 | 0.778731 |
| 20260730 | 0.758851 | 0.777253 |
| mean | 0.768016 | 0.778628 |

Mean macro-F1 相对 P0 仅 `+0.017366`，低于 `+0.020` 晋级线；ensemble
在统一 macro threshold 的 negative-event FP mass 为 3.740368，P0 为
2.742803，也违反 FP safeguard。

容量控制同样没有挽救它：

- 非 nested 随机 256D 单 seed：AP 0.780817、macro-F1 0.781912；
  但前 128 维已改变，只能算子空间敏感性，并且 FP 仍更高；
- strict nested-256：前 128 维 bit-exact，只追加正交 128 维；
  AP 0.766638、macro-F1 0.774334，未过门槛。

因此随机 256D 的好数值不能被包装成“加维有效”或最终 winner。

## 7. 可以讲的 research story

### 问题

现有 methane classifier / EO FM 更擅长单时刻 appearance 或早期 token
融合。甲烷 plume 是弱、局部、短暂证据；EO revisit 又不规则、缺测、可能
重复，普通视频的固定帧率假设和通用 attention 都没有显式区分：

- methane-sensitive appearance；
- acquisition-induced nuisance；
- 真正的 transient change。

### 方法

TEMPO 把它们变成两个独立 expert：

1. response/appearance expert 保留现有 methane-sensitive decision
   surface；
2. transient expert 显式编码 signed/magnitude delta，并用真实时间差、
   role、质量和相似度选择历史；
3. 两个 expert 分开训练、只在 logit 层固定共识；
4. 多 sensor 时继续遵守 sensor-native evidence first，最后按
   availability route；
5. 局部 correspondence/onset 作为待外层验证的第二尺度，而不是强行加入。

### 实验回答

- 历史被打乱后 L89、legacy360、EMIT 都下降：不是参数量假象；
- 普通 cross-attention、hard SlowFast、learned NormWear liaison 都没有
  胜过更简单的 acquisition-aware/fixed evidence 结构；
- direct residual 与 appearance-free transient 都失败：证明必须“独立
  expert + late consensus”；
- legacy one-sensor 条件收益最大：时间历史可以在 sensor 缺失时补证据，
  但不是把异质 sensor token 直接混在一起。

### Claim 边界

单独的 temporal difference、motion excitation、two-stream、late fusion、
local correspondence、MIL 和 missing-modality routing 都不是新原理。
可防守的贡献只能是：

> 面向 methane EO 的特定组合：保留 methane-response expert，独立学习
> irregular-acquisition transient evidence，用 availability-aware fixed
> consensus 连接 time/sensor 两个轴，并用 event-disjoint、branch-isolated
> shuffle、matched-compute 和 all-negative-event alarms 验证。

当前这是有开发实验支持的组合型方法贡献，不是已经确认的 SOTA。

## 8. 尚缺的唯一关键结果

Legacy360 的新 R4/TEMPO exactly-once evaluator 已冻结：

- sealed payload：
  `/diniuvol/yuyao/methanefuse_two_axis_legacy360_v1/formal_v5/sealed_test_once/legacy360_sealed_test_merged.features.pt`
- lock SHA：
  `62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92`

开发 dry-run 已精确重放 P0、R4 四 seed与 availability router。该 payload
此前已被上面报告的 compact scale-aware checkpoint 读取并完成一次旧协议
locked evaluation；**尚未运行的是新 R4/TEMPO 候选**。只有得到对该 exact
path、exact lock 和“新候选唯一一次评测”的字面授权后，才能执行，无论
结果好坏均不得重跑。

L89 下一步真正有科学价值的是新的、未参与任何历史错误筛选的
event-disjoint outer split。现有 hard-event-filtered outer cohort 是
model-conditioned engineering check，不能充当 clean SOTA confirmation。

## 9. 主要可复现产物

- 中文架构说明：
  `research/tempo_20260728/TEMPO_TECHNICAL_ARCHITECTURE_CN.md`
- 独立结果审计：
  `research/tempo_20260728/FINAL_INDEPENDENT_AUDIT.md`
- L89 global：
  `research/tempo_20260728/TEMPO_L89_GLOBAL_EXPERIMENT_LOG.md`
- Fresh clean-L89：
  `research/tempo_20260728/L89_CLEAN_FRESH_ITERATION_LOG.md`
- L89 patch：
  `research/tempo_20260728/TEMPO_L89_PATCH_RESULTS.md`
- legacy360：
  `research/tempo_20260728/TEMPO_LEGACY360_GLOBAL_RESULTS.md`
- 跨 sensor 审计：
  `research/tempo_20260728/CROSS_SENSOR_TEMPO_TECHNICAL_AUDIT.md`
- locked evaluator protocol：
  `research/tempo_20260728/TEMPO_LOCKED_EVALUATOR_PROTOCOL.md`
- 核心实现：
  `research/tempo_20260728/tempo_l89_global.py`
  `research/tempo_20260728/tempo_l89_patch.py`
  `research/tempo_20260728/tempo_legacy360_global.py`
  `research/tempo_20260728/tempo_locked_evaluator.py`

所有本轮报告中的 L89/legacy development 结果都保留 checkpoint、prediction、
manifest 和 SHA；没有用最好 seed 替代预声明多 seed，也没有把看到结果后
搜索的权重伪装成 confirmatory result。
