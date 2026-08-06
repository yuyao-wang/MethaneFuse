# TEMPO 2026-07-28 独立结果审计

日期：2026-07-28 UTC  
审计角色：独立结果审计；未参与候选模型选择  
范围：legacy360、L89 global/patch、EMIT、S5P、S2 数据就绪状态，以及
MethaneFuse ICDM 论文和 `Upgraded_dataset/README.md` 中的历史基线  
数据边界：**本审计没有读取任何 test、sealed 或 holdout payload**

## 一、执行结论

当前没有一个结果能够被严谨地表述为“在四个 sensor 上取得新的 SOTA”。
可以保留的最强技术结论是：

> 在 L89 的六次不规则重访上，把现有 methane-response expert 与独立训练的
> acquisition-aware global transient expert 分开，并使用固定 late-logit
> consensus，在当前开发集的固定三-seed 集成中改善了
> event-balanced row ranking；patch-local onset 具有互补信号，但尚未通过
> 原始晋级协议。

不同实验的最终判定如下。

| 实验 | Promote / reject | 可以说什么 | 不能说什么 |
|---|---|---|---|
| L89 P5 + global D1 | **PROMOTE：只晋级到新的、无条件化 outer lock** | 六时刻 matched history 对 event-balanced row AP/AUC 有可重复贡献；固定 late fusion 优于 P5 | 已经是 clean SOTA；macro-F1 已显著提高；hard-event-filtered test 能作无偏确认 |
| L89 D7 SparseSlowFast | **REJECT：不扩 seed** | fast/slow history 确实被使用，F1 point 有 trade-off | 可以在看过结果后把 F1 改成主指标；硬时间 bin 优于连续 acquisition gate |
| L89 patch readout-zero | **REJECT formal promotion；探索性保留** | local matching、history normality 和 sparse MIL 学到了互补局部信号 | 已确认优于 P5+D1；已满足 patch protocol；可以用 post-hoc two-scale 当预注册结果 |
| legacy360 R4 / availability gate | **REJECT SOTA claim；保留工程候选** | 在旧开发协议上 F1 超过 0.90，matched history 有作用 | 可与论文 360 m 的 83.77 F1 直接比较；clean SOTA |
| EMIT R4 | **REJECT 主模型** | transient branch 相对弱 P0 有信号；与强 role-only 可能互补 | R4 优于强 temporal baseline；融合增益已显著 |
| S5P approximate-native | **REJECT** | 当前近似表示不足以验证方法 | R4 在 S5P 有效；这是真正 native/per-time 特征实验 |
| S2 | **BLOCKED** | 现有 cache 不支持同协议检验 | 已完成 S2 六时刻 TEMPO 实验 |

因此，论文技术主线最多应冻结为：

```text
response/appearance expert
        +
independent acquisition-aware global transient expert
        +
fixed late evidence consensus
        +
optional patch-local onset component（待外层验证）
```

不是新 attention、不是预训练框架，也不是已经验证的统一多传感器模型。

## 二、结果总表

### 2.1 L89：当前最有价值的六时刻证据

所有数值来自 event-disjoint train/dev；train 为 10,033 行、723 个
canonical events，dev 为 9,614 行、124 个 events。六个角色为
`t0/prev1/prev2/prev3/seasonal/year`。

| 系统 | Event-balanced row AP | AUC | Macro-F1 | Positive-F1 |
|---|---:|---:|---:|---:|
| exact P0 | 0.749149 | 0.842789 | 0.761262 | 0.691468 |
| fixed P5 | 0.765148 | 0.848388 | 0.772047 | 0.713650 |
| D1，三 seed mean | 0.761963 | 0.848658 | 0.777726 | 0.723346 |
| P5+D1，三次单-seed fusion mean | 0.769946 | 0.852054 | 0.776842 | 0.720921 |
| P5 + D1 三-seed logit ensemble | **0.772943** | **0.853505** | 0.777219 | 0.724457 |
| patch，formal Gate-B 三-seed ensemble | 0.770202 | 0.850207 | 0.778641 | 0.722103 |
| P5+D1+formal patch，固定 equal-thirds | **0.773516** | 0.853237 | 0.778612 | 0.727889 |
| formal patch two-scale `.25/.25/.50`，post-hoc | 0.773192 | 0.852762 | **0.779861** | **0.728902** |
| patch，另一个四-seed robustness ensemble | 0.770319 | 0.850673 | 0.780399 | 0.734171 |

P5+D1 三-seed ensemble 相对 P5 的 2,000 次 paired canonical-event
bootstrap：

| 指标 | Point delta | 95% CI | 审计判断 |
|---|---:|---:|---|
| AP | +0.007795 | [+0.001459, +0.015550] | 支持 |
| AUC | +0.005117 | [+0.000609, +0.009796] | 支持 |
| positive-F1 | +0.010807 | [+0.001963, +0.020793] | 支持，但受阈值选择影响 |
| macro-F1 | +0.005172 | [-0.001493, +0.012264] | 未确认 |

D1 的 base-fixed、cross-event history shuffle 使三 seed mean AP 下降
`0.014475`、macro-F1 下降 `0.012268`。代码路径保持原始 base logits，
只把完整 history bundle（feature、mask、time gap、quality）换成其他
event 的 donor，因此它是有效的 branch-isolated 因果诊断。

这个 shuffle 能证明“匹配的历史包有信息”，但不能单独证明贡献一定来自
feature content，因为 acquisition mask、time gap 和 quality 也一起被换掉。

### 2.1.1 SparseSlowFast 视频控制：有历史信号，但按预设主指标否决

D7 把 `prev1/prev2/prev3` 硬分为 fast branch，把
`seasonal/year` 硬分为 slow branch；两支使用独立 acquisition gates、
共享 signed/absolute delta encoder，最后按 availability 做固定归一化求和。
它是最接近视频 SlowFast 思想的参数匹配控制：

- D7 参数 `413,744`；
- D1 参数 `412,801`；
- 差异仅 `+943`，即 `+0.228%`。

| 系统 | Event-balanced AP | AUC | Macro-F1 | Positive-F1 | FP mass / FP events |
|---|---:|---:|---:|---:|---:|
| P5 | 0.765148 | 0.848388 | 0.772047 | 0.713650 | 3.871 / 16 |
| D7 SparseSlowFast，seed20260728 | 0.761416 | 0.849759 | 0.781642 | 0.722372 | 3.063 / 19 |
| fixed P5+D7 | 0.772062 | **0.854687** | **0.783478** | **0.733233** | 4.146 / 21 |
| fixed P5+D1 三-seed | **0.772943** | 0.853505 | 0.777219 | 0.724457 | 3.931 / 18 |

D7 在 epoch 2 达到最佳 AP `0.761416`，epoch 3 AP 降至 `0.744371`，
已出现明确过拟合并停止。固定模型/阈值 clean history shuffle 使 AP
下降 `0.033508`、macro-F1 下降 `0.030897`，说明它确实使用 fast/slow
历史。

但是预先写下的 fusion 晋级线是 AP `>=0.777794`；P5+D7 只有
`0.772062`，并且 FP mass/FP events 比 P5 和 P5+D1 都更差。因此按
primary ranking gate 否决，不扩 seed。P5+D7 的 macro/positive-F1 point
更高是一个值得记录的 trade-off，但在看到结果后把 primary metric 从 AP
换成 F1 会构成 metric cherry-pick。

这个控制支持的结论是：在当前稀疏、非规则采集上，硬编码 fast/slow role
bins 没有优于连续 acquisition-aware D1 的 primary ranking 结果；视频
思想应该保留“多时间尺度证据”概念，而不是照搬规则帧率分支。

### 2.2 L89 patch：有信号，但未通过原始晋级门槛

128-D readout-zero 原协议 Gate-B 三 seed 的结果为：

| Seed | Event-balanced row AP | Macro-F1 |
|---:|---:|---:|
| 20260728 | 0.772408 | 0.779900 |
| 20260729 | 0.772789 | 0.778731 |
| 20260730 | 0.758851 | 0.777253 |
| arithmetic seed mean | 0.768016 | 0.778628 |
| fixed equal-logit ensemble | 0.770202 | 0.778641 |

原始 `PATCH_PROTOCOL.md` 要求的是“across three seeds 的 mean
macro-F1”至少比 P0 高 `0.020`，不是 ensemble metric。formal 三-seed
mean 相对 P0 只高 `0.017366`，距门槛 `0.002634`；即使改用 ensemble
macro-F1，增益也只有 `0.017379`。

此前先运行的 `17/42/73/20260728` 四-seed robustness set 并非协议 Gate-B
指定 seeds，属于补充性、post-hoc robustness evidence。它的 arithmetic
mean macro-F1 为 `0.778953`，相对 P0 `+0.017690`；其更有利的 ensemble
macro-F1 为 `0.780399`，相对 P0 `+0.019137`。两种算法都仍未达到
`+0.020`，后补的 formal seeds 排除了“只是 seed 选择导致失败”的解释。

在各自已选择阈值下：

- formal patch ensemble 的 all-negative-event FP mass 为 `3.7404`；
- exact P0 的 FP mass 为 `2.7428`；
- P5 的 FP mass 为 `3.8710`；
- formal patch 的 FP events 为 `18/27`，P0 为 `14/27`，P5 为
  `16/27`；
- 补充四-seed patch 更差：FP mass `4.4362`，FP events `21/27`。

这同时违反了原协议的“all-negative-event FP 不增加”要求。即使后续
history-shuffle 证明 patch 使用了 matched history，也不能追溯性地抹掉
这两个失败的门槛。

唯一 protocol-declared null-loss 控制 A2（`null_weight=0.1`、
readout-zero、seed20260728）在 epoch 3 得到 AP `0.766228`、
AUC `0.842916`、macro-F1 `0.777292`。相对 matched A1 同 seed：

- AP `-0.006180`；
- AUC `-0.008546`；
- macro-F1 `-0.002608`；
- 在各自 macro threshold 的 positive-F1 `-0.008371`；
- 若分别为 positive-F1 再选阈值，则仍为 `-0.008941`。

在各自 positive-F1-selected threshold 下，两者的 all-negative-event FP
mass 都恰好为 `4.861201`，null loss 没有改善该协议下的主误报指标。在
各自 macro-F1-selected threshold 下，A2 的 FP mass/FP events/FP rows
为 `3.868939 / 17/27 / 78`，A1 为 `4.136201 / 20/27 / 85`；这说明
A2 存在误报—排序 trade-off，但不能抵消 AP/AUC/F1 的共同退化。按协议
否决 A2，不扩 seed。

第一版 cross-event history shuffle（harsh shuffle）使用补充四-seed
robustness ensemble，在固定 checkpoint、固定 ensemble 和固定原
threshold 下得到：

- AP `-0.022223`；
- AUC `-0.009444`；
- macro-F1 `-0.011917`；
- positive-F1 `-0.014912`；
- all-negative FP mass `+0.412500`。

但该实现保留 target `unique_mask/time gap/quality`，同时用一个未匹配
availability 的 donor row 替换所有 history tokens。由于无效 visit 的
cache token 是零，44,521 个 target-valid history slots 中有 3,070 个
（`6.896%`）被 donor-invalid zero token 替换，涉及 3,070/9,614 rows。
所以这个大幅 shuffle drop 混合了“破坏 matched history content”和
“把零 token 当成有效 history”的效应，只能标记为 **harsh diagnostic**，
不能作为正式 Gate-C 机制证据。

随后对同一个补充四-seed ensemble 完成的 clean exact-pattern shuffle
修复了这个问题：

- 9,614/9,614 rows 都有 eligible donor；
- 4 种完整 `unique_mask` pattern 分开构造 cross-event derangement；
- donor 与 target 的 canonical event 不同；
- donor/target availability mismatch 为 0；
- t0、P0 base logit、target mask、time gap、quality 保持不变；
- 四个 checkpoint 共用 donor map，原 ensemble threshold 固定且不重拟合。

正式 clean shuffle 的 delta 为：

| 指标 | Clean shuffle − original |
|---|---:|
| AP | -0.019023 |
| AUC | -0.010643 |
| macro-F1 | -0.016876 |
| positive-F1 | -0.020117 |
| all-negative FP mass | +0.546212 |

这使 patch architecture 通过“history shuffle removes gain”的机制检查：
局部分支确实依赖同一对象的 matched historical patch content，而不是只靠
P0 或 acquisition metadata。严格说，这个 shuffle replay 使用的是补充
四-seed ensemble，而不是 formal Gate-B 三-seed ensemble；但 patch 已经
因正式 mean macro-F1 与 FP safeguard 失败，无需再用 GPU 补一个不会改变
晋级结论的三-seed replay。机制证据保留，promotion 仍否决。

使用 formal Gate-B 三-seed patch 的三专家 equal-thirds 相对 P5：

- AP `+0.008369`，95% CI `[+0.001688,+0.015772]`；
- AUC `+0.004849`，95% CI `[+0.000032,+0.009593]`；
- positive-F1 `+0.014239`，95% CI `[+0.003769,+0.025826]`；
- macro-F1 的区间跨 0。

但真正需要比较的是已晋级的 P5+D1。三专家 equal-thirds 相对 P5+D1：

- AP 只有 `+0.000573`，95% CI `[-0.003184,+0.004915]`；
- macro-F1 `+0.001393`，95% CI `[-0.003649,+0.006286]`；
- AP、AUC、macro-F1、positive-F1 的 bootstrap CI 均跨 0。

two-scale `.25 P5 + .25 D1 + .50 patch` 是看到 equal-thirds 结果之后
增加的权重规则，应明确标记为 **post-hoc exploratory**。它相对 P5+D1：

- AP `+0.000249`，95% CI `[-0.005305,+0.006619]`；
- AUC `-0.000743`，CI 跨 0；
- macro-F1 `+0.002642`，CI 跨 0；
- positive-F1 `+0.004445`，CI 跨 0。

四项都未确认，不能把它包装成预注册的整体胜利。

#### 256-D 宽度控制：独立随机投影的高分没有通过严格嵌套复核

最初的 256-D 控制重新抽取了一套与 128-D 不同的随机投影。唯一 seed
`20260728` 得到 AP `0.780817`、AUC `0.858378`、macro-F1 `0.781912`，
相对同 seed 128-D readout-zero 分别为 AP `+0.008409`、macro-F1
`+0.002012`。这个 point result 看似越过单-seed continuation gate，但它
同时改变了投影子空间，无法区分“保留更多 patch 信息”和“抽到了更有利
的随机基”。

严格复核随后构造了 nested 256-D 投影：前 128 列与原始 128-D basis
array-equal（最大误差 `0.0`），只额外加入与之正交的 128 列；seed、训练
协议和选择指标保持不变。结果为：

| Single-seed width control | AP | AUC | Macro-F1 | Positive-F1 |
|---|---:|---:|---:|---:|
| 128-D matched A1 | 0.772408 | 0.851462 | 0.779900 | 0.734644 |
| 256-D independent random subspace | **0.780817** | **0.858378** | **0.781912** | 0.731637 |
| 256-D strict nested extension | 0.766638 | 0.844087 | 0.774334 | 0.724195 |
| fixed P5 + nested-256 late fusion | 0.769042 | 0.847763 | 0.776167 | 0.725893 |

strict nested-256 相对 matched 128-D 的 AP 为 `-0.005770`、macro-F1
为 `-0.005566`，同时低于预先冻结的 `AP >= 0.774408` 和
`macro-F1 >= 0.779900` 两条线；在 unified macro operating point 下，
all-negative FP mass 为 `3.8466`、FP events 为 `18/27`。按协议停止，
不扩 seed。

因此，独立 256-D 的高分只能记录为 **random-subspace / single-seed
sensitivity**，不能作为“更宽 patch representation 因果改善”的证据。
即使 nested 控制通过，它也会增加下游输入层参数，仍需要
matched-compute multi-seed 复核；本次 nested fail 已使该问题无需继续。

### 2.3 Legacy360：F1 超过 0.90，但不是 clean SOTA

开发集为 12,621 行，训练 core 为 113,843 行；当前 train/dev event 和
plume 均无重叠。锁定 evaluator 的开发集 replay 为：

| 系统 | Binary F1 | Macro-F1 | AP | AUC |
|---|---:|---:|---:|---:|
| P0 locked replay | 0.905424 | 0.894916 | 0.960035 | 0.960381 |
| R4 四-seed equal-logit | 0.910235 | 0.900769 | **0.961690** | **0.962527** |
| one-sensor R4, otherwise P0 | **0.910745** | **0.901134** | 0.961190 | 0.962070 |

availability gate 相对 P0 的 10,000 次 event bootstrap：

- binary F1 `+0.005275`，95% CI `[+0.002109,+0.008604]`；
- macro-F1 `+0.006171`，95% CI 为正；
- AP 的区间窄幅跨 0。

R4 history shuffle 使 F1 下降 `0.003638`、AP 下降 `0.002098`，所以
matched history 确实参与了预测。但是：

| Legacy dev sensor stratum | P0 F1 | R4 mean F1 | ΔF1 | P0 AP | R4 mean AP | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.93319 | 0.93513 | +0.00194 | 0.97159 | 0.97241 | +0.00083 |
| L89 | 0.90487 | 0.91282 | +0.00795 | 0.95911 | 0.95931 | +0.00020 |
| EMIT | 0.89372 | 0.89259 | **-0.00113** | 0.95217 | 0.95264 | +0.00046 |
| S5P | 0.90105 | 0.90769 | +0.00664 | 0.95469 | 0.95974 | +0.00505 |

这些 stratum F1 使用每个 sensor 自己在同一 dev 选择的阈值，只能作为
诊断。它也直接否定“所有 sensor F1 都提高”：EMIT 的 point F1 下降，
而其他三个 stratum 的增益仍处在污染开发协议内。

1. warm-start checkpoint 来自包含当前 dev 子集的旧 source-train pool；
2. availability gate 是在同一 dev 做 subgroup audit 后提出的 post-hoc
   冻结规则；
3. 同一 dev 上已经筛过 P/Q/R、R1/R4/R5、attention、add、specialist、
   blend、loss 和超参数；
4. 四-seed ensemble 的推理计算量高于单个 P0；
5. R1 plain delta 的 F1 为 `0.91017`，与 R4 的 `0.910235` 基本相同，
   且 R1 macro-F1 更高。

所以这里能支持的是“appearance + explicit temporal difference 是一个
有效旧协议工程改进”，不能支持“motion excitation 是决定性新机制”。

### 2.4 EMIT：强 temporal baseline 改变结论

EMIT train/dev 为 11,688/5,201 行、900/129 events，event overlap 为 0；
但只有 `t0/prev1/seasonal` 三个角色。

| 系统 | Event-balanced row AP | AUC | Macro-F1 | Positive-F1 |
|---|---:|---:|---:|---:|
| exact t0 P0 | 0.750368 | 0.835329 | 0.757995 | 0.685169 |
| R4 四-seed ensemble | 0.765439 | 0.847151 | 0.769199 | 0.693404 |
| role-only 三-seed ensemble | **0.784186** | 0.858284 | 0.777879 | 0.703098 |
| fixed role-only+R4 | 0.784821 | **0.860149** | **0.781784** | **0.709314** |

R4 相对弱 P0 有明确历史信号，history shuffle 使 AP/AUC 分别下降
`0.015808/0.012188`。但 R4 明显低于更强的 role-only baseline。

固定融合相对 role-only 的 AP、AUC、macro-F1 和 positive-F1 四个 paired
bootstrap CI 全部跨 0。并且融合使用 3 个 role-only heads 加 4 个 R4
heads，而基线只有 3 个 heads；这不是 matched-compute comparison。

因此 EMIT 只能作为“transient evidence 可能互补”的探索性证据。

### 2.5 S5P 与 S2

S5P 结果：

| 系统 | Row AP | Row AUC | Event-balanced row AP | Event-balanced row AUC |
|---|---:|---:|---:|---:|
| exact P0 | 0.576047 | 0.529720 | 0.577098 | 0.524090 |
| raw delta | **0.584436** | **0.538068** | **0.581257** | **0.526015** |
| R4 | 0.581062 | 0.535119 | 0.578915 | 0.525756 |

当前 3×3 frame 是对已上采样 224×224 产品的 adaptive average，不是
真正 native feature，也不是 learned per-time CLS。AUC 接近随机，R4
低于 raw delta，应停止扩 seed。

S2 现有 UniverSat shard 只有 `[N,1536]` collapsed feature，六次 visit
已经折成一个向量，无法恢复 per-time delta、matched-history shuffle 或
同协议 R4。没有新的 event-disjoint inner-dev per-time cache 就不能给出
S2 结论。

## 三、与上一篇论文及 README 基线的可比性

MethaneFuse ICDM 论文采用 2025-05-16 chronological cutoff，并在 crop
构造前按 event 切分。论文 360 m 结果为：

| 论文系统/尺度 | F1 | Accuracy | FPR | Recall | AUROC |
|---|---:|---:|---:|---:|---:|
| MethaneFuse，360 m | 83.77% | 83.52% | 13.07% | 80.48% | 90.16% |
| MethaneFuse，480 m | 84.87% | 84.21% | 14.87% | 83.40% | 93.62% |

当前 legacy360 的 `0.910745` 不能写成“比论文 360 m 提高 7.3 个 F1
点”，原因包括：

- cohort、切分和 cache 身份不同；
- 当前 warm start 看过旧 source pool 中的 dev；
- 当前数字是反复开发后的 dev，论文数字是 held-out test；
- 当前 F1 使用 development-optimized operating threshold，论文阈值协议
  不等价；
- 当前四-seed ensemble 与论文单模型计算量不等价。

`Upgraded_dataset/README.md` 中的历史数字同样只能作 guardrail：

| 数据/历史设置 | 历史 F1 |
|---|---:|
| S5P old test | 0.6375 |
| L89 old split | 0.7119 |
| L89 cutoff version | 0.8047 |
| EMIT old split | 0.7146 |
| EMIT full strict cutoff | 0.5667 |
| L89 hard-event-filtered，old checkpoint | 0.7760 |
| L89 hard-event-filtered，from scratch | 0.7544 |

这些结果使用不同 row/event aggregation、cohort、crop、split、checkpoint、
threshold 和模型计算量。不能把当前 event-balanced dev AP/F1 与其中某一
个数字直接组成 SOTA 表。

L89 的 lineage 需要更精确地区分 train 与 test：

- 当前 inner train/dev 都来自 source-train；
- `L89_temporal_train_hard_event_filtered.csv`、
  `L89_temporal_train_drop_3_hard_events.csv` 和
  `L89_temporal_train_cutoff_2025_10.csv` 均为 62,389 行，且文件
  SHA-256 完全相同：
  `04c5e0f27256d24d14dfb7f91a43801f09277ab6ae59dd9c4b6b9f3ab7bfd7c0`；
- 因此，那 9 个根据旧 checkpoint test error 删除的 difficult events
  **没有直接改变当前 inner train/dev pool**；
- 但是 `hard-event-filtered test` 本身正是 performance-conditioned
  payload，不能作为 clean confirmatory outer test。

所以当前 L89 inner mechanism evidence 比目录名暗示的更干净；它的主要
问题是同一 dev 被大量自适应复用。任何以 hard-event-filtered test 做的
one-shot replay，最多仍是工程外测，不能解除 test-set conditioning。

## 四、泄漏、选择偏差和统计风险

### 4.1 泄漏等级

| 数据 | 当前 train/dev event overlap | 上游/历史污染 | 审计等级 |
|---|---:|---|---|
| L89 inner train/dev | 0 | source-train 未被 9-event 删除改动；dev 多轮复用 | 较强开发证据，仍非 confirmatory |
| L89 hard-filtered outer | 与 train 为 0 | test payload 根据旧模型错误删过 9 events | model-conditioned 外测 |
| EMIT | 0 | 当前 cache 边界可审计；同一 dev 用于选择 | 开发证据 |
| S5P | 0 | 表示近似且 dev 无 all-negative events | 低可信否决实验 |
| legacy360 | 0 | warm-start source pool 包含当前 dev 子集 | 污染的工程开发证据 |
| S2 | 未建立同协议 pair | collapsed cache 无法审计 temporal branch | blocked |

### 4.2 开发集反复使用

Bootstrap 只处理有限 124/129/291 个 event 的采样不确定性，不能修复：

- 用同一 dev 选择 checkpoint；
- 用同一 dev 选择 architecture；
- 用同一 dev 选择 projection width、radius、top-k、normality、
  initialization 和 fusion rule；
- 多 seed 之后再选择 ensemble；
- 看 subgroup 后再设计 availability gate；
- 看 equal-thirds 后再提出 two-scale 权重。

因此现有 bootstrap 区间应标记为 **post-selection descriptive CI**，
而不是完全独立的 confirmatory p-value/CI。

### 4.3 Seed 和计算量

- L89 D1 三-seed fusion 的单-seed AP 为
  `0.763524/0.775794/0.770521`，其中至少一个低于 P5 `0.765148`；
- patch 单 seed20260728 的 scalar 版本曾给出 AP `0.770474`、三专家
  AP `0.777571`，但四 seed scalar ensemble 只有 `0.767654/0.772482`；
  只报最好 seed 会明显夸大效果；
- formal readout-zero Gate-B 的 seed20260730 AP 只有 `0.758851`，
  明显低于前两个 formal seeds 的 `0.772408/0.772789`；正式表必须保留
  这个 seed，不能换成表现更好的 17/42/73；
- 当前候选常把 3–7 个 heads 与单 head baseline 比较。最终论文必须增加
  matched-parameter、matched-training-budget 和 matched-inference-cost
  baseline，或明确承认 ensemble 成本。

### 4.4 阈值

AP/AUC 是当前最稳妥的主证据。macro-F1 与 positive-F1 往往分别在 dev
上选择各自最优阈值，不一定对应同一个部署 operating point。最终外层
评估必须：

1. 在 inner-dev 冻结一个阈值规则；
2. 不在 outer/test 重新优化阈值；
3. 同时报告 AP、AUC、positive-F1、macro-F1、recall、FPR；
4. 报告 all-negative-event FP mass 和 FP event count。

### 4.5 “Event-balanced”不等于“每个 event 一个预测”

当前 `event_balanced_*` 指标仍然在 row label/probability 上计算，只是
sample weight 令每个 canonical event 的总权重相等；它不是先把一个
event 的所有 rows 聚合成一个 event probability。paired bootstrap 则以
event 为 cluster 重采样。论文必须使用“event-balanced row metric”这个
准确名称，不能简写成“event-level classification AP/F1”。若应用场景的
最终决策单位真的是 event，还应另外预注册 event aggregation（例如 max、
top-k mean 或 noisy-OR）并在 outer split 一次评估。

## 五、机制主张审计

### 有实验支持

1. **显式 matched temporal evidence 有用。**  
   L89 D1、legacy R4 和 EMIT R4 的 history shuffle 都使结果下降。

2. **appearance 与 transient 分开训练，再做固定 late fusion，比把
   residual 直接接在强 P5 上更可靠。**  
   direct-P5 D1 在 epoch 1/2 已下降；appearance-free D6 也明显过弱。

3. **不规则采集信息值得显式建模。**  
   L89 D1 使用 role、time gap、quality 和 valid mask；分组诊断表明收益
   随 history coverage/quality 改变。

4. **局部弱瞬态可能需要 patch correspondence + sparse MIL。**  
   same-pixel control 回落到 P0；3×3 优于 same-pixel，5×5 没有带来
   平衡收益；history normality 在 64-D ablation 中增加约 0.0031 AP。

### 没有实验支持或已经否决

- “t0 作 query、历史作 K/V”是新的；
- 新的 attention 原理；
- 统一的四 sensor 六时刻模型；
- 大规模不可见光 foundation-model pretraining；
- hard linear background trajectory subtraction；
- appearance-free transient classifier；
- NormWear-style learned liaison；
- learned reliability fusion；
- R4 一定优于 plain delta；
- patch 已稳定优于 P5+D1；
- segmentation 能力提升；
- 所有 sensor 的 classification F1 都提高；
- clean SOTA。

### Novelty 边界

各个单独部件都已有明显的相邻工作：

- appearance/motion two-stream、feature difference 和 motion excitation
  是成熟的视频识别设计；
- Siamese/local correspondence、current-reference change 和背景残差属于
  变化检测/HACD 的既有家族；
- top-k MIL 是常见的弱监督稀疏目标聚合；
- mixture-of-experts、late fusion 和 missing-modality routing 不是新原则；
- AnySat、ALISE、masked temporal EO models 已覆盖异质 EO 和不规则时序
  表示学习；
- NormWear 能启发 axis-separated experts，但不能作为“首次双轴融合”的
  novelty 依据。

所以不能把论文写成“发明 temporal difference / local matching / late
fusion”。如果最后被 outer evaluation 确认，较窄的技术 delta 是这些
部件在 methane EO 约束下的特定组合：

1. 保留 methane-sensitive response expert，不再依赖通用可见光预训练；
2. 对 irregular/missing acquisition 单独形成 global transient evidence；
3. 用 history-to-history normality 与小邻域 correspondence 定义
   patch-local onset；
4. 不在小数据上学习复杂 liaison，而用固定 evidence consensus；
5. 用 event-disjoint split、branch-isolated shuffle、all-negative-event
   false alarms 和 matched-compute controls 验证机制。

这仍然是“组合型方法贡献”，不是基础原理级 novelty。若 patch 未被外层
确认，novelty 会进一步收窄到 independent acquisition-aware temporal
evidence + fixed late consensus。

## 六、推荐的论文技术表述

最窄且目前证据可支撑的版本是：

> Existing methane classifiers primarily encode response/appearance in a
> current or early-fused representation. Sparse EO revisits are not regular
> video: observations are irregular, missing, sensor-dependent, and a plume
> may occupy only a few patches. TEMPO therefore separates methane-sensitive
> response evidence from acquisition-aware transient evidence. A global
> expert models signed and magnitude changes over valid revisits; an optional
> local expert compares small current regions with locally matched historical
> background and normalizes onset by history-to-history variation. Experts are
> trained independently and combined only through fixed late-logit consensus.

审计限定语必须紧跟：

> Current results establish the global mechanism on reused L89 development
> data and show exploratory local complementarity. They do not yet establish
> cross-sensor SOTA or a unified six-visit model.

如果 locked outer evaluation 没有确认 P5+D1 或 patch 增益，论文主线应
退回“negative but informative”结论：普通视频式 early fusion、learned
liaison 和直接 residual correction 不适合小样本、不规则 EO revisit；
独立 late evidence 是更稳定的工程方案。不能用 legacy 污染开发结果替代
外层验证。

## 七、晋级、冻结和否决清单

### 晋级到一次性 outer evaluation

- L89 `P5 + D1 three-seed equal-logit ensemble`；
- 固定 checkpoint、固定 equal-logit 权重、固定 threshold rule；
- paired canonical-event cluster bootstrap；
- P5、D1 mean seed、P5+D1 ensemble 的 matched report；
- history-shuffle 作为 mechanism diagnostic，不作为选择 outer 结果的
  新依据。

这里的“晋级”只对新的、没有参与历史错误筛选的 outer events 成立。
如果实际执行的是现有 `hard-event-filtered test`，应将结果标为
model-conditioned engineering replay，不得升级为 clean SOTA。

### 仅作为探索性附加候选

- 128-D readout-zero patch four-seed ensemble；
- P5+D1+patch equal-thirds；
- two-scale `.25/.25/.50`，必须标为 post-hoc；
- EMIT role-only+R4 fixed fusion；
- legacy360 availability gate。

### 明确否决

- S5P 当前 approximate-native R4；
- L89 scalar-zero patch 作为最终版本；
- direct-P5 residual；
- appearance-free D6；
- hard-binned D7 SparseSlowFast 作为正式 winner；
- learned liaison/reliability gate；
- hard normality subtraction；
- 用最好单 seed 代替 multi-seed；
- 把 0.910745 与论文 0.8377 直接相减。

### 当前阻塞

- S2 同协议结果；
- 四 sensor 统一六时刻结果；
- segmentation transfer；
- clean SOTA；
- 任意正式 test claim。

### Sealed / outer 授权边界

本审计没有授权、读取或运行任何 sealed/test/holdout payload。

- **L89：**当前只有
  `research/tempo_20260728/l89_lock_template_v4/LOCK_MANIFEST.json`，
  状态是 `development_template_not_authorizable`。若最终采用 P5+D1，
  仍须先构建并人工复核 final manifest，再由实验负责人单独授权它的
  **exact SHA-256**，producer 和 evaluator 才能各运行一次。现有
  hard-event-filtered outer cohort 即使锁定运行，也仍是
  model-conditioned engineering replay，不是 clean confirmatory test。
- **legacy360：**locked evaluator 已冻结，manifest SHA-256 为
  `62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92`。
  它仍等待实验负责人同时明确授权这个 exact digest 和一个
  **exact absolute sealed/test cache path**；开发 dry-run 不构成授权。

在这些条件满足前，任何 agent 都不应猜测 outer 路径或运行一次性入口。

## 八、Patch 控制的最终判据

截至本报告当前版本，bounded controls 状态为：

1. frozen-base、fixed-threshold、cross-event patch history shuffle：
   harsh 版本受 6.896% valid-slot/zero-donor 混杂；clean exact-pattern
   版本已在补充四-seed ensemble 上完成，0 mismatch，AP `-0.019023`，
   architecture-level mechanism check 通过；
2. 唯一 protocol-declared A2 `null_weight=0.1`：已完成并否决；AP、
   AUC、macro-F1、positive-F1 均低于 matched A1，不扩 seed；
3. 256-D seed20260728 宽度控制：非嵌套版本 AP `0.780817`，但更换了
   随机投影子空间；前 128 列 bit-exact 的 strict nested 版本只有 AP
   `0.766638`、macro-F1 `0.774334`，两条 continuation gate 均失败，
   已停止且不扩 seed。

这些控制只回答机制和误报问题，不能把已经 miss 的原始总体 promotion
gate 改写掉。A2 只能作为误报—排序 trade-off；非嵌套 256-D 的单-seed
高点只能说明随机子空间敏感，不能成为最终模型。

## 九、审计产物索引

- 技术架构与张量流说明：  
  `/home/yuyao/panopticon/research/tempo_20260728/TEMPO_TECHNICAL_ARCHITECTURE_CN.md`
- L89 global report：  
  `/home/yuyao/panopticon/research/tempo_20260728/TEMPO_L89_GLOBAL_EXPERIMENT_LOG.md`
- L89 exactly-once outer protocol（当前不可授权 template）：  
  `/home/yuyao/panopticon/research/tempo_20260728/L89_OUTER_EXACTLY_ONCE_PROTOCOL.md`
- L89 patch protocol：  
  `/home/yuyao/panopticon/research/tempo_20260728/PATCH_PROTOCOL.md`
- L89 patch report：  
  `/home/yuyao/panopticon/research/tempo_20260728/TEMPO_L89_PATCH_RESULTS.md`
- legacy360 report：  
  `/home/yuyao/panopticon/research/tempo_20260728/TEMPO_LEGACY360_GLOBAL_RESULTS.md`
- cross-sensor technical audit：  
  `/home/yuyao/panopticon/research/tempo_20260728/CROSS_SENSOR_TEMPO_TECHNICAL_AUDIT.md`
- L89 D1 aggregate：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/eventbase_d1_multiseed_fixed/aggregate.json`
- L89 P5+D1 bootstrap：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/d1_sidecar_p5_equal_logit_05_paired_event_bootstrap_2000.json`
- L89 D7 SparseSlowFast aggregate：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/eventbase_d7_sparse_slowfast_seed20260728_v1/aggregate.json`
- L89 fixed P5+D7 audit：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1/p5_d7_sparse_slowfast_equal_logit_05_seed20260728_audit.json`
- L89 patch readout-zero audit：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_multiseed_fixed_v1/fixed_multiseed_audit.json`
- L89 patch formal Gate-B 三-seed audit：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_gateb3_fixed_v1/fixed_multiseed_audit.json`
- L89 patch A2 null-loss control：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/heads/p0_a2_readout_seed20260728/result.json`
- L89 patch harsh history-shuffle（受 availability 混杂）：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_history_shuffle_v1/history_shuffle_audit.json`
- L89 patch clean exact-pattern history-shuffle：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim128_v1/audits/readout_history_shuffle_maskmatched_v1/history_shuffle_audit.json`
- L89 patch 256-D independent-subspace single-seed：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim256_v1/heads/p0_a1_readout_seed20260728/result.json`
- L89 patch 256-D strict nested control：  
  `/diniuvol/yuyao/methanefuse_research_20260728/tempo_l89_patch_p0_dim256_nested128_v1/heads/p0_a1_readout_nested_seed20260728/result.json`
- legacy360 locked evaluator：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/legacy360_global_v1/locked_evaluator_v9/LOCK_MANIFEST.json`
- legacy360 sealed authorization protocol：  
  `/home/yuyao/panopticon/research/tempo_20260728/TEMPO_LOCKED_EVALUATOR_PROTOCOL.md`
- EMIT role/R4 audit：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/cross_sensor_r4_v1/emit_role_r4_fixed_fusion_v1/audit.json`
- S5P aggregate：  
  `/diniuvol/yuyao/methanefuse_tempo_20260728/cross_sensor_r4_v1/s5p_approx_native_seed20260728_v2/aggregate.json`

## 十、最终审计判词

目前最值得保留的不是“又换了一种 attention”，而是一个可以被实验
证伪、并且已有部分统计支持的架构原则：

> **Methane response and sparse transient evidence should be learned as
> independent experts and fused late; irregular acquisition and local
> background normality should be represented explicitly rather than delegated
> to generic early temporal attention.**

但截至本审计，唯一达到下一阶段资格的是 L89 global P5+D1 的锁定外层
候选。Patch 是有吸引力的技术部件，却没有通过原始 promotion gate；
legacy360 的 0.91 是工程结果，不是论文 SOTA 证据；EMIT 只给出弱互补
信号；S5P 被否决；S2 未完成同协议实验。

任何更强的结论都应等待一次未参与模型选择的、event-disjoint、固定阈值、
matched-compute outer evaluation。
