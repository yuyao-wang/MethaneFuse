# 下一篇 Journal：实验与 Claim Lock

**日期：** 2026-07-28  
**状态：** 预注册方案；不是新结果，不授权读取任何新 outer test  
**适用边界：** 四传感器 verified-512 重建后的下一轮正式实验

## 0. Executive lock

下一篇最有吸引力、同时与现有证据一致的故事不是“又一种 residual/MAE/
ranking”，也不是“更复杂的 cross-attention”，而是：

> **保护已经有用的 scene-semantic base；用 sensor-native observation
> operator 训练一个可被忽略的 methane-response residual sidecar；再用
> cross-fitted event-null calibration，把排序信号转成不增加负事件误报的
> 决策证据。**

现有证据只授权把这句话写成**待检验假设**：

| 已完成证据 | 现在允许的解释 | 不能推出的结论 |
|---|---|---|
| scale-aware dual-axis dev F1 `0.90558` | downstream engineering 能力强，scale/acquisition conditioning 有用 | dual-axis attention 是 journal novelty |
| locked old-test overall/S2 F1 `0.86485/0.90616` | old protocol 下的 faithful engineering comparison | clean benchmark SOTA；RCTP 有效 |
| in-place P5 − P0 event macro-F1 `−0.02715`，95% CI 完全低于 0 | synthetic specialization 原位改写 encoder 会损害真实 decision geometry | “更多 pretraining”会自然修复 |
| frozen final-patch P5 选择 epoch 0 | 现有 final-token local readout 没有救回 P5 | patch-local signal 已被普遍否定 |
| Sidecar P5 event AP − P0 `+0.015998`、− P4 `+0.010767`，两者 CI 均高于 0 | preserve-base sidecar 有值得新 holdout 复验的 ranking partial signal | response-specific success、方法 promotion 或 SOTA |
| Sidecar P5 − P4 macro-F1 `+0.000892`，CI 跨 0；all-negative FP mass `3.871`，差于 P4 的 `2.531` | 当前瓶颈是 decision specificity / event-null behavior | AP 正信号足以进入 sealed test |

因此当前 verdict 仍是 **promotion no-go**。本方案的任务是一次性回答：

1. formal sensor response 是否真的优于 matched scrambled response；
2. event-null calibration 能否保留 AP，同时消除 all-negative-event FP 代价；
3. 同一 residual 是否同时迁移到 classification 与真实 plume segmentation；
4. 该能力在自然 missing sensor/time 下是否仍存在。

## 1. 标题与 abstract 逻辑骨架

### 1.1 建议标题

若所有正式 gate 通过，首选：

> **Preserve the Scene, Calibrate the Plume: Sensor-Native Residual
> Pretraining for Multisensor Methane Detection**

更强调物理条件的版本：

> **From Sensor Response to Event Evidence: Representation-Preserving
> Pretraining for Transient Methane Detection**

若 response-specific 或 segmentation gate 失败，应改成分析型标题，不保留
正向方法暗示：

> **When Better Ranking Is Not a Safer Decision: A Controlled Study of Weak
> Methane Transients in Earth-Observation Models**

### 1.2 Abstract 逻辑骨架

最终 abstract 按以下六句写，结果槽在 single locked test 前保持空白：

1. **问题：** 通用 EO encoder 擅长 scene semantics，但 methane 是弱、局部、
   短暂且 sensor-dependent 的观测响应。
2. **诊断：** matched experiments 显示 in-place counterfactual continuation
   损害真实 macro-F1；preserve-base sidecar 只产生 dev-only AP partial signal，
   尚未带来 false-positive-safe decision。
3. **方法：** 冻结 scene-semantic base，按正式 SRF/PSF/product operator
   训练 bounded response sidecar，并用 outer-train cross-fitted all-negative
   events 做 sensor/acquisition-stratified null calibration。
4. **协议：** 从 verified 512 sources 统一 recrop，按 global event/source
   connected components 切分；以 matched P0/P4/P5，在四 sensor 上同时评估
   classification、可信 real-mask segmentation 与自然 missingness。
5. **结果：** 只填 locked outer-test 的 P5 相对 P0、P4 和 strongest collision
   baseline 的 paired effect、CI、negative-event FP 与 missingness strata。
6. **边界：** 贡献若成立，是上述组合在泄漏安全、matched-compute 双任务
   transfer 下的证据；不是 residual、ranking、MAE 或 cross-attention 的首次提出。

禁止把当前 Sidecar AP delta、legacy `0.90616`，或任何 current recent-dev
结果填进第 5 句作为正式方法效果。

### 1.3 最新 train-only event-null audit：static post-hoc calibration no-go

在不读取 holdout 的 5-fold group cross-fitting audit 中，所有 fold/calibrator
都选择了 identity transform；预先由 train OOF 锁定的 threshold 原样应用到
dev 后，P0/P4/P5 的 event macro-F1 分别为
`0.7342/0.7571/0.7560`，all-negative-event FP mass 分别为
`1.8331/2.3188/2.7922`。P5 − P4 macro-F1 为
`−0.0011`，paired 95% CI `[−0.0117, +0.0092]`。

因此，仅按 static availability/quarter strata 拟合 offset 的 post-hoc
calibration 是明确 **no-go**：它既没有分开 P5/P4，也没有控制 P5 的负事件
尾部。下一方法必须把 **event-null selectivity 写进 train-time objective 与
sampling**，再用 cross-fitted calibration 做锁定的 decision mapping；不能把
post-hoc threshold/offset 当成 sidecar 的救援机制。这项 audit 不授权 outer-test。

## 2. 下一方法：Preserve–Respond–Calibrate

### 2.1 Preserve：scene-semantic base 不可被改写

- 使用 provenance-clean 的公开/重训 Panopticon base；formal run 不使用历史
  test-selected checkpoint。
- base encoder 全冻结、eval mode、byte-identical；每个 run 保存参数 SHA。
- P0/P4/P5 downstream 都接收相同形状 `[z_base, 0.1 × residual]`。
- sidecar 为 bias-free rank-8、末层 exact-zero initialization；`z_base` 与
  residual 分通道保存，禁止 in-place 相加。
- classification 与 segmentation head 都可以把 residual 权重学到 0；不能
  迫使下游牺牲已有 scene representation。

这部分的主张是防遗忘的实验设计，不宣称 residual adapter 本身有 novelty。

### 2.2 Respond：formal sensor-native response sidecar

sidecar 的共享语义是“该 sensor 下 methane 应怎样改变观测”，不是“把不同
sensor 的 raw tokens 变成一样”。

| Sensor | 正式 native operator | 禁止的捷径 |
|---|---|---|
| S2 | 12-band reflectance；正式 SRF、CH4 sensitivity、GSD/PSF；先在物理平面注入再 native sample | 在 224/512 插值图上画细 plume；用 sensor ID 替代 response |
| L8/9 | 7 SR bands；分别记录 L8/L9 product、SRF、30 m PSF/GSD | 把重复 acquisition 当多份时间证据 |
| EMIT | native 32-band methane-window reflectance；高光谱 transmission 后 SRF/PSF 聚合 | 把 legacy 16-band WV3 simulation 与 native 32-band 混为一臂 |
| S5P | 原始 native 3×3 XCH4、footprint/QA；可得时使用 averaging kernel，否则显式标记近似 | 上采样成 224 texture；生成虚构的高分辨率 segmentation target |

对同一 outer-train background、plume column field、strength、visit、mask、
noise 和 validity，P4/P5 唯一差别是：

- **P4：** response identity 在预锁定的 sensor/product stratum 内打乱；
- **P5：** 使用正确 SRF/PSF/product response。

所有 synthetic backgrounds 只来自 outer-train。P4/P5 共享 draws、初始化、
参数量、optimizer、update count、batch plan 与 downstream head。

### 2.3 Select then calibrate：train-time null selectivity + cross-fitted calibration

现有 P5 的问题不是 AP 为零，而是高分尾部在 all-negative events 上失控。
最新 audit 又排除了 static availability/quarter offsets 的 post-hoc rescue。
因此 P0/P4/P5 downstream 首先使用 matched、strata-balanced all-negative-event
batches，并在训练时直接约束 negative-event score tail / soft FP mass；该
null-selectivity loss 的权重、tail quantile 与 sampling ratio必须在 protocol
lock 时固定。之后 calibration 才作为正式 decision mapping 与独立消融，而
不是 test 后调 threshold。

1. 在 outer-train 按 global event/source component 做 K-fold cross-fitting；
2. 只用 held-out-fold predictions 构造 empirical null；
3. 预先冻结 strata：
   `sensor × product/operator version × acquisition/quality bucket ×
   current/temporal coverage bucket`；
4. 每个小 stratum 低于预锁定样本数时按固定父层级 shrink/pool，禁止看 dev
   后重新分箱；
5. 把每个 sensor score 转为 null-tail evidence、校准概率及 uncertainty；
6. threshold、fusion weight 和 fallback hierarchy 全部由 outer-train OOF
   predictions 冻结；new dev/outer test 不重新拟合。

all-negative-event FP mass、Brier/ECE、recall at locked FPR 与 event macro-F1
是 calibration 的共同指标。只提高 AP 不算 calibration 成功。
必须另保留无 train-time null loss 的 matched ablation；若只有 post-hoc
mapping 改善，不能声称 residual 学到了 null selectivity。

### 2.4 Multisensor fusion：operator/strata first，attention second

多传感器融合不在 raw pixel/token 层假装四个 sensor 同质：

```text
native observation
  -> sensor-native operator + frozen base
  -> bounded response residual
  -> within-sensor temporal comparison
  -> sensor/acquisition-stratified null evidence
  -> availability-aware event evidence fusion
```

- classification 融合的是校准后的 event evidence 与 reliability，不是未经
  校准的 raw CLS；缺失分支贡献 neutral evidence，而不是 zero observation。
- S2/L89/EMIT segmentation 在各自 native grid 解码，再投影到同一物理 query
  footprint 评估；S5P 只贡献 coarse cell/event context。
- cross-attention 可作为 matched readout control，但不是方法的物理接口或
  novelty。若只加 cross-attention 就达到相同结果，应缩小 method claim。

## 3. 数据重置：四 sensor verified-512 recrop

### 3.1 “512”定义

从四 sensor 的 **verified 512 source artifacts** 重新裁出同一 WGS84 centre、
metre-scale footprint 与 Carbon Mapper label。`512` 是 source/provenance
层级，不表示把所有输出强制 resize 成 512 或 224；输出保持 sensor-native。

每个 usable slot 必须同时具备：

- embedded product/acquisition identity 与 manifest 一致；
- native query coverage 与逐 channel/cell validity；
- deterministic alias collapse；provenance conflict quarantine；
- value/validity crop readback SHA；
- frozen operator、normalization、crop geometry 与 FOV version。

S5P 从原始 NC/cell 重取 3×3。任何 unresolved provenance conflict、假
compatibility band、all-invalid query 或重复 acquisition 都不能伪装成有效输入。

### 3.2 Global event/source split

先建 connected-component graph，再生成任何 crop 或 synthetic pair。以下任一
关系相连即必须同 split：

- canonical event / plume lineage；
- source scene、product、acquisition ID 或 path hash；
- 同一 background donor 或派生 crop parent；
- 预注册的 near-duplicate site/time relation。

在四 sensor 的 union 上一次性分 outer-train / new-dev / locked outer-test；
同一 component 不得通过不同 sensor 落到两侧。硬 gate：

```text
event overlap = 0
source/product/acquisition/path-hash overlap = 0
synthetic donor overlap = 0
```

current recent-dev 已被反复查看，只能用于历史诊断，不能成为 confirmatory
new-dev。若现有数据无法提供从未用于设计的 outer-test，则必须冻结新增
time-forward cohort；不能把旧 test 重命名为新 test。

### 3.3 任务 eligibility

- canonical manifest 保留自然 missing sensor/time；不做 complete-case-only。
- classification：该 sensor 的 `t0` 必须 query-usable；fused task 至少一个
  sensor 的 t0 usable。
- temporal objective：至少两个 verified unique visits。
- segmentation：只使用 provenance-trusted real plume masks；zero
  compatibility mask 不是 negative target。
- 每个 split 报告 label-conditioned sensor/time mask 分布；eligibility filter
  后重新验证所有 global overlap 仍为 0。

## 4. 最小实验矩阵

### 4.1 核心 3×2 factorial

| Arm | Residual | Calibration U | Calibration C | 用途 |
|---|---|---:|---:|---|
| P0 | exact zero；base-only | 必跑 | 必跑 | base 与 calibration 独立效果 |
| P4 | response-scrambled sidecar | 必跑 | 必跑 | 参数/augmentation/ranking matched control |
| P5 | correct-response sidecar | 必跑 | 必跑 | 候选方法与 calibration load-bearing test |

`U/C` 分别为未做 event-null calibration / 使用锁定 cross-fitted calibration。
calibration 在 frozen predictions 上拟合，因此不得改变 P0/P4/P5 encoder、
head 或 checkpoint。每臂三 seeds；outer-test 只评估一个预先锁定的
seed-ensemble/aggregation rule。

正式 3×2 主矩阵使用锁定的 train-time null-selectivity loss（记为 `N1`）；
另加 `P4-N0-C/P5-N0-C` 两个无该 loss 的 matched ablation，以区分
“训练时学会压制 null tail”和“事后重映射分数”。static quarter/availability
offset 不再作为候选 rescue。

所有六个组合都跑：

- S2、L89、EMIT、S5P per-sensor event classification；
- availability-aware fused classification；
- S2/L89/EMIT real-mask segmentation；
- S5P native 3×3 cell localization（单列，不并入高分辨率 pixel macro）；
- natural missingness 与 deterministic observed-one-to-zero dropout curves。

### 4.2 Level-2 novelty collision controls

survey verdict 是 **Level 2 — High Overlap (fragile)**。若要保留窄组合 claim，
同 split/matched head 下至少还需覆盖：

1. current-only 与简单 masked mean/max/role temporal aggregator；
2. raw cross-fitted temporal residual；
3. pointwise rank regression；
4. score distillation；
5. within-stratum rank permutation；
6. classical HACD / compact state-space temporal detector；
7. AnySat/MAE-family generic masked objective（在其可支持 sensor 上用公开模型，
   全四 sensor 用 matched masked-latent objective）。

这些都是 collision controls，不是本文发明。先做 one-seed new-dev screen，
每个 family 的 strongest train-only-locked member进入三-seed formal comparison；
不得对 weight/temperature/architecture 做 dev 后网格。

## 5. 预注册 gates

所有 paired CI 按 global canonical event/source component cluster bootstrap
（至少 2,000 replicates）计算。P5 必须分别胜过 P0 与 P4；不能只比较事后选出的
较弱 arm。classification 与 segmentation 是 intersection gate：任一失败，
都不能写 survey 的双任务组合 claim。

### G0 — Data/provenance

- verified identity、readback、mask、alias 与 split tests 全部通过；
- unresolved usable provenance conflict 为 0；
- 三个 split 的 global event/source/donor overlap 均为 0；
- outer-test manifest/SHA 在训练前封存。

失败即停止模型实验。

### G1 — Renderer/shortcut

- P4/P5 的 injected energy、area、visit、noise 与 validity matched；
- mask-out/non-response-band synthetic-vs-clean discriminator AUROC `<= 0.60`；
- synthetic/real train plume 的 response SNR 与 morphology 分布有预定义重叠；
- alternate renderer 不反转 train-only capability 方向。

失败只允许修 renderer 并重新版本化，不允许用更多 epoch 补救。

### G2 — Response capability

在 new-dev 前先用 outer-train OOF：

- P5 methane-vs-matched-nuisance ranking `> 0.55`；
- P5 必须优于 P4；paired lower CI `> 0`；
- sidecar norm 不能只由 sensor ID、missingness 或 injected strength shortcut
  解释；
- base SHA、base features与 epoch-0 equality checks 全部通过。

失败则停止 P5，不进入 formal downstream。

### G3 — New-dev promotion

本方案默认 rank 8；若 protocol lock 最终保留 `{4, 8}` 候选，也只能在
outer-train group-CV 内选择并冻结。event-hash new-dev 不选择 rank、epoch、
threshold、calibration 或 fusion rule，只执行一次下述 promotion gate。

候选 `P5-C` 相对 `P0-C`、`P4-C` 和 strongest collision control 必须同时满足：

**Classification**

- four-sensor event-balanced macro-F1 `Δ >= +0.020`，paired 95% lower CI `> 0`；
- event-balanced AP `Δ >= +0.010`，paired 95% lower CI `> 0`；
- 至少 2/4 sensor macro-F1 各提高 `>= +0.020`；
- 其余 sensor 不得下降超过 `0.010`；
- all-negative-event FP mass 不增加，且 paired 95% upper CI `<= 0`。

**Calibration load-bearing**

- `P5-C` 相对 `P5-U` 的 all-negative-event FP mass 显著下降；
- event AP 非劣界为 `−0.005`，macro-F1 不下降；
- 若 C 同样改善 P0/P4，但 P5 不优于 P4，只能 claim calibration engineering，
  不能 claim response-specific sidecar。

**Segmentation**

- S2/L89/EMIT macro pixel AUPRC `Δ >= +0.020`，paired lower CI `> 0`；
- macro IoU 不下降；
- negative-mask false-positive area 不增加；
- P5 必须优于 P4；S5P coarse-cell localization 只作独立 secondary result。

**Missingness**

- 自然主要 availability strata 中无一 macro-F1 下降超过 `0.010`；
- sensor/time dropout degradation-curve AUC 不差于 strongest comparator；
- 若要 claim robustness，其 paired lower CI 必须 `> 0`，且增益不能只来自一个
  complete-case stratum。

任一 co-primary 条件失败都为 **no-go**，不得打开 outer-test。

### G4 — Single locked outer-test

G3 通过后，冻结：

- manifests/SHA、代码/容器、P0/P4/P5 checkpoints；
- calibration strata/fallback、fusion rule、threshold；
- seed aggregation、metrics、bootstrap seed 与 failure policy。

outer-test 仅运行一次，并原样重复 G3 的 primary comparisons。运行失败只允许
预先定义的 infrastructure retry；不得改模型、threshold、strata 或选另一个
checkpoint。outer-test 未复现则 paper claim 降级为 dev-only，不作 SOTA。

## 6. 结果—叙事决策表

| 结果模式 | 允许叙事 | 禁止叙事 |
|---|---|---|
| P5 仍只有 AP 正信号，macro/FP 失败 | preserve-base ranking partial signal 再次出现；decision no-go | RCTP success、SOTA |
| C 降低 FP，但 P5 ≈ P4 | event-null calibration 有用的 engineering result | sensor-response specificity |
| P5 > P4 classification，但 segmentation 失败 | sensor-conditioned classification result | dense-transfer combination novelty |
| 双任务通过，但 missingness 失败 | verified available-sensor result | missing-sensor robustness |
| P5-C 胜过 P0/P4/collision controls，双任务、FP、missingness 和 locked test 全通过 | leakage-safe 的 narrow combination claim | residual/ranking/MAE 的首创 |

## 7. Claim matrix

| Candidate claim | 当前状态 | 成立所需证据 | 决定性 falsifier | 最强允许措辞 |
|---|---|---|---|---|
| scene base 应被保护 | 有强负证据：in-place P5 macro `−2.72 pt`；sidecar 未改 base | 新 split 上 P5 sidecar > matched in-place 或至少 base 不退化 | sidecar 仍系统性损害 P0 | “representation preservation is necessary in this setting” |
| response sidecar 有真实 specificity | 未成立；P4 synthetic 曾优于 P5，现有 macro P5≈P4 | P5 > P4 的 capability、classification、segmentation paired CI 均过线 | P5≈P4 或 P4更好 | “formal sensor response contributes beyond matched augmentation” |
| train-time null selectivity + calibration 解决 FP | static post-hoc 版本已 no-go：5-fold calibrator 全选 identity，P5 FP `2.7922`、macro 比 P4 `−0.0011` | N1 相对 N0 保留 AP、提高 macro-F1且 FP upper CI `<=0`；C 只做锁定 mapping | N1 仍只改善 AP，或 FP/macro失败 | “train-time null selectivity plus cross-fitted mapping yields safer decisions” |
| multisensor 增益来自 native physics/strata | 未成立 | operator/strata arm > sensor-ID/raw-attention matched controls | raw cross-attention 相同或更好 | “fusion occurs through sensor-native evidence, not assumed token equivalence” |
| classification transfer | 当前只有 used-dev partial signal | global-split、三 seed、locked-test primary gate | new-dev/outer-test不复现 | 报 paired effect，不写 SOTA |
| segmentation transfer | 未完成 | 可信 real masks 上 P5>P0/P4/collision controls | localization不升或 FP area增加 | 只有通过后才写 “dense transfer” |
| missingness robustness | 未完成 | natural strata + dropout curve 通过 | gain只在 complete cases | 只报告经验证 availability patterns |
| narrow journal novelty | Level 2，high-overlap/fragile | 击败 residual、rank、distill、permutation、HACD/state-space、AnySat/MAE controls，并双任务复现 | 任一简单 control 解释全部增益 | “experimentally validated combination of cross-fitted temporal residuals, sensor/acquisition calibration strata, irregular six-visit dense pretraining, and dual-task transfer” |
| SOTA | 当前不成立 | clean contemporary benchmark、同 protocol 强 baselines、single locked test | 任何 protocol 不可比或 selection 污染 | 在满足前完全不写 |

## 8. 最小预算与停止纪律

预算以 derived data 已可读为前提；recrop 是 CPU/I/O critical path，不用 GPU
时间掩盖数据问题。

| 阶段 | 上限 |
|---|---:|
| verified-512 provenance/recrop pilots 与 global split | `0` GPU；约 `2–5` wall-days CPU/I/O |
| 四 sensor renderer/shortcut gates | `<= 10` A100-GPUh |
| P4/P5 × 3 seeds formal sidecar pretraining | `<= 36` A100-GPUh |
| P0/P4/P5 核心 classification + segmentation + missingness | `<= 30` A100-GPUh |
| Level-2 collision controls，先 screen 后 strongest 三 seed | `<= 38` A100-GPUh |
| locked inference、replication 与预留 | `<= 16` A100-GPUh |
| **总 hard cap** | **`<= 130` A100-GPUh** |

两张 A100 在数据就绪后理论最短约 3 wall-days；实际排期需给 I/O 与失败 gate
留余量。新增 derived cache 上限 `1 TiB`，base features 每 sensor/version 只
缓存一次，synthetic pairs 优先 deterministic on-the-fly。

在 new-dev 结果出现后，禁止：

- 增加 seed、epoch、rank、residual scale 或 response loss 权重；
- 重新定义 strata、missingness bucket 或 FP mass；
- 把失败的 P4 换成较弱 scrambled control；
- 用 current recent-dev 或 legacy test 做选择；
- 因预算未用完而继续 sweep。

## 9. 必须始终保留的 non-claims

无论结果多好，都不能 claim：

- residual/background suppression novelty；
- anomaly/tail ranking、pairwise nonconformity 或 pointwise ranking novelty；
- multisensor temporal MAE、mask-one-sensor/predict-another novelty；
- cross-attention / dual-axis attention novelty；
- current Sidecar partial AP signal 是 SOTA；
- legacy overall/S2 `0.86485/0.90616` 是 clean RCTP benchmark；
- S5P 3×3 是高分辨率 plume segmentation；
- 一个正 AP delta 足以证明 response causality、calibrated decision 或部署安全。

这篇 paper 的最高合法 claim 仍然是 **Level-2 fragile 的组合性贡献**。只有
formal response falsifier、event-null safety、classification、segmentation、
missingness 与 single locked test 全部闭环，才可把它从方案升级为结果。
