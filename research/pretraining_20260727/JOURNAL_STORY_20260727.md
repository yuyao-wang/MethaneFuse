# MethaneFuse Journal research story：弱瞬态不应被“解释掉”

日期：2026-07-27

状态：validation-only、可复现实验结论；不是最终 SOTA 声明

暂定方法名：**TransientQuery**

## 一页结论

这轮实验得到的最重要结果不是“又找到一个更复杂的 MAE”，而是一个
可以写成论文的问题—诊断—方法闭环：

> 通用 EO foundation model 的重建、背景预测和同地点一致性目标擅长学习
> 跨时刻稳定的地表内容；甲烷 plume 却是只在当前观测中短暂出现、面积小、
> 光谱效应弱的瞬态。对这类目标，预训练越成功地解释稳定背景，越可能把
> 真正需要保留的当前异常当成 nuisance 丢掉。

三类 matched-control 结果支持这个诊断：

1. 四传感器 shared pixel MAE 的 reconstruction/transfer 审计全部通过，
   但相对 scratch 的 macro AP 为 `-0.0081`；validity-masked MAE 为
   `-0.0098`。
2. L89 历史预测器已经达到 validation cosine `0.8496`，但
   innovation-only 分类相对 t0-only 平均掉 `0.0534 AP`；把 innovation
   与 t0 拼接仍掉 `0.0152 AP`。
3. label-free 同地点 correspondence pretext 从 `62.7%` 学到 `96.5%`
   accuracy，但 downstream AP 在 100% labels 下掉 `0.0229`，在 10%
   labels 下掉 `0.0466`；而完全 matched 的 permuted-label control 并未
   出现这种语义收益。

因此本轮不应宣称“成功的 temporal pretraining”。真正的正结果是一个
更窄、但更可靠的训练原则：**保留 t0 为唯一决策 query，只让去重后的历史
观测作为有角色的比较 context；不重建 t0，不把 t0 拉向历史，也不先压缩成
残差。**

在 frozen official Panopticon CLS features 上，参数和训练预算完全匹配的
两层 current-query cross-attention head 得到：

| Sensor | t0-only AP | TransientQuery AP | `ΔAP` | `ΔAUROC` | `Δmacro-F1` |
|---|---:|---:|---:|---:|---:|
| L89，3 seeds | 0.7243 | **0.7652** | **+0.0409** | **+0.0367** | **+0.0300** |
| EMIT，3 seeds | 0.7627 | **0.7858** | **+0.0232** | **+0.0143** | **+0.0106** |
| S5P native-grid approximation，3 seeds | 0.5655 | **0.5775** | **+0.0120** | **+0.0103** | **+0.0080** |
| 三传感器 macro | 0.6841 | **0.7095** | **+0.0253** | **+0.0204** | **+0.0162** |

L89 的三个随机种子中，`role_only - t0_only` 的 event-cluster bootstrap
AP 95% CI 全部严格大于零。EMIT 三个种子的点估计均为正，但单种子
event-cluster CI 只有一个严格大于零；所以 EMIT 是一致的跨传感器复现，
还不是足够强的独立统计结论。S5P 是方向性证据，而且当前实现是对上采样
产品做 finite-mask adaptive pooling 得到的近似 native `3×3` grid，不应
写成最终 native-product SOTA。

最关键的机制对照也成立：

| Sensor | TransientQuery AP | 跨 event 打乱历史训练 AP | `ΔAP` |
|---|---:|---:|---:|
| L89 | 0.7652 | 0.7182 | **+0.0469** |
| EMIT | 0.7858 | 0.7584 | **+0.0274** |
| S5P | 0.5775 | 0.5602 | **+0.0173** |

这说明收益来自与当前地点匹配的历史内容，而不只是参数量、额外 token 或
训练正则。反过来，加入精确 `Δdays` Fourier encoding 没有在 L89 或 EMIT
上胜过离散 role-only，且预注册 gate 均失败。因此目前不应把“irregular
continuous-time encoding”写成 load-bearing novelty。

## 推荐的论文标题

首选：

> **Preserving Weak Transients in Earth-Observation Foundation Models:
> Current-Conditioned Temporal Comparison for Methane Detection**

更有记忆点的版本：

> **Do Not Explain Away the Plume: Learning from Irregular Earth-Observation
> Histories for Methane Detection**

如果最终 physics-conditioned pretraining 和 segmentation 都通过，再使用：

> **Do Reconstruction Objectives Erase Weak Transients? Pretraining
> Irregular Multisensor Earth-Observation Sequences for Methane
> Classification and Segmentation**

当前证据只足以支撑前两个标题，尚不足以支撑第三个标题中的正向
“pretraining”贡献。

## 可直接改写的 abstract

Earth-observation foundation models commonly learn by reconstructing masked
content, predicting nearby acquisitions, or enforcing consistency across
observations of the same location. These objectives favor persistent surface
semantics, whereas methane plumes are weak, spatially sparse, and transient.
We study this objective–target mismatch using leakage-free, matched-compute
experiments on irregular multi-visit Landsat 8/9, EMIT, and Sentinel-5P
sequences. Successful pretext optimization does not imply useful methane
transfer: shared masked reconstruction reduces macro average precision,
a cross-fitted background predictor reaches high latent similarity while its
innovation representation loses classification signal, and a temporal
correspondence task reaches 96.5% pretext accuracy yet degrades downstream AP
by 2.3 points with full labels and 4.7 points in the low-label regime. We
therefore introduce TransientQuery, a current-conditioned temporal comparator
that retains the current acquisition as the decision query and uses only
unique, role-tagged historical observations as masked context, without
reconstructing or enforcing invariance on the current evidence. Across three
seeds, TransientQuery improves AP by 4.1 points on Landsat 8/9 and 2.3 points
on EMIT, with a further directional 1.2-point gain on a native-grid Sentinel-5P
screen. Cross-event history permutation removes the gains, whereas precise
continuous-time encoding does not improve over acquisition roles. These
results identify persistent-scene bias as a concrete failure mode of generic
temporal pretraining for weak EO transients and motivate objectives that
preserve, rather than explain away, current evidence.

上面 abstract 暂时不能加入 “state of the art”。必须先完成 clean S2、锁定的
outer-test、强 backbone comparator，以及 classification/segmentation 双任务
确认。

## 论文故事如何复用 MethaneFuse 的逻辑

上一篇 MethaneFuse 的叙事链是：

1. 现实问题：实际部署时 sensor 经常部分缺失；
2. 现有方法缺口：模型默认所有 sensor 同时存在；
3. 数据与方法：建立现实 benchmark，并针对 partial-sensor inference 设计
   模型；
4. 量化结果：在 F1/AUROC 上取得数个点的改善。

下一篇可以严格复用同一逻辑，但把问题换成“时间上的弱瞬态”：

1. **现实问题**：plume 只短暂出现在 t0，六个 role 又不规则、缺失且可能
   重复，并不是规则视频。
2. **现有方法缺口**：generic EO pretraining 偏向恢复/对齐跨时刻稳定内容；
   这种 persistent-scene prior 与 methane transient 的统计结构冲突。
3. **实验证据**：MAE、背景预测残差、同地点 correspondence 三条直观路线
   都在严格对照下负迁移，而且 pretext 本身确实学成功。
4. **方法原则**：当前证据不可被重建或不变性约束吞掉；t0 必须保留为 query，
   历史只提供去重、缺失感知、有角色的条件比较。
5. **结果**：L89/EMIT 得到 2–4 AP 点的稳定提升，打乱历史后收益消失；这比
   “多加三帧”更接近一条可验证的能力提升。

## 方法：TransientQuery

设 frozen 或可训练的 sensor-native encoder 输出
`z = [z0, zprev1, ..., zyear]`。模型执行：

1. 给每个有效 observation 加 acquisition-role embedding；
2. 用 acquisition identity 去重，同一 granule 不允许被当成多份证据；
3. missing/invalid observation 通过 attention mask 排除；
4. 以 `z0` 为唯一 query，使用两层 cross-attention 从历史 context 读取信息；
5. 分类器只读取更新后的 current query。

本轮四个 matched arms 共享完全相同的模型参数形状、初始化、batch order、
optimizer steps 和验证选择规则：

- `t0_masked`：只有当前 acquisition；
- `role_only`：完整去重历史和离散 role；
- `delta_time`：在 role 上再加入真实 `Δdays`；
- `history_shuffle_train`：保留 t0，但把完整历史替换为另一 canonical event
  的历史。

`role_only` 是当前应该推进的版本。`delta_time` 在 L89/EMIT 的预注册目标
“必须比 role-only 至少高 0.005 AP”上三个种子全部失败；更大 `model_dim=512`
也没有超过 `256` 维版本。因此当前收益不能简单归因于连续时间编码或 head
容量。

需要明确：本轮只训练 temporal head，输入是 frozen Panopticon CLS；它验证
了 temporal decision rule，但还不是完整 FM backbone pretraining。

## 完整结果

### L89：主结果

严格 train/validation 为 `10,033 / 9,614` rows，`723 / 124` canonical
events，event overlap 为零；六个 roles 均经过 acquisition 去重。三种子均值
± sample SD：

| Arm | AP | AUROC | macro-F1@0.5 |
|---|---:|---:|---:|
| t0-only | 0.7243 ± 0.0112 | 0.7670 ± 0.0104 | 0.7022 ± 0.0110 |
| **role-only** | **0.7652 ± 0.0099** | **0.8037 ± 0.0092** | **0.7323 ± 0.0105** |
| role + `Δdays` | 0.7619 ± 0.0118 | 0.7979 ± 0.0124 | 0.7222 ± 0.0164 |
| shuffled history | 0.7182 ± 0.0046 | 0.7661 ± 0.0100 | 0.6998 ± 0.0059 |

Event-cluster bootstrap（每个 seed 2,000 replicates）：

- role-only minus t0-only AP：
  - seed 20260727: `+0.0428`, CI `[0.0271, 0.0615]`
  - seed 20260728: `+0.0410`, CI `[0.0255, 0.0588]`
  - seed 20260729: `+0.0388`, CI `[0.0210, 0.0585]`
- role-only minus shuffled-history AP：
  - seed 20260727: `+0.0450`, CI `[0.0282, 0.0659]`
  - seed 20260728: `+0.0422`, CI `[0.0259, 0.0607]`
  - seed 20260729: `+0.0535`, CI `[0.0270, 0.0822]`

### EMIT：跨传感器确认

严格 train/validation 为 `11,688 / 5,201` rows，validation 有 129
canonical events，event overlap 为零。cache 中重复 acquisition 被显式
抑制：train `3,416` 个，validation `1,787` 个。三种子均值 ± sample SD：

| Arm | AP | AUROC | macro-F1@0.5 |
|---|---:|---:|---:|
| t0-only | 0.7627 ± 0.0054 | 0.8144 ± 0.0041 | 0.7368 ± 0.0090 |
| **role-only** | **0.7858 ± 0.0040** | **0.8287 ± 0.0040** | **0.7475 ± 0.0018** |
| role + `Δdays` | 0.7803 ± 0.0016 | 0.8247 ± 0.0017 | 0.7407 ± 0.0113 |
| shuffled history | 0.7584 ± 0.0013 | 0.8129 ± 0.0052 | 0.7192 ± 0.0236 |

每个 seed 的 role-only minus t0-only AP 为 `+0.0195`、`+0.0180`、
`+0.0320`。对应 event-cluster CI 前两个跨零，第三个为
`[0.0106, 0.0524]`。role-only minus shuffled-history AP 为 `+0.0302`、
`+0.0226`、`+0.0295`；第一、第三个 CI 严格大于零，第二个下界为
`-0.0005`。这应表述为一致且有机制支持的 replication，而不是三个独立的
显著结果。

### S5P：只作为方向性结果

三种子均值：

| Arm | AP | AUROC | macro-F1 |
|---|---:|---:|---:|
| t0-only | 0.5655 | 0.5290 | 0.5225 |
| **role-only** | **0.5775** | **0.5394** | **0.5305** |
| role + `Δdays` | 0.5765 | 0.5389 | 0.5320 |
| shuffled history | 0.5602 | 0.5249 | 0.5208 |

排序信号只略高于随机；在真正 native S5P 文件上复现前，不进入 headline。

## 三个关键负结果

### 1. Reconstruction 不是答案

在全球 canonical-event purge 后，unmasked MAE 与 validity-masked MAE
相对 matched scratch 的 macro AP 分别为 `-0.0081` 和 `-0.0098`。
transfer coverage、head 初始化、更新预算、manifest 和 metric 重算均通过
审计。这不是 checkpoint 没载入或数据泄漏导致的假阴性。

### 2. “预测背景，再看 residual”会丢信号

三折 event-level cross-fitting 的 L89 predictor：

- 小模型：OOF cosine `0.8287`，validation cosine `0.8269`；
- shape-matched 256 维模型：OOF cosine `0.8516`，validation cosine
  `0.8496`。

但下游两种子均值：

| Readout | AP |
|---|---:|
| t0-only | 0.7222 |
| predicted background only | 0.6654 |
| innovation only | 0.6688 |
| t0 + innovation | 0.7070 |
| shuffled innovation | 0.6885 |

预测得好的 stable component 并不等于 methane-discriminative component；
显式 residual 也没有增加可用信息。

### 3. 同地点 correspondence 学得越好，迁移反而越差

pretext 使用 label-free、跨 event 的全局双射 donor，t0 不进入 K/V，
correspondence 与 permuted-label control 使用相同初始化、batch plan、
237 optimizer steps 和 60,198 examples。correspondence accuracy 达到
`0.9654`，permuted control 保持在 `0.5226`，说明任务确实被学会。

| Labeled events | Scratch AP | Correspondence AP | `ΔAP` |
|---|---:|---:|---:|
| 10% | 0.6609 | 0.6143 | **-0.0466** |
| 100% | 0.7495 | 0.7267 | **-0.0229** |

这不是“预训练没收敛”，而是 pretext semantics 与 weak-transient target
错位。它为 persistent-scene bias 提供了最直接的机制证据。

## 与相关工作的边界

不能声称以下任何一项本身是新概念：

- masked satellite pretraining：已有
  [SatMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)；
- heterogeneous EO masked/teacher-latent transfer：已有
  [AnySat](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)；
- irregular acquisition-aware satellite latents：已有
  [ALISE](https://arxiv.org/abs/2407.08448)；
- same-location temporal/seasonal contrast：已有
  [SeCo](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)
  和
  [CACo](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html)；
- video temporal ordering/contrast：已有
  [Shuffle and Learn](https://arxiv.org/abs/1603.08561) 和
  [CORP](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.html)；
- background prediction、residual anomaly detection、anomaly ranking 和
  pairwise nonconformity 均是已占据的 family。

可以防守的 delta 是：

1. 在弱甲烷瞬态上系统展示 reconstruction、background prediction 和
   correspondence 三种 persistent-scene objective 的可控负迁移；
2. 在不规则、缺失、重复 acquisition 的多传感器序列上，验证
   current-conditioned / role-aware / duplicate-masked comparison；
3. 同时用 current-only、跨 event history shuffle、continuous-time、
   capacity 和 matched pretext-label permutation 排除 shortcut；
4. 最终把同一原则迁移到 event classification 和 plume segmentation。

第 4 点尚未完成。因此 novelty 仍应标为 survey 给出的
**Level 2 — High Overlap (fragile)**，不能升级为“新 residual principle”或
“新 temporal correspondence principle”。

## 下一轮只做四件事

### Gate A：把正结果从 head 提升到完整模型

在 L89 和 EMIT 上，各跑一个 matched pair：

- current-only full backbone finetuning；
- TransientQuery full backbone finetuning。

固定 3 epochs、相同 optimizer updates、相同 train-only normalization、
相同 checkpoint-selection metric。promotion 条件：

- 每个 sensor `ΔAP >= +0.02`；
- 两者 macro-F1 非负；
- history-shuffle 重训至少抹掉一半 AP 增益；
- 不再调 architecture。

### Gate B：完成 clean S2，而不是复用受污染的旧 crop

必须从 512 source 重新 crop，统一地点、time role、canonical plume/event ID，
再统一 split。当前任务中一次只读 `rg` 范围过宽，意外显示了另一份 S2
all-rows 文件里带 `test` 标记的行；没有打开目标 S2 test manifest、没有使用
任何数值，但从保守 protocol 出发，本轮所有 S2 结果和模型选择全部排除。

### Gate C：把同一 current-query context 接到 segmentation decoder

只允许历史改变 t0 decoder 的 context，不允许历史直接生成 plume mask。
主指标为 pixel AUPRC / IoU，并包含历史打乱对照。只有 classification 与
segmentation 都提高，才能使用 survey 中那条窄 combination delta。

### Gate D：如果仍要做 FM pretraining，只测试“瞬态保留”目标

不要再增加 MAE、correspondence 或 residual predictor 的 epochs。唯一值得
进入新 pilot 的方向是 sensor-response-aware 的 counterfactual transient
pretraining：在 train-only background histories 中，用传感器 spectral
response 和空间 plume prior 注入弱、局部、仅存在于 t0 的 counterfactual，
训练 current-query encoder 定位/分类注入，而不是重建它。必须配：

- spectrum-shuffled injection；
- spatial-mask permutation；
- matched generic augmentation；
- classification 与 segmentation transfer；
- 完整 physics/realism audit。

这只是由当前负结果推导出的下一候选，不是已验证贡献。若 1-seed L89 pilot
不能比 scratch 和 role-only 至少高 `0.02 AP`，立即停止。

## 当前不能写的内容

- “我们已经得到四传感器 SOTA”；
- “六时刻 MAE 提升 methane detection”；
- “background residual / anomaly ranking 是首次提出”；
- “continuous irregular time encoding 是增益来源”；
- “EMIT 三个种子都达到 event-level significance”；
- “S5P 已经解决”；
- 把 validation 指标与旧 README 中不同 split 的 test 指标直接比较；
- 再次打开已用于历史评估的 sealed tests 来做方法选择。

## 权威产物

核心代码：

- `research/pretraining_20260727/l89_ragged_cls_experiment.py`
- `research/pretraining_20260727/emit_ragged_cls_experiment.py`
- `research/pretraining_20260727/l89_innovation_pretrain_experiment.py`
- `research/pretraining_20260727/l89_correspondence_pretrain_experiment.py`
- `research/pretraining_20260727/evaluate_l89_ragged_gate.py`

核心结果目录：

- `/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260727`
- `/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260728`
- `/diniuvol/yuyao/methanefuse_research_20260727/results/l89_ragged_cls_v1_seed20260729`
- `/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260727`
- `/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260728`
- `/diniuvol/yuyao/methanefuse_research_20260727/results/emit_ragged_cls_v1_seed20260729`
- `/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_innovation_v1b`
- `/diniuvol/yuyao/methanefuse_research_20260727/experiments/l89_correspondence_v1_seed20260727`

Correspondence 正式审计：

- `run_status=complete`
- 269 项只读检查通过，0 error
- `summary.json` SHA-256:
  `8f7d13b659b25be89e477edab6a717b482843abd5b9b3b5a1f77d95d745c02b9`
- exact launch script SHA-256:
  `a9da431dc0d23fd636a221be6470818c79af0fee028a66a4602259d34d2e4137`

EMIT evaluator：

- evaluator SHA-256:
  `c2cb5ecc42b22f43cdd714227ce1e80375f6dd32a9d0431e3c369a088a4c0eb4`
- evaluator CPU tests: 10/10
- EMIT runner CPU tests: 4/4
- 每个 EMIT seed 的 gate integrity: all checks pass
