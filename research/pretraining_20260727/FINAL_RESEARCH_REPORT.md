# MethaneFuse Journal pretraining：实验结论与 research story

**日期：** 2026-07-28  
**状态：** legacy360 downstream 已锁定；RCTP L89 in-place、final-patch 与 Sidecar v2 follow-up 均已完成；promotion **no-go**，但 Sidecar 保留一个 representation-preserving ranking partial signal  
**推荐标题：**

> **Pretraining for What the Eye Cannot See: Sensor-Response
> Counterfactual Learning for Transient Methane Detection**

## 1. 一句话结论

现有 PTH 加 scale-aware time×sensor head 已把 legacy360 development F1
推到 `0.90558`，exactly-once locked test 的 overall/S2-containing F1 为
`0.86485/0.90616`；这证明 downstream 实现具有竞争力，但没有证明新
attention 或 RCTP。真正可发表的问题不是“怎样再改一次 cross-attention”，
而是：

> 现有 EO encoder 虽能接收 SWIR，却没有被训练去保留“某个 sensor 下物理上
> 符合甲烷响应的弱瞬态变化”。

拟议解法是 Response-Conditioned Counterfactual Transient Pretraining
(RCTP)：在同一背景上生成 exact clean/methane/matched-nuisance
counterfactual，只对齐 methane-induced representation difference，而不是
整幅 scene embedding。首轮 matched real-L89 transfer 已完成，但
in-place correct-response P5 没有胜过 base P0 或 response-scrambled P4，
frozen final-patch readout 也没有救回。保持 base encoder 不变的 Sidecar v2
则让 P5 的 event-balanced AP 达到 `0.76515`，paired ranking interval
相对 P0/P4 均高于 0；但 macro-F1 幅度与 lower-CI、all-negative-event FP
control 和 synthetic response-specific control 都没有过锁定门槛。因此当前
只能说“preserve-base residual 值得在新 holdout 继续验证”，不能说 RCTP
已经成功，更不能进入 sealed test。

## 2. 为什么 TransientQuery 不是主线 novelty

上一篇 MethaneFuse 已经是在 Panopticon representation 后做 attention
fusion。把 attention 显式拆成 time 与 sensor 两个 axis 是合理的 engineering
control，但它仍只聚合已有特征，并没有定义什么变化是 methane。

完整 train-cache 的消融进一步支持这一边界：

| Development arm | Dev-best F1 | F1@0.5 | Macro F1@0.5 | AP |
|---|---:|---:|---:|---:|
| scale-aware time×sensor, d64, LR 1e-4 | **0.90558** | **0.90313** | **0.89622** | 0.96003 |
| gated delta, LR 3e-4 | 0.90430 | 0.90036 | 0.89427 | **0.96042** |
| plain time×sensor, d64, LR 3e-5 | 0.89952 | 0.89616 | 0.88982 | 0.95580 |

scale-aware 明显优于 plain axial，说明 acquisition/scale conditioning 有用；
但这不是一条足以单独发表的 attention novelty。

## 3. Exactly-once legacy downstream 结果

唯一 promoted checkpoint、epoch 3、development threshold
`0.3910266161` 在查看 test metric 前已经冻结。31,564-row old-legacy test
只评估一次：

| Scope | Binary F1 | Macro F1 | AP | AUC |
|---|---:|---:|---:|---:|
| overall | **0.86485** | 0.85063 | 0.93313 | 0.93042 |
| S2-containing | **0.90616** | 0.90140 | 0.95842 | 0.96368 |
| L89-containing | 0.83623 | 0.83273 | 0.91275 | 0.91435 |
| EMIT-containing | 0.85082 | 0.80467 | 0.92817 | 0.90326 |
| S5P-containing | 0.86739 | 0.83363 | 0.94298 | 0.92622 |

相对历史 universal full-test F1 `0.82832`，overall 提升 `+3.65` points；
相对上一篇论文 360 m F1 `0.8377`，提升 `+2.71` points。S2 达到要求的
0.90 guardrail，但 overall 尚未到 0.90。

必须保留两个限制：

1. immutable old split 有 706 个 canonical events 同时出现在 source
   train/test；
2. warm-start PTH 来自历史 test-selected workflow。

因此这些数是有价值的 faithful engineering comparison，不是 clean benchmark
SOTA，也不能填入 RCTP abstract 的提升位置。

### 3.1 S2 near-0.90 归因审计

独立地，对历史 S2 old checkpoint（epoch 2）做 FP32 inference，并按
canonical event 是否与 train 重叠分层：

| Old-test stratum | Rows | F1 | AUC |
|---|---:|---:|---:|
| overlap events | 12,319 | 0.894292 | 0.956153 |
| non-overlap events | 2,635 | **0.901430** | 0.953434 |

权威产物为：

- `/home/yuyao/panopticon/Upgraded_dataset/diagnostics/s2_old_near90_event_overlap/old_checkpoint_overlap_fp32.json`  
  SHA-256
  `3702516c8cabaa4038dab73e642ce42f6c5ea79409516e1e00738c55af481440`
- `/home/yuyao/panopticon/Upgraded_dataset/diagnostics/s2_old_near90_event_overlap/old_checkpoint_nonoverlap_fp32.json`  
  SHA-256
  `414b4c5ad4178f2a36386b1e16122aff94f9f17357402ddb3a76c753ebbc865f`

706-event overlap 仍然破坏 benchmark independence，是必须披露的 protocol
limitation；但 non-overlap subset 的 F1 并未下降，反而略高，因此现有证据
不支持把历史约 0.90 F1 归因于 event overlap。这个分层同样不能把旧
test-selected checkpoint 变成 clean benchmark。

在 strict GEE source/split 上，相同训练族的 1/3/6-time best-threshold F1
分别约为 `0.8118/0.8147/0.8124`；修正后的 v14 strict split 为
`0.80886`。三种时间输入几乎持平，所以“多放几个 history visit”也不是
near-0.90 的主要解释。约 0.90 与约 0.81 的剩余差距，应优先审计
crop geometry/FOV、source product 与 preprocessing、data generation/split
细节和 training protocol；这些目前只是待检验的候选解释。另线程的 FOV audit
尚未完成，本报告不提前写入其结论。

## 4. 现有实验怎样定位真正的 gap

项目内 matched diagnostics 已多次出现“pretext 成功但 transfer 失败”：

- pixel MAE 与 validity-masked MAE 相对 matched scratch 的四传感器 macro AP
  约为 `-0.0081/-0.0098`；
- temporal correspondence pretext accuracy 达 `96.5%`，downstream AP
  却下降 `0.0229`，10% labels 时下降 `0.0466`；
- innovation-only classification 相对 t0 下降 `0.0534 AP`；
- 相反，把 matched history 用作 current observation 的 context，在 L89/EMIT
  frozen features 上分别提高 `0.0409/0.0232 AP`。

这些结果支持的不是“EO FM 忽略不可见光”，而是：

> reconstruction/consistency objective 主要奖励稳定、占面积大的 scene
> content；能接收 methane-relevant wavelength 不等于目标函数会保留微弱的
> methane response。

## 5. RCTP 方法与可证伪预测

对共享 plume column field \(C\) 和 sensor-native observation operator
\(O_s\)：

```text
x0_s = O_s(background)
x+_s = O_s(background + C)
x-_s = O_s(background + equal-energy nuisance)
```

\(O_s\) 必须包含或显式条件于 SRF/CH4 response、产品和单位、GSD、PSF/footprint、
validity、visit time、quality 与缺失。训练目标是：

1. 找到唯一被注入的 visit 和局部 support；
2. 让正确 methane response 高于 exact clean 与等能量错误 response；
3. 对齐跨 sensor 的 methane-induced \(\Delta z\)，而不是 scene embedding；
4. 用 clean anchor 限制非 plume 表示漂移；
5. S5P 保持 native 3×3 cells，不伪装成 224×224 texture。

最关键的 causal control 是 P4 response-scrambled 与 P5 correct-response：
二者初始化、render draws、监督、updates、参数量和 downstream head 完全一致，
唯一差别是 response identity。

## 6. 已完成的 synthetic mechanism evidence

冻结 Panopticon 的 L89 screen 在 4,096/2,048 train/dev controls 上得到：

| Panel | AP | AUC | Paired win |
|---|---:|---:|---:|
| all strengths | 0.7476 | 0.8382 | 0.7607 |
| weak, ≤2% | 0.5472 | 0.7076 | 0.6029 |
| history injection | 0.7400 | 0.8331 | 0.7521 |
| t0 injection | 0.7868 | 0.8632 | 0.8041 |

chance AP/paired-win 均为 1/3。这个 screen 只证明 approximate L89 methane
direction 可与两个等能量 nuisance 区分，不能称为真实分类、encoder pretraining
或 multi-sensor 结果。

在 matched continue-pretraining 中：

| Arm | Best epoch | Synthetic AP | AUC | Paired win | Selected F1 |
|---|---:|---:|---:|---:|---:|
| P4 response-scrambled | 2 | **0.97199** | **0.98145** | **0.97266** | **0.91245** |
| P5 correct response | 2 | 0.95244 | 0.96947 | 0.94727 | 0.87338 |

因此不能声称 P5 mechanism 优于 P4；scrambled response 反而更容易成为
synthetic discriminator。下面的真实 L89 P0/P4/P5 comparison 进一步检验
它是否仍有可迁移的物理优势。

## 7. Matched real L89 transfer

P0/P4/P5 的完整真实 L89 train/recent-dev cache 与同初始化 `role_only`
3-epoch heads 已完成。三组 prediction identity 完全一致：9,614 rows、
360 plumes、124 canonical events；head 参数签名和初始状态一致。以下数值直接
来自 SHA-256
`1eeea4013f10ac5cb42631d9393b3cae382b7293c243c5e18fcc555439f851d6`
的权威 `comparison.json`。

| Arm | Best epoch | Threshold | Event-balanced AP | AUC | Macro F1 (95% CI) | Positive F1 (95% CI) | Row AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 base PTH | 3 | 0.47655 | **0.74193** | **0.83867** | **0.75739** [0.72436, 0.78757] | **0.70958** [0.66461, 0.74857] | 0.75072 |
| P4 response-scrambled | 1 | 0.22905 | 0.72827 | 0.82078 | 0.74944 [0.71526, 0.78158] | 0.69766 [0.64900, 0.74034] | 0.73944 |
| P5 correct-response | 2 | 0.20781 | 0.73748 | 0.81886 | 0.73024 [0.69438, 0.76283] | 0.68651 [0.63944, 0.72766] | **0.75584** |

区间来自 2,000-replicate canonical-event-cluster bootstrap
（seed `20260728`，每个 replicate 不重新拟合 threshold）。关键 paired
delta 为：

| Contrast | Δ Macro F1 (95% CI) | Δ Positive F1 (95% CI) |
|---|---:|---:|
| P4 − P0 | −0.00795 [−0.03190, 0.01722] | −0.01192 [−0.04166, 0.01725] |
| P5 − P0 | **−0.02715 [−0.04779, −0.00565]** | −0.02307 [−0.04814, 0.00312] |
| P5 − P4 | −0.01920 [−0.04016, 0.00047] | −0.01115 [−0.03398, 0.01071] |

固定 threshold 0.5 的独立 audit 得到：

| Arm | Macro F1@0.5 (95% CI) | Positive F1@0.5 (95% CI) |
|---|---:|---:|
| P0 | **0.75732** [0.72328, 0.78847] | **0.70691** [0.66088, 0.74807] |
| P4 | 0.72078 [0.68331, 0.75762] | 0.61475 [0.55245, 0.66992] |
| P5 | 0.72859 [0.68934, 0.76506] | 0.63726 [0.57537, 0.69287] |

其中 P5 − P0 的 macro/positive F1 delta 分别为
`−0.02873 [−0.06000, −0.00073]` 与
`−0.06964 [−0.11412, −0.03019]`，结论与 dev-selected threshold 一致。

P5 的 row-level AP 比 P0 高 `+0.00512`，但这个结果让 observation 较多的
event 获得更大权重；换成每个 canonical event 等权后，P5 AP 比 P0 低
`−0.00445`，而 primary event-balanced macro F1 低 `−0.02715`，paired
95% CI 完全低于 0。P5 的最高 row AP 因而只是一个值得诊断的排序信号，
不能抵消 event-balanced classification no-go，也不能表述成有效 transfer。

本轮明确是 train/recent-dev-only engineering screen：
`test_or_sealed_read=false`，没有读取 test 或 sealed split，也没有进行 test
selection。

### 7.1 Event-balanced-loss CPU post-hoc follow-up

为检查上述 no-go 是否只是 row-heavy training objective 造成，完成了一次
**exploratory post-hoc** CPU follow-up。所有 arm 都使用 inverse
canonical-event-size BCE，使每个 event 具有相同基础 loss weight；checkpoint
按 event-balanced AP 选择，threshold 按 event-balanced macro F1 选择。三组的
initial state、batch plan、seed、optimizer、参数签名与 3-epoch cap 均匹配。
结果直接来自 SHA-256
`1ab631e2ed31aaed6c10368b121272142a70ccfe22ab0e22471f0008d817294a`
的 post-hoc `comparison.json`：

| Arm | Best epoch | Event AP | Event AUC | Event macro F1 (95% CI) | Event positive F1 (95% CI) |
|---|---:|---:|---:|---:|---:|
| P0 base PTH | 3 | **0.74281** | **0.83401** | **0.76403** [0.73144, 0.79528] | **0.70204** [0.65173, 0.74512] |
| P4 response-scrambled | 2 | 0.73399 | 0.82099 | 0.74263 [0.70840, 0.77610] | 0.67437 [0.62160, 0.72158] |
| P5 correct-response | 3 | 0.72861 | 0.81959 | 0.73931 [0.70481, 0.77235] | 0.67548 [0.62417, 0.72092] |

P5 − P0 的 event macro F1 为
`−0.02472 [−0.04610, −0.00485]`，event positive F1 为
`−0.02655 [−0.05676, 0.00077]`；同时 event AP/AUC 分别下降
`0.01420/0.01443`。因此把 loss、checkpoint selection 和 threshold selection
全部改为 event-balanced 后仍未救回 P5，反而复现了 primary no-go。P5 与 P4
也没有分开：macro F1 delta 为
`−0.00332 [−0.02115, 0.01575]`。

这项分析由已经查看过的 dev 结果触发，只能标记为 exploratory diagnostic，
不能升级为 confirmatory evidence。其审计字段为
`test_or_sealed_read=false`，没有读取 test 或 sealed artifact。它只排除了
“event-balanced head training 即可救回当前 P5”的解释；patch-local 方法由
下面独立的 frozen follow-up 检验。

### 7.2 Frozen final-patch follow-up

随后冻结 P0/P4/P5 encoder 与各自 role-only base logit，只给一个
zero-initialized、无 bias 的 192→1 local head 同位置 final-token
`t0 − mean(valid unique history)` evidence；epoch 0 必须 bit-exact 等于 base。
这个 train/dev-only diagnostic 的结果为：

| Arm | Selected epoch | Event AP base→best | Event macro F1 base→best | Row AP base→best |
|---|---:|---:|---:|---:|
| P0 | 0 | 0.741929→0.741929 | 0.757392→0.757392 | 0.750716→0.750716 |
| P4 scrambled | 1 | 0.728275→0.735145 | 0.749437→0.756928 | 0.739443→0.737900 |
| P5 correct | 0 | 0.737478→0.737478 | 0.730239→0.730239 | 0.755838→0.755838 |

2,000-replicate paired canonical-event bootstrap（124 events，seed
`20260728`）显示：

- P4 local-best − P4 base AP：
  `+0.006870 [−0.000446, +0.015446]`；
- P4 local-best − P0 AP：
  `−0.006783 [−0.055320, +0.049051]`；
- P5 best − P0 AP：
  `−0.004450 [−0.039888, +0.036818]`。

P5 selector 保留 epoch 0，即 selected local model 与自己的 frozen base
完全相同。最终 patch token readout 没有发现能救回 P5 的局部瞬态证据；P4
的小幅恢复区间也跨 0。因此 frozen final-patch 是 no-go，而不是
patch-local RCTP 的正结果。

权威 artifacts：

- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/comparison.json`  
  SHA-256
  `d72fc6525d8ae06f114c704f4832b40bfa93b45e38f75d50e6a71aa6ab9f3131`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/paired_event_bootstrap_seed20260728.json`  
  SHA-256
  `6afbf1d50637b6934e73076da9c7fbcf47d52343c0856d8087d462b623152f45`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_patch_local_followup_v1/comparison_seed20260728/PAIRED_EVENT_BOOTSTRAP.md`  
  SHA-256
  `42b687436d1bd7144e8618cd053fab7db6065bcb75bdef0be972dd0863217a53`

所有 artifacts 均声明 `test_or_sealed_read=false`；没有读取 holdout。

### 7.3 Sidecar-RCTP v2：partial ranking signal，但 promotion no-go

Sidecar v2 不再原位更新 Panopticon：base encoder byte-identical 且全冻结，
只训练 zero-initialized rank-8 response-conditioned residual（12,416 transferable
parameters），downstream 使用 `[z_base, 0.1 × residual]`；P0 用
`[z_base, exact_zero]` 保持相同 1,536-D head。Disposable probe 只能看到
residual，不能通过 frozen base feature 走捷径。P4/P5 的 draws、初始化、
1,024 updates、参数量和 optimizer 匹配；真实 L89 head 使用 inverse-event-size
BCE、event-AP checkpoint selection、event-macro-F1 threshold selection 与
2,000-replicate paired event bootstrap。

Synthetic dev 首先没有支持 response specificity：

| Arm | Synthetic dev AP |
|---|---:|
| P4 response-scrambled | **0.593971** |
| P5 correct-response | 0.387171 |

真实 recent-dev 的 point metrics 为：

| Arm | Event AP | Event AUC | Positive F1 | Macro F1 |
|---|---:|---:|---:|---:|
| P0 | 0.749149 | 0.842789 | 0.691468 | 0.761262 |
| P4 scrambled | 0.754381 | 0.842398 | 0.701309 | 0.771155 |
| P5 correct | **0.765148** | **0.848388** | **0.713650** | **0.772047** |

固定 predictions 的 paired canonical-event audit 给出：

| Contrast | Δ Event AP (95% CI) | Δ Positive F1 (95% CI) | Δ Macro F1 (95% CI) |
|---|---:|---:|---:|
| P5 − P0 | **+0.015998 [+0.002681, +0.030047]** | **+0.022182 [+0.002006, +0.044495]** | +0.010785 [−0.003346, +0.026284] |
| P5 − P4 | **+0.010767 [+0.000714, +0.020273]** | +0.012341 [−0.005099, +0.032569] | +0.000892 [−0.012323, +0.015677] |
| P4 − P0 | +0.005232 [−0.007088, +0.018610] | +0.009842 [−0.007626, +0.026766] | +0.009893 [−0.001459, +0.021052] |

这说明 preserve-base sidecar 中的 P5 有一个可重复的 event-ranking partial
signal，并非“response signal 完全为零”；相对 P0 的 positive F1 interval
也高于 0。但它没有形成可靠的 decision/specificity gain：P5 相对最强 control
P4 的 macro F1 只高 `0.000892`，区间跨 0。

更关键的是，在 27 个 all-negative canonical events 上，以每个 arm 已锁定
threshold 计算的 hard false-positive mass 为：

| P0 | P4 | P5 |
|---:|---:|---:|
| 2.742803 | **2.531088** | 3.871023 |

P5 相对 P0/P4 分别增加 `+1.128220/+1.339935`。预锁定 promotion rule 要求
`P5 macro-F1 − max(P0,P4) >= +0.020`、paired lower CI > 0，且 all-negative
FP mass 不增加；三项都没有同时满足。因此 verdict 仍为 **no-go**，不得进入
test/sealed evaluation。AP 的 partial positive 只能支持
representation-preserving residual 是比 in-place encoder update 更合理的
后续方向，不能支持 response-specific success，也不能包装成 anomaly-ranking
novelty。

权威 artifacts：

- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/comparison.json`  
  SHA-256
  `7fe0fff947c195ea3b9096e2d9b499d55b3302ae2ece9402d5ee459827a34daf`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/COMPARISON.md`  
  SHA-256
  `1beb227164425056b99e0de2d270ba063e53bc24102e3e4a28fe7258ac923310`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/ARTIFACT_SHA256SUMS.txt`  
  SHA-256
  `0c5426791b9015510bedba692d59756ab92adbb171965c7cfa1abd50b44b9987`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/sidecar_all_negative_event_fp_audit.json`  
  SHA-256
  `19f58f091915415ea73084142c1be37c26055279ccef2ed5b31d86b25e2937fd`
- `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/downstream_event_balanced_seed20260728/sidecar_all_negative_event_fp_audit.md`  
  SHA-256
  `36ed323b5b0370ac6e35aa450c970a68a23861dc904ab561bcbfee76b3dc2692`

Sidecar pretraining、downstream 和 read-only audit 均为 train/recent-dev-only；
`test_or_sealed_read=false`、`holdout_read=false`。

### 7.4 Train-only event-null calibration audit

最后用 CPU 对 frozen Sidecar checkpoints 做 post-hoc calibration
falsification；没有重新训练模型或提取特征。723 个 train canonical events
被严格分为 deterministic 5-fold event groups，event 不跨 fold。每个 fold
只用另外四折的 all-negative events 拟合 availability pattern、acquisition
quarter 或二者组合的 q75/q90 null offset；candidate 与 threshold 都只由
train OOF 选择，再一次性应用于 dev。由于继承的 heads 已经由此前 dev AP
选择，这仍是 exploratory diagnostic，不是 confirmatory run。

Train-only selector 对 P0/P4/P5 **全部选择 identity**：

| Arm | Selected calibrator | Dev event AP | Dev macro F1 | Train-OOF threshold | All-negative FP mass |
|---|---|---:|---:|---:|---:|
| P0 | identity | 0.7491 | 0.7342 | 0.6951 | **1.8331** |
| P4 scrambled | identity | 0.7544 | **0.7571** | 0.8532 | 2.3188 |
| P5 correct | identity | **0.7651** | 0.7560 | 0.5475 | 2.7922 |

在 train-OOF threshold 固定的 2,000-replicate event-cluster bootstrap 中，
P5 − P0 macro F1 为 `+0.0218 [+0.0056, +0.0403]`，但关键的
P5 − P4 为 `−0.0011 [−0.0117, +0.0092]`。相对最强 matched control 的
`+0.020` 幅度、paired lower CI > 0 与 no-FP-increase 三个 promotion clauses
全部失败。

这说明静态 availability/quarter offset 不足以把 P5 的 ranking partial
signal 转成 selectivity-safe decision；下一步需要在 train-time
representation/objective 内学习 event-null selectivity，而不是继续 post-hoc
挪 threshold。L89 是单 sensor cohort，本 audit 不能验证
sensor-conditioned calibration。

Audit：
[EVENT_NULL_CALIBRATION_AUDIT.md](/home/yuyao/panopticon/research/pretraining_20260727/EVENT_NULL_CALIBRATION_AUDIT.md)
（SHA-256
`9a969abf16f45ea7c30c72bf27def84a306a15c7f3769df3c3b04ec52519f4c1`）。
Scope 为 CPU-only train/recent-dev post-hoc；
`test_or_sealed_read=false`、`holdout_read=false`。

## 8. Research story 的 go/no-go 边界

完整证据链现在是：

1. in-place P5 显著损害 event-balanced macro F1；
2. event-balanced head training 不能救回；
3. frozen final-patch local readout 的 P5 选择 exact epoch-0 base，仍无救回；
4. preserve-base Sidecar v2 的 P5 ranking AP 相对 P0/P4 有正 paired interval，
   但 macro-F1 相对 P4 仅 `+0.000892`、区间跨 0，且 all-negative-event FP
   mass 明显增加；
5. 两代 synthetic matched control 都是 P4 scrambled 优于 P5 correct，
   response-specific causal mechanism 仍未成立；
6. 严格 train-only 5-fold event-null audit 对三组都选择 identity；static
   availability/quarter offsets 不能修复 P5 对 P4 的 decision/selectivity
   差距。

所以最终 verdict 仍是 **no-go**。不能写“RCTP 已成功”，也不能用 legacy360
`0.86485` 或 S2 `0.90616` 替代 RCTP evidence。最诚实的 partial conclusion
是：

> 在不可见弱瞬态任务上，保护已有 EO representation、把 counterfactual
> residual 放在可忽略的 side channel，可能比原位改写 encoder 更合理；当前
> dev-only 结果显示了稳定的 ranking signal，但尚未形成 response-specific、
> false-positive-safe 的分类能力，所缺的更像是 train-time event-null
> selectivity，而不是一个事后静态 offset。

这只是 representation-preserving design signal，不是方法成功、SOTA、
sensor-response causality 或 anomaly-ranking novelty。当前 recent-dev 已被反复
查看，Sidecar 也是由失败结果触发的 exploratory fallback，因此不得打开 sealed
test。

下一篇论文若继续这条线，应先做而不是事后补做：

1. 从当前 training events 冻结新的 deterministic event-hash holdout，并保存
   manifest/SHA；
2. 用 versioned formal SRF/PSF/product renderer 替换 approximate response
   vector，同时保留 response-scrambled matched control；
3. 在 train/inner-dev 内加入显式 event-null constraint，并锁定 calibrator 与
   threshold；static availability/quarter offset 已不足。L89 单 sensor 不能
   验证 sensor-conditioned calibration，因此该条件必须在 verified
   multi-sensor recrop 上检验；
4. 在 event-hash holdout 上一次性检验
   `P5 − max(P0,P4) >= +0.020 macro-F1`、paired lower CI > 0 和
   all-negative-event FP 不增加；
5. 只有上述条件通过，才进入 real-mask localization 与最终 outer sealed
   test。

当前应严格保留两条分开的结果：

- scale-aware time×sensor 是强 downstream engineering model；
- Sidecar v2 有 exploratory ranking partial signal，但完整 RCTP
  response-specific promotion 失败。

legacy360 two-axis 的结果仍然只是 faithful engineering comparison：它有旧
split event overlap 和历史 test-selected warm start，既不是 clean SOTA，
也不是 RCTP 有效性的证据。

Journal promotion 的建议硬线仍是：

- P5 相对 strongest matched comparator 至少 `+2.0` absolute macro F1，
  且 paired lower CI > 0；
- P5 明确优于 P4 response-scrambled；
- all-negative-event FP mass 不增加；
- provenance-trusted real-mask localization 同时提升；
- missing visits/sensors 和弱 sensor strata 不塌陷；
- 全局 event/source split 后，从四 sensor 的 verified 512 sources 重新 crop，
  再做一次 locked outer test。

## 9. Claim boundary

不能声称 RCTP 是首次 residual/anomaly ranking、synthetic anomaly、
multi-sensor temporal MAE、cross-sensor prediction 或 pairwise ordering。
HACD/RankMask、AnySat、ALISE、S4 与 Mixed-Modality MAE 已覆盖这些宽泛家族。

在完整证据出现前，最窄、最诚实的潜在 delta 是：

> cross-fitted irregular temporal references + sensor/product-conditioned
> counterfactual calibration + dense weak-transient pretraining + transfer to
> both classification and plume localization, under leakage-free
> matched-compute controls.

这仍是 provisional combination claim，不是已经成立的 novelty。
