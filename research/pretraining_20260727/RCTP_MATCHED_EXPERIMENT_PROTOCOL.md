# RCTP 首轮 matched experiment protocol

**Protocol lock draft:** 2026-07-28 UTC  
**Scope:** 四个六时刻单传感器数据集、RCTP 首六个 objective、legacy360
time × sensor 双轴工程验证。  
**状态:** 预注册方案；本文档不声称已经得到 RCTP 结果。

## 1. 要验证的唯一主故事

已有结果支持一个问题，但尚未支持一个新预训练方法：

- shared residual MAE 相对 matched scratch 的 four-sensor macro AP 分别下降
  `0.0081` 和 `0.0098`；
- current-conditioned TransientQuery frozen head 相对 t0-only 的 AP 提升为
  L89 `+0.0409`、EMIT `+0.0232`、S5P native-grid approximation
  `+0.0120`；
- 因而当前证据只说明 scene reconstruction / correspondence 容易偏向稳定背景，
  而保留 t0 弱瞬态可能更合适。

首轮 RCTP 要证伪或支持的命题是：

> 在相同初始化、输入、更新数和 downstream readout 下，学习“同一场景加入
> 甲烷后应产生的 sensor-response-conditioned 差分”，应比重建 scene、
> 预测 latent、重建时间残差和等能量错误光谱响应更能迁移到真实甲烷分类与
> plume localization。

证据链必须完整：

```text
正确 CH4 response > 等能量错误 response 与 generic objectives
                    ↓
真实 event classification + 真实 mask localization 都提高
                    ↓
在缺 sensor / 缺 visit 条件下仍保留增益
```

双轴 attention 只作为固定 downstream readout 和 robustness probe，不是 novelty。
如果 full RCTP 与错误 response 相同，最多只能说 synthetic augmentation 有用；
如果只提高 classification 而不提高真实 mask localization，不能声称 dense
transient transfer；如果只有双轴 head 提高，论文应写 architecture result，而
不是 RCTP result。

## 2. 数据账本与可用 claim 边界

下表数字来自各目录的 `split_audit.json`、已准备的 train-only recent-dev
audit，以及 2026-07-28 对六个 role path 列的只读集合审计。

| Sensor | Named outer train / test | 已锁 train-core cap8 / recent dev | 已确认的隔离 | 当前 caveat 与用途 |
|---|---:|---:|---|---|
| S2 | 109,212 / 23,413 | 18,747 / 16,546 | 1,376 / 210 outer events，event overlap=0；outer train 最晚 2025-11-23，test 最早 2025-11-24；六 role exact-path overlap=0 | 六个 acquisition date 唯一；但尚未完成 source-acquisition/site connected-component audit。当前可做 train/dev screen；formal 前必须从已验证 512 source 统一 recrop |
| L89 | 62,389 / 7,942 | 10,049 / 9,614 | outer event overlap=0；outer train 最晚 2025-08-05，test 最早 2025-08-06；六 role exact-path overlap=0 | test 中 9 个 event 是根据旧 checkpoint 的 test errors 删除的，属于 model-conditioned test；只能做 engineering screen，绝不能作为 journal outer-test |
| EMIT32 | 34,592 / 6,138 | 11,688 / 5,201 | 1,029 / 158 outer base events，event/sample overlap=0；严格 cutoff 2025-06-09；六 role exact-path overlap=0 | 当前最接近可用的 independent temporal test；formal 前仍需 acquisition/path-hash 与 mask provenance gate。主线必须用 native 32 bands，不与 legacy WV3-simulated 16 bands 混报 |
| S5P | 25,345 / 3,213 | 18,772 / 3,805 | outer plume/event assignment overlap=0；train 最晚 2025-12-24，test 最早 2025-12-25 | 六个 raw-NC role 列中有 **573 个 distinct granule path** 同时出现在 outer train 与 test；当前 outer test 不是 acquisition-disjoint。必须重切 connected components。模型只能读 native 3×3 cells，224 upsample 仅可作旧实现诊断 |

四个 recent-dev 均从各自 outer-train 的最近 canonical events 产生，target
fraction 为 15%，train/dev/test canonical-event overlap 均为 0；screen
train 每个 `event × label` 最多 8 行，dev 不 cap。准备好的 audit 位于：

```text
/diniuvol/yuyao/methanefuse_research_20260727/manifests/
  s2_split_audit.json
  l89_split_audit.json
  emit_split_audit.json
  s5p_split_audit.json
```

### 2.1 首六实验使用的 split

1. 预训练 background、renderer calibration、normalization 和所有 synthetic
   pair 只允许来自各 sensor 的 `train_core_full`，不能读取 recent-dev 或 outer
   test payload。
2. 一种 objective 的短 screen 使用固定 `train_core_cap8` 和完整 recent-dev；
   objective 晋级后用同一 `train_core_full` 重跑。所有 arm 使用完全相同的行、
   seed 和 sample order。
3. recent-dev 只选 checkpoint、epoch、threshold 和是否晋级；任何 outer test
   数字都不能决定继续训练、改 loss 或选 arm。
4. 四个现有 outer test 均先保持 sealed。L89 与 S5P 即使最终打开，也只能标作
   legacy/engineering result；journal 主表须使用下面的 global clean split。

### 2.2 Journal formal split

正式 joint-sensor 数据从同一 512 source 重新 crop，以一个 global assignment
同时服务四 sensor。connected component 至少连接：

- 相同 canonical/base event、plume ID 或 sample ID；
- 任意六访中相同 source acquisition / granule ID / path hash；
- 相同原始 scene 与 crop source；
- 明确要声称 geographic generalization 时，再连接预注册的 site cluster；
  若不做 site-disjoint，只能称 temporal event generalization。

现有统一 source index
`/diniuvol/yuyao/methanefuse_research_20260727/manifests/multisensor_6time_512_wide.csv`
的 SHA-256 为
`2ada7e41e397689710440d242a1975271e3bb0c5bd2cc260122170da12bf0703`；
它有 19,834 plume rows，global train/dev/test 为
12,131/1,895/5,808 rows，event overlap=0。但只有 27 rows 同时拥有四 sensor，
且 v2 provenance pilot 仍缺 301/410 个生产字段并有一个 EMIT `prev3` conflict。
因此当前不能用“只保留四 sensor 完整行”的小子集做 paper result，也不能在
provenance gate 通过前 materialize formal crops。正式模型必须支持 partial
sensor sets。

## 3. 首六个 matched experiments

所有可训练 arm 从同一个 **clean Panopticon PTH** 初始化。历史上由 test 选出的
MethaneFuse/legacy360 PTH 只另跑 engineering warm-start，不进入 paper 主表。
`P0` 是 epoch-0 reference；`P1–P5` 是首轮五个 matched continue-pretraining
arm。

| ID | Objective | 唯一允许改变的内容 | 必答问题 |
|---|---|---|---|
| P0 | PTH direct transfer | 不做 continue-pretraining；只训练与其他 arm 相同的 downstream head | 新预训练是否真的优于已有 encoder |
| P1 | normalized pixel MAE | 对相同 valid native tokens 做 masked pixel/cell reconstruction；S5P 为 3×3 cell，不是 224 image | generic scene reconstruction 是否足够 |
| P2 | masked EMA-latent prediction | 用同一 mask、visible-token budget 和 EMA teacher 预测 latent | 避免 pixel loss 后，generic latent target 是否足够 |
| P3 | validity-masked residual MAE | 重建 `t0 − valid history`；duplicate visit 严格 mask | 已有 temporal residual objective 是否已经解释增益 |
| P4 | response-scrambled RCTP | 与 P5 相同 pair、plume field、mask、strength、loss、参数和 updates；只把 methane positive 的 band-response 与 metadata 做固定、可复现的 within-sensor `kappa/SRF` permutation，并保持总能量与 mask area 匹配 | 增益是否只是 synthetic shape/energy，而不是正确不可见光响应 |
| P5 | full RCTP | 正确 SRF/`kappa`/product unit/GSD/PSF；exact clean–methane pair；`L_loc + L_cf + L_col + L_visit + L_xsensor + L_bg + L_anchor` | 正确 sensor response 是否产生可迁移选择性 |

为避免 data augmentation confound，P1–P5 都读取同一 deterministic stream：
相同 background、plume field、injected visit、strength、validity、augmentation
和 hard-negative count。P1–P3 可以看 `x0/x+/x-`，但不接收 pair identity、
response label、mask、strength 或 injected-visit supervision。

P5 通过首轮后、在任何 novelty claim 前还必须补两个 confirmatory retrain：

- `P5 − L_xsensor`：验证跨 sensor response-difference alignment 是否
  load-bearing；
- `P5 sensor-ID-only`：保留正确 pixels，但移除 SRF/`kappa` metadata，验证
  模型不是只记 sensor ID。

这两个不是首六实验的一部分，但缺少它们时不能声称“response-conditioned
cross-sensor pretraining”。

## 4. Matched-compute lock

P1–P5 固定：

- initialization、encoder、response adapter、feature tap、T×S downstream
  readout、native sensor tokenizer；
- unique backgrounds、synthetic pairs、mask ratio、visible native tokens、
  sensor-balanced batches和 optimizer update 数；
- Stage-1 为 5,000 updates，2,000 updates 做一次 capability gate；
- AdamW、adapter/head LR `1e-4`、warmup 5%、cosine、bf16、clip 1.0；
- Panopticon trunk 冻结；objective-specific pretrain head downstream 全部丢弃；
- trainable encoder/adapter parameters 相同，objective decoder 参数和实测
  FLOPs 单独报告；每 arm 总 FLOPs 控制在最强 arm 的 ±5%；
- screen seed `20260727`；formal seeds
  `20260727, 20260728, 20260729`；
- 每个 objective downstream 都使用同一已锁的 head、LR grid 和数据顺序。

先在 EMIT32 与 L89 运行 P0–P5，因为 EMIT 最能验证光谱机制，L89 计算较低且
已有稳定 TransientQuery 对照；P5 未通过这两个 sensor 的 gate 时，不把失败
实现扩展到 S2/S5P 或更长 Stage-2。

## 5. 1–3 epoch downstream 规则

每个 sensor 与 fused task 都按以下顺序：

1. **Epoch 0:** 评估冻结 encoder + zero-residual/base boundary；P0 也保留这一
   条记录。
2. **Epoch 1:** 训练相同 small head。若相对 P0，event-balanced F1 和 AP 都
   下降超过 1.0 absolute point，且四 sensor 没有一个正向结果，该 arm
   engineering screen 立即停止。
3. **Epoch 2:** 必跑给未触发上条的 arm。若 train F1 上升而 dev primary
   metric 连续两次下降，判为过拟合并停止。
4. **Epoch 3:** 仅当 epoch 2 相对 epoch 1 的 dev primary F1 至少提高
   0.25 point，或已距 promotion line 不超过 1 point 且 AP 仍上升时运行。
   三个 epoch 后无条件停止。
5. checkpoint 与 threshold 只按 recent-dev 选择；同时保留固定 threshold
   0.5 的指标。坏结果不得靠几十个 epoch 等待，下一次迭代必须改变一个明确
   假设并使用新 run ID。

上面是 one-seed screen 的节省规则。进入 formal matched comparison 的 arm
都固定跑满相同 3 epochs，再在 epoch 0–3 中由 dev 选择，避免 differential
early-stop 形成 compute confound。

Pretraining 的更早停止线：

- renderer shortcut gate：只看非 methane bands 或 mask 外区域的
  synthetic-vs-clean discriminator AUROC 必须 `≤0.60`；
- 2,000 updates 后 P5 的 methane-vs-equal-energy-hard-negative ranking
  `≤55%`，停止并修 renderer/loss；
- 5,000 updates 后 P5 与 P4 的 capability 差小于 5 percentage points，
  不进入 encoder LoRA/Stage-2；
- representation collapse、有效 token 数异常、或 response norm 只由 sensor
  ID 决定，立即停止。

## 6. 锁定指标

### 6.1 Classification

每 sensor 主指标为 **event-balanced positive-class F1**：对 event `g` 中每行
赋权 `1 / n_g` 后计算 weighted TP/FP/FN；threshold 在该 sensor recent-dev
锁定。必须同时报告：

- positive F1 at locked threshold 与 F1@0.5；
- row-weighted positive F1、macro-F1、AUPRC、AUROC；
- recall at 1% 和 5% FPR；
- event-cluster paired bootstrap 95% CI；
- 每 sensor 独立值，以及
  `MacroSensorF1 = mean(F1_S2, F1_L89, F1_EMIT, F1_S5P)`。

不能用 S2 的样本量给 macro 指标加权。fused model 另报告 availability stratum
和 sensor/visit dropout degradation curve，不能用 full-row 平均掩盖最弱
sensor。

### 6.2 Real-mask localization proxy

为了低成本验证 dense transfer，冻结 encoder，给每种预训练权重接完全相同的
单层 patch/cell head，最多训练 3 epochs：

- S2、L89、EMIT 只使用 provenance 可信、非空的真实 plume mask；mask
  downsample 到 encoder patch grid，主指标为 event-balanced pixel AUPRC；
- 支持指标为 patch IoU/F1、top-score patch 命中 mask 的 pointing accuracy，
  以及 object recall at fixed false alarms；
- S2 zero/compatibility mask 不得充当真实 negative mask；
- S5P 没有可辩护的 224 plume mask，只报告 native 3×3 injected-cell AUPRC
  作为 capability，不计入“真实 segmentation transfer”。

这个 proxy 不是 full segmentation SOTA，但足以证伪“encoder 学到了可定位
transient”。若 proxy 通过，再用同一锁定 encoder 跑正式 decoder。

## 7. Promotion magnitude：什么结果才够写 journal

### 7.1 One-seed mechanism promotion

P5 只有同时满足以下条件才进入三 seed：

- 对 P4，methane-vs-wrong-response ranking 至少 `+10` percentage points；
- 对 `max(P1,P2,P3,P4)`，四 sensor dev `MacroSensorF1 ≥ +1.0` point 且
  macro AP `≥ +1.0` point；
- 至少 3/4 sensor 的 F1 delta 非负；
- S2/L89/EMIT 中至少一个真实-mask pixel AUPRC `≥ +2.0` points。

否则先判断 renderer shortcut、loss scale 或 local patch evidence；不通过
后仍重复时放弃 RCTP 主线。

### 7.2 Journal go criterion

正式三 seed、clean split 的 P5 相对最强 matched comparator 必须满足：

1. 四 sensor `MacroSensorF1 ≥ +2.0` absolute points，event-bootstrap 95% CI
   下界 `>0`；
2. 至少两个 sensor 各提高 `≥2.0` F1 points，其他 sensor 不得系统性下降
   超过 `1.0` point；或者最难 sensor 提高 `≥4.0` points且 macro 提高
   `≥2.0` points；
3. P5 相对 P4 至少提高 `1.0` MacroSensorF1 point，并满足 response
   capability 的 `+10` percentage-point gate；
4. 至少一个高分辨率 sensor 的真实-mask pixel AUPRC 提高 `≥3.0` points、
   patch IoU/F1 提高 `≥2.0` points，且另两个可评估 sensor 没有
   `>1.0` point 的系统性下降；
5. confirmatory `P5 − L_xsensor` 或 sensor-ID-only 至少一个产生预注册的
   明显回落，否则收窄 cross-sensor/response metadata claim。

只有 `0.2–0.5` point 的不稳定提升不足以支撑 journal 主方法。

绝对数值另设工程线，不能替代上面的因果对照：

- clean S2：`F1 ≈0.895` 是需要逼近/超过的 legacy guardrail；历史
  `0.8955` checkpoint 由旧 test 反复选择，只是参考，不是 formal SOTA；
- clean full multisensor：`≥0.88` 为强结果，`≥0.90` 为 stretch；
- 若只有 S2 达到 0.90、但 P5≈P4 或其他 sensor 退化，不能宣称 RCTP 成功；
- 若未达到 0.90、但上述 matched effect、真实 localization 与 falsifier
  全部通过，可以讲方法 story，但不能写“达到 90 F1 / SOTA”。

## 8. 双轴实验如何接入、而不混淆 story

当前 legacy360 dual-axis launcher 尚未产生 head-sweep 结果。其固定比较应为：

```text
current-only sensor fusion
vs T×S two-axis
vs scale-aware T×S（S5P history masked）
```

所有 arm 使用同一 cached `[row, sensor, time-role, 768]`、相同 base、3 epochs
和 dev threshold。其用途是先锁定一个 readout；之后 P0–P5 全部复用该 readout，
不能让 RCTP 独享更强 architecture。

legacy360 只能做 engineering evidence：

- source train/test 有 706 个 canonical-event overlap；
- historical universal/S2 PTH 曾由 test 选择，且当前 dev 来自该 PTH 原先见过的
  source-train；
- sealed test 不能用于 head、epoch 或 threshold 选择。

双轴 readout 的 promotion line 为：相对 current-only，clean dev fused F1
至少 `+1.0` point，并改善至少两个 sensor-containing strata；否则固定使用
更简单 readout。正式双轴结果只能来自统一 512 recrop global split，并需报告
T-only、S-only、T→S 与 missing-sensor/visit curve。无论结果多高，都不把
“两个 attention axis”写成 RCTP novelty。

## 9. 一次性 outer-test 与审计产物

在完成 P0–P5、两个 confirmatory ablation、三 seed 和 readout lock 后：

1. 冻结每 sensor/fused threshold、三个 checkpoint SHA、代码 SHA、manifest
   SHA、normalization、renderer config 和唯一 winner；
2. 以一个命令批量执行三个已锁 seed，对 clean outer-test 只读取一次；
3. 按 event 做 paired bootstrap，不能再根据 test 改 epoch、threshold、loss
   或 seed；
4. 旧 L89/S5P/legacy360 test 结果单列为 engineering，不与 clean global
   test 混表。

每个 arm 最少保存：

```text
split_audit.json
source_manifest.sha256
renderer_config.json + renderer_config.sha256
init_checkpoint.sha256
run_config.json
capability_history.json
downstream_metrics_epoch_0_to_3.json
event_predictions_dev.csv
localization_predictions_dev/
checkpoint_best_dev.pth
selection_lock.json
sealed_test_result_once.json   # 仅最终 winner
```

## 10. 最终判决模板

```text
P5 <= P4:
  response-conditioned 主张失败；停止或只写 synthetic augmentation

P5 > P4，但 P5 <= strongest generic objective:
  物理信号可学，但没有证明比 generic pretraining 更适合任务

classification 提高、real-mask proxy 不提高:
  只能写 classification，不写 dense transfer / segmentation

classification + localization 提高，但 no-L_xsensor 不回落:
  写 per-sensor RCTP；删除 cross-sensor alignment 主张

P5 同时通过 matched objective、response falsifier、real-mask transfer、
clean split 与 3-seed CI:
  才进入 SOTA 调参与 journal 主表
```

这套 protocol 的目的不是保证一个 0.90 数字，而是保证：如果最后出现亮眼
数字，能够明确归因于适合不可见甲烷响应的预训练，而不是旧 PTH、test threshold、
错误 split、更多参数或双轴 attention。
