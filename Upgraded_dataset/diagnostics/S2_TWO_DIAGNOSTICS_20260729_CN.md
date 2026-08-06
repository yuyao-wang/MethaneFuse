# S2 两项诊断实验结果（2026-07-29）

## 结论摘要

1. `cdse_compat` 在本次单次运行中没有改善 matched 新 GEE 训练，
   结果反而明显更低。
   第 3 epoch 的 F1/AUROC 为 `0.7555/0.8478`，同协议 raw
   基线为 `0.8247/0.9028`。
2. 第 3 个 epoch 在本次 raw 运行中带来明显提升。raw 基线从 epoch 2 的
   `0.7770/0.8707` 上升到 epoch 3 的 `0.8247/0.9028`。
3. 旧 360 m checkpoint 直接评测新 2026 cohort 时接近失效：
   固定阈值 F1 `0.6667`，test-oracle 最优阈值 F1 `0.6689`，
   AUROC `0.5685`。这不是阈值校准问题。
4. 此前“旧 checkpoint 在 matched 新 GEE 上仍有约 0.90 F1”的证据
   引用了错误的 CSV。两个旧 JSON 实际评测的是 legacy 原图，不是
   nested matched-GEE crops。真正补跑后，旧 checkpoint 在新 GEE 上的
   AUROC 只有 `0.5114`（event-disjoint）和 `0.5228`（row-random）。
5. 已知的旧 t0 辐射/通道 compat 变换可把真正新 GEE
   event-disjoint 的 AUROC 从 `0.5114` 提高到 `0.6748`。这支持该
   已知契约差异是影响因素之一，但该变换远不足以恢复到旧水位。
6. 使用真正新 GEE 训练到 epoch 3 的 checkpoint 评测 2026 cohort，
   固定阈值 F1 `0.7633`、test-oracle F1 `0.7691`、AUROC `0.8375`。
   当前 2026 cutoff 组合协议下的结果更低；该比较同时包含 cohort、
   crop/FOV、geometry 和 split 差异。旧 checkpoint 的极低分还叠加了
   明显的输入域/契约失配。

## 实验一：`cdse_compat` 完整三 epoch

数据为真正的新 GEE matched cohort：

`Upgraded_dataset/s2_legacy360_matched_gee_splits/matched_gee_crops36_splits/row_random_80_20`

兼容变换只作用于 t0：base dataset 先完成归一化，wrapper 再反归一化
回 DN，执行变换后重新归一化：

- 有效 retained bands 加 `1000 DN`；
- 将新数据的 B8A 移入旧契约的 B8 位置；
- 将旧契约中的 B8A/B9 槽位置零。

训练 recipe、split、初始化权重、batch size、optimizer 和 scheduler
均与 raw 基线一致。raw 从自身 epoch-2 `ckpt_latest` 恢复并续跑
epoch 3；corrected compat 从自身 epoch-1 `ckpt_latest` 恢复并续跑
epoch 2–3。两者都保留了各自的 optimizer/scheduler 状态。

| 版本 | Epoch | Train acc | Test F1 | Test AUROC |
|---|---:|---:|---:|---:|
| raw | 1 | 0.6642 | 0.7354 | 0.8132 |
| raw | 2 | 0.7620 | 0.7770 | 0.8707 |
| raw | 3 | 0.8126 | **0.8247** | **0.9028** |
| cdse_compat | 1 | 0.6348 | 0.6774 | 0.7573 |
| cdse_compat | 2 | 0.7058 | 0.6790 | 0.8127 |
| cdse_compat | 3 | 0.7468 | **0.7555** | **0.8478** |

同 epoch 3 比较：

- F1：`cdse_compat - raw = -0.0692`
- AUROC：`cdse_compat - raw = -0.0550`

本次单次运行未支持“+1000/B8/B9 契约补偿可把新训练恢复到约
0.90 F1”的假设。另一方面，raw 从 epoch 2 到 epoch 3 的提升说明
2 epochs 尚不是这次运行的最佳终点。

结果文件：

- `Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/row_random_80_20.cdse_compat_corrected.metrics.jsonl`
- `Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/row_random_80_20.metrics.jsonl`

## 实验二：checkpoint 直接评测新 2026 cohort

目标 test：

`Upgraded_dataset/s2_gee_legacy_notebook_6time/temporal_cutoff_split/cutoff_2025-12-22/test.csv`

该集合有 `18,245` 行，标签为 `9,080/9,165`，使用
`path_t0,path_seasonal,path_year` 作为旧 3-time 对应输入。

### 旧 360 m checkpoint，原始输入

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.5114 |
| 固定阈值 F1 | 0.6667 |
| test-oracle 最优阈值 F1 | 0.6689 |
| 最优阈值 | 0.1170 |
| AUROC | 0.5685 |
| Recall | 0.9726 |
| FPR | 0.9541 |

模型几乎把所有样本判为正类。即使在 test 上直接搜索最优阈值，F1
也只有 `0.6689`，所以这不是单纯的 calibration 问题。

### 旧 360 m checkpoint，已知旧 t0 辐射/通道 compat

固定阈值 F1 为 `0.6724`，AUROC 为 `0.6777`。该变换提高了本次
旧 checkpoint 的 AUROC，但不能恢复旧模型。

### 新 GEE matched-cohort epoch-3 checkpoint

| 指标 | Matched 新 GEE test | 新 2026 cohort |
|---|---:|---:|
| 固定阈值 F1 | 0.8247 | 0.7633 |
| test-oracle 最优阈值 F1 | 未计算 | 0.7691 |
| AUROC | 0.9028 | 0.8375 |

两侧都使用新 GEE 12-band 输入时，2026 cutoff 组合协议的固定阈值
F1 低 `0.0614`、AUROC 低 `0.0653`。该差值共同包含 cohort、
32/36 px crop/FOV、geometry 和 split 差异，不能单独归因于 cohort；
同时也不能用这组组合差异单独解释旧 checkpoint 的近随机表现。

结果文件：

- `Upgraded_dataset/diagnostics/s2_old_checkpoint_on_new_2026_cohort_with_threshold_fp32.json`
- `Upgraded_dataset/diagnostics/s2_old_checkpoint_on_new_2026_cohort_legacy_t0_cdse_contract_fp32.json`
- `Upgraded_dataset/diagnostics/s2_new_gee_matched_epoch3_checkpoint_on_new_2026_cohort_fp32.json`

## 关键证据纠错：原来的 0.90 不是新 GEE 结果

原有两个文件：

- `Upgraded_dataset/s2_legacy360_matched_gee_splits/row_random_80_20/old_checkpoint_fp32.json`
- `Upgraded_dataset/s2_legacy360_matched_gee_splits/event_disjoint_80_20/old_checkpoint_fp32.json`

都引用外层 split。其三时相影像路径 100% 指向
`finalDataset_query/legacy_param_360m/crops`。真正的新 GEE 影像位于
nested `matched_gee_crops36_splits`，而且是在上述两个 JSON 生成之后
才完成。原比较实际是：

- 旧 checkpoint → legacy 原图：F1 `0.9052/0.8987`
- 新训练 → nested 新 GEE：F1 `0.7770/0.7557`（当时均仅 2 epochs）

这不是同一影像源上的 matched transfer 对照。

本次对真正 nested 新 GEE 的补跑结果为：

| Split / 输入 | 固定阈值 F1 | test-oracle F1 | AUROC |
|---|---:|---:|---:|
| event-disjoint / raw | 0.6525 | 0.6782 | 0.5114 |
| row-random / raw | 0.6509 | 0.6744 | 0.5228 |
| event-disjoint / 已知旧 t0 辐射/通道 compat | 0.6865 | 0.6982 | 0.6748 |

结果文件：

- `Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/event_disjoint_80_20.old_checkpoint_true_new_gee_fp32.json`
- `Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/row_random_80_20.old_checkpoint_true_new_gee_fp32.json`
- `Upgraded_dataset/s2_legacy360_matched_gee_splits/reproduction_results/event_disjoint_80_20.old_checkpoint_true_new_gee_legacy_t0_cdse_contract_fp32.json`

## 当前最稳妥的判断

- 不能再说“新 GEE 影像质量已被旧 checkpoint 的 0.90 结果排除”；
  那条证据的输入路径有误。
- 已知的 radiometric/channel compat 提高了旧 checkpoint 的 AUROC，
  但在本次重新 finetune 运行中结果低于 raw。
- 训练不足是一个独立问题：raw 第 3 epoch 带来明显提升。
- 当前 2026 cutoff 组合协议的结果低于 matched 36 px 协议；差值混合了
  cohort、crop/FOV、geometry 和 split 因素，不能只归因于 cohort。
- 下一步应把注意力放在新旧 t0 的剩余处理差异（空间重采样、crop
  geometry、配准和像素级残差）以及更充分的 raw 训练，而不是继续
  把 `+1000/zero-band` 当作单一根因。

注：`test-oracle` F1 只用于诊断排序/校准上限，不能作为正式 sealed
test 指标或部署阈值。
