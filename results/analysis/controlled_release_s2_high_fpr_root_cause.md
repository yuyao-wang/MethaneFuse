# S2 原表负样本高 FPR 根因审计

审计日期：2026-07-31

## 结论

原表测试的高 FPR 不是单一阈值问题，也不能完全归因于一次预处理错误。

1. **负标签证据不足。** 进入本次 S2 评估的 45 个表内负事件中，只有 1 个能在官方逐次释放表中直接核验为 Sentinel-2 的计量零释放。其余 44 个不能称为“经地面真值确认的无甲烷场景”。
2. **测试物化程序确有 resize 几何差异，但它不是高 FPR 主因。** 按训练生成器的 `torch.nn.functional.interpolate(..., align_corners=False)` 重建全部数据后，原表 FPR 仍为 97.8%（44/45）。
3. **主因是训练标签语义和当前测试任务不一致。** 训练时的负样本几乎都是“同一个甲烷正场景中远离已知源中心的 crop”，模型主要学成了“crop 是否包含已知中心/设施及其固定视觉背景”，而不是“同一设施在这个日期是否存在 plume”。
4. **域偏移和分数校准进一步放大问题。** 表内负样本也在设施中心附近裁剪，模型对它们给出饱和高分。事后调阈值可以降低 FPR，但无法同时保住 recall，不能把当前模型变成可靠的 temporal non-release classifier。

## 1. 负样本来源核验

原始表有 60 个负标签；其中 45 个具备完整的 S2 三时相输入并进入当前评估。逐事件核验结果如下。

| 证据类别 | 数量 | 能否作为严格无甲烷真值 |
|---|---:|---|
| 官方 S2 逐次释放表中的计量零释放 | 1 | 是 |
| 控制释放实验期外，假设没有受控释放 | 28 | 只能说明不在该实验期，不能排除现实甲烷源 |
| 实验期内，但找不到对应 S2 计量记录 | 2 | 否 |
| MethaneAIR L4point 可用期内目录无记录 | 5 | 否；检测目录缺项不等于 non-detection |
| 日期超出 MethaneAIR L4point 可用期 | 9 | 否 |

### Ehrenberg 2021

Scientific Reports 论文明确说明受控释放期为 2021-10-16 至 2021-11-03，释放量包含零值；论文和官方配套 CSV 只有一个 Sentinel-2 零释放观测，即 2021-11-01。当前进入评估的 21 个 Ehrenberg 负样本全部在实验期外，原表也没有包含 2021-11-01 这个真正的 S2 零释放事件。

来源：

- https://www.nature.com/articles/s41598-023-30761-2
- https://github.com/esherwin/SatelliteTesting/blob/master/Data/matchedDF_Satellites_230130.csv

### Casa Grande 2022

AMT 论文明确说明实验期为 2022-10-10 至 2022-11-30，并包含计量零释放。当前 10 个 Casa Grande 负样本中，只有 2022-11-25 能匹配官方 S2 零释放记录；2 个实验期内日期找不到对应 S2 计量行，另外 7 个在实验期外。论文中的“true negative”和“0 false positives”是针对计量零释放或远低于传感器检出限的正式实验观测，并不能自动赋给同地点的任意其他日期。

来源：

- https://amt.copernicus.org/articles/17/765/2024/
- https://doi.org/10.5281/zenodo.10149991

### MethaneAIR

Google Earth Engine 的 MethaneAIR L4point 产品是**高排放甲烷点源检测目录**，不是包含 survey coverage、non-detection 或 no-emission 标签的完整观测表。其 schema 只有 plume、flux、flight 和时间等检测属性，而且官方注明并非每个航班都有全部产品。因此“目录中没有点”不能转换成负标签。产品可用期为 2021-07-30 至 2023-10-13；当前 14 个 MethaneAIR 负样本中还有 9 个日期在可用期之外。

来源：

- https://developers.google.com/earth-engine/datasets/catalog/EDF_MethaneSAT_MethaneAIR_L4point

逐事件审计见 `results/analysis/controlled_release_45_negative_provenance_audit.csv`。

## 2. 数据处理核验

历史 360m 训练 crop 生成提交 `b955193c8df71d3790f050f100b5e6ffbea4af55` 使用：

```python
F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
```

旧测试物化程序曾使用 NumPy endpoint-aligned resize，几何上不完全一致。现已增加严格训练合同并重新生成全部输入：

- 原始 crop：36×36
- resize：224×224，bilinear，`align_corners=False`
- 时相顺序：t0、t-90、t-360
- t0 band 布局：`B1,B2,B3,B4,B5,B6,B7,B8A,0,0,B11,B12`
- 历史时相布局：`B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`
- 存储：float32
- 归一化：复用训练/推理数据加载器的 S2 预计算统计量

严格重跑的原表结果：

| 数量 | TP/FN | TN/FP | FPR | Recall | Accuracy | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 正 + 45 负 | 9/0 | 1/44 | 97.8% | 100.0% | 18.5% | 0.699 | 0.383 |

旧 resize 版本的 FPR 是 95.6%；严格修正后反而是 97.8%。因此 resize 差异是真实的可复现性缺陷，但不是模型无法识别表内负样本的原因。唯一经官方确认的 2022-11-25 S2 零释放事件仍被预测为正，正类概率 0.9183。

另一个需要记录的合同问题是：这些 S2 GeoTIFF 的地面像元实际为 20m，36 像素约对应 720m，而训练目录把它称为“360m”是因为生成器按 10m GSD 命名。训练和本次严格测试都沿用了相同的 36 像素合同，因此它不是本次 train/test 不一致的来源，但“360m”这个物理尺度名称并不准确。

## 3. 模型到底学成了什么

训练生成器的标签规则是：

- 正样本：随机偏移，但 crop 必须包含已知 plume/source 中心；
- 负样本：仍取自同一个甲烷正场景，但随机到 512×512 图像其他位置，并拒绝任何包含中心的 crop。

训练 manifest 的 S2 部分共有 59,635 个 crop：

| 标签 | 数量 | 距中心半径中位数 | 最大半径 | 半径 ≤31 px |
|---|---:|---:|---:|---:|
| 正 | 30,296 | 14.0 px | 24.0 px | 100.0% |
| 负 | 29,339 | 190.7 px | 333.8 px | 0.99% |

所以训练分布把“靠近中心”和正标签几乎绑定在一起。当前表内负样本为了与正样本构造一致，也采用正样本式偏移，其半径中位数为 18.1 px；这正好落入训练中几乎从未作为负类出现的区域。

严格预处理后的因果对照使用完全相同的 144 个正 crop：

| 负样本构造 | TP/FN | TN/FP | FPR | Recall | Accuracy | AUROC | AP | 负类分数均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 同一正场景，远离中心 | 136/8 | 126/18 | 12.5% | 94.4% | 91.0% | 0.965 | 0.955 | 0.148 |
| 表内负日期，按正样本式中心偏移 | 136/8 | 15/129 | 89.6% | 94.4% | 52.4% | 0.608 | 0.587 | 0.806 |

唯一改变的是负样本的语义和空间位置。结果说明：

- 模型非常擅长解决它在训练中被教会的任务：区分“包含已知中心的 crop”和“远离中心的 crop”；
- 模型只保留了较弱的 plume 排序信号，不能可靠区分“同一设施有释放日期”和“同一设施无释放日期”；
- 模型输入中没有经纬度，因此严格地说它没有记住数值地理坐标；它学到的是可从图像读出的**设施/地表纹理、中心构图、固定背景、静态光谱和时相伪特征**，这些是地理位置的视觉代理；
- AMT 论文也警告，若没有已知源位置，真实 plume 可能和背景、云及高反射/吸收地物造成的伪影难以区分。这会进一步加剧模型对中心和场地先验的依赖。

## 4. 阈值是不是原因

checkpoint 没有保存单独拟合的阈值；原训练/评估规则是二分类 logit argmax，等价概率分界为 0.5。这里没有人为改成新的固定阈值。

阈值失配确实存在，但不是根因：

- 原表上事后选择 Youden 阈值 0.9012，FPR 仍为 37.8%，Recall 降到 77.8%；
- 平衡的中心负样本集上事后阈值 0.8857，FPR 仍为 47.9%，Recall 为 75.7%。

正负分数有一定排序差异，所以 AUROC 高于 0.5；但分布严重重叠且整体饱和在高分区。没有一个阈值能同时恢复低 FPR 和高 recall。用测试集事后选阈值也会造成数据泄漏。

## 5. 建议的修正方向

1. 不再把 MethaneAIR 检测目录缺项或实验期外日期直接视作 gold negative。
2. 用逐次过境的计量零释放、现场确认未释放、或有完整 survey coverage 的 non-detection 构建 temporal negatives。
3. 训练集中加入“同一设施、同一中心、不同日期、无释放”的 hard negatives，使正负样本的中心偏移、场地、季节和亮度分布匹配。
4. 按 site/source 分组切分 train/validation/test，禁止同一设施视觉背景跨集合泄漏。
5. 在独立 validation set 上重新校准阈值；在完成 hard-negative retraining 前，不应把调阈值当作修复。

## 可复查产物

- 负样本逐事件来源审计：`results/analysis/controlled_release_45_negative_provenance_audit.csv`
- 严格原表 manifest：`data/controlled_release_test/legacy360_torch_exact/original_table/manifest.csv`
- 严格原表结果：`results/eval/controlled_release_s2_legacy360_torch_exact_original_universal360.json`
- 严格中心负样本结果：`results/eval/controlled_release_s2_legacy360_torch_exact_balanced_universal360.json`
- 严格远离中心对照结果：`results/eval/controlled_release_s2_legacy360_torch_exact_distance_universal360.json`

## 6. 追加因果实验：亮度是放大器，但不是主因

使用 9 个可信正样本和 14 个 2018 同地点负样本，对已经物化的 224×224
legacy 输入做严格配对亮度干预。crop、纹理、时相和标签完全不变，只把所有有效
band 同比缩放；t0 的两个缺失 band 始终保持为零。

| 全局亮度倍数 | 正样本均分 | 负样本均分 | AUROC |
|---:|---:|---:|---:|
| 0.70 | 0.9040 | 0.8682 | 0.627 |
| 0.85 | 0.9214 | 0.8887 | 0.679 |
| 1.00 | 0.9197 | 0.8962 | 0.667 |
| 1.15 | 0.9170 | 0.9206 | 0.444 |
| 1.30 | 0.9112 | 0.9381 | 0.286 |

14 个负样本的亮度—分数斜率全部为正，中位 Spearman rho 为 0.987。增亮
30% 后负样本均分增加 0.0419，14/14 增加，配对 `p=0.000122`；压暗 30%
后均分下降 0.0280，14/14 下降，`p=0.000122`。因此亮度依赖是因果关系。

但压暗 30% 后 14 个负样本仍全部被判为正，所以亮度只能放大错误，不能解释
100% FPR。完整结果见：

- `results/eval/s2_legacy360_brightness_probe_summary.json`
- `data/controlled_release_test/brightness_probe_legacy360_v1/dataset_audit.json`

## 7. 追加因果实验：模型依赖可见光地表纹理，不依赖甲烷 SWIR plume

对同一批 23 个源样本构造 12 组机制干预和 15 组逐 band/逐时相消融。
空间常数化保留每个 frame/band 的均值，只删除该 band 的空间纹理；因此它与
前面的全局亮度实验互补。

| 干预 | 正样本均分 | 负样本均分 | FPR | Recall | AUROC |
|---|---:|---:|---:|---:|---:|
| 原图 | 0.9197 | 0.8962 | 100% | 100% | 0.667 |
| 全部空间纹理常数化 | 0.1468 | 0.1615 | 0% | 0% | 0.274 |
| 只常数化 t0 全部 band | 0.1211 | 0.0265 | 0% | 11.1% | 0.825 |
| 只常数化两个历史 frame | 0.1005 | 0.0639 | 0% | 0% | 0.413 |
| 三时相 B01–B04 可见光常数化 | 0.1015 | 0.1609 | 0% | 0% | 0.143 |
| 只把 t0 B01–B04 常数化 | 0.1819 | 0.1013 | 0% | 11.1% | 0.524 |
| 三时相 red-edge/NIR 常数化 | 0.9056 | 0.8874 | 100% | 100% | 0.544 |
| 三时相 B11 常数化 | 0.9258 | 0.9122 | 100% | 100% | 0.639 |
| 三时相 B12 常数化 | 0.9091 | 0.9154 | 100% | 100% | 0.532 |
| 三时相 B11/B12 同时常数化 | 0.8996 | 0.9321 | 100% | 100% | 0.365 |
| B11/B12 互换 | 0.9314 | 0.8983 | 100% | 100% | 0.702 |
| 14×14 patch 位置随机重排 | 0.9287 | 0.8954 | 100% | 100% | 0.754 |
| 所有像素随机重排 | 0.6077 | 0.5767 | 100% | 100% | 0.698 |

这些结果给出四个直接结论：

1. 只保留全局均值不够，局部空间变化是必要输入；所以模型不是简单的亮度阈值器。
2. 删除 B11/B12 的空间结构、甚至交换 B11/B12，输出几乎不受影响。模型没有依赖
   Sentinel-2 甲烷反演所需的 SWIR plume 对比。
3. 删除 t0 可见光 B01–B04 纹理后，原来饱和为正的样本几乎全部变成负；可见光中
   甲烷 plume 本身不可见，因此驱动模型的是设施、道路、裸地、植被、边缘和局部
   反射率等场地视觉代理。
4. 整块 14×14 patch 的位置打乱几乎不改变分数，说明模型主要使用“有哪些局部
   patch”，而不是 plume 的全局位置、方向或连通形状。

训练参考的简单统计也支持同一结论。只使用 80 个亮度、局部梯度、patch 内方差、
SWIR/visible 统计量的逻辑回归，在固定 512 正 + 512 负训练参考上取得 OOF AUROC
0.688；它把本次 14 个 2018 负样本全部判为训练式正类。训练正类相对负类具有更高
的 patch 内对比和局部梯度，这些差异来自中心设施/远处背景构造。

完整产物：

- `results/eval/s2_legacy360_mechanism_probe_summary.json`
- `results/eval/s2_legacy360_band_time_probe_summary.json`
- `results/analysis/s2_legacy360_visible_shortcut_summary.json`
- `data/controlled_release_test/mechanism_probe_legacy360_v1/dataset_audit.json`
- `data/controlled_release_test/band_time_probe_legacy360_v1/dataset_audit.json`

同一组逐 band/逐时相干预又使用最新本地 S2-only 360m checkpoint 独立重跑。
结果与 Universal 完全同向：原图负样本均分为 0.9412；只常数化 t0 可见光后降至
0.1123；常数化三时相 B11/B12 后反而分别为 0.9519/0.9571；互换 B11/B12 后仍为
0.9494。由此可以排除 Universal 多传感器 fusion 是该现象的来源；问题来自两个模型
共享的训练标签和 Panopticon channel/time 表示。

- `results/eval/s2_legacy360_band_time_probe_s2only_latest_summary.json`

## 8. 架构问题：所谓三时相输入没有显式时间身份

当前 loader 把 t0、t-90、t-360 直接拼成 36 个 channel，但三个 frame 的 12 个
`chn_ids` 完全重复。`PanopticonPE` 先逐 channel patchify，再用基于波长 ID 的
cross-attention 对 channel 集合做聚合；代码中没有 frame/time embedding，也没有
把同一 frame 的 band 绑定成一个时间组。

因此模型对具有相同波长 ID 的重复 channel 在数学上近似置换不变：它可以看到三个
同波长观测值的集合，却不知道哪个是当前、90 天前或 360 天前，更不能稳定学习
“当前减历史”的有方向甲烷残差。实测交换 t-90 和 t-360 后，23 个样本的最大概率
变化仅 0.000488（FP16 一个量化步长），平均变化严格为零。

t0 的 `+1000 DN` 和两个零 band 会让模型通过数据伪特征猜测哪个观测可能是 t0，
但这不是可靠的时间编码，也不能把同一 frame 的 B11/B12 配成一组。要做真正的
temporal plume classifier，应分别编码三个 frame，加入明确的时间/间隔 embedding，
再做有方向的 temporal fusion。

## 9. 追加预处理复核

随机取当前测试的一个样本，直接比较 evaluator loader 输出与手工
`(raw_DN - training_mean) / training_std`：最大绝对误差为 `1.91e-6`。同时确认：

- 每个 frame 为 `12×224×224`，batch 为 `36×224×224`；
- t0 channel 8、9 全零；
- 三个 frame 的 channel ID 完全重复；
- 所有机制测试都通过 12-band shape、float32、t0 零 band 和原 manifest 哈希审计。

所以当前高 FPR 不能归因于这次测试 loader 的 band 顺序或归一化错误。

## 10. 更新后的根因排序

1. **训练标签的空间捷径是主因。** 原生成器把“crop 包含中心 10×10 区域”定义为
   正，把“不包含中心”定义为负。严格 S2 子集有 30,296 正和 29,339 负：正样本
   偏移半径中位数 14.0 px、最大 24.04 px；负样本中位数 190.67 px，仅 0.99%
   位于 31 px 内。模型被训练成中心设施/远处背景分类器。
2. **三时相架构缺少时间身份。** 它无法可靠学习当前相对历史的有方向变化，只能聚合
   三个重复波长观测的无序集合。
3. **可见光局部纹理和绝对辐射成为主要代理。** 局部设施/地表纹理决定基础高分，
   亮度进一步抬高 2018 负样本分数。
4. **测试负标签可信度不足会增加噪声，但不是唯一解释。** 即使是正式计量零释放样本，
   只要仍在设施中心，当前模型也容易判正。
5. **严格预处理已复核通过。** resize 历史差异需要记录，但用训练合同重建后仍高 FPR；
   当前 loader 的归一化和 band contract 没有发现错误。

## 11. 训练正样本的 mask 内是否真的有一致 SWIR plume 信号

最后对固定的 512 个训练正参考做了 mask 内/局部背景对比，并将同一 mask 平移到
同一图像的其他位置作为空白对照。512/512 个 crop 的 plume mask 非空，509 个具有
足够的局部背景像素。

| 特征 | 真实 mask 对比中位数 | 真实 mask 为正的比例 | 随机平移中位数 |
|---|---:|---:|---:|
| `(B11-B12)/(B11+B12)` | -0.128 | 41.7% | +0.037 |
| B11 | +0.179 | 65.2% | -0.078 |
| B12 | +0.211 | 66.6% | -0.085 |
| B01–B04 均值 | +0.335 | 71.5% | -0.117 |

这里的对比定义为 `(mask 均值 - 邻近膨胀环均值) / 邻近环标准差`。原始 SWIR
归一化差异没有一致的正方向，而 mask 所在区域呈现更强、更稳定的可见光地表对比；
可见光真实 mask 与随机位置的配对差异为 0.546，`p=5.19e-24`。这不能证明每个
S2 正样本都没有甲烷信息，但说明训练标签位置同时携带一个更容易学习的地表/设施
信号。结合 B11/B12 消融几乎不影响模型输出，可以确认当前 checkpoint 实际选择了
这个可见光捷径，而不是较弱且时刻未必匹配的 SWIR plume 信号。

- `results/analysis/s2_legacy360_training_positive_mask_swir_contrast.json`
- `results/analysis/s2_legacy360_training_positive_mask_swir_contrast.csv`

## 12. 可见光纹理具体由哪个 band 驱动

对 t0 的 B01、B02、B03、B04 枚举全部 16 种纹理删除子集，并用精确 Shapley
分解“删除空间纹理造成的正类分数下降”。该分解覆盖所有 band 交互和冗余，不是只做
四次独立 leave-one-out。

针对本次 14 个 2018 负样本：

| 模型 | Band | Shapley 分数下降 | 占四个可见光总下降 | 单独删除后的均分 | 只保留该 band 纹理的均分 |
|---|---|---:|---:|---:|---:|
| Universal 360m | B02 | 0.7059 | 88.8% | 0.2923 | 0.8888 |
| Universal 360m | B03 | 0.0501 | 6.3% | 0.8855 | 0.1653 |
| Universal 360m | B01 | 0.0335 | 4.2% | 0.9001 | 0.1518 |
| Universal 360m | B04 | 0.0054 | 0.7% | 0.8992 | 0.1170 |
| S2-only 最新 | B02 | 0.7424 | 89.6% | 0.2731 | 0.9124 |
| S2-only 最新 | B03 | 0.0488 | 5.9% | 0.9149 | 0.1656 |
| S2-only 最新 | B01 | 0.0279 | 3.4% | 0.9431 | 0.1566 |
| S2-only 最新 | B04 | 0.0098 | 1.2% | 0.9383 | 0.1284 |

结论是 **t0 B02（蓝光）空间纹理是决定性特征**：单独删除它就消除约三分之二的
原始正类分数；仅保留 B02、删除另外三个可见光 band 时，负样本仍保持接近原图的
饱和高分。B03 有少量辅助作用，B01/B04 对这些假阳性几乎可忽略。两个独立
checkpoint 得到近乎相同的 89% B02 归因，说明这不是某一个分类头的偶然权重。

这里的“贡献”严格指当前 23 个受控释放/2018 场景、当前 legacy 输入合同和当前两个
checkpoint 下，对空间纹理常数化干预的因果贡献；不能直接解释成 B02 对所有地区、
所有 S2 plume 任务的一般物理重要性。

- `results/eval/s2_legacy360_visible_subset_probe_universal360_summary.json`
- `results/eval/s2_legacy360_visible_subset_probe_s2only_latest_summary.json`
- `data/controlled_release_test/visible_subset_probe_legacy360_v1/dataset_audit.json`
