# RCTP：面向不可见光甲烷弱瞬态的响应条件反事实预训练方案

**状态：方法与执行方案，尚未产生 RCTP 实验结果。**  
**日期：2026-07-27**  
**方法暂名：Response-Conditioned Counterfactual Transient Pretraining（RCTP）**

---

## 0. 先把“事实、建议、待验证假设”分开

### 0.1 仓库中已经确认的事实

1. Panopticon 的 patch embedding 会先对每个空间 patch 的各波段分别投影，再以波长/SRF 中心 embedding 做 channel cross-attention。因此它已经具备“在 patch 内融合任意光谱通道”的合适起点；它并不显式知道某个 band 对 CH4 的响应强度。
2. 现有 `MethaneResidualMAE` 对 `x_t0-x_history` 做 masked pixel reconstruction，使用 sensor/lag embedding，并在若干层以 CLS token 做先时间、后 sensor 的 hierarchical fusion。它优化的是残差像素均方误差，不是甲烷响应选择性。
3. 已补的 validity-aware 版本只修正无效像素参与 reconstruction loss 的问题；它没有改变“主要 loss 容易被大面积背景/配准/云残差支配”这一目标偏置。
4. 四个六时刻数据的当前输入约定不同：

   | Sensor | 当前正式单 sensor 数据 | 输入语义 | 当前规模（train/test） |
   |---|---|---|---:|
   | S2 | 12 bands，六访，当前 CSV 指向 224 crop | 多光谱反射率；旧审计发现部分构建版本含 zero compatibility bands/masks | 117,792 / 20,544 |
   | L89 | 7 bands，六访，224 crop | 多光谱 SR；时间重复不可忽略 | 62,389 / 7,942（hard-event filtered） |
   | EMIT | 32 bands，约 2138–2493 nm，六访，224 crop | methane-window hyperspectral reflectance | 34,592 / 6,138 |
   | S5P | 六访，每访本质为 3×3 XCH4 cell | retrieved methane product，不是高分辨率反射率图像 | 25,345 / 3,213 |

5. S5P 当前 NPZ 把 3×3 信息插值到 224×224。仓库中的 native-grid screen 明确说明：从该 224 场做 finite-mask pooling 只能得到“近似 native-scale summary”，不是原始 3×3 的精确逆变换。正式预训练应从原始 NC/cell 重新取 3×3。
6. EMIT 存在两种不要混淆的表示：新的单 sensor 数据是原生 32-band methane-window cube；legacy 360 m 多 sensor 数据中的 `emit` 是由 EMIT 按 WV3 SRF 模拟的 16 bands。RCTP 主实验应使用 32-band native EMIT，16-band 只保留为与旧 MethaneFuse 兼容的实验臂。
7. 现有单 sensor 记录（不同 split，不可直接横比）为：L89 hard-event filtered 从头训练最好 F1 0.7544；EMIT full temporal cutoff F1 0.5667；S5P F1 0.6375。S2 legacy 360 m 的历史 checkpoint 在其旧 temporal split 上约 F1 0.8955。后者不能被当作新六时刻协议的结果。
8. hard-event filtered L89 的 9 个事件是根据旧 test error 选出的，仓库 audit 已明确标注该 test 是 model-conditioned；它可以用于快速工程筛选，不能作为 journal 的最终无偏 test。
9. 历史 MethaneFuse checkpoint 曾用 test accuracy 选 best epoch。它适合现在的工程 warm start，但正式论文必须从干净 outer-train 重新训练，并只用 inner validation 选 checkpoint/threshold。

### 0.2 本文建议

用已有 Panopticon/MethaneFuse encoder 做 **continue-pretraining**，不从头训练 foundation model；新增：

- sensor response metadata adapter；
- 物理/产品算子生成的 clean–methane counterfactual pairs；
- dense transient、剂量、visit 与跨 sensor response-difference 目标；
- 小权重 frozen-anchor，避免破坏已有空间/光谱表征；
- sensor-native tokenizer，尤其禁止把 S5P 继续当 224 图像做 MAE。

### 0.3 必须由实验回答、现在不能当结论的假设

1. RCTP 会比 generic pixel MAE、masked latent prediction 和现有 residual MAE 更能保留弱甲烷瞬态。
2. SRF/CH4-response conditioning，而不是仅仅增加参数或 synthetic augmentation，是增益的 load-bearing variable。
3. 用 synthetic counterfactual 学到的 response direction 能迁移到真实 classification 和 plume segmentation。
4. 跨 sensor 对齐的是“甲烷导致的表示差分”，不是地点/地表语义，因此能改善缺 sensor 和 sensor shift。
5. 最终是否能逼近或超过 0.90 F1 只能由严格同 split 结果判断；不能由旧 S2 checkpoint 外推到四 sensor。

---

## 1. 论文故事：不是“更会重建”，而是“知道不可见气体怎样改变观测”

### 1.1 Research gap

通用 EO foundation model 和 MAE/JEPA 类目标主要奖励对大面积、稳定、可预测地表的建模。甲烷 plume 恰好相反：

- 信号主要位于不可见的 SWIR/吸收窗口或已经反演的 XCH4 产品；
- plume 面积小、对比弱、只在某次 acquisition 出现；
- background、云、季节、BRDF 和配准误差在 pixel/latent loss 中占据绝大多数能量；
- 四个 sensor 对同一甲烷柱浓度的响应方向、空间 PSF、单位和产品算子完全不同。

因此，普通 masked reconstruction 的最优捷径可能是“恢复稳定地表并平滑掉小瞬态”。仅把 t0 与历史做 cross-attention 或 residual 也没有解决根因：模型仍未被告知 **什么变化符合 CH4 的 sensor-specific response，什么变化只是亮度、云、条带或地表变化**。

### 1.2 RCTP 的窄而清晰的贡献

对同一个 train-only clean/reference scene，构造共享同一 plume column field 的一对观测：

```text
x0_s = sensor_s(background)
x+_s = sensor_s(background + CH4 column field)
```

其中 `sensor_s` 不是一个 sensor ID，而是由以下条件定义的观测算子：

```text
SRF / CH4 sensitivity + unit/product type + GSD/PSF/footprint
+ acquisition time/role + quality/validity
```

模型看到 exact paired control：地表、日期、空间纹理、云和噪声保持不变，只有甲烷响应变化。预训练不要求它重建全部 scene，而要求它：

1. 定位哪一访、哪些 patch/cell 出现了符合 CH4 response 的弱瞬态；
2. 区分 CH4 与同能量的 achromatic、wavelength-shuffled、cloud/shadow、striping 等 nuisance；
3. 使不同 sensor 在同一物理 plume 下的 **response difference** 可比较；
4. 保持非 plume 区域表示稳定，并用 frozen checkpoint 约束不遗忘已有背景能力。

### 1.3 可以与不可以声称的 novelty

可以在验证后声称的窄命题：

> Sensor-response-conditioned counterfactual transient pretraining，比 scene reconstruction 更适合把不可见、sensor-dependent、只在一次 acquisition 出现的弱甲烷扰动写入可迁移 encoder。

不能声称：

- 首个 background prediction / residual anomaly detection；
- 首个 synthetic anomaly pretraining；
- 首个物理启发 methane detection；
- 首个 multisensor/temporal MAE；
- 首个 time×sensor 双轴 attention；
- 首个跨 sensor prediction 或 anomaly ranking。

广义 synthetic anomaly、HACD/residual、MethaneMapper、AnySat/ALISE 等路线都已占位。最终 novelty 应落在 **SRF/产品响应条件 + exact counterfactual pair + irregular transient visit + native heterogeneous sensors + classification/segmentation transfer** 的组合，并继续做精确 collision audit。

### 1.4 暂定标题与一句话

> **Pretraining for What the Eye Cannot See: Sensor-Response Counterfactual Learning for Transient Methane Detection**

一句话：

> 通用 EO 模型学习“scene 长什么样”；RCTP 学习“同一个 scene 加入甲烷后，不同 sensor 应该怎样改变”。

---

## 2. 可复用的现有基础与最小代码改造边界

### 2.1 保留的组件

- Panopticon 的 per-band patchification、波长 embedding 和 patch-local channel attention；
- 已有 DINO/Panopticon spatial trunk；
- 已有四 sensor reader、train-only normalization、validity/missing/duplicate audit；
- `MethaneResidualMAE` 中的 sensor/lag embedding 与 hierarchical fusion，可作为 baseline 或初始化来源；
- 当前并行开发的 T×S downstream head，只作为下游 readout，不作为 RCTP 的 novelty。

### 2.2 新增的最小模块

1. `ResponseMetadataEncoder`
   - 输入每 band 的 SRF summary/curve projection、`kappa_CH4`、bandwidth、unit/product type；
   - 输入 sensor GSD、PSF/footprint、query scale；
   - 输入 role、真实 `delta_days`、quality、valid/duplicate mask；
   - 输出与 Panopticon channel token、sensor token 相加的 residual adapter；
   - adapter 末层 zero-init，保证加载 PTH 后第 0 step 与原模型一致。

2. `CounterfactualRenderer`
   - 高光谱/反射率传感器走 spectral radiative response + SRF integration + PSF/GSD；
   - S5P 走 XCH4 product-space operator；
   - 保存 renderer version、气体 cross-section source/version、SRF hash、seed 和全部参数。

3. `TransientPretrainHeads`
   - dense response/localization head；
   - visit head；
   - column amount/strength head；
   - paired response-difference projector；
   - downstream 时全部可丢弃，只转移 encoder/adapters。

### 2.3 初始化规则

工程验证：

- 可以从现有 universal 360 m 或 per-sensor PTH warm start，快速判断机制是否有信号；
- 必须在结果中标记该 PTH 是否历史上由 test 选取。

论文正式结果：

- 首选公开 Panopticon pretrained weights；
- 或在新 outer-train 上按同一 protocol 重新训练 MethaneFuse baseline PTH；
- 禁止把当前 test-selected MethaneFuse PTH 当成 formal clean initialization。

---

## 3. Response metadata：不能只给一个 sensor ID

### 3.1 Optical/hyperspectral band metadata

对 sensor `s` 的 band `c` 保存：

```text
R_s,c(lambda)       完整/离散 SRF（归一化）
mu_s,c              SRF 中心波长
sigma_s,c           有效带宽
kappa_s,c           给定参考大气状态下的 CH4 integrated sensitivity
unit_s              reflectance / scaled SR / XCH4
product_s           L2A / L2SR / EMIT reflectance / S5P retrieved XCH4
```

建议的 conditioning scalar：

```text
                       ∫ R_s,c(lambda) * sigma_CH4(lambda; P0,T0) d lambda
kappa_s,c = ----------------------------------------------------------------
                              ∫ R_s,c(lambda) d lambda
```

但 renderer 不能只用这个 scalar。正式 renderer 应先在高光谱网格上做 transmission，再按 SRF 积分；`kappa` 只用于告诉网络该 band 的大致甲烷响应方向与强度。

CH4 cross-section 应固定到一个公开、带版本的数据库/光谱表，并记录温压假设。若使用多组温压 profile，profile 只由 outer-train 地区/月分布标定。

### 3.2 Spatial/acquisition metadata

每次 acquisition 还需要：

```text
sensor_id, role, acquisition_time, true delta_days
GSD, PSF/footprint, query_size_m
valid_band_mask, valid_pixel/cell_mask, cloud/quality
duplicate_acquisition_mask, registration_confidence
```

`acquisition_id` 只用于去重和 attention masking，不能学习 granule-specific embedding。

### 3.3 必做的 metadata negative controls

- 在同一 sensor 内随机打乱 `kappa` 与 band 的对应；
- 只保留 `mu`/bandwidth，移除 `kappa`；
- 只给 sensor ID，不给 SRF；
- 给正确 SRF 但错误 GSD/PSF；
- 给固定 role index，移除真实 `delta_days`。

若上述控制与 full RCTP 相同，则“物理响应条件”并未发挥作用，论文不能以此为主张。

---

## 4. Counterfactual 数据构造

### 4.1 背景库只能来自 outer-train

候选 clean/reference background：

1. negative-labeled/off-plume crop 的 t0 与历史访；
2. positive event 的历史访，但仅在该 acquisition 与排放事件不重合、且质量合格时使用；
3. positive t0 的 mask 外区域，只用于局部背景统计，不作为整图 clean target。

这些只能称为 **reference controls**，不能称为 confirmed methane-free。可用 robust matched-filter/XCH4 tail score 删除最可疑的污染项，但阈值只在 outer-train 定义。

按以下字段匹配/分层采样：

```text
sensor, site/region, month, surface brightness/land cover,
cloud/valid fraction, view geometry（可得时）, revisit gap
```

### 4.2 物理 plume column field

在以米为单位的公共物理平面生成 `C(p)`：

- 中心线方向由 train-only 风向分布采样；若无可靠风数据，用均匀方向并标为 synthetic；
- 宽度、长度、曲率、断裂和浓度衰减从真实 train plume mask/IME 分布拟合；
- 使用一组 Gaussian plume / advected filament / empirical-mask deformation renderer，防止只学一种模板；
- plume strength 以 log-uniform 覆盖低于、接近和高于真实 detection limit 的范围；
- synthetic mask 保存 soft column field、binary support 和 integrated mass proxy；
- 同一个 `C(p)` 分别通过各 sensor 的 GSD/PSF/footprint，保证跨 sensor pair 的物理原因相同而像素形态不同。

### 4.3 反射率/辐亮度 sensor 的正样本

高保真形式：

```text
L+(lambda,p) = L0(lambda,p)
               * exp[-m(p) * sigma_CH4(lambda; P,T)]

x+_s,c(p) = ∫ R_s,c(lambda) L+(lambda,p) d lambda
            / ∫ R_s,c(lambda) d lambda
```

然后施加：

```text
sensor PSF -> native GSD/grid -> noise/quantization -> validity mask
```

当只有 multispectral band 值、没有高光谱 `L0(lambda)` 时，首版近似：

```text
x+_s,c(p) = clip(x0_s,c(p) * exp[-alpha * kappa_s,c * C(p)] + epsilon_s,c)
```

这个近似必须作为 renderer version 单独标记，并用 EMIT 高光谱 scene 做误差校准：先在 EMIT 高光谱上 full render，再聚合到 S2/L89 SRF，与 `kappa` 近似比较。

### 4.4 S5P 的正样本

S5P 输入是 retrieved XCH4，不是 reflectance band，不能复用 Beer–Lambert image renderer：

```text
XCH4+_cell = XCH4_0_cell + K_retrieval,s5p(C, footprint, averaging_kernel)
```

首版若缺 averaging kernel，可用 footprint-integrated column enhancement，并把 kernel 缺失标为近似。正式版本应从 NC 中读取原始 3×3 cell、qa/missing、footprint/averaging-kernel 可用字段。

S5P synthetic supervision只有：

- 3×3 coarse cell support；
- injected visit；
- cell-level strength/order。

绝不生成或监督 224×224 plume segmentation。

### 4.5 在 irregular six-visit 中注入

- 默认只在一个 **唯一 acquisition** 注入，通常是 t0；
- 以小概率注入某个历史访，用于 visit-identification，避免模型把“role=t0”当 label；
- 另设 `none` 类，六访均不注入；
- 重复 acquisition 在注入前去重，不能把同一 observation 注入两次；
- 保留真实 `delta_days`、missing pattern 和 quality mask；
- 正负 pair 使用完全相同的 augmentation、noise 与 invalid mask。

### 4.6 Hard negatives

每个 methane positive 至少配一个等能量 hard negative：

1. **wavelength-shuffled response**：同一 `C(p)`，打乱 band–`kappa` 对应；
2. **achromatic brightness**：所有 band 同方向变化；
3. **sign-reversed response**：使用 `-kappa`；
4. **non-CH4 spectral response**：水汽/矿物/植被或平滑低阶谱变化；
5. **cloud/shadow/BRDF**：空间形态相似但谱方向不同；
6. **striping/missingness**：EMIT/S5P 产品常见伪影；
7. **CutPaste morphology**：作为非物理 synthetic anomaly baseline。

hard negatives 与 methane positive 的 mask 面积、能量、visit、位置和数量尽量匹配。否则模型可能只学强度或形状。

---

## 5. 四 sensor 的 native 处理

### 5.1 S2

- 12-band reader与 Panopticon 波长 ID 保持现有约定；
- spatial patch + per-band channel attention；
- CH4 response 重点会落在 SWIR 相关 band，但模型仍看全谱；
- 对全零/compatibility band 做 `valid_band_mask`，不能让 raw zero 经 normalization 变成有效强信号；
- classification/SSL 可使用 zero compatibility mask 样本；segmentation 只使用 provenance 标记为真实 plume mask 的样本；
- 首选从原始 32×32 native crop 做物理 injection，再按现有 downstream 输入需要 resize，而不是先在 224 插值图上生成细 plume。

### 5.2 L89

- 正式输入 7 SR bands，使用 Landsat 8/9 的 SRF，而不是只给中心波长；
- 30 m GSD 下，先在物理平面生成 plume，再 PSF/downsample；
- 去重六 role，尤其处理 `prev1=prev2`；
- B6–B7 与非 RGB 的现有记录表明不可见 band 有信号，但这是监督分类观察，不等于 RCTP 已有效；
- hard-event filtered split 只做快速机制 screen；formal split 需重新按 event/time 构建无偏 test。

### 5.3 EMIT

- 主线用 32-band 2138–2493 nm native methane-window reflectance；
- spectral grouping/tube masking可以作为 auxiliary，不应主导 loss；
- full spectral renderer最适合先在 EMIT 验证 `kappa` 近似和 hard negatives；
- 86% 左右的历史 role 重复问题在旧审计中出现过，必须按 acquisition time/path hash 去重；
- native 32-band 与 legacy 16-band WV3-simulated arm分开报告，不混合训练统计；
- segmentation mask 只使用有可靠 airborne plume projection/provenance 的样本。

### 5.4 S5P

- 从原始 NC 读取每访 3×3 XCH4 与 cell validity/qa；
- token lattice 为 `time × 9 cells`，附带 cell相对位置、footprint 和真实 `delta_days`；
- 不对 3×3 上采样后做 ViT small-patch MAE；
- 只提供 coarse atmospheric context、visit、cell anomaly 和 strength supervision；
- 与高分辨率 sensor 融合时，S5P 进入 sensor axis 的 global/coarse token，不参与 patchwise 224 对齐。

---

## 6. 模型输入与 factorized T×S

每个有效 observation 先独立得到 dense tokens：

```text
Z_s,t = E_panopticon(x_s,t;
                     wavelength/SRF,
                     response metadata,
                     validity)
```

时间轴只在同一 sensor 内操作：

```text
T_s: {Z_s,t, delta_t, role, quality}_t -> transient-aware sensor representation
```

sensor 轴再融合：

```text
S: {T_s, GSD/PSF, availability, reliability}_s -> fused representation
```

首版坚持 factorized，而不是 full `(sensor×time)^2` attention：

- 缺 sensor/缺访更容易 mask；
- S5P coarse token 不被伪装成 224 patch；
- 可独立消融 T-only、S-only、T→S、S→T；
- 参数/计算更可控。

但 **T×S 是 readout/architecture，不是 RCTP 的方法主张**。RCTP 必须在相同 T×S 架构下与 generic objectives 做 matched comparison。

局部 plume 不能只靠 global CLS。每个 sensor representation至少保留：

- t0/local query patch tokens；
- GeM 或 top-k patch evidence；
- global CLS/context；
- coarse S5P cells。

---

## 7. 预训练目标

令 `x0` 为 clean/reference control，`x+` 为 methane counterfactual，`x-` 为 matched hard negative；`M` 为 plume support，`v*` 为 injected visit，`a` 为 injected column/strength；`E_frozen` 为初始化 checkpoint 的冻结副本。

### 7.1 Dense localization

对每个高分辨率 sensor 输出 `q_s,t,p`：

```text
L_loc = Focal(q, M) + Dice(q, M)
```

S5P 改成 3×3 valid-cell focal/BCE，不计算 224 Dice。所有 loss 均乘 valid pixel/cell mask。

### 7.2 Counterfactual ordering

令 `g(x)` 为 masked top-k/GeM transient score：

```text
L_cf = softplus(m1 - g(x+) + g(x0))
     + softplus(m2 - g(x+) + g(x-))
```

`m1,m2` 可随 injected strength 分桶，但只在 outer-train 调整。它直接要求 methane response 高于 exact clean control 和等能量 nuisance。

### 7.3 Column/strength regression

```text
L_col = Huber(log(1 + a_hat), log(1 + a))
```

高分辨率 sensor 可同时回归 integrated column proxy；S5P 回归 footprint-weighted enhancement。强度仅用于 synthetic pretraining，不把它宣传成真实排放率反演。

### 7.4 Injected-visit classification

```text
L_visit = CE(v_hat, v*),  v* ∈ {unique visits, none}
```

它迫使模型使用观测内容和真实时间，而不是默认 t0 必为异常。

### 7.5 Cross-sensor response-difference alignment

只对由同一 background 与同一 `C(p)` 渲染的 synthetic matched group：

```text
Delta z_s = P_s(pool_M(E(x+_s)) - stopgrad(pool_M(E(x0_s))))

L_xsensor = supervised_contrastive(
              Delta z_s / response_scale_s,
              positive = same plume field/strength,
              negative = different field/strength or hard negative)
```

对齐的是 paired response difference，不是 scene embedding；因此不会强迫 S2 patch 与 S5P 3×3 cell 逐像素一致。若移除 `x0` 后直接对齐 `E(x+)`，地点/地表语义会成为捷径，该版本只作为负控制。

### 7.6 非 plume 背景稳定性

```text
L_bg = mean_{p outside dilate(M)}
       [1 - cosine(E_p(x+), stopgrad(E_p(x0)))]
```

它限制 synthetic injection 不应改变非 plume 表示。mask 外再 dilation，避免 PSF 边界被误当背景。

### 7.7 Frozen-anchor

```text
L_anchor = mean_valid [1 - cosine(E(x0), E_frozen(x0))]
```

只在 clean control/非 plume区域使用，避免 anchor 把 methane response 也压回去。

### 7.8 首轮总 loss

```text
L_RCTP =
    1.00 L_loc
  + 0.50 L_cf
  + 0.25 L_col
  + 0.25 L_visit
  + 0.15 L_xsensor
  + 0.10 L_bg
  + 0.05 L_anchor
```

这些权重是 **建议起点，不是已调优结论**。第一轮只允许根据 train/inner-val 的 gradient norm 与 metric scale 做一次预注册式重标定，不能围绕 test 调权。

Generic masked latent loss可作为低权重 auxiliary（不超过总 gradient norm 的 20%），但 RCTP 主臂首先应验证不依赖它也能工作。

---

## 8. 两阶段训练

### Stage 0：renderer 与协议 Gate（不开大训练）

目标不是看 classifier F1，而是证明 synthetic signal 没有明显捷径。

1. 从每 sensor outer-train 随机取 1,000–2,000 background；
2. 每个生成 methane、wavelength-shuffle、achromatic、sign-reverse；
3. 比较 real train plume 与 synthetic 的：
   - per-band/spectral-ratio 分布；
   - matched-filter SNR；
   - plume area/length/width；
   - 非 methane band statistics；
   - valid/cloud/missing 分布；
4. 训练一个只看非 methane bands 或 mask 外像素的 synthetic-vs-clean discriminator。

**Gate 0 通过条件：**

- 非 methane/mask-out discriminator AUROC 不高于 0.60；
- synthetic 与真实 train plume 的 matched-filter SNR/area 有充分重叠，而非完全分离；
- wavelength-shuffled 与 methane 在总能量、mask area 上匹配；
- alternate renderer（Gaussian vs empirical deformation）不改变结论方向。

若不通过，先修 renderer，禁止用“更多 epoch”掩盖 domain gap。

### Stage 1：冻结 backbone 的 response-adapter 预训练

- 初始化：现有 PTH 的工程 arm + clean PTH 的 formal arm；
- 冻结 Panopticon trunk；
- 训练 response metadata adapter、T/S modules 和 transient heads；
- 约 5,000 optimizer updates；
- batch 按 sensor/label/strength 平衡，单 step 可只含一种 native sensor，跨 sensor contrast通过 matched group queue 或周期性 group batch；
- adapter/head LR `1e-4`，AdamW，warmup 5%，cosine decay；
- bf16，gradient clip 1.0。

Stage 1 的任务是判断 counterfactual target 是否可学，不期望它完成最终 domain transfer。

### Stage 2：受限更新 encoder

只在 Stage 1 通过 capability gate 后开启：

- trunk 前 8 blocks 冻结；
- 最后 4 blocks 加 LoRA（建议 rank 8–16）或直接以 `1e-5` 小 LR 更新；
- adapter/head LR `1e-4`；
- 10,000–20,000 updates；
- 保留 `L_anchor`；
- sensor-balanced sampling，防止样本最多的 S2 支配；
- 每 1,000 updates 做 inner-val frozen probe，不持续窥视 outer-test。

### Downstream transfer

每个 sensor 与 fused task 均跑：

1. frozen linear/小 MLP probe；
2. adapter/T×S head，encoder 冻结，2–3 epochs；
3. 若 inner-val 仍上升，LoRA last-4 再 2 epochs；
4. segmentation 只在真实可信 mask 上训练/评估。

用户要求的“几个 epoch 即判断”适用于 downstream。预训练以 update/capability curve 判断，不应把一次 full pass 当作固定单位。

---

## 9. 严格 ablation：必须完整重训，不能只在 inference 关模块

### 9.1 Objective baselines（同 encoder、数据、updates）

| Arm | 目标 |
|---|---|
| B0 | 无 methane-specific pretraining，直接用 PTH |
| B1 | normalized pixel MAE |
| B2 | validity-masked residual MAE |
| B3 | masked EMA latent prediction |
| B4 | past→t0 latent prediction |
| R0 | CutPaste/synthetic anomaly，未使用物理 response |
| R1 | full RCTP |

### 9.2 RCTP mechanism ablations

- RCTP 去掉 `kappa/SRF`，只给 sensor ID；
- wavelength-shuffled `kappa`；
- achromatic injection；
- 去掉 PSF/GSD，只在统一 224 grid 注入；
- 去掉 `L_cf`；
- 去掉 `L_visit`；
- 去掉 `L_xsensor`；
- 对齐 raw scene embedding 而不是 response difference；
- 去掉 `L_bg`；
- 去掉 frozen anchor；
- 只 Gaussian renderer / 只 empirical renderer / 混合 renderer；
- clean-only、positive-history-only、balanced mixed background pool；
- PTH 初始化 vs random init；
- 32-band native EMIT vs 16-band simulated EMIT；
- S5P native 3×3 vs 224 pseudo-image。

### 9.3 Temporal/sensor architecture ablations

- t0 only；
- T only；
- S only；
- T→S；
- S→T；
- joint full attention（matched parameters）；
- 去掉真实 `delta_days`；
- history shuffle；
- duplicate unmasked；
- sensor dropout / visit dropout；
- CLS-only vs CLS+GeM/top-k patch evidence。

### 9.4 关键 falsifier

RCTP 只有同时满足以下条件才支持 story：

1. full RCTP > CutPaste/achromatic/wavelength-shuffled；
2. 去掉/打乱 SRF response 后 capability 与 downstream 显著回落；
3. synthetic-vs-clean 捷径检测不成立；
4. 真实 classification **和**真实 segmentation 至少各有一个稳定增益；
5. 在 matched compute 下优于 generic MAE/latent objective。

如果只比 B0 高、但与 CutPaste 或 wavelength-shuffle 相同，则结论只是“synthetic augmentation 有用”，不能声称 response-conditioned pretraining。

---

## 10. 评估、成功线与停止线

### 10.1 主指标

Classification：

- positive-class F1（固定 0.5 与 inner-val 锁定 threshold 都报告）；
- macro F1；
- AUPRC、AUROC；
- recall at fixed FPR；
- event-balanced 与 row-weighted 两种汇总。

Segmentation：

- pixel AUPRC 为主；
- IoU/F1；
- object/event detection recall at fixed false alarms；
- 按 plume size/strength 分层。

Capability：

- methane vs wavelength-shuffled ranking accuracy；
- injected visit accuracy；
- column-strength Spearman；
- missing-sensor/visit degradation curve；
- sensor-held-out response transfer；
- real-vs-synthetic spectral/SNR calibration。

### 10.2 F1 目标要在同一 protocol 下定义

已知 legacy S2 checkpoint 约 0.8955 F1，但 full multisensor 360 m 历史量级约 0.83。二者不是同一个任务。

建议设三条线：

1. **工程 stretch target**：严格复现 legacy S2 协议时达到/超过 0.895，并确认不是 threshold/test selection 造成；
2. **full multisensor target**：先超过同 split strong PTH baseline至少 2 个绝对 F1 points；0.88 为强目标，0.90 为 stretch，不预先承诺；
3. **journal go criterion**：
   - 至少两个 sensor 各提高 ≥2.0 F1 points，且其余 sensor 无超过 1.0 point 的系统性下降；或
   - 一个最困难 sensor 提高 ≥4 points，同时四 sensor macro 平均提高 ≥2 points；
   - 并且 response falsifier 与 segmentation transfer同时成立。

只有 0.2–0.5 point 的不稳定增益不足以支撑该 story。

### 10.3 早停与 pivot

Downstream：

- 第 2–3 epoch inner-val F1 仍低于原 PTH 1 point以上：立即停；
- train F1 上升、inner-val 连续两个 evaluation下降：判定过拟合；
- 同一 arm 不用几十个 epoch“等奇迹”。

Pretraining：

- 2,000 updates 后 methane-vs-hard-negative ranking ≤55%：停并检查 renderer/loss；
- 5,000 updates 后 full RCTP 与 wavelength-shuffle capability相同：停掉该实现；
- feature variance collapse、有效 token 数异常或 response norm 只由 sensor ID 决定：停；
- Stage 1 无有效 downstream signal：不进入 Stage 2。

迭代优先级：

```text
数据/renderer捷径 -> loss scaling -> local patch evidence
-> SRF/PSF metadata -> LoRA范围 -> 才是更长训练
```

---

## 11. 泄漏控制与 formal protocol

1. 先按 canonical/base plume event 建 outer train/dev/test；共享路径、acquisition、site cluster 的 connected component 不跨 split。
2. temporal split 中同一 plume/event 的所有 positive/negative crop必须同侧。
3. normalization、renderer strength/shape distribution、hard-negative参数、threshold、temperature、early stopping只用 outer-train/inner-dev。
4. synthetic background只来自 outer-train；不得对 dev/test 图像做预训练或 renderer calibration。
5. SRF 与公开 CH4 cross-section 是外部物理常量，可以跨 split；由数据拟合的温压/噪声/PSF校准不可跨 split。
6. 按 `(sensor, acquisition_id/path hash, crop center)` 去重；重复 role只保留一份 evidence。
7. S2 zero compatibility mask不能作为真实 negative segmentation target。
8. L89 hard-event filtered test不作为 final claim；必须另建未由模型错误筛选的 test。
9. 历史 test-selected PTH 仅标为 engineering warm start；formal PTH 必须在当前 outer-train重训。
10. dev 选一次 checkpoint与threshold，outer-test锁定后只跑一次；不以 test F1 决定继续几 epoch。
11. matched-compute 对照固定：
    - 相同 initialization；
    - 相同 unique backgrounds/synthetic pairs；
    - 相同 optimizer updates；
    - 相同 encoder、feature tap、downstream head；
    - 至少 3 seeds；
    - paired bootstrap 按 event 重采样。

---

## 12. 两张 A100 与 `/diniuvol` 的执行预算

### 12.1 Cache 结构

建议：

```text
/diniuvol/yuyao/methanefuse_research_20260727/rctp/
  manifests/              # split + SHA256 + audit
  metadata/               # SRF/kappa/PSF/product versions
  background_index/       # 不复制全部 raw，只存索引与质量统计
  synthetic_shards/       # WebDataset/LMDB shards，按 sensor/version
  feature_cache/          # frozen PTH feature smoke
  checkpoints/
  logs/
```

- synthetic pair尽量 deterministic on-the-fly；只 cache renderer结果或 feature shard，不无界复制所有远程 TIFF；
- shard 目标 1–4 GB，原子写 `.part -> final`；
- 每个 artifact 保存 source manifest SHA、renderer config SHA、git commit、seed；
- DataLoader 16–24 workers/进程起步，遥远 I/O 与 rendering 分开线程池；
- 监控 `/diniuvol` free space，至少保留 100 GB safety margin。

### 12.2 GPU 分工

Gate 0/Stage 1：

- GPU0：EMIT32 + L89，先验证 spectral renderer 与不可见 band；
- GPU1：S2 + native S5P，并行验证 broad-band 与 product-space分支。

Stage 2：

- GPU0：full RCTP；
- GPU1：最强 matched baseline/关键 falsifier（通常是 wavelength-shuffle 或 CutPaste）；
- 显存不足才单进程；若 backbone冻结且显存富余，一张卡可并行 2 个小 head，但需限制 CPU/I/O 竞争。

### 12.3 10 小时内可完成的里程碑（估计，不是保证）

| 时间 | 交付 |
|---:|---|
| 0–1 h | split/hash/duplicate audit；SRF/shape/units manifest |
| 1–2.5 h | EMIT full renderer + L89/S2 `kappa` 近似 + S5P product renderer smoke |
| 2.5–3.5 h | Gate 0 shortcut/SNR/area 报告 |
| 3.5–6 h | Stage 1 frozen adapter：RCTP、wavelength-shuffle 两臂并行 |
| 6–8 h | 每 sensor 2–3 epoch downstream probe；淘汰无效臂 |
| 8–10 h | 胜者 LoRA short continuation或关键 ablation；汇总 paired metrics |

若 raw data/renderer准备超过预算，优先完成 EMIT32+L89 的机制验证，不能为了“跑起来”退化成四 sensor 224 pixel MAE。

---

## 13. 可执行里程碑与交付物

### M0 — Protocol lock

交付：

- `rctp_split_audit.json`
- 四 sensor manifest SHA；
- event/acquisition/path overlap=0；
- train-only normalization provenance；
- PTH provenance（是否 test-selected）。

通过后才允许生成 synthetic。

### M1 — Response metadata

交付：

- `sensor_response_metadata.json`
- 每 band `mu/sigma/kappa/SRF hash`；
- GSD/PSF/product/unit；
- EMIT32 与 legacy EMIT16 显式分开；
- S5P product operator version。

单测：

- SRF积分归一；
- `kappa` 符号/排序 sanity；
- wavelength shuffle 可复现；
- zero/invalid band被 mask。

### M2 — Renderer Gate

交付：

- 每 sensor clean/methane/6种 hard-negative contact sheet；
- `renderer_calibration.json`；
- synthetic-vs-clean discriminator；
- real/synthetic matched-filter SNR与形态分布；
- alternate renderer sensitivity。

Gate 0 不通过就停。

### M3 — RCTP frozen-adapter screen

交付：

- RCTP vs wavelength-shuffle vs CutPaste；
- 5k-update curves；
- capability metrics；
- 每 sensor 2–3 epoch frozen downstream；
- exact configs/checkpoints/logs。

只有 full RCTP 在 capability 与至少一个真实 downstream task都胜出才进入 M4。

### M4 — LoRA short continuation

交付：

- last-4 LoRA；
- anchor/no-anchor ablation；
- classification + segmentation；
- 3 seeds shortlist；
- event-level paired bootstrap。

### M5 — Formal multisensor experiment

交付：

- clean outer split 重训的 baseline PTH；
- full four-sensor RCTP；
- T-only、S-only、T×S matched heads；
- missing sensor/visit robustness；
- sealed test一次性评估；
- paper table、failure cases 与限制。

---

## 14. 最小决策树

```text
Renderer Gate 0 失败
  -> 修 SRF/PSF/单位/捷径；不训练

Gate 0 通过，但 RCTP ≈ wavelength-shuffle
  -> response conditioning 未生效；检查 metadata/loss
  -> 仍相同则放弃 RCTP 主线

RCTP capability 胜出，但真实 downstream 不升
  -> synthetic-to-real gap
  -> 调 renderer/calibration，不延长 epoch

classification 升、segmentation 不升
  -> 只能写 classification method，不能宣称 dense transfer

classification + segmentation 都升，且关键 ablation 回落
  -> 进入 formal 3-seed/matched-compute

只提升 <0.5 F1 point
  -> 不足以支撑 journal 主方法；保留为负结果/分析
```

---

## 15. 最终建议

先做 EMIT32 与 L89，是因为它们分别提供最强的光谱机制验证条件和较便宜的下游筛选；随后迁移到 S2 broad-band 和 S5P product-space。不要先做四 sensor 大一统 MAE。

RCTP 的价值不应由“多了几个 attention block”证明，而应由三个结果链证明：

```text
正确 CH4 response > 等能量错误 response
         ↓
真实弱瞬态 classification/segmentation 提升
         ↓
缺 sensor/跨 sensor 时仍保留该能力
```

只要其中任一箭头断裂，就应缩小或放弃主张。若三者都成立，论文故事会比“六时刻 + 双轴 cross-attention”强得多：模型提高的不是一般 scene reconstruction，而是 **对不可见甲烷扰动的 sensor-conditional selectivity**。

---

## 16. 本方案审阅过的本地实现与数据依据

- `thirdparty/dinov2/models/panopticon.py`
- `src/data/sensor_transforms.py`
- `/home/yuyao/NormWear/modules/methane_residual_mae.py`
- `/home/yuyao/NormWear/methane_pipeline/dataset.py`
- `research/pretraining_20260727/methane_residual_mae_validity.py`
- `research/pretraining_20260727/query360_data.py`
- `research/pretraining_20260727/s5p_native_grid_experiment.py`
- `research/pretraining_20260727/emit_ragged_cls_experiment.py`
- `Upgraded_dataset/dino_classifier_head_s2_temporal_satmae.py`
- `Upgraded_dataset/dino_classifier_head_l89_temporal_satmae.py`
- `Upgraded_dataset/dino_classifier_head_emit32_temporal_satmae.py`
- `Upgraded_dataset/dino_classifier_head_s5p_temporal_satmae.py`
- `/home/yuyao/methane_train/preprocess_dataset_query_multi/`
- `/home/yuyao/methane_train/preprocess_dataset_L89/landsat9_oli_srf.csv`
- `/home/yuyao/methane_train/preprocess_dataset_EMIT/WV3_VNIR_SWIR_response.csv`
- `Upgraded_dataset/README.md` 与四个 split audit

