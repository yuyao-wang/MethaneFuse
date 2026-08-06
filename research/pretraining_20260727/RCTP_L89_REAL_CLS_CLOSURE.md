# RCTP continue-pretrain → real L89 classification：最小闭环

**状态：代码和 CPU smoke 已完成；未启动 GPU。**  
**范围：train/recent-dev only；不读取 test/sealed。**

## 1. 审计结论

现有 `rctp_l89_screen.py` 是有效的 frozen-encoder mechanism screen，但不能
直接回答真实 classification transfer：

1. Panopticon 在整个 screen 中始终 `requires_grad_(False)`；3.5 MB 的
   `checkpoint_best_dev_ap.pt` 只含 `probe_state`，没有更新后的 encoder。
2. screen 只在线编码被 injection 的一个 clean visit 和三个变体，另外五个
   clean visit 来自 base-PTH frozen CLS cache。
3. 如果只把最后 blocks 解冻、但继续使用上述 cache，第一次 optimizer update
   后便会在同一 sequence 中混合 current-encoder feature 和 stale base-encoder
   feature。这不是近似误差，而是表示空间不一致。
4. 现有 L89 downstream `train-heads` 只消费 frozen CLS cache；因此任何更新后
   encoder 都必须重新抽取真实 train/dev CLS，或改成昂贵的逐 epoch online
   image training。一次抽取后训练 small head 明显更省算力。
5. screen 的 response adapter 位于 objective-specific probe 内，不能单独接到
   普通真实 L89 classifier；真正可迁移的最小参数是最后 1–2 个 backbone
   blocks。probe downstream 全部丢弃。

因此不能把已有 screen checkpoint 称为“RCTP pretrained PTH”，也不能直接拿
它跑真实 L89 classification。正确的最小闭环是重新从相同 clean Panopticon
PTH 独立训练 P4/P5，在线重编码全部六个 clean visits，再导出完整 backbone。

## 2. 已实现的最小修复

新增
`rctp_l89_continue_pretrain.py`：

- P4/P5 都从
  `weights/panopticon_vitb14_teacher.pth`
  开始；
- 默认首轮中，旧 cache 只提供 `id/event_id`、`unique_mask`、`delta_days`、
  role 与 input contract；
- 完成 SHA/row-alignment 审计后，`features` 永远从 temporal/probe payload
  中删除；默认 weight=0 时直接丢弃，显式开启 fallback 时只保留在单独命名的
  CPU detached-target handle，因此不能成为 context/probe input；
- 每个 planned row 的六个有效 clean visits 与三个 matched variants 在一次
  same-checkpoint forward 中编码；
- 只解冻 ViT blocks 10–11，其他 trunk 保持 frozen/eval；
- 实际 Panopticon 参数数：
  - last 1 block：7,089,408 trainable；
  - last 2 blocks：14,178,816 trainable；
  - objective probe：900,707 trainable；
- backbone LR `1e-5`，probe LR `1e-4`，AdamW、5% warmup、cosine、bf16、
  clip 1.0；
- 可选 `--clean-anchor-weight` 默认严格为 `0`；零权重不加载 target、不做
  anchor 算术，并原样返回原 pretext loss tensor/graph；
- 显式启用时，只对 validity-masked online clean CLS施加 cosine 或 L2
  distillation；variants不参与，base target始终 detach；
- best dev AP checkpoint 保存完整 `backbone` state；现有
  `load_backbone()` 能严格直接加载；
- epochs 被 parser 硬限制为 1–3。

P4 与 P5 使用完全相同的三张 render、plume、strength、visit、loss、初始化、
update 数和参数形状：

| Arm | 正样本 | 两个 nuisance | response metadata |
|---|---|---|---|
| P4 | fixed wavelength-scrambled response | achromatic + correct response | 同一 scrambled response |
| P5 | correct L89 response | achromatic + scrambled response | correct response |

实现方式只是对 renderer 的相同三个 tensors 做 `(2,1,0)` 或 `(0,1,2)`
重排，因此不会引入额外 image/FLOP/data confound。

## 3. 数据量与真实 downstream

已审计的 six-visit manifests/cache：

| Split | Rows | Labels 0/1 | Canonical events | Unique observations |
|---|---:|---:|---:|---:|
| train | 10,033 | 5,585 / 4,448 | 723 | 56,059 |
| recent-dev | 9,614 | 5,454 / 4,160 | 124 | 54,135 |

train/dev canonical-event overlap 为 0，所有六时刻路径已经位于
`/diniuvol/yuyao`，不需要直接读远程盘。

默认 bounded continue-pretrain：

- negative reference backgrounds；
- 每 arm、每 epoch：2,048 train rows；
- 固定 512-row dev panel；
- 2 epochs；
- 每行在线编码 `6 clean + 3 variants = 9` images；
- 每 arm 两 epoch约 36,864 train image forwards，加 9,216 dev forwards。

真实 classification 使用完整 10,033/9,614 rows，而不是 synthetic label：

1. P0 复用已经由 base PTH 严格抽取的 full train/dev cache；
2. P4 用导出的 P4 PTH重新抽取完整真实 train/dev CLS；
3. P5 用导出的 P5 PTH重新抽取完整真实 train/dev CLS；
4. 三个 arm 都训练同一个 locked `role_only` temporal head，seed、initial state、
   parameter shapes、batch order、LR 与 3-epoch cap完全相同。

`role_only` 在 RCTP 结果出现前已由三 seed base experiment锁定：此前 dev AP
约 `0.755–0.775`、macro-F1@0.5 约 `0.720–0.739`，因此不会依据 P5 的结果
选择一个更有利的 readout。

`l89_ragged_cls_experiment.py` 已补充：

- canonical-event-balanced AP/AUC；
- positive F1 与 macro-F1@0.5；
- dev-selected event-balanced positive/macro F1；
- event-balanced threshold；
- prediction CSV 中保留该 threshold 的 prediction。

最终 `audit_rctp_l89_real_cls_loop.py` 还会验证 P0/P4/P5：

- dev IDs、plumes、events、labels完全一致；
- 每个 cache 的 `weights_sha256` 与对应 PTH完全一致；
- downstream initial-state SHA 与参数签名一致；
- P4/P5 base SHA、probe init、trainable init、plan SHA、optimizer steps一致；
- 以 canonical event 为 cluster 做 2,000 次 paired bootstrap，报告
  `P5−P0`、`P5−P4` 和 `P4−P0` 的 F1 置信区间。

## 4. 精确执行与 GPU 预算

闭环 launcher 是 dry-run by default：

```bash
bash research/pretraining_20260727/run_rctp_l89_real_cls_loop.sh
```

确认 legacy extraction 已释放两张 GPU 后才显式运行：

```bash
bash research/pretraining_20260727/run_rctp_l89_real_cls_loop.sh --run
```

launcher 在检测到任意 active GPU compute process 时默认拒绝启动。P4/P5
分别占一张 GPU；head training 和统计 audit 在 CPU。

历史 full cache 抽取实测：

- train：522 秒；
- recent-dev：394 秒。

硬 timeout 预算：

| Stage | P4 | P5 | 合计 GPU-min |
|---|---:|---:|---:|
| continue-pretrain | ≤20 min | ≤20 min | ≤40 |
| full train CLS refresh | ≤12 min | ≤12 min | ≤24 |
| full dev CLS refresh | ≤10 min | ≤10 min | ≤20 |
| downstream head/audit | CPU | CPU | 0 |
| **总硬上限** | **42 min** | **42 min** | **84** |

两 arm 并行时 wall time 硬上限约 42 分钟，加 CPU head/audit；预期实际
GPU 总量约 50–70 GPU-min。任何 stage timeout 或 non-finite 会 fail closed，
atomic artifact 不会留下被误认为完成的 checkpoint/cache。

## 5. 1–3 epoch 判定

当前最小 screen 默认 2 pretrain epochs、3 downstream epochs：

- epoch 1 出现 representation collapse、dev AP 接近随机或 P5 capability
  不高于 P4时，不延长；
- epoch 2 后必须进入真实 classification，不以更多 synthetic epoch掩盖
  transfer failure；
- 第 3 pretrain epoch只允许在 epoch 2 的 P5 capability 仍明显上升、P5>P4，
  且重新计算后总 GPU 预算仍低于 90 分钟时运行；
- downstream formal comparison固定三个 arm 都跑满相同 3 epochs，再按 dev
  AP选 epoch；不会 differential early-stop。

初始 engineering promotion：

- P5 相对 P0 的 event-balanced F1/AP 至少 `+1.0` absolute point；
- P5 相对 P4 同时为正，且 paired event-bootstrap不能显示明显退化；
- 若只提升 `0.2–0.5` point或 CI 大幅跨零，不能写成 RCTP transfer result。

## 6. Transfer 弱时唯一允许的最小 fallback

只有在首轮出现 **P5 synthetic capability 明显高于 P4，但 P5 的真实
classification transfer 接近/低于 P0** 时，才运行一次保守 fallback。若
P5 capability 本身不高于 P4，anchor不能修复 response objective，应停止而
不是调正则。

fallback 同时独立重跑 P4/P5，固定：

- 只解冻最后 1 block（7,089,408 parameters）；
- backbone LR 从 `1e-5` 降到 `3e-6`；
- probe LR仍为 `1e-4`；
- `clean_anchor_weight=0.10`，metric=`cosine`；
- 其余 rows、plans、render、updates、head与首轮完全相同；
- 不做 weight grid，避免在同一 dev 上反复寻优。

目标来自已审计 base-PTH six-visit CLS：

- train feature SHA：
  `6f730ece195cf7bcf9398a39c45854a737e77376a19f901cd2a4b4431a8fc917`；
- dev feature SHA：
  `93e0ffb39b6350dd3851dacb48b016632636f5a98eb055eeffe696a1d9c3fd84`；
- base PTH SHA：
  `55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e`。

dry-run：

```bash
RCTP_RUN_TAG=fallback_last1_anchor010 \
RCTP_TRAIN_LAST_BLOCKS=1 \
RCTP_BACKBONE_LR=3e-6 \
RCTP_CLEAN_ANCHOR_WEIGHT=0.10 \
RCTP_CLEAN_ANCHOR_METRIC=cosine \
bash research/pretraining_20260727/run_rctp_l89_real_cls_loop.sh
```

释放 GPU 后才在同一命令末尾加 `--run`。summary/run config/transfer PTH
metadata 都记录 weight、metric、train/dev target SHA；每个 epoch同时记录
raw anchor loss、weighted anchor loss与不含 anchor 的 pretext loss。

这个 fallback 的解释只是“限制最后 block 对 clean EO representation 的
漂移，同时保留 RCTP gradient”，不是新的 objective，也不能给 P5 单独使用。

## 7. CPU 验证

```bash
python -m research.pretraining_20260727.test_rctp_l89_continue_pretrain_cpu
python -m research.pretraining_20260727.test_l89_ragged_cls_experiment_cpu
```

当前结果：新闭环 8/8、原 L89 cache/head regression 8/8，共 16/16 PASS。
新增 anchor regression 明确检查：

- weight=0 返回同一个 pretext tensor，graph/gradient路径不变；
- target detach、invalid visits梯度为零；
- variants不进入 anchor；
- anchor gradient仍到达最后 block，冻结 blocks无梯度。

其余测试继续覆盖：

- 只有最后 blocks 获得 gradient；
- P4/P5 tensor 集合相同且只交换 response role；
- 不含 `features` key 仍可完成一次在线 optimizer step；
- exported PTH 满足现有 `load_backbone` contract；
- event-balanced threshold；
- paired event bootstrap。

## 8. 不能越界的解释

这个闭环能回答“正确 L89 spectral response 的 counterfactual
continue-pretraining 是否迁移到真实 L89 classification”，但仍有三个限制：

1. 当前 L89 response `[0,0,0,0,0,0.15,1]` 和 wavelength shuffle 是 screening
   approximation，不是 versioned SRF integral；正结果只能先作为 engineering
   evidence。
2. 单 sensor L89 内 response metadata 为常数，不能证明 multi-sensor
   response conditioning 或 fusion；该主张仍需至少两种真实 sensor operator。
3. continue-pretrain checkpoint、classification checkpoint 和 threshold 都使用
   recent-dev；event bootstrap不能消除这种 adaptive selection optimism。协议
   冻结后仍需一次独立、明确授权的 final evaluation。

P1–P3 generic objectives也不在这个最小闭环内。即使 P5>P0，也必须先证明
P5>P4，再决定是否值得补齐 P1–P3 与 formal SRF renderer。
