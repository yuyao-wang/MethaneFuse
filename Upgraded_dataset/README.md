# 新数据集结果记录

给自己看的简单记录：单个 sensor，用 Panopticon 跑新数据。

| 数据 | epoch | train_loss | train_acc | train_f1 | test_loss | test_acc | test_f1 | recall | fpr | auroc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S5P 新数据 | 4 | 0.6000 | 0.6847 | 0.6948 | 0.7160 | 0.6085 | 0.6375 | 0.6583 | 0.4462 | 0.6503 |
| S5P 新数据（截止 2025-10） | 1 | 0.6918 | 0.5261 | 0.5497 | 0.6815 | 0.5538 | 0.6222 | 0.7101 | 0.6138 | 0.5876 |
| L89 新数据 | 3 | 0.4483 | 0.8126 | 0.7876 | 0.5283 | 0.7474 | 0.7119 | 0.6877 | 0.2030 | 0.8272 |
| L89 新数据（截止 2025-10） | 4 | 0.4445 | 0.8120 | 0.7883 | 0.4456 | 0.8309 | 0.8047 | 0.8242 | 0.1641 | 0.8991 |
| EMIT 32bands 新数据 | 9 | 0.3381 | 0.8863 | 0.8531 | 0.6013 | 0.7963 | 0.7146 | 0.6236 | 0.0842 | 0.8438 |
| EMIT 32bands 新数据（截止 2025-10） | 8 | 0.4177 | 0.8330 | 0.7865 | 0.5729 | 0.7477 | 0.6714 | 0.6305 | 0.1713 | 0.7886 |
| EMIT 32bands full（到 2026-05，时间 cutoff） | 12 | 0.3778 | 0.8642 | 0.8285 | 0.7499 | 0.6841 | 0.5667 | 0.4796 | 0.1611 | 0.7111 |

EMIT 32bands full（到 2026-05）数据路径：

- train：`Upgraded_dataset/emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/emit32_temporal_train.csv`
- test：`Upgraded_dataset/emit32_full_through_2026_05_strict_temporal_cutoff_2025_06_09/emit32_temporal_test.csv`

简单结论：L89 截止 2025-10 的版本整体最好；这次重新划分的 EMIT full test F1 是 0.5667。

## L89 波段实验

| 数据 | 最佳 epoch | test_acc | test_f1 | recall | fpr | auroc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 去掉 3 个事件，全波段 | 2 | 0.7638 | 0.7165 | 0.6653 | 0.1561 | 0.8369 |
| B1-5 | 2 | 0.7175 | 0.6579 | 0.5986 | 0.1836 | 0.7964 |
| B6-7 | 2 | 0.7300 | 0.6975 | 0.6859 | 0.2334 | 0.8008 |
| 非 RGB（B1、B5、B6、B7） | 5 | 0.7609 | 0.7139 | 0.6573 | 0.1530 | 0.8387 |

同一个 checkpoint 直接测试：full test F1 0.6770，去掉 3 个事件后 F1 0.7081（+0.0311）。三个事件有影响，但不能解释截止 2025-10 与全量数据之间的全部差距。

## L89 full 排查

- 模型确实学到了非可见光信息，B6-7 比 B1-5 更有效。
- full 变差的主因是不同 event 的 patch 数量差别太大，而且大 event 更难，不只是那 3 个 event。
- 旧模型 raw full：F1 0.7315；每个 `event × label` 最多保留 16 行后：F1 0.7702。
- 同样处理后，截止 2025-10 F1 0.7870，之后 F1 0.7630，差距明显缩小。
- 影像亮度、冬季和地区有次要影响，但单独过滤只能提高约 0.01。
- cap16 重新训练最佳 F1 0.7615，没有超过旧模型。

## L89 hard-event filtered split

- 动态删除 9 个困难 event，共 3056 行。
- 新 train：62389 行（88.71%）；新 test：7942 行（11.29%）。
- train/test event 泄漏为 0。
- 旧 checkpoint 验证：F1 0.7760，AUROC 0.8745。
- 从头训练 5 个完整 epoch，最佳是 epoch 2：test_acc 0.7872，test_f1 0.7544，
  recall 0.7372，fpr 0.1730，auroc 0.8531。
- epoch 3-5 的 test_f1 为 0.7422、0.7499、0.7388，未超过旧 checkpoint；
  第 6 个 epoch 已停止。
- 最佳 F1 checkpoint：
  `checkpoints/l89_hard_event_filtered/l89_hard_event_filtered_seed20251031/ckpt_best_test_f1.pth`。
- 目录：`l89_6time_temporal_hard_event_filtered_split/`。

注意：这 9 个 event 是根据旧 checkpoint 的 test 错误选出的，因此这个 split
适合排查和后续实验，但不是完全无偏的独立 test。
