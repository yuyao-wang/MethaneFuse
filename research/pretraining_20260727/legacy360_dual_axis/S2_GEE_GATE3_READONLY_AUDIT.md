# S2 GEE six-visit gate3 read-only audit

Audit date: 2026-07-27 UTC. No process was started/stopped and no training
file was changed.

## Bottom line

This run is **six visits concatenated as channels**, but it is not an
explicit temporal-attention model. It reshapes
`[B, 6, 12, H, W] -> [B, 72, H, W]`, repeats the 12 wavelength IDs six
times, and passes all 72 planes through Panopticon's learned-query channel
cross-attention. The concat branch drops the timestamp tensor and has no
visit/slot embedding. Consequently it learns a six-frame channel set rather
than preserving which content was `t0`, `prev1`, ..., `year` in the model.

It strictly loads `weights/panopticon_vitb14_teacher.pth` and
`--train_backbone` makes the complete Panopticon backbone trainable, with
backbone LR `1e-4` and head LR `1e-3`. Thus this is Panopticon-initialized
full fine-tuning, not a frozen probe. One qualification is that the Noam
`LambdaLR` multiplier peaks at only `1/sqrt(4000) = 0.01581`, so the actual
peak backbone/head LRs are about `1.58e-6`/`1.58e-5`; it is nominally full
fine-tuning but at a much smaller effective LR than the command-line values
suggest.

## Evaluation protocol

- `test_f1` is positive-class binary F1 from `argmax` predictions, equivalent
  to a 0.5 positive probability threshold in this binary softmax.
- `test_macro_f1` is macro-F1 at threshold 0.5.
- `test_best_f1` and its threshold are optimized directly on the test PR
  curve every epoch.
- There is no development loader. `ckpt_best_test.pth` is selected by test
  accuracy, while the misleadingly named `ckpt_best_val_ap.pth` is selected
  by **test** AP. If a human reports the largest F1 across the three logged
  epochs, that is another layer of test-set model selection.

Therefore the run is useful as an engineering screen, but its best epoch,
best threshold, or best checkpoint is not a sealed-test/SOTA result.

## Split audit

The CSV split contains 103,122 train rows and 18,245 test rows, with nearly
balanced labels. It is clean on the declared identifiers:

- sample-ID, row-ID, plume-ID, and event-group overlap: all zero;
- exact path overlap for each of the six slots, and across all slot columns:
  zero;
- train maximum event time: 2025-12-22 14:00:16 UTC;
- test minimum event time: 2025-12-24 04:38:14 UTC.

It is event-disjoint, but not spatially/site-disjoint. Among the 651 unique
test plume coordinates, 213 have a train coordinate within 0.1 km, 366
within 1 km, and 456 within 5 km. Exact coordinates rounded to six decimals
do not overlap. This does not prove duplicated pixels, and exact exported
paths do not overlap, but it leaves a material site/background-familiarity
risk. The result should be described as a temporal event split, not a
geographic generalization split.

## Comparability

The historical legacy360 S2 `0.895503` is not an apples-to-apples target for
this run. It used three roles `(t0,t-90,t-360)` concatenated into 36 channels,
59,635/14,954 old train/test rows, a different legacy360 crop/source/split,
and a checkpoint repeatedly selected on old test accuracy. That old split
also has 706 canonical-event overlaps. The shared Panopticon initialization
and concat/full-FT recipe make it a useful engineering guardrail, but a gain
or loss here cannot be attributed to “six visits” because the data and split
changed simultaneously.

The formal dual-axis experiment is even less directly comparable. It keeps
legacy360 features as `[row, sensor, time-role, 768]`, explicitly models
three time roles and four sensor roles with missingness masks, and uses
train_core/dev selection followed by one locked test evaluation. The present
gate3 run is S2-only, collapses visit identity inside channel fusion, fully
fine-tunes the image backbone, and consults test every epoch. It cannot test
the sensor axis. Only a same-manifest, same-base, same-selection ablation can
measure the value of the two-axis mechanism.

