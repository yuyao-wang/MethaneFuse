# TEMPO legacy360 global pilot protocol

## Question

Can explicit global onset evidence improve the already promoted scale-aware
legacy360 classifier without changing its Panopticon features or sacrificing
its development result?

## Frozen data boundary

- Complete `train_core`: `113,843 × 4 sensors × 3 roles × 768`.
- Development: `12,621 × 4 × 3 × 768`.
- Roles are current, approximately 90-day history, and approximately 360-day
  history.
- Inputs are the existing float16 universal Panopticon cache on `/diniuvol`.
- The runner rejects paths containing `test` or `sealed` and validates the
  train/dev event and plume identity boundary.
- No test/sealed cache, prediction, label, threshold, or metric may be read.

## P0 and matched arms

P0 is the frozen promoted `scale_aware_two_axis_query`, dimension 64,
development-selected epoch 3. Its known development positive F1 is `0.90558`.
The runner reproduces its logits from the immutable checkpoint and cache.

Every TEMPO arm has the same parameter signature, steps, seed-specific
initialization, optimizer, and zero-initialized final residual classifiers.
Epoch zero must therefore exactly equal P0.

- P1: current-minus-90-day and current-minus-360-day feature deltas; masked
  mean across histories and sensors.
- P2: P1 replaced by a signed temporal innovation and an excess-magnitude
  channel. The innovation subtracts the history-to-history trend scaled by
  `90/(360-90)`.
- P3: P2 plus a learned nominal-gap/similarity/reliability gate. S5P history is
  excluded because the stored S5P context is approximately 10.5 km rather
  than plume-local 360 m evidence.
- P4: P3 plus learned evidence-token fusion across available sensors.

After the hard-coded trend screen, a second matched screen tests:

- Q1: raw P1 deltas with a learned history gap/similarity gate while retaining
  every sensor.
- Q2: a shared per-feature mixer learns two evidence channels from short and
  long deltas plus signed and absolute historical change. It does not assume
  that background evolution is linear.
- Q3: Q2 plus evidence-token fusion whose gate also sees the frozen promoted
  model's per-sensor logit and absolute confidence.

The final raw-delta optimization screen uses R1 (longer P1), R2 (raw deltas
with both learned gates), and R3 (R1 plus an equal-event-weight all-negative
training penalty). It screens learning rate `1e-4/3e-4`, residual cap
`0.5/1.0`, and null-loss weight `0.02/0.05`. Weight `0.1` was predeclared but
pruned after both lower weights reduced F1. A setting is not promoted unless
AP does not regress and all-negative FP rows are at most P0's 29.

R4 is a single bounded video-inspired follow-up, not another broad search. For
each sensor/history pair it projects current appearance, signed delta, and
absolute delta. Signed/absolute motion generates a feature-wise sigmoid gate
that excites current appearance, then signed motion is added residually:

`e = appearance(t0) * sigmoid(g(delta, |delta|)) + motion(delta)`.

It retains the frozen promoted logits and zero-initialized residual classifier.
R1 and R4 instantiate the same complete module set and seed-specific initial
state. R4 is stopped after epoch 2 unless it has already exceeded matched R1.

R4-add is the strict active-capacity control for R4. It invokes the same
appearance projection, signed-motion projection, magnitude projection,
excitation MLP, pooling, residual MLP, and classifiers, but replaces
multiplicative excitation with

`e = appearance(t0) + motion(delta) + tanh(g(delta, |delta|))`.

For each arm the runner performs a deterministic forward/backward audit and
records every parameter for which autograd materializes a gradient. A
zero-valued gradient still counts as graph-active because the final
classifiers intentionally start at zero. At `D=768`, R4 and R4-add both
activate exactly `123,314` of `134,210` declared parameters, have the same
active-name digest, and execute `924,144` learned-linear MACs per row. Their
only treatment difference is the fixed pointwise fusion operator. They use
the same seed-42 initial-state digest, optimizer, data order, batch size,
maximum of four epochs, and patience one.

R5 keeps R1's per-sensor raw-delta tokens unchanged and replaces mean sensor
pooling with one small masked multi-head set-attention layer. A learned CLS
liaison queries only available sensor tokens; missing sensors are key-masked.
It deliberately does not use a base-logit scalar gate. R1/R4/R5 instantiate
the same complete module set, so their recorded parameter signature and
seed-specific initialization are identical.

R6-attn is the direct cross-attention baseline requested by the architecture
question. Within each sensor, t0 is the query and valid short/long histories
are keys/values; sensor outputs are mask-averaged. Three independent
`768→48` projections, a bias-free attention output, and a post-attention MLP
make its graph-active parameter count exactly equal R4's `123,314`. It uses
the same seed-42 initialization policy, optimizer, steps, maximum of three
epochs, and patience one. Epoch zero remains exact P0 and the original
unshuffled threshold is frozen for history shuffle.

## Screen and promotion

1. Run P1–P4 for seed 42 and two epochs.
2. P0 remains selection-eligible. Report both the best trained epoch and the
   deployable best including exact-P0 fallback.
3. Promote only arms that improve development F1/AP without increasing the
   equal-event-weight all-negative canonical-event FP mass.
4. Run promoted arms for seeds 17, 42, and 73. Do not expand weak arms merely
   because compute remains.
5. Evaluate a history-shuffle control at the original selected threshold;
   never re-optimize a threshold on shuffled development.

Primary outputs are positive-class F1, macro F1 at that threshold, AP, AUC,
sensor strata, all-negative-event FP mass, and history-shuffle deltas.

This is a diagnostic engineering screen on the old legacy split. It cannot be
presented as a leakage-free SOTA result.

## Completed result

R4 is the selected video-inspired arm. Across the frozen seeds
17/42/73/101, its development F1 is
`0.911286/0.909452/0.910796/0.907121` (mean `0.909664 ± 0.001864`), versus
P0 `0.905582`. Mean macro F1 improves from `0.895073` to `0.900867`; mean AP
from `0.960034` to `0.960883`; mean AUC from `0.960381` to `0.961758`.

The equal-logit four-seed R4 ensemble reaches F1 `0.910235`, AP `0.961690`,
and AUC `0.962527`. Its P0-constrained operating point reaches F1 `0.910074`
and macro F1 `0.901761`, while reducing all-negative event alarms from
`12/27` events and 29 rows to `11/27` events and 28 rows, and increasing
positive/mixed event any-detection from `257/264` to `259/264`.

At 10,000 fixed-threshold canonical-event bootstrap resamples, the
three-seed-mean R4 minus P0 deltas are:

- binary F1 `+0.004024`, 95% CI `[+0.000581,+0.007410]`, win probability
  `0.9894`;
- macro F1 `+0.005744`, 95% CI `[+0.002189,+0.009302]`, win probability
  `0.9993`;
- AP `+0.000863`, 95% CI `[-0.000610,+0.002379]`, win probability `0.8767`.

History shuffling at the original ensemble threshold reduces F1 by `0.003638`
and AP by `0.002098`, without threshold refitting. Thus the gain depends on
matched temporal evidence. R5 is stopped: its best seed-42 F1 is `0.909592`
and its unguarded all-negative FP rows increase to 30.

The strict R4-add capacity control peaks at epoch 1 with F1 `0.907662`,
macro F1 `0.897494`, AP `0.960093`, and AUC `0.961240`; matched R4 peaks at
epoch 2 with `0.909452/0.901787/0.960753/0.962081`. In 10,000 fixed-threshold
paired canonical-event resamples, R4-add minus R4 has binary-F1 delta
`-0.001778`, 95% CI `[-0.005098,+0.001587]`, but macro-F1 delta `-0.004313`,
95% CI `[-0.007905,-0.000869]`. Therefore the primary supported mechanism is
the dual-stream use of current appearance and signed/magnitude motion.
Multiplicative excitation is the best tested fusion variant and has stronger
macro-F1 evidence, but one seed does not isolate it through primary binary F1.

The direct t0-query/history-KV attention control peaks at epoch 1 with F1
`0.909078`, macro F1 `0.899392`, AP `0.960161`, and AUC `0.961535`, below
matched seed-42 R4 on all four metrics. Shuffling history at its original
threshold reduces F1 by `0.002457` and AP by `0.000692`, so it uses history
but does not explain R4's stronger result.

Finally, the predeclared no-training availability rule applies R4 only to
exactly-one-sensor rows and otherwise returns P0, retaining both locked global
thresholds. It reaches F1 `0.910745`, macro F1 `0.901134`, AP `0.961190`, and
AUC `0.962070`. It is the higher-F1 deployment policy; global R4 remains the
better ranking model with AP `0.961690`.

All results are development-only and post-selection. No test/sealed artifact
was read.

## Exactly-once locked evaluation

The final evaluation program freezes the following before any sealed/test
input is named or opened:

- the promoted P0 checkpoint and its SHA-256;
- R4 checkpoints for seeds 17, 42, 73, and 101 and their SHA-256 values;
- equal logit weights `[0.25, 0.25, 0.25, 0.25]`;
- the global P0 and R4 development thresholds;
- the deployment rule `exactly one current sensor -> R4; otherwise -> P0`;
- all row- and event-level metrics.

The immutable manifest SHA-256 is
`62a7f6f047ab4cd773aea8c7aa309db0517508ef8c2d042493e1916a29b4ac92`.
Execution requires this exact digest, a literal confirmation token, a new
output directory, and an explicitly marked sealed/test input path. It first
streams the source cache once into local staging while hashing it. P0, global
R4, and the availability rule then share the same single in-memory payload;
there is no threshold, epoch, weight, or subgroup selection.

The CPU development dry-run reproduced P0 F1 `0.905424`, global R4 F1
`0.910235`, and availability-gate F1 `0.910745`. Unit and regression tests
passed `20/20`. The sealed/test command has not been executed and must remain
unused until the experiment owner explicitly authorizes the lock digest.
