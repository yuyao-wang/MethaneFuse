# Sensor-wise BCE + RankNet fallback plan

Last updated: 2026-07-27

## Authorization boundary

> **This fallback is authorized only if the completed
> validity-masked-reconstruction promotion gate returns `FAIL` /
> `do_not_promote`.**

Until that decision exists, do not implement this objective, modify the
runner, or launch any experiment described here. If the validity-masked gate
passes, this fallback is unnecessary and must remain unimplemented.

This is the only planned fallback after validity-masked MAE. It is a
supervised, AP-aligned classification objective, **not self-supervised
pretraining**. A successful result may support a claim about sensor-native
temporal classification with ranking-aware training. It must not be described
as a new pretrained foundation model or as evidence that self-supervised
pretraining helps.

If a self-supervised pretraining contribution is mandatory, the recommendation
after a failed validity-masked gate is to stop this objective search rather
than substitute an unbounded repository or pretext-task search.

## Motivation

The globally purged unmasked MAE experiment is a valid negative result:
macro AP changed by `-0.008082`, with material regressions on S2 and L89. If
validity-masked MAE also fails, invalid-pixel reconstruction is no longer a
sufficient explanation for the transfer failure.

The validation-only individual experiments also provide little support for
another residual-centric self-supervised pretext:

- S2 current-only, raw temporal, and residual temporal-token results favor
  current/raw input over the tested residual representation;
- the matched L89 residual pilot is below its raw-temporal pilot;
- S5P raw temporal input is stronger than its tested residual input;
- EMIT is the only sensor with a small, repeatable residual-over-raw ranking
  advantage.

A pixel reconstruction loss allocates most of its gradient to spatial,
spectral, and background elements that are not the downstream decision.
Sensor-wise pairwise ranking instead penalizes a positive methane crop being
ranked below a negative crop. This is closer to the primary AP evaluation
than pixel MSE, while retaining pointwise BCE to preserve probability scale
and fixed-threshold behavior.

## Exactly one objective change

Keep the current sensor-native model unchanged:

- sensor-specific patch stems;
- one shared encoder;
- sensor-specific classification heads;
- normalized current, recent-residual, and seasonal-residual streams;
- current-query temporal fusion.

For one sensor batch \(s\), let:

- \(z_i\) be the pre-sigmoid classification logit;
- \(P_s = \{i : y_i = 1\}\);
- \(N_s = \{j : y_j = 0\}\).

The within-sensor RankNet term is:

\[
L_{\mathrm{rank},s}
=
\frac{1}{|P_s||N_s|}
\sum_{i \in P_s}
\sum_{j \in N_s}
\operatorname{softplus}
\left(
\frac{z_j-z_i}{\tau}
\right).
\]

The complete sensor loss is:

\[
L_s
=
L_{\mathrm{weighted\ BCE},s}
+ \lambda L_{\mathrm{rank},s}.
\]

The settings are preregistered and fixed:

- \(\lambda = 0.5\);
- \(\tau = 1.0\);
- all positive-negative pairs in the current sensor batch are used;
- no hyperparameter sweep;
- no hard-negative mining;
- no projection head, decoder, memory queue, or additional data view.

At random initialization, both BCE and the pairwise logistic term are normally
near `0.693`. The fixed `0.5` multiplier makes ranking an auxiliary signal
without allowing it to dominate BCE calibration.

If either class is absent from a batch, define the ranking term as a
graph-connected exact zero, for example `logits.sum() * 0.0`, and continue
with BCE. The run must log how often this occurs.

## Positive and negative construction

- A positive is a training row with classification label `1`.
- A negative is a training row with classification label `0`.
- Pairs are formed only within the same sensor and current mini-batch.
- Every valid positive-negative pair receives equal weight.
- Different sensors must never be paired because their score distributions,
  native measurements, class prevalence, and calibration can differ.
- No validation example, validation score, validation threshold, pseudo-label,
  or model-selected hard negative may enter training.

Canonical-event duplicates within the training partition do not create
train/validation leakage because no event crosses the globally purged
boundary and event identifiers are not model inputs. They can nevertheless
cause event-level pseudo-replication. The run should report the relevant
event-count audit, and later uncertainty estimates must be grouped by event
rather than by crop row. This fallback must not bundle a new event sampler or
event-weighting rule with the loss change.

## Matched one-epoch protocol

The fallback changes only the supervised loss. It must use:

- `four_sensor_global_val_purge_v1`;
- the same purged train and recent-validation manifests;
- zero global canonical-event train/validation overlap;
- the same normalization-statistics file and hash;
- seed `20260727`;
- the same current/residual representation and clipping;
- the same architecture, initialization procedure, augmentation, batch size,
  optimizer, learning rate, and weight decay;
- 157 balanced rounds;
- 628 optimizer steps;
- one complete supervised epoch;
- full validation for all four sensors;
- no initialization from either failed MAE checkpoint;
- no test access.

The existing one-epoch BCE scratch artifact may be the comparator only if a
machine check confirms identical manifest hashes, normalization hash, event
protocol fingerprint, encoder signature, initial head hash, seed, batch
configuration, balanced rounds, and update count. If that exact comparison
cannot be established, the result is not eligible for promotion; do not
silently compare incompatible runs.

The objective signature must record at least:

```text
objective = sensorwise_weighted_bce_plus_all_pairs_ranknet_v1
rank_weight = 0.5
rank_temperature = 1.0
pair_scope = within_sensor_current_batch
pair_selection = all_positive_negative_pairs
one_class_policy = connected_zero_rank_loss
```

The training history must separately record, per sensor:

- total loss;
- weighted BCE;
- RankNet loss;
- positive count;
- negative count;
- pair count;
- number and fraction of one-class batches.

At least 99% of training batches must contain a valid positive-negative pair.
A lower fraction invalidates the screen as a sampling failure rather than a
negative model result.

## Required CPU tests before any GPU launch

1. **Hand-computed loss.** On a small fixed logit/label tensor, verify the
   implementation against a manually enumerated \(P \times N\) calculation.
2. **Permutation invariance.** Permuting batch rows must not change the
   pairwise loss or pair count.
3. **Gradient direction and shift invariance.** Positive logits must receive
   gradients that increase them, negative logits gradients that decrease
   them, and adding one constant to every logit must not change RankNet loss.
4. **Numerical stability.** Logits at least as extreme as `-100` and `+100`
   must produce finite loss and gradients through `softplus`; do not implement
   the loss with explicit exponentials.
5. **One-class behavior.** All-positive and all-negative batches must return
   an exact graph-connected zero ranking term, finite BCE, and finite
   backward gradients.
6. **Sensor isolation.** Synthetic batches from two sensors must be reduced
   independently. A test must fail if a positive from one sensor can pair with
   a negative from the other.
7. **Legacy equivalence.** Setting the ranking weight to zero must reproduce
   the legacy supervised BCE path exactly, or within a documented
   floating-point tolerance.
8. **Tiny end-to-end backward.** A small CPU S2 and S5P model step must give
   finite loss and finite gradients in both the shared encoder and the
   corresponding classification head.
9. **Objective/resume signature.** Changing objective version, weight,
   temperature, or pair scope must reject resume. An old BCE-only checkpoint
   must not be resumed as if it were the RankNet objective.
10. **Matched-comparator audit.** A synthetic comparator must reject changed
    manifests, normalization, protocol fingerprint, encoder signature,
    initial head hash, seed, round count, or optimizer-step count, while
    allowing only the declared loss-objective difference.
11. **Metric recomputation.** Saved validation probabilities and labels must
    reproduce per-sensor and macro AP, AUROC, best F1, and F1@0.5.
12. **Current-only inference ablation.** The unchanged model must support a
    finite validation forward using only the current stream, so temporal-use
    attribution can be audited without training another model.

All tests must pass before a bounded GPU smoke test. A GPU smoke may check only
finite execution and logging; it must not be used to tune \(\lambda\),
\(\tau\), batch composition, or any other setting.

## Promotion and hard-stop gate

All protocol and integrity checks must pass. Relative to the matched
one-epoch BCE scratch comparator, the RankNet run must satisfy every criterion:

1. macro AP delta is at least `+0.010`;
2. at least three of four per-sensor AP deltas are non-negative;
3. the worst per-sensor AP delta is at least `-0.010`;
4. macro best-F1 delta is at least `-0.005`;
5. macro F1@0.5 delta is at least `-0.010`;
6. at least 99% of training batches contain a valid positive-negative pair;
7. full temporal input exceeds the same checkpoint's current-only inference
   ablation by at least `+0.005` macro AP, with a positive AP contribution on
   at least two sensors, before the result may be called transient-aware.

S5P promotion must be supported by AP/AUROC and non-degenerate probability
behavior; an all-positive best-F1 operating point is not evidence of useful
classification.

Failure of any criterion gives:

```text
status = fail
decision = do_not_promote
```

After failure:

- do not tune the ranking weight or temperature;
- do not add epochs;
- do not add seeds;
- do not introduce hard-negative mining;
- do not add another pretraining or contrastive objective;
- record the negative result and stop this mainline.

Passing this one-epoch gate permits a later multi-seed confirmation. Passing
does not retrospectively turn the method into self-supervised pretraining, and
it does not by itself justify a paper-level claim.
