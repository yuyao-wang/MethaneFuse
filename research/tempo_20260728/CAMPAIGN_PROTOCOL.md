# TEMPO 12-hour experiment protocol

Started: 2026-07-28 UTC

## Question

Can a small supervised video-style transient module improve methane
classification without foundation-model pretraining, while retaining the
appearance competence of the existing Panopticon checkpoints?

TEMPO is evaluated as a zero-initialized residual on top of a frozen appearance
baseline:

`final_logit = frozen_base_logit + alpha * transient_logit`, with `alpha = 0`
at initialization.

The transient path must distinguish current-to-history change from ordinary
history-to-history variation, account for irregular time gaps and missing
observations, and fuse sensors only after each sensor has produced an evidence
token.

## Data-use boundary

- Model and threshold selection use train and event-disjoint development data
  only.
- Paths containing `test`, `sealed`, or `holdout` are rejected by experiment
  runners.
- No test result is reopened during this campaign.
- The old legacy360 development result is an engineering guardrail, not a
  leakage-free SOTA claim, because its warm-start checkpoint saw the old source
  pool that contains that development subset.

## Locked references

- L89 six-time, event-disjoint validation:
  - role-only: macro-F1@0.5 0.7380378, AP 0.7651179, AUC 0.8092584.
  - delta-time: macro-F1@0.5 0.7166529, AP 0.7621929, AUC 0.7976557.
- Legacy360 four-sensor engineering development:
  - promoted scale-aware TwoAxis best binary F1 0.9055816.
  - AP 0.9600343, AUC 0.9603806.
  - locked threshold 0.3910266161.
- Historical locked test values are context only and will not be read for
  selection: overall F1 0.86485 and S2-containing F1 0.90616.

## Staged ablation

All comparisons use the same cache, seed, initialization, batches, optimizer
updates, and frozen base.

1. P0: exact frozen baseline / zero residual.
2. P1: current-minus-history deltas.
3. P2: onset excess = current-history change minus history-history normality.
4. P3: P2 plus time-gap, validity, and quality gates.
5. P4: P3 plus event-balanced / all-negative-event null loss.
6. P5: independent per-sensor evidence tokens followed by masked evidence
   fusion (legacy360).
7. Patch stage: local historical matching and top-k MIL, only if a global arm
   passes the gate.

## Promotion gate

Initial one-seed screens run for at most 2--3 epochs. An arm is stopped if its
validation AP/F1 has clearly peaked or deteriorated. Survivors are repeated
with three seeds.

An arm may be promoted only if:

- it does not reduce the matched baseline's primary F1;
- AP and AUC do not show a material regression;
- all-negative canonical-event false positives do not increase materially;
- history shuffling reduces its performance, showing that the history is used;
- gains are not produced solely by post-hoc threshold search.

The preferred claim requires a positive canonical-event clustered confidence
interval versus the matched baseline. If that does not hold, the result is
reported as a negative experiment rather than a new method.

## Resource contract

- Physical GPU0: TEMPO experiments.
- Physical GPU1: reserved for the independently running S2 reproduction until
  explicit release.
- Feature caches and outputs live under `/diniuvol/yuyao`.
- Global-head screens precede any expensive patch extraction.
- Every long run must emit a status file and metrics by epoch; dead or
  overfitting runs are stopped rather than allowed to consume the full budget.
