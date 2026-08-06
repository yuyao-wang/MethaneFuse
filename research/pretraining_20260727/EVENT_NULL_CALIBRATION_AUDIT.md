# Sidecar-RCTP v2: train-only event-null calibration audit

Date: 2026-07-28 UTC

Status: **complete, exploratory, no-go**

Decision: **the tested train-only event-null calibrators do not rescue P5**

## Bottom line

P5's positive dev ranking signal remains real within this frozen exploratory
run (`event-balanced AP=0.7651`, versus P4 `0.7544` and P0 `0.7491`), but the
tested train-only availability/acquisition null corrections did not convert it
into a cleaner decision rule. The calibration candidate selected without
looking at dev labels was **identity for all three arms**. With thresholds also
locked from train out-of-fold predictions, P5 had:

- event-balanced macro F1 `0.7560`, versus best control P4 `0.7571`
  (`delta=-0.0011`, event-cluster 95% CI `[-0.0117, +0.0092]`);
- all-negative-event FP mass `2.7922`, versus P4 `2.3188`;
- no `+0.02` macro-F1 margin, no positive lower confidence bound, and no
  all-negative-event FP improvement.

The promotion gate therefore fails all three required clauses. This audit does
**not** support a claim that post-hoc event-null calibration solves the current
P5 selectivity problem.

## Scope and leakage guard

Only the following Sidecar-RCTP v2 artifacts were read:

- the P0/P4/P5 `train.pt` and `val.pt` feature caches under
  `/diniuvol/yuyao/methanefuse_research_20260727/rctp_l89_sidecar_fallback_v1/cache`;
- the frozen P0/P4/P5 downstream head checkpoints under
  `downstream_event_balanced_seed20260728`;
- that run's configuration and existing validation predictions for
  consistency checks.

No file whose role is test, sealed, or holdout was read. No original
Sidecar-RCTP artifact was changed. The only repository output from this audit
is this report.

There was no saved train-prediction CSV. This was not a blocker because the
complete train/dev feature caches and frozen downstream checkpoints were
available. Train and dev logits were reconstructed by CPU-only inference; no
features were re-extracted and no model was trained.

Important limitation: the inherited downstream checkpoints had already been
selected by dev event-balanced AP in the preceding exploratory experiment.
This audit did not introduce any further dev-label fitting, but it remains a
post-hoc diagnostic rather than confirmatory evidence.

## Audited data

The three arms have identical row IDs, canonical event IDs, and labels.

| Split | Rows | Canonical events | All-negative events |
|---|---:|---:|---:|
| Train | 10,033 | 723 | 167 |
| Dev | 9,614 | 124 | 27 |

L89 is a single-sensor cohort, so this run cannot test sensor-conditioned
calibration. It can test only acquisition/availability strata. Train contained
five six-visit availability patterns and acquisition quarters Q1--Q4; dev
contained four of those patterns and quarters Q1--Q3.

Representative provenance:

| Artifact | SHA-256 |
|---|---|
| P0 downstream checkpoint | `e5924ecff3db6ed9fe99105d355e082229b3c8d3a15a15da715e2bf0d2a5fe56` |
| P4 downstream checkpoint | `2268de582e181f987470cd720aae77cf009fa08e3c0ff398faafbb526260a3f3` |
| P5 downstream checkpoint | `5a10899243183d64eebb939f50b3dc2b6a54364f781c7691efc91e2a5f46d70c` |
| P0 train cache | `e82dc41976d59b5c5aa660afd5a4aa3828f607e3e882a93fb3d6af42a71e9edf` |
| P0 dev cache | `ee7665fe29557505f4136a52f2caf30ae80224ba5deb605d34218fd840812fd5` |

## Train-only calibration protocol

The protocol was identical for P0, P4, and P5.

1. Canonical events were divided into five deterministic folds
   (`seed=20260728`), stratified by whether an event contained any positive
   row. Rows from one event never crossed folds.
2. For each training fold, a null location was estimated only from
   **all-negative events in the other four folds**. Every event contributed
   equal total weight.
3. Candidate null locations were the 75th or 90th percentile of the negative
   logit distribution, conditioned on:
   - six-visit availability pattern;
   - t0 acquisition quarter;
   - availability pattern x quarter.
4. Each stratum offset was shrunk toward the global null location by
   `n_events / (n_events + 12)`. The offset was subtracted from the frozen
   classifier logit. Identity was an explicit control.
5. The classification threshold was selected only on the resulting train OOF
   probabilities by event-balanced macro F1.
6. A candidate was eligible only if its train-OOF all-negative-event FP mass
   was no greater than identity. Among eligible candidates, selection maximized
   train-OOF macro F1, then preferred lower null FP and the simpler candidate.
7. Dev probabilities were the mean prediction from the five fold calibrators.
   The train-OOF threshold was then applied once. Dev labels were used only to
   report the final exploratory metrics, never to fit an offset, select a
   candidate, or choose a threshold.

The all-negative-event FP mass is the sum, over all-negative canonical events,
of each event's false-positive row fraction. Event-balanced metrics give each
canonical event equal total weight.

## Locked train-only selection result

All three arms selected `identity`. Thus a properly train-selected calibrator
did not justify altering any arm's scores.

| Arm | Selected calibrator | Dev event AP | Dev event AUC | Dev positive F1 | Dev macro F1 | Train-OOF threshold | All-negative FP mass |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | identity | 0.7491 | 0.8428 | 0.6345 | 0.7342 | 0.6951 | 1.8331 |
| P4 | identity | 0.7544 | 0.8424 | 0.6763 | 0.7571 | 0.8532 | 2.3188 |
| P5 | identity | **0.7651** | **0.8484** | 0.6743 | 0.7560 | 0.5475 | 2.7922 |

These F1 values differ from the preceding Sidecar report because that report
selected its threshold on dev. Here the threshold is deliberately locked from
train OOF predictions.

Paired canonical-event bootstrap (`2,000` replicates, seed `20260728`,
threshold fixed):

| Contrast | Macro-F1 point delta | 95% CI |
|---|---:|---:|
| P5 - P0 | +0.0218 | [+0.0056, +0.0403] |
| P5 - P4 | -0.0011 | [-0.0117, +0.0092] |

P5 improves over the weak P0 control, but not over the stronger matched P4
control.

## What the unsuccessful candidates reveal

A fixed `75th-percentile x availability-pattern` correction was the only tested
P5 variant that both lowered dev null FP and raised dev macro F1:

| P5 rule | Dev AP | Dev macro F1 | All-negative FP mass |
|---|---:|---:|---:|
| Identity | 0.7651 | 0.7560 | 2.7922 |
| Null q75 by availability | 0.7638 | 0.7594 | 2.7021 |

This is only a post-hoc observation, not a selectable result: on train OOF it
*increased* null FP mass from `7.369` to `8.244`, so the locked train-only rule
correctly rejected it. Moreover, applying the same fixed correction to P4 gave
macro F1 `0.7618` and FP mass `2.2807`; P5 still did not beat P4 and remained
less selective.

The highest dev macro F1 among the tested P5 corrections was `0.7648` (q75 by
availability x quarter), but its FP mass increased to `3.2091`. This is the
opposite of the intended event-null behavior and cannot be promoted.

## Promotion gate

The predeclared gate is P5 greater than `max(P0,P4)` by at least `0.02`
event-balanced macro F1, paired event-cluster lower CI above zero, and no
all-negative-event FP increase.

| Clause | Result |
|---|---|
| P5 macro F1 at least +0.02 over best control | **Fail** (`-0.0011` vs P4) |
| Paired lower 95% CI above zero | **Fail** (`-0.0117`) |
| No all-negative-event FP increase | **Fail** (`2.7922` vs P4 `2.3188`) |
| Overall gate | **FAIL / NO-GO** |

## Interpretation and next implication

The useful research signal is narrower than “calibration fixes P5.” P5
preserves better correct-response ranking, but simple static offsets based on
visit availability and acquisition quarter cannot distinguish genuine weak
transients from event-level null excursions. The missing capability appears to
be **event-null selectivity learned in the representation/objective**, not a
post-hoc threshold shift.

A next confirmatory framework should therefore combine the frozen semantic
base and response-conditioned residual sidecar with an explicit train-time
event-null constraint (for example, cross-fitted negative-event tail control or
sensor/acquisition-stratified conformal calibration), then evaluate it on the
verified four-sensor recrop. The multi-sensor experiment is necessary before
claiming sensor-conditioned calibration. It must lock the calibrator and
threshold on train/inner-dev, retain P0/P4/P5 matched controls, and touch the
outer test only once after the full `+0.02 / lower-CI / no-FP-increase` gate is
met.
