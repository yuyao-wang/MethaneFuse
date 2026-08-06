# Fresh clean-L89 TEMPO iteration log

Date: 2026-07-28 UTC  
Scope: inner development only; no test, sealed, holdout, or outer artifact read

## Protocol

- Source: `L89_temporal_train_full_event_balanced.csv`.
- Train: 18,682 rows / 672 canonical events.
- Development: 4,677 rows / 176 canonical events.
- Train/development canonical-event overlap: 0.
- Public Panopticon PTH SHA-256:
  `55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e`.
- Fresh train cache SHA-256:
  `8fac069afdc376b3f671b0d3c1f6aea0ad0db68b8ec3cdf0364b09306984842a`.
- Fresh development cache SHA-256:
  `f66e7ecc5f8d92602198a10d4e2ceadfc186c2269bc2b58e2ce6cbf6af82e7dc`.
- Both caches have zero read errors and zero invalid `t0`.
- Metrics below are row predictions weighted by inverse canonical-event size.
  Threshold-dependent metrics use the development event-balanced macro-F1
  maximizer. They are not event-aggregated predictions.

## Fresh P0

P0 is the exact role-attention appearance head above
`[base_768, exact_zero_768]`. It was selected by event-balanced AP only.

| Epoch | AP | AUC | Macro-F1 | Positive-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.715835 | 0.812593 | 0.744208 | 0.662970 |
| **2** | **0.752329** | **0.841389** | **0.765197** | **0.686078** |
| 3 | 0.743421 | 0.819714 | 0.747984 | 0.669588 |

Epoch 3 already regressed, so longer head training is rejected.

## D1 bounded one-seed screen

All runs use the same frozen P0, event-balanced batches, at most four epochs,
and patience one. `shuffle delta` is shuffled-history minus real-history AP;
negative is the desired direction.

| Config | dim | LR | dropout | D1 AP | AUC | Macro-F1 | Positive-F1 | FP mass | shuffle delta AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c0 default | 192 | 8e-4 | 0.15 | 0.742921 | 0.840997 | 0.769276 | 0.702339 | 7.033 | **-0.016110** |
| c1 low LR | 192 | 4e-4 | 0.15 | **0.747942** | 0.843609 | **0.772173** | **0.704725** | 6.583 | +0.008160 |
| c2 high LR | 192 | 1.2e-3 | 0.15 | 0.735211 | 0.839231 | 0.764155 | 0.703446 | 9.008 | +0.003514 |
| c3 small | 128 | 8e-4 | 0.15 | 0.742460 | 0.839395 | 0.768440 | 0.697651 | 6.508 | +0.010607 |
| c4 large | 256 | 8e-4 | 0.15 | 0.738636 | 0.838750 | 0.765762 | 0.698197 | 7.733 | -0.000459 |
| c5 dropout | 192 | 8e-4 | 0.30 | 0.747758 | **0.844076** | 0.770671 | 0.700214 | 6.271 | +0.000813 |

The AP-only screen would select c1. That choice is not promoted: shuffling its
history increases AP, and its fixed P0+D1 fusion AP is only 0.750807. The
mechanism-first c0 was fixed before this screen; it is the only run with a
large branch-isolated AP drop under history shuffle and has the best fixed
fusion AP. This negative result is why no further D1 hyperparameter grid was
opened.

## Fixed one-seed P0 + D1 diagnostic

The predeclared arithmetic is
`0.5*logit(P0) + 0.5*logit(D1)`; the weight was not fit.

| System | AP | AUC | Macro-F1 | Positive-F1 | FP mass |
|---|---:|---:|---:|---:|---:|
| P0 | 0.752329 | 0.841389 | 0.765197 | 0.686078 | 5.196 |
| D1 seed 20260728 | 0.742921 | 0.840997 | 0.769276 | 0.702339 | 7.033 |
| fixed P0+D1 | **0.753066** | **0.844106** | **0.770988** | **0.701214** | 6.408 |

Paired canonical-event bootstrap, 2,000 replicates, point thresholds fixed:

- AP versus P0: `+0.000736`, 95% CI
  `[-0.008718,+0.011110]`.
- AUC versus P0: `+0.002717`, 95% CI
  `[-0.001542,+0.007081]`.
- Macro-F1 versus P0: `+0.005791`, 95% CI
  `[-0.003226,+0.015703]`.
- Positive-F1 versus P0: `+0.015135`, 95% CI
  `[+0.002316,+0.029356]`.
- Negative-event FP mass versus P0: `+1.2125`, 95% CI
  `[+0.500,+2.075]` (worse).

The one-seed result establishes complementarity, but also exposes the false
alarm cost and cannot be promoted by itself.

## Mechanism-first c0, three seeds

The c0 configuration was repeated at seeds 20260727/28/29. D1 point metrics:

| Seed | AP | AUC | Macro-F1 | Positive-F1 | FP mass |
|---:|---:|---:|---:|---:|---:|
| 20260727 | 0.735967 | 0.834654 | 0.767534 | 0.696113 | 6.458 |
| 20260728 | 0.743011 | 0.840985 | 0.770193 | 0.704052 | 7.033 |
| 20260729 | 0.742814 | 0.837188 | 0.765403 | 0.693078 | 6.458 |
| mean | 0.740597 | 0.837609 | 0.767710 | 0.697748 | 6.650 |

The fixed candidate first averages the three D1 logits and then forms
`0.5*logit(P0) + 0.5*mean_seed(logit(D1))`.

| System | AP | AUC | Macro-F1 | Positive-F1 | FP mass |
|---|---:|---:|---:|---:|---:|
| P0 | 0.752329 | 0.841389 | 0.765197 | 0.686078 | 5.196 |
| mean-seed D1 | 0.748303 | 0.841200 | 0.770990 | 0.702631 | 6.621 |
| fixed P0+mean-D1 | **0.752429** | **0.842090** | **0.772695** | **0.699453** | 5.571 |

Paired canonical-event bootstrap, 2,000 replicates, point thresholds fixed:

- AP versus P0: `+0.000100`, 95% CI
  `[-0.004696,+0.004955]`.
- AUC versus P0: `+0.000701`, 95% CI
  `[-0.001426,+0.002914]`.
- Macro-F1 versus P0: **`+0.007498`**, 95% CI
  **`[+0.001151,+0.014078]`**.
- Positive-F1 versus P0: **`+0.013375`**, 95% CI
  **`[+0.004366,+0.023155]`**.
- Negative-event FP mass versus P0: `+0.375`, 95% CI
  `[-0.125,+0.875]`.

Thus the reproducible gain is a classification operating-point gain, not a
ranking/AP gain. The three-seed mean also reduces the false-alarm penalty from
`+1.2125` to `+0.375`.

## Fusion-weight ceiling diagnostic

A post-hoc 21-point curve was run only to test whether equal fusion is
pathological. It is not used for model selection:

- best AP on the same development panel is 0.752849 at D1 weight 0.35;
- best macro-F1 is 0.773772 at D1 weight 0.75;
- fixed 0.5 gives AP 0.752429 and macro-F1 0.772695.

The surface is flat; fitting a liaison or a precise weight is not justified.
The confirmatory candidate therefore remains fixed 0.5/0.5.

A deterministic history-availability gate was also checked: 1,591 rows have
four unique histories and 3,086 have five. Applying fusion only to the partial
rows gives AP/macro-F1 `0.749327/0.766580`; applying it only to full-history
rows gives `0.748845/0.768889`. Both are below all-row fixed fusion
`0.752429/0.772695`, so no L89 availability router is retained.

## False-alarm penalty control

`d3_gated_null_delta` kept the D1 architecture and added all-negative-event
top-k penalties of 0.03, 0.10, and 0.30.

| null weight | D3 AP | Macro-F1 | Positive-F1 | FP mass | fixed P0+D3 AP | fixed P0+D3 Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.745588 | 0.769760 | 0.704542 | 7.521 | 0.751405 | 0.767520 |
| 0.10 | **0.748963** | 0.768720 | 0.703461 | 7.871 | **0.752039** | **0.769913** |
| 0.30 | 0.747619 | 0.765743 | 0.694312 | 6.708 | 0.751083 | 0.767438 |

None beats fixed P0+mean-D1 or reliably fixes false alarms. This branch is
rejected; no wider null-loss grid is warranted.

## Video-inductive-bias matched controls

Four one-seed controls use the same P0, initialization, optimizer, batches,
loss, and early stopping:

| Control | Video analogue | AP | Macro-F1 | FP mass | shuffle delta AP | fixed P0+control AP | fixed P0+control Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 uniform delta | frame differencing | 0.741557 | 0.770273 | 6.208 | +0.009793 | 0.749007 | 0.770677 |
| P2 onset | background-normalized change | **0.747732** | 0.765969 | 8.908 | -0.004186 | **0.752062** | 0.767186 |
| P3 gated onset | change + acquisition gate | 0.742054 | 0.768508 | 8.308 | +0.008156 | 0.749377 | 0.768048 |
| D7 hard SlowFast | fixed recent/seasonal bins | 0.747096 | **0.770714** | **5.508** | +0.003690 | 0.750776 | **0.769517** |

P2 has real history sensitivity but excessive false alarms; hard SlowFast has
the best control false-alarm mass but its history shuffle does not reduce AP
and its fixed fusion is weaker. None reaches fixed P0+mean-D1 macro-F1
0.772695. The useful video borrowing is therefore the *appearance/transient
division of labor*, while EO time is handled by a continuous real-gap gate
rather than fixed frame-rate or hard fast/slow bins.

## Supervised last-block matched control

An exploratory runner is implemented and CPU-tested:

```text
shared frozen Panopticon blocks 0--10
├── current-only block 11 + final norm + exact-zero TEMPO head
└── temporal block 11 + final norm + exact-zero TEMPO-D1 head
```

The two block-11 copies and heads are byte-identical at epoch zero; both arms
share one frozen-trunk pass and one event-by-label row plan. Only the history
mask differs. Blocks 0--10 remain frozen and are executed under `no_grad`,
while the final norm remains differentiable. The runner caps training at three
epochs with patience one and a 28-GiB allocation guard. Seven CPU contract and
optimizer tests pass. A GPU0 run is pending execution permission/driver
visibility; it has not produced a metric yet.

## Current interpretation

The fresh clean replicate supports only this narrow statement:

> Irregular-history D1 is not a stronger standalone ranker than appearance
> P0. Its independently trained errors are complementary enough that fixed
> late-logit consensus gives a reproducible `+0.75` macro-F1 point and
> `+1.34` positive-F1 points on event-disjoint development, while AP remains
> unchanged.

This is consistent with the proposed appearance/transient two-stream story,
but it is not yet a clean SOTA claim. Promotion still requires the frozen
formal replicate and a separately authorized outer/sealed evaluation.
