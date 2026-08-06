# Fresh clean-L89 exploratory P0 + D1 audit

This is a one-seed, inner-development-only diagnostic. The fusion is
predeclared as `0.5*logit(P0) + 0.5*logit(D1)`; no fusion weight was
searched.

| System | Event-balanced AP | AUC | Macro-F1 | Positive F1 | Negative-event FP mass |
|---|---:|---:|---:|---:|---:|
| p0 | 0.752329 | 0.841389 | 0.765197 | 0.686078 | 5.195833 |
| d1 | 0.742921 | 0.840997 | 0.769276 | 0.702339 | 7.033333 |
| fixed_p0_d1 | 0.753066 | 0.844106 | 0.770988 | 0.701214 | 6.408333 |

Paired canonical-event-cluster bootstrap (2,000 replicates; each
system's point threshold is fixed across replicates):

| Comparison | Metric | Delta | 95% CI | Win probability |
|---|---|---:|---:|---:|
| fusion_minus_p0 | event_balanced_ap | +0.000736 | [-0.008718, +0.011110] | 0.5545 |
| fusion_minus_p0 | event_balanced_auc | +0.002717 | [-0.001542, +0.007081] | 0.9000 |
| fusion_minus_p0 | event_balanced_positive_f1_selected | +0.015135 | [+0.002316, +0.029356] | 0.9920 |
| fusion_minus_p0 | event_balanced_macro_f1_selected | +0.005791 | [-0.003226, +0.015703] | 0.8915 |
| fusion_minus_p0 | all_negative_fp_mass | +1.212500 | [+0.500000, +2.075000] | 0.0000 |
| fusion_minus_d1 | event_balanced_ap | +0.010144 | [+0.000351, +0.019614] | 0.9795 |
| fusion_minus_d1 | event_balanced_auc | +0.003109 | [-0.000939, +0.006892] | 0.9310 |
| fusion_minus_d1 | event_balanced_positive_f1_selected | -0.001126 | [-0.009311, +0.006590] | 0.3940 |
| fusion_minus_d1 | event_balanced_macro_f1_selected | +0.001712 | [-0.004263, +0.007431] | 0.7070 |
| fusion_minus_d1 | all_negative_fp_mass | -0.625000 | [-1.125000, -0.125000] | 0.9955 |

Interpretation guardrail: this result can motivate the architecture
and the preregistered multi-seed/formal replicate, but it cannot be
used as a test-set or SOTA claim.
