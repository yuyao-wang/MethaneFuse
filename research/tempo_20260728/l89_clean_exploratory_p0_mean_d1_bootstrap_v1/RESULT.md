# Fresh clean-L89 P0 + three-seed mean-D1 audit

**Exploratory inner development only; not a test or SOTA claim.**

The D1 configuration was the mechanism-first default fixed before the
CPU hyperparameter screen. D1 seeds are averaged in logit space, then
combined with P0 using the fixed formula
`0.5*logit(P0) + 0.5*mean_seed(logit(D1))`.

| System | Event-balanced AP | AUC | Macro-F1 | Positive F1 | Negative-event FP mass |
|---|---:|---:|---:|---:|---:|
| p0 | 0.752329 | 0.841389 | 0.765197 | 0.686078 | 5.195833 |
| mean_seed_d1 | 0.748303 | 0.841200 | 0.770990 | 0.702631 | 6.620833 |
| fixed_p0_mean_d1 | 0.752429 | 0.842090 | 0.772695 | 0.699453 | 5.570833 |

Paired canonical-event-cluster bootstrap (2,000 replicates; point
thresholds fixed):

| Comparison | Metric | Delta | 95% CI | Win probability |
|---|---|---:|---:|---:|
| fusion_minus_p0 | event_balanced_ap | +0.000100 | [-0.004696, +0.004955] | 0.5055 |
| fusion_minus_p0 | event_balanced_auc | +0.000701 | [-0.001426, +0.002914] | 0.7285 |
| fusion_minus_p0 | event_balanced_positive_f1_selected | +0.013375 | [+0.004366, +0.023155] | 0.9985 |
| fusion_minus_p0 | event_balanced_macro_f1_selected | +0.007498 | [+0.001151, +0.014078] | 0.9900 |
| fusion_minus_p0 | all_negative_fp_mass | +0.375000 | [-0.125000, +0.875000] | 0.0575 |
| fusion_minus_mean_d1 | event_balanced_ap | +0.004126 | [-0.001202, +0.009023] | 0.9410 |
| fusion_minus_mean_d1 | event_balanced_auc | +0.000890 | [-0.001233, +0.002866] | 0.8165 |
| fusion_minus_mean_d1 | event_balanced_positive_f1_selected | -0.003177 | [-0.010959, +0.004252] | 0.2060 |
| fusion_minus_mean_d1 | event_balanced_macro_f1_selected | +0.001706 | [-0.004269, +0.007412] | 0.7175 |
| fusion_minus_mean_d1 | all_negative_fp_mass | -1.050000 | [-1.725000, -0.449688] | 1.0000 |

Guardrail: the three-seed mean reduces initialization noise, but all
three seeds still share one inner-development split. Promotion
requires the frozen formal replicate and a separately authorized
outer/sealed evaluation.
