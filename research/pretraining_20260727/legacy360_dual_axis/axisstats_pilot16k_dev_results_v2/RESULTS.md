# Legacy360 AxisStats dev-only control

This is a strong low-dimensional engineering control, not RCTP novelty and not a sealed-test result.

- Train cache rows: 15,973 (`train_core`)
- Canonical dev rows: 12,621 (`dev`)
- Test/sealed inputs: **not accepted and not read**
- Selected model: `logreg_universal_fused_c0p03`
- Selected dev threshold: `0.48300340`
- Strongest direct-PTH reference: `reference_universal_base` at dev-best F1 `0.897926`
- Fitted-control delta vs strongest reference: `+0.000000`

| Candidate | Family | Features | Iter | F1@0.5 | Dev-best F1 | Threshold | AP | AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| reference_universal_base | reference | fused_logit | 0 | 0.886986 | 0.897926 | 0.362209 | 0.954828 | 0.954632 |
| reference_hybrid_base | reference | fused_logit | 0 | 0.881834 | 0.890312 | 0.369221 | 0.949749 | 0.949890 |
| logreg_universal_fused_c0p03 | logistic_regression | universal_fused | 7 | 0.897007 | 0.897926 | 0.483003 | 0.954828 | 0.954632 |
| logreg_fused_c0p03 | logistic_regression | fused | 7 | 0.897261 | 0.897920 | 0.477569 | 0.954983 | 0.954853 |
| logreg_fused_c0p3 | logistic_regression | fused | 7 | 0.897261 | 0.897915 | 0.472981 | 0.954980 | 0.954852 |
| logreg_logits_c0p03 | logistic_regression | logits | 14 | 0.895675 | 0.896696 | 0.452344 | 0.949991 | 0.951844 |
| logreg_logits_c0p3 | logistic_regression | logits | 22 | 0.894660 | 0.896660 | 0.418762 | 0.950436 | 0.951929 |
| logreg_base_c0p03 | logistic_regression | base | 24 | 0.891898 | 0.893066 | 0.490222 | 0.945592 | 0.950122 |
| logreg_base_c0p3 | logistic_regression | base | 37 | 0.891111 | 0.891790 | 0.475436 | 0.936696 | 0.945482 |
| logreg_full_c0p03 | logistic_regression | full | 35 | 0.863498 | 0.865584 | 0.464699 | 0.917072 | 0.926313 |
| histgb_logits_lr0p05_leaf7 | hist_gradient_boosting | logits | 100 | 0.893605 | 0.894268 | 0.552753 | 0.948457 | 0.951980 |
| histgb_base_lr0p05_leaf7 | hist_gradient_boosting | base | 100 | 0.893766 | 0.894013 | 0.504481 | 0.948773 | 0.952195 |
| histgb_base_lr0p1_leaf15 | hist_gradient_boosting | base | 81 | 0.889097 | 0.890945 | 0.557087 | 0.941929 | 0.948805 |
| histgb_full_lr0p05_leaf7 | hist_gradient_boosting | full | 100 | 0.890491 | 0.892545 | 0.458675 | 0.952671 | 0.953330 |
| histgb_full_lr0p1_leaf15 | hist_gradient_boosting | full | 100 | 0.883057 | 0.883188 | 0.500978 | 0.941263 | 0.944916 |

## Locked winner by sensor-containing stratum

| Sensor | Rows | F1 | Macro F1 | AP | AUC | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| s2 | 6057 | 0.931865 | 0.928339 | 0.969446 | 0.975286 | 0.541852 |
| l89 | 4068 | 0.901488 | 0.897324 | 0.951370 | 0.953534 | 0.525565 |
| emit | 3644 | 0.876812 | 0.843698 | 0.948500 | 0.932550 | 0.627881 |
| s5p | 1558 | 0.875773 | 0.855239 | 0.946395 | 0.936552 | 0.582798 |

Selection and threshold optimization both used canonical dev. The lock is suitable for a later authorized one-shot evaluation, but this script intentionally provides no test command.
