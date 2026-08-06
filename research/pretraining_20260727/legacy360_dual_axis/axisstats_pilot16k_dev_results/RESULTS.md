# Legacy360 AxisStats dev-only control

This is a strong low-dimensional engineering control, not RCTP novelty and not a sealed-test result.

- Train cache rows: 15,973 (`train_core`)
- Canonical dev rows: 12,621 (`dev`)
- Test/sealed inputs: **not accepted and not read**
- Selected model: `histgb_base_lr0p05_leaf7`
- Selected dev threshold: `0.50448129`

| Candidate | Family | Features | Iter | F1@0.5 | Dev-best F1 | Threshold | AP | AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| reference_universal_base | reference | fused_logit | 0 | 0.886986 | 0.897926 | 0.362209 | 0.954828 | 0.954632 |
| reference_hybrid_base | reference | fused_logit | 0 | 0.881834 | 0.890312 | 0.369221 | 0.949749 | 0.949890 |
| logreg_base_c0p03 | logistic_regression | base | 24 | 0.891898 | 0.893066 | 0.490222 | 0.945592 | 0.950122 |
| logreg_base_c0p3 | logistic_regression | base | 37 | 0.891111 | 0.891790 | 0.475436 | 0.936696 | 0.945482 |
| logreg_base_c3 | logistic_regression | base | 46 | 0.889635 | 0.890014 | 0.523073 | 0.927389 | 0.936992 |
| histgb_base_lr0p05_leaf7 | hist_gradient_boosting | base | 100 | 0.893766 | 0.894013 | 0.504481 | 0.948773 | 0.952195 |
| histgb_base_lr0p05_leaf15 | hist_gradient_boosting | base | 100 | 0.881839 | 0.889538 | 0.505062 | 0.941950 | 0.948604 |
| histgb_base_lr0p1_leaf15 | hist_gradient_boosting | base | 81 | 0.889097 | 0.890945 | 0.557087 | 0.941929 | 0.948805 |
| logreg_full_c0p03 | logistic_regression | full | 35 | 0.863498 | 0.865584 | 0.464699 | 0.917072 | 0.926313 |
| logreg_full_c0p3 | logistic_regression | full | 72 | 0.860285 | 0.862139 | 0.398704 | 0.858926 | 0.882488 |
| logreg_full_c3 | logistic_regression | full | 100 | 0.858382 | 0.859117 | 0.373422 | 0.829196 | 0.859989 |
| histgb_full_lr0p05_leaf7 | hist_gradient_boosting | full | 100 | 0.890491 | 0.892545 | 0.458675 | 0.952671 | 0.953330 |
| histgb_full_lr0p05_leaf15 | hist_gradient_boosting | full | 100 | 0.873524 | 0.887856 | 0.535592 | 0.944504 | 0.948208 |
| histgb_full_lr0p1_leaf15 | hist_gradient_boosting | full | 100 | 0.883057 | 0.883188 | 0.500978 | 0.941263 | 0.944916 |

## Locked winner by sensor-containing stratum

| Sensor | Rows | F1 | Macro F1 | AP | AUC | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| s2 | 6057 | 0.926821 | 0.924477 | 0.964935 | 0.973940 | 0.523692 |
| l89 | 4068 | 0.900142 | 0.896107 | 0.950507 | 0.953948 | 0.523845 |
| emit | 3644 | 0.873170 | 0.837778 | 0.945045 | 0.929698 | 0.634193 |
| s5p | 1558 | 0.857942 | 0.833338 | 0.938520 | 0.928452 | 0.588575 |

Selection and threshold optimization both used canonical dev. The lock is suitable for a later authorized one-shot evaluation, but this script intentionally provides no test command.
