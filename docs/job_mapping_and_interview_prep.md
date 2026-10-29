# Mapping to the Applied AI / Predictive Maintenance Research Assistant role
Student dropout is **not** predictive maintenance. The project demonstrates transferable research skills on a comparable supervised-classification problem (rare failure/dropout event, imbalanced classes, costly missed detections).

| Requirement | Evidence in this project |
|---|---|
| AI methodology selection | Documented choice of baseline + Random Forest + majority floor; metrics chosen for imbalance |
| Data preparation | Audit, de-duplication, imputation, encoding, scaling inside a Pipeline; leakage decisions |
| Model training and testing | Stratified reproducible split; training-only CV; single final test evaluation |
| Model benchmarking | `model_comparison.csv` / `results.md` with a shared metric set |
| Research analysis | EDA, single-feature AUC, error analysis, permutation importance, leakage sensitivity run |
| Technical documentation | README, report outline, tests, reproduction commands |
| Planning and communication | Staged workflow (audit, EDA, modelling, reporting); report and presentation outlines |

Predictive-maintenance parallel: machine failure is also rare and costly to miss, so the same recall/precision trade-off, imbalance handling and leakage vigilance (e.g. features recorded after failure) apply.

# Likely interview questions
1. **Why not just report accuracy?** With about 15% positives, predicting "stayed" for everyone gives about 85% accuracy and catches no one. Precision, recall, F1 and PR-AUC show whether at-risk cases are found.
2. **What is data leakage and where did you look for it?** Features that contain outcome information unavailable at prediction time. I dropped IDs, excluded the derived `risk_level` by default, checked single-feature AUCs, and ran a with/without comparison.
3. **Why a pipeline?** Fitting scalers/encoders before splitting lets test information leak into training; a Pipeline refits them on training folds only.
4. **How did you choose between models?** Cross-validated F1 on the training set; the test set was used once for reporting.
5. **How would you improve it?** Tune the threshold on validation data, calibrate probabilities, use repeated CV, examine subgroup errors, and use time-based splits if timestamps exist.
