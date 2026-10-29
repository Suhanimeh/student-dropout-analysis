# Academic report outline (6-8 pages)
1. **Abstract** – question, data, methods, headline result, main limitation.
2. **Introduction** – why early identification of at-risk students matters; research question.
3. **Data** – source, size, variables, target definition, class balance, audit results (missing, duplicates).
4. **Methods** – leakage handling (IDs, `risk_level`), preprocessing pipeline, split, models, CV, metrics and why.
5. **Results** – model-comparison table (from `reports/results.md`), confusion matrices, PR-AUC, permutation importance.
6. **Error analysis** – who the FN/FP are (`error_analysis.csv`); cost of each error type.
7. **Leakage sensitivity** – results with vs without `risk_level`.
8. **Discussion and limitations** – feature timing, imbalance, single split, fairness, generalisation.
9. **Conclusion and future work.**
10. **References** – cite only sources actually used (dataset, scikit-learn documentation).

# Presentation outline (8 slides, ~8 minutes)
1. Title and question  2. Data and target  3. Data audit and leakage decision  4. Method (pipeline, split, models)
5. Metrics: why not accuracy  6. Results table and confusion matrix  7. Error analysis and limitations  8. Conclusion and next steps
