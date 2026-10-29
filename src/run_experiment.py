"""Run the dropout experiment: baseline vs Random Forest.

Usage:  python -m src.run_experiment --data data/student_engagement.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

from src.evaluation import (cross_validate_model, error_analysis, evaluate,
                            plot_confusion_matrix, to_markdown)
from src.model import build_pipeline, get_models
from src.preprocessing import (RANDOM_STATE, TARGET, audit_data, load_data, prepare_xy,
                               split_data)


def run(data, target=TARGET, out="reports", model_dir="models", seed=RANDOM_STATE,
        include_leakage_suspects=False, threshold=0.5) -> pd.DataFrame:
    out, model_dir = Path(out), Path(model_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data)
    (out / "data_audit.json").write_text(json.dumps(audit_data(df, target), indent=2))
    X, y = prepare_xy(df, target, include_leakage_suspects)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)

    rows, fitted, preds = {}, {}, {}
    for name, est in get_models(seed).items():
        pipe = build_pipeline(X_train, est)
        cv = cross_validate_model(pipe, X_train, y_train, seed)
        pipe.fit(X_train, y_train)
        metrics, y_pred = evaluate(pipe, X_test, y_test, threshold)
        rows[name] = {**cv, **metrics}
        fitted[name], preds[name] = pipe, y_pred
        slug = name.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
        plot_confusion_matrix(y_test, y_pred, name, out / "figures" / f"cm_{slug}.png")
        (out / f"classification_report_{slug}.txt").write_text(
            classification_report(y_test, y_pred, target_names=["Stayed", "Dropped out"],
                                  zero_division=0))

    results = pd.DataFrame(rows).T
    results.to_csv(out / "model_comparison.csv")

    # Model selection uses training-set CV F1 only (never the test set).
    candidates = results.drop(index=[n for n in results.index if "floor" in n])
    best = candidates["CV f1 (mean)"].idxmax()
    joblib.dump(fitted[best], model_dir / "best_model.joblib")

    error_analysis(X_test, y_test, preds[best]).to_csv(out / "error_analysis.csv")
    pi = permutation_importance(fitted[best], X_test, y_test, scoring="f1",
                                n_repeats=5, random_state=seed)
    pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False) \
        .to_csv(out / "permutation_importance.csv", header=["importance"])

    shown = ["CV f1 (mean)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
             "TN", "FP", "FN", "TP"]
    note = ("with" if include_leakage_suspects else "without")
    md = (f"# Model comparison (test set, threshold={threshold}, seed={seed}, "
          f"{note} leakage-suspect columns)\n\n"
          f"Train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"positive rate (test): {y_test.mean():.3f}\n\n"
          f"{to_markdown(results[shown])}\n\nSelected model (by training-CV F1): **{best}**\n")
    (out / "results.md").write_text(md)
    print(md)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/student_engagement.csv")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--out", default="reports")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--include-leakage-suspects", action="store_true",
                   help="Keep risk_level as a feature (for a leakage sensitivity check).")
    a = p.parse_args()
    run(a.data, a.target, a.out, "models", a.seed, a.include_leakage_suspects, a.threshold)
