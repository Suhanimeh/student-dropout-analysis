"""Evaluation helpers: metrics, cross-validation, confusion matrix, error analysis."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.exceptions import UndefinedMetricWarning

from src.preprocessing import RANDOM_STATE

# The majority-class floor never predicts a positive; precision is then undefined (set to 0).
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def predict_scores(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def evaluate(pipe, X_test, y_test, threshold: float = 0.5) -> dict:
    scores = predict_scores(pipe, X_test)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": average_precision_score(y_test, scores),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }, y_pred


def cross_validate_model(pipe, X_train, y_train, seed: int = RANDOM_STATE, folds: int = 5) -> dict:
    """Stratified CV on the TRAINING set only (used for model selection)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring={"f1": "f1", "recall": "recall", "precision": "precision",
                                  "pr_auc": "average_precision"})
    return {f"CV {k} (mean)": out[f"test_{k}"].mean() for k in ("f1", "recall", "precision", "pr_auc")} | \
           {"CV F1 (std)": out["test_f1"].std()}


def plot_confusion_matrix(y_true, y_pred, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stayed", "Dropped out"], yticklabels=["Stayed", "Dropped out"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def error_analysis(X_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    """Mean numeric feature values for TP / FP / FN / TN groups."""
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    outcome = pd.Series("TN", index=y_test.index)
    outcome[(y_test == 1) & (y_pred == 1)] = "TP"
    outcome[(y_test == 0) & (y_pred == 1)] = "FP"
    outcome[(y_test == 1) & (y_pred == 0)] = "FN"
    num = X_test.reset_index(drop=True).select_dtypes(include="number")
    summary = num.groupby(outcome).mean()
    summary.insert(0, "count", outcome.value_counts())
    return summary.reindex(["TP", "FP", "FN", "TN"])


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(["Model"] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(int(v)) if c in {"TN", "FP", "FN", "TP"} else
                 (floatfmt.format(v) if isinstance(v, float) else str(v))
                 for c, v in row.items()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)
