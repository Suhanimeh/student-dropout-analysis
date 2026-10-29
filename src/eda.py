"""Exploratory data analysis and leakage checks.

Usage:  python -m src.eda --data data/student_engagement.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (ID_COLUMNS, LEAKAGE_SUSPECTS, TARGET, audit_data,
                               encode_target, load_data)


def single_feature_auc(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """AUC of each numeric feature used alone. Values near 1.0 suggest leakage."""
    y = encode_target(df[target])
    rows = []
    for c in df.select_dtypes(include="number").columns:
        if c == target or c in ID_COLUMNS:
            continue
        x = df[c].fillna(df[c].median())
        auc = roc_auc_score(y, x)
        rows.append({"feature": c, "auc": auc, "auc_direction_free": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc_direction_free", ascending=False)


def derived_label_check(df: pd.DataFrame, col: str) -> float | None:
    """How well do the other numeric features reproduce `col` using a depth-3 tree?
    Accuracy near 1.0 means `col` is a formula of the features, not new information."""
    others = [c for c in df.select_dtypes(include="number").columns
              if c not in (TARGET, col) and c not in ID_COLUMNS]
    if col not in df.columns or not others:
        return None
    X = df[others].fillna(df[others].median())
    return float(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42),
                                 X, df[col].astype(str), cv=5).mean())


def run_eda(df: pd.DataFrame, target: str, out_dir: Path) -> dict:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates()
    numeric = [c for c in df.select_dtypes(include="number").columns
               if c != target and c not in ID_COLUMNS]
    summary = {"audit": audit_data(df, target)}

    fig, ax = plt.subplots(figsize=(4, 3.5))
    df[target].value_counts().sort_index().plot.bar(ax=ax, color=["#4c78a8", "#e45756"])
    ax.set_title("Target distribution"); ax.set_xlabel(target); fig.tight_layout()
    fig.savefig(fig_dir / "target_distribution.png", dpi=150); plt.close(fig)

    ncols = 2
    nrows = -(-len(numeric) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows))
    for ax, c in zip(axes.ravel(), numeric):
        sns.histplot(data=df, x=c, hue=target, stat="density", common_norm=False,
                     element="step", ax=ax)
        ax.set_title(f"{c} by {target}")
    for ax in axes.ravel()[len(numeric):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(fig_dir / "feature_distributions_by_target.png", dpi=150)
    plt.close(fig)

    corr = df[numeric + [target]].assign(**{target: encode_target(df[target])}).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix"); fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=150); plt.close(fig)

    summary["single_feature_auc"] = single_feature_auc(df, target).round(4).to_dict("records")
    for col in LEAKAGE_SUSPECTS:
        if col in df.columns:
            ct = pd.crosstab(df[col], df[target], normalize="index").round(4)
            summary[f"{col}_vs_{target}_rate"] = ct.to_dict()
            summary[f"{col}_reproducible_from_features_cv_accuracy"] = derived_label_check(df, col)

    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/student_engagement.csv")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    s = run_eda(load_data(a.data), a.target, Path(a.out))
    print(json.dumps(s["audit"], indent=2))
    print("Wrote figures and eda_summary.json to", a.out)
