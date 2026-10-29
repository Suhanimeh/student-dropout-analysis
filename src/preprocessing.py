"""Data loading, auditing and preprocessing for the dropout experiment.

Leakage controls:
  * identifier columns are dropped;
  * columns that may be derived from the outcome (LEAKAGE_SUSPECTS) are dropped
    unless explicitly requested;
  * imputation, scaling and encoding live inside an sklearn Pipeline, so they are
    fitted on training data only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "dropped_out"
ID_COLUMNS = ["student_id"]
# risk_level looks like a derived score/label (not a raw measurement). It is
# excluded by default until its derivation is confirmed. See README.
LEAKAGE_SUSPECTS = ["risk_level"]
RANDOM_STATE = 42


def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV there or pass --data <path>."
        )
    return pd.read_csv(path)


def encode_target(y: pd.Series) -> pd.Series:
    """Return a 0/1 integer target from 0/1, bool or yes/no/true/false values."""
    if not (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)):
        mapping = {"yes": 1, "true": 1, "1": 1, "dropout": 1, "no": 0, "false": 0, "0": 0}
        mapped = y.astype(str).str.strip().str.lower().map(mapping)
        if mapped.isna().any():
            raise ValueError(f"Unrecognised target values: {sorted(y[mapped.isna()].unique())}")
        return mapped.astype(int)
    return y.astype(int)


def audit_data(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Basic data-quality summary: types, missing values, duplicates, class balance."""
    missing = df.isna().sum()
    counts = df[target].value_counts(dropna=False)
    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in missing.items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()) if id_col else None,
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "target_positive_rate": float(encode_target(df[target].dropna()).mean()),
    }


def prepare_xy(df: pd.DataFrame, target: str = TARGET, include_leakage_suspects: bool = False):
    """Drop duplicates, rows with missing target, IDs and (by default) suspect columns."""
    df = df.drop_duplicates().dropna(subset=[target])
    drop = [target] + [c for c in ID_COLUMNS if c in df.columns]
    if not include_leakage_suspects:
        drop += [c for c in LEAKAGE_SUSPECTS if c in df.columns]
    return df.drop(columns=drop), encode_target(df[target])


def split_data(X, y, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Reproducible stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
