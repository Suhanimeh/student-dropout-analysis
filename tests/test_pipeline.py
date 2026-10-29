import numpy as np
import pandas as pd

from src.evaluation import evaluate, error_analysis
from src.model import build_pipeline, get_models
from src.preprocessing import (LEAKAGE_SUSPECTS, audit_data, encode_target, prepare_xy,
                               split_data)


def test_audit_reports_missing_and_target_rate(synthetic_df):
    a = audit_data(synthetic_df, "dropped_out")
    assert a["missing_values"]["avg_quiz_score"] == 11
    assert 0 < a["target_positive_rate"] < 1


def test_ids_target_and_leakage_suspects_removed(synthetic_df):
    X, y = prepare_xy(synthetic_df)
    for col in ["student_id", "dropped_out", *LEAKAGE_SUSPECTS]:
        assert col not in X.columns
    assert len(X) == len(y)


def test_leakage_suspects_can_be_kept_explicitly(synthetic_df):
    X, _ = prepare_xy(synthetic_df, include_leakage_suspects=True)
    assert "risk_level" in X.columns


def test_split_is_reproducible_and_stratified(synthetic_df):
    X, y = prepare_xy(synthetic_df)
    a = split_data(X, y, seed=1)
    b = split_data(X, y, seed=1)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.01


def test_scaler_is_fitted_on_training_data_only(synthetic_df):
    X, y = prepare_xy(synthetic_df)
    X_tr, X_te, y_tr, _ = split_data(X, y)
    pipe = build_pipeline(X_tr, get_models()["Logistic Regression (baseline)"]).fit(X_tr, y_tr)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    cols = X_tr.select_dtypes("number").columns
    assert np.allclose(scaler.mean_, X_tr[cols].mean(), atol=0.2)  # median-imputed -> approx
    assert not np.allclose(scaler.mean_, X[cols].mean(), atol=1e-12)


def test_all_models_train_and_return_metrics(synthetic_df):
    X, y = prepare_xy(synthetic_df)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    for name, est in get_models().items():
        pipe = build_pipeline(X_tr, est).fit(X_tr, y_tr)
        m, pred = evaluate(pipe, X_te, y_te)
        assert {"Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"} <= m.keys()
        assert m["TN"] + m["FP"] + m["FN"] + m["TP"] == len(y_te)
        ea = error_analysis(X_te, y_te, pred)
        assert list(ea.index) == ["TP", "FP", "FN", "TN"]


def test_target_encoding_handles_strings():
    assert encode_target(pd.Series(["Yes", "no", "TRUE"])).tolist() == [1, 0, 1]
