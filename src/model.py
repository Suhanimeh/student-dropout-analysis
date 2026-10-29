"""Model definitions: a trivial floor, a baseline, and one additional model."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import RANDOM_STATE, build_preprocessor


def get_models(seed: int = RANDOM_STATE) -> dict:
    return {
        # Sanity floor: always predicts the majority class.
        "Majority-class (floor)": DummyClassifier(strategy="most_frequent"),
        # Baseline.
        "Logistic Regression (baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed),
        # Additional model.
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
    }


def build_pipeline(X_train, estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
