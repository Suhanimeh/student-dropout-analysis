import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df():
    """SYNTHETIC data used only to test the code. Never used for reported results."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "student_id": np.arange(n),
        "attendance_rate": rng.uniform(0.3, 1.0, n),
        "avg_quiz_score": rng.uniform(30, 100, n),
        "assignment_completion": rng.uniform(0, 1, n),
        "engagement_score": rng.exponential(0.3, n),
    })
    logit = -2.0 - 2.0 * (df.attendance_rate - 0.65) - 0.02 * (df.avg_quiz_score - 65)
    df["dropped_out"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["risk_level"] = pd.cut(df.engagement_score, [-1, 0.2, 0.5, 99], labels=["High", "Medium", "Low"])
    df.loc[5:15, "avg_quiz_score"] = np.nan
    return df
