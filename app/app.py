"""Optional demo: score a CSV with the model saved by src.run_experiment.
Run from the repo root:  streamlit run app/app.py   (requires requirements-app.txt)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
DROP = ["student_id", "dropped_out", "risk_level"]  # same exclusions as training

st.title("Student dropout risk scoring (demo)")
st.caption("Academic demo only; not validated for real decisions.")

file = st.file_uploader("Upload CSV with the same feature columns as the training data")
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run: python -m src.run_experiment")
    elif st.button("Score students"):
        model = joblib.load(MODEL_PATH)
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        df["dropout_risk"] = model.predict_proba(X)[:, 1]
        st.dataframe(df.sort_values("dropout_risk", ascending=False))
