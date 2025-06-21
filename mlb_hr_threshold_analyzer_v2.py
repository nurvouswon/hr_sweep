import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# --- Upload CSVs ---
st.markdown("#### Upload Training Event-Level CSV (with hr_outcome)")
train_file = st.file_uploader("Upload Training Event-Level CSV", type="csv", key="trainfile")
st.markdown("#### Upload Today's Event-Level CSV (with merged features, NO hr_outcome)")
live_file = st.file_uploader("Upload Today's Event-Level CSV", type="csv", key="livefile")

min_thr = st.number_input("Min HR Prob Threshold", 0.00, 0.5, 0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.20, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # Show all intersect columns for debug
    intersect = [c for c in train_df.columns if c in live_df.columns]
    st.write("Intersect columns:\n", intersect)

    # --- Fix: Only use *numeric* columns for model ---
    # Check numeric in both train and live
    numeric_cols = [
        c for c in intersect
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(live_df[c])
        and c != "hr_outcome"
    ]

    # Drop all-null or constant columns in BOTH files
    def is_valid_numeric(col):
        return (
            train_df[col].notnull().any() and train_df[col].nunique(dropna=True) > 1 and
            live_df[col].notnull().any() and live_df[col].nunique(dropna=True) > 1
        )
    model_features = [c for c in numeric_cols if is_valid_numeric(c)]
    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # Also show dtypes for double-debug
    st.write(pd.DataFrame({
        'col': model_features,
        'train_dtype': [str(train_df[c].dtype) for c in model_features],
        'live_dtype': [str(live_df[c].dtype) for c in model_features],
        'train_null': [train_df[c].isnull().mean() for c in model_features],
        'live_null': [live_df[c].isnull().mean() for c in model_features],
        'train_unique': [train_df[c].nunique(dropna=True) for c in model_features],
        'live_unique': [live_df[c].nunique(dropna=True) for c in model_features],
    }))

    # Dropped (all null or constant) columns
    dropped_null_or_const = [c for c in intersect if (
        train_df[c].isnull().all() and live_df[c].isnull().all()
    ) or (
        train_df[c].nunique(dropna=True) <= 1 and live_df[c].nunique(dropna=True) <= 1
    )]
    st.write("Dropped (all null both):", dropped_null_or_const)
    st.write("Dropped (constant in both):", dropped_null_or_const)

    # --- Model Fit & Predict ---
    from sklearn.linear_model import LogisticRegression

    X = train_df[model_features].fillna(0)
    y = train_df['hr_outcome'].astype(int)
    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X, y)

    X_live = live_df[model_features].fillna(0)
    hr_probs = model.predict_proba(X_live)[:, 1]
    live_df["hr_pred_prob"] = hr_probs

    # Output sweep
    st.markdown("### Results: HR Bot Picks by Threshold")
    thresholds = np.arange(min_thr, max_thr + step_thr, step_thr)
    for thr in thresholds:
        picks = live_df.loc[live_df["hr_pred_prob"] >= thr, "player_name"].tolist()
        st.write(f"Threshold {thr:.2f}: {picks}")

    st.markdown("Done! These are the official HR bot picks for today at each threshold.")
