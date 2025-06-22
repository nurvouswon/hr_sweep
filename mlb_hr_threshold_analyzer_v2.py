import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# File uploaders
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_file"
)
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_file"
)

# Threshold sliders always visible
col1, col2, col3 = st.columns(3)
with col1:
    min_thresh = st.number_input("Min HR Prob Threshold", 0.0, 1.0, 0.05, 0.01)
with col2:
    max_thresh = st.number_input("Max HR Prob Threshold", 0.0, 1.0, 0.20, 0.01)
with col3:
    thresh_step = st.number_input("Threshold Step", 0.001, 0.5, 0.01, 0.001)

if train_file is not None and live_file is not None:
    train = pd.read_csv(train_file)
    live = pd.read_csv(live_file)

    st.success(f"Training file loaded! {train.shape[0]:,} rows, {train.shape[1]} columns.")
    st.success(f"Today's file loaded! {live.shape[0]:,} rows, {live.shape[1]} columns.")

    # Find the HR outcome column (case-insensitive)
    outcome_col = [c for c in train.columns if c.lower() == "hr_outcome"]
    if not outcome_col:
        st.error("No 'hr_outcome' column found in training CSV.")
        st.stop()
    outcome_col = outcome_col[0]

    # Feature intersection and audit
    train_cols = {c.lower(): c for c in train.columns}
    live_cols = {c.lower(): c for c in live.columns}
    feature_names = [
        c for c in train.columns
        if c.lower() in live_cols and c.lower() != outcome_col.lower()
    ]
    missing_in_live = [c for c in train.columns if c.lower() not in live_cols]
    missing_in_train = [c for c in live.columns if c.lower() not in train_cols]

    # Dtype/null/unique diagnostics
    diag_table = []
    for c in feature_names:
        tcol = train_cols[c.lower()]
        lcol = live_cols[c.lower()]
        diag_table.append({
            "feature": c,
            "train_dtype": str(train[tcol].dtype),
            "live_dtype": str(live[lcol].dtype),
            "train_null_frac": train[tcol].isna().mean(),
            "live_null_frac": live[lcol].isna().mean(),
            "train_unique": train[tcol].nunique(),
            "live_unique": live[lcol].nunique(),
        })
    diag_df = pd.DataFrame(diag_table)

    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    st.dataframe(diag_df, use_container_width=True)
    st.markdown("**Features in train missing from live:**")
    st.write(missing_in_live)
    st.markdown("**Features in live missing from train:**")
    st.write(missing_in_train)

    # Dtype mismatches summary
    dtype_mismatch = diag_df[diag_df["train_dtype"] != diag_df["live_dtype"]]
    if not dtype_mismatch.empty:
        st.warning("⚠️ Dtype mismatches detected!")
        st.dataframe(dtype_mismatch)

    # Prepare features for modeling
    X_train = train[[train_cols[c.lower()] for c in feature_names]].copy()
    X_live = live[[live_cols[c.lower()] for c in feature_names]].copy()

    # Label encode string columns (safe even if not needed)
    for c in X_train.columns:
        if X_train[c].dtype == object or X_live[c].dtype == object:
            enc = LabelEncoder()
            values = pd.concat([X_train[c], X_live[c]]).astype(str)
            enc.fit(values.fillna("NA"))
            X_train[c] = enc.transform(X_train[c].fillna("NA").astype(str))
            X_live[c] = enc.transform(X_live[c].fillna("NA").astype(str))

    # Convert to float for sklearn
    X_train = X_train.astype(float)
    X_live = X_live.astype(float)

    y_train = train[outcome_col]

    # Fit logistic regression
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_live)[:, 1]
        live["hr_prob"] = preds
        st.success(f"Model fit and predictions complete. Model used {X_train.shape[1]} features.")

        # Threshold sweep table
        ths = np.arange(min_thresh, max_thresh + thresh_step, thresh_step)
        results = []
        for t in ths:
            count = (live["hr_prob"] >= t).sum()
            results.append({"Threshold": t, "Count": count})
        st.dataframe(pd.DataFrame(results))

        # Show top predictions above current min threshold
        st.write("### Top HR Candidates (hr_prob ≥ min threshold)")
        st.dataframe(
            live[live["hr_prob"] >= min_thresh]
            .sort_values("hr_prob", ascending=False)
            .head(50)
        )
    except Exception as e:
        st.error(f"Model fitting or prediction failed: {e}")

else:
    st.info("Please upload both files to enable model and threshold controls.")
