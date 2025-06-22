import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# Upload CSVs
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_csv"
)
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_csv"
)

# Threshold controls
col1, col2, col3 = st.columns(3)
with col1:
    min_thresh = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
with col2:
    max_thresh = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
with col3:
    thresh_step = st.number_input("Threshold Step", min_value=0.001, max_value=1.0, value=0.01, step=0.01)

if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)

    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")

    # Identify numeric/categorical features
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())

    # List differences
    missing_from_live = sorted(list(train_cols - live_cols))
    missing_from_train = sorted(list(live_cols - train_cols))
    st.write("🩺 **Feature Diagnostics Table (train/live overlap)**")
    st.write("**Features in train missing from live:**")
    st.write(missing_from_live)
    st.write("**Features in live missing from train:**")
    st.write(missing_from_train)

    # Find columns with dtype mismatch
    dtype_mismatches = []
    for col in set(df_train.columns) & set(df_live.columns):
        if df_train[col].dtype != df_live[col].dtype:
            dtype_mismatches.append((col, df_train[col].dtype, df_live[col].dtype))
    if dtype_mismatches:
        st.warning("⚠️ Dtype mismatches detected!")
        st.write(pd.DataFrame(dtype_mismatches, columns=["Feature", "Train Dtype", "Live Dtype"]))

    # Display null fractions
    def null_frac_report(df, label):
        rep = []
        for col in df.columns:
            frac = df[col].isnull().mean()
            rep.append({"feature": col, f"{label}_null_frac": frac, f"{label}_dtype": str(df[col].dtype)})
        return pd.DataFrame(rep)
    st.write("**Null Fraction (train):**")
    st.dataframe(null_frac_report(df_train, "train"))
    st.write("**Null Fraction (live):**")
    st.dataframe(null_frac_report(df_live, "live"))

    # Find overlapping features (case-insensitive)
    overlap = [c for c in df_train.columns if c.lower() in [x.lower() for x in df_live.columns]]
    features_to_use = []
    for c in overlap:
        if c not in ["hr_outcome"]:
            features_to_use.append(c)

    # Only use non-null features in both
    features_to_use = [c for c in features_to_use if df_train[c].notnull().sum() > 0 and df_live[c].notnull().sum() > 0]

    # Remove "object" columns if not strictly needed (LogisticRegression can't take strings)
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()

    # Handle categorical columns safely
    for c in X_train.columns:
        if X_train[c].dtype == object or X_live[c].dtype == object:
            all_vals = pd.concat([X_train[c], X_live[c]]).astype(str).fillna("NA")
            enc = LabelEncoder()
            enc.fit(all_vals)
            X_train[c] = enc.transform(X_train[c].astype(str).fillna("NA"))
            X_live[c] = enc.transform(X_live[c].astype(str).fillna("NA"))

    # Final feature audit after encoding
    st.write("**Final feature audit (non-null features used):**")
    st.write(X_train.columns.tolist())

    # Build model
    y = df_train["hr_outcome"].values if "hr_outcome" in df_train.columns else None

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y)
        st.success(f"Model trained with {X_train.shape[1]} features.")

        # Run predictions on live
        preds = model.predict_proba(X_live)[:, 1]

        # Threshold sweep
        results = []
        thresholds = np.arange(min_thresh, max_thresh + thresh_step, thresh_step)
        for t in thresholds:
            count = np.sum(preds >= t)
            results.append({"Threshold": t, "CountAbove": count})
        st.write("### Threshold Results")
        st.dataframe(pd.DataFrame(results))

        # Output predictions if desired
        df_out = df_live.copy()
        df_out["hr_prob"] = preds
        st.write("### Sample predictions")
        st.dataframe(df_out.head(10))

    except Exception as e:
        st.error(f"Model fitting or prediction failed: {e}")
        st.stop()
