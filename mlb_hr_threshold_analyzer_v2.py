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

    # --- Feature audit
    st.markdown("### 🔍 Audit Report:")

    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]

    # -- Enhanced: Try to cast to float where possible, to treat numeric/categorical IDs as numerics if possible
    for col in intersect:
        try:
            train_df[col] = pd.to_numeric(train_df[col], errors="ignore")
        except Exception:
            pass
        try:
            live_df[col] = pd.to_numeric(live_df[col], errors="ignore")
        except Exception:
            pass

    # Show audit for all intersecting columns
    audit_rows = []
    for col in intersect:
        audit_rows.append({
            "col": col,
            "train dtype": str(train_df[col].dtype),
            "live dtype": str(live_df[col].dtype),
            "train unique": train_df[col].nunique(dropna=True),
            "live unique": live_df[col].nunique(dropna=True),
            "train null": train_df[col].isnull().mean(),
            "live null": live_df[col].isnull().mean(),
        })
    st.write("Intersect columns:")
    st.dataframe(pd.DataFrame(audit_rows))

    # --- Improved: Keep features if they are numeric OR IDs, NOT all-null in both, NOT constant in both
    # Use: if not all-null in BOTH, and not constant in BOTH (allows constant-in-live for 1 game)
    numeric_like = []
    for col in intersect:
        t_type = train_df[col].dtype
        l_type = live_df[col].dtype
        is_num = (np.issubdtype(t_type, np.number) or np.issubdtype(l_type, np.number))
        is_id = "id" in col.lower() or col in ["mlb_id", "batter_id", "pitcher_id"]
        if is_num or is_id:
            numeric_like.append(col)

    model_features = []
    for col in numeric_like:
        train_all_null = train_df[col].isnull().all()
        live_all_null = live_df[col].isnull().all()
        train_unique = train_df[col].nunique(dropna=True)
        live_unique = live_df[col].nunique(dropna=True)
        # Keep if not all-null in both, and not constant in both
        if not (train_all_null and live_all_null) and not (train_unique <= 1 and live_unique <= 1):
            if col != "hr_outcome":
                model_features.append(col)

    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # Diagnostics for what was dropped
    dropped_all_null = [col for col in numeric_like if train_df[col].isnull().all() and live_df[col].isnull().all()]
    dropped_constant_both = [col for col in numeric_like if train_df[col].nunique(dropna=True) <= 1 and live_df[col].nunique(dropna=True) <= 1]
    st.write("Dropped (all null both):", dropped_all_null)
    st.write("Dropped (constant in both):", dropped_constant_both)

    # --- Model Fit & Predict
    from sklearn.linear_model import LogisticRegression

    X = train_df[model_features].fillna(0)
    y = train_df['hr_outcome'].astype(int)
    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X, y)

    # Score live
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

    # --- Audit Table Download ---
    audit_df = pd.DataFrame(audit_rows)
    st.markdown("### 📋 Detailed Model Audit Table")
    st.dataframe(audit_df)
    st.download_button("⬇️ Download Audit Table (.csv)", audit_df.to_csv(index=False), file_name="mlb_hr_model_audit_table.csv")
