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

min_thr = st.number_input("Min HR Prob Threshold", 0.00, 0.5, 0.01, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.15, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # ------ Feature Harmonization: Robust Numeric Inclusion ------
    # Find all columns present in BOTH files
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]

    # Force conversion to numeric in BOTH dataframes for all intersect columns
    for c in intersect:
        train_df[c] = pd.to_numeric(train_df[c], errors='ignore')
        live_df[c] = pd.to_numeric(live_df[c], errors='ignore')

    # Candidate features: Must be numeric in at least one, not all-null in both
    candidate_features = []
    for c in intersect:
        train_is_num = pd.api.types.is_numeric_dtype(train_df[c])
        live_is_num = pd.api.types.is_numeric_dtype(live_df[c])
        train_allnull = train_df[c].isnull().all()
        live_allnull = live_df[c].isnull().all()
        if (train_is_num or live_is_num) and not (train_allnull and live_allnull):
            candidate_features.append(c)

    # Exclude meta/id/target columns (edit as needed)
    exclude = {"player_name", "game_date", "team_code", "position", "weather", "stadium", "city", "mlb_id", "batting_order", "time", "hr_outcome"}
    model_features = [c for c in candidate_features if c not in exclude]

    # ------ Full Audit Report ------
    st.markdown("### 🔍 Audit Report:")
    st.write("Intersect columns:")
    st.write(intersect)
    for c in intersect:
        st.write(f"Col: {c} | train dtype: {train_df[c].dtype} | live dtype: {live_df[c].dtype} | "
                 f"train unique: {train_df[c].nunique(dropna=False)} | live unique: {live_df[c].nunique(dropna=False)} | "
                 f"train null: {train_df[c].isnull().mean():.4f} | live null: {live_df[c].isnull().mean():.4f}")

    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # Features dropped: all-null in both files
    dropped_allnull = [c for c in intersect if train_df[c].isnull().all() and live_df[c].isnull().all()]
    st.write("Features dropped (all-null in BOTH):", dropped_allnull)

    st.write("Null count in live file (top 20):")
    st.write(live_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write("Null count in train file (top 20):")
    st.write(train_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write(f"Train events: {len(train_df)}, Live events: {len(live_df)}")

    # --- Model Fit & Predict ---
    from sklearn.linear_model import LogisticRegression

    # Prepare modeling data: force all to numeric (even if they weren't before), fillna
    X = train_df[model_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = train_df['hr_outcome'].astype(int)
    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X, y)

    # Score live
    X_live = live_df[model_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    hr_probs = model.predict_proba(X_live)[:, 1]
    live_df["hr_pred_prob"] = hr_probs

    # Output sweep
    st.markdown("### Results: HR Bot Picks by Threshold")
    out_report = {}
    thresholds = np.arange(min_thr, max_thr + step_thr, step_thr)
    for thr in thresholds:
        picks = live_df.loc[live_df["hr_pred_prob"] >= thr, "player_name"].tolist()
        out_report[thr] = picks
        st.write(f"Threshold {thr:.2f}: {picks}")

    st.markdown("Done! These are the official HR bot picks for today at each threshold.")

    # --- Enhanced Audit Table (Optional) ---
    audit_table = []
    for c in model_features:
        audit_table.append({
            'feature': c,
            'train_dtype': str(train_df[c].dtype),
            'live_dtype': str(live_df[c].dtype),
            'train_null_frac': train_df[c].isnull().mean(),
            'live_null_frac': live_df[c].isnull().mean(),
            'train_nunique': train_df[c].nunique(dropna=False),
            'live_nunique': live_df[c].nunique(dropna=False),
        })
    audit_df = pd.DataFrame(audit_table)
    st.markdown("### 📋 Detailed Model Audit Table")
    st.dataframe(audit_df)

    st.download_button("⬇️ Download Audit Table (.csv)", audit_df.to_csv(index=False), file_name="mlb_hr_model_audit_table.csv")
