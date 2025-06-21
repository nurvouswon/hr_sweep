import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="traincsv")
uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="livecsv")

min_thresh = st.number_input("Min HR Prob Threshold", 0.0, 1.0, 0.01, 0.01)
max_thresh = st.number_input("Max HR Prob Threshold", 0.0, 1.0, 0.15, 0.01)
thresh_step = st.number_input("Threshold Step", 0.001, 0.2, 0.01, 0.01)

if uploaded_train and uploaded_live:
    train_df = pd.read_csv(uploaded_train)
    live_df = pd.read_csv(uploaded_live)

    # --- Column normalization for reliability ---
    train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
    live_df.columns = [c.strip().lower().replace(" ", "_") for c in live_df.columns]

    # --- Identify candidate features ---
    ycol = "hr_outcome"
    drop_cols = [
        "hr_outcome", "batter_id", "player_name", "game_date", "mlb_id",
        "events", "description", "stand", "p_throws"
    ]
    # Exclude non-numeric and all-null columns from training
    train_features = [
        c for c in train_df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(train_df[c])
        and not train_df[c].isnull().all()
    ]
    # Remove constant features in train
    train_features = [c for c in train_features if train_df[c].nunique(dropna=True) > 1]

    # --- Only use features present in BOTH and are numeric in live
    model_features = [
        c for c in train_features
        if c in live_df.columns
        and pd.api.types.is_numeric_dtype(live_df[c])
        and not live_df[c].isnull().all()
    ]

    # ==== Audit Report Construction ====
    audit = {}
    audit['model_features'] = model_features
    audit['features_in_history_not_live'] = sorted([c for c in train_features if c not in live_df.columns])
    audit['features_in_live_not_history'] = sorted([c for c in live_df.columns if c not in train_features])
    audit['features_constant_train'] = sorted([c for c in train_features if train_df[c].nunique(dropna=True) <= 1])
    audit['features_constant_live'] = sorted([c for c in model_features if live_df[c].nunique(dropna=True) <= 1])
    audit['features_allnull_train'] = sorted([c for c in train_features if train_df[c].isnull().all()])
    audit['features_allnull_live'] = sorted([c for c in model_features if live_df[c].isnull().all()])
    audit['train_nulls'] = train_df[model_features].isnull().sum().sort_values(ascending=False).head(20)
    audit['live_nulls'] = live_df[model_features].isnull().sum().sort_values(ascending=False).head(20)
    audit['train_n'] = len(train_df)
    audit['live_n'] = len(live_df)

    # === Display Audit Summary ===
    st.subheader("🔍 Audit Report:")
    st.write(f"Model features used ({len(model_features)}): {model_features}")
    st.write(f"Features in history but missing from live: {audit['features_in_history_not_live']}")
    st.write(f"Features in live but missing from history: {audit['features_in_live_not_history']}")
    st.write(f"Features dropped (constant in train): {audit['features_constant_train']}")
    st.write(f"Features dropped (constant in live): {audit['features_constant_live']}")
    st.write(f"Features dropped (all-null in train): {audit['features_allnull_train']}")
    st.write(f"Features dropped (all-null in live): {audit['features_allnull_live']}")
    st.write(f"Null count in live file (top 20):")
    st.dataframe(audit['live_nulls'])
    st.write(f"Null count in train file (top 20):")
    st.dataframe(audit['train_nulls'])
    st.write(f"Train events: {audit['train_n']}, Live events: {audit['live_n']}")

    # === Downloadable Audit Report ===
    audit_report = pd.DataFrame({
        "features_used": pd.Series(model_features),
        "features_missing_in_live": pd.Series(audit['features_in_history_not_live']),
        "features_missing_in_history": pd.Series(audit['features_in_live_not_history'])
    })
    st.download_button("⬇️ Download Audit Report CSV", data=audit_report.to_csv(index=False), file_name="mlb_hr_model_audit_report.csv")

    # ==== Modeling ====
    st.markdown("---")
    X_train = train_df[model_features].fillna(0)
    y_train = train_df[ycol].astype(int)
    X_live = live_df[model_features].fillna(0)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    live_df['hr_prob'] = model.predict_proba(X_live)[:, 1]

    # ==== Threshold Sweep & Picks ====
    st.subheader("Results: HR Bot Picks by Threshold")
    picks_by_thresh = {}
    for t in np.arange(min_thresh, max_thresh + thresh_step, thresh_step):
        picks = (
            live_df.loc[live_df['hr_prob'] >= t, 'player_name'].tolist()
            if 'player_name' in live_df.columns
            else live_df.loc[live_df['hr_prob'] >= t].index.tolist()
        )
        picks_by_thresh[round(t, 3)] = picks

    st.write("All Picks (Threshold Sweep):")
    for t in sorted(picks_by_thresh.keys()):
        st.write(f"Threshold {t}: {picks_by_thresh[t]}")

    # === Download Full Results ===
    st.download_button("⬇️ Download HR Bot Probabilities",
        data=live_df.to_csv(index=False),
        file_name=f"hr_bot_probs_{datetime.now().strftime('%Y_%m_%d')}.csv"
    )
else:
    st.warning("Please upload BOTH a training event-level CSV (with hr_outcome) and a live/today event-level CSV (with merged features).")
