import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# ---- File Uploaders ----
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train")
live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live")

if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")
    
    # ---- Threshold controls ----
    st.subheader("Min HR Prob Threshold")
    min_thresh = st.number_input("Min HR Prob Threshold", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
    st.subheader("Max HR Prob Threshold")
    max_thresh = st.number_input("Max HR Prob Threshold", value=0.20, min_value=0.0, max_value=1.0, step=0.01)
    st.subheader("Threshold Step")
    thresh_step = st.number_input("Threshold Step", value=0.01, min_value=0.001, max_value=1.0, step=0.01)
    
    # ---- Feature Alignment and Diagnostics ----
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    missing_in_live = sorted([c for c in df_train.columns if c.lower() not in live_cols])
    missing_in_train = sorted([c for c in df_live.columns if c.lower() not in train_cols])
    
    # Auto-match features: strict intersection, then clean up dtypes
    features_to_use = sorted([c for c in df_train.columns if c in df_live.columns and c != "hr_outcome"])
    dtype_problems = {}
    for c in features_to_use:
        if str(df_train[c].dtype) != str(df_live[c].dtype):
            dtype_problems[c] = f"train={df_train[c].dtype}, live={df_live[c].dtype}"
            # Try to fix most common mismatch (int vs float)
            if (np.issubdtype(df_train[c].dtype, np.integer) and np.issubdtype(df_live[c].dtype, np.floating)) or (np.issubdtype(df_train[c].dtype, np.floating) and np.issubdtype(df_live[c].dtype, np.integer)):
                df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
                df_live[c] = pd.to_numeric(df_live[c], errors="coerce")
            # Or fallback: force both to string if object
            elif df_train[c].dtype == "O" or df_live[c].dtype == "O":
                df_train[c] = df_train[c].astype(str)
                df_live[c] = df_live[c].astype(str)

    # Drop features that are all null in train or live
    features_to_use = [c for c in features_to_use if df_train[c].notnull().sum() > 0 and df_live[c].notnull().sum() > 0]

    # Detect categorical/object features (not to be dropped, but to be encoded)
    cat_feats = [c for c in features_to_use if df_train[c].dtype == "O" or df_live[c].dtype == "O"]

    # Show diagnostics
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    st.write("**Features in train missing from live:**")
    st.write(missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**")
    st.write(missing_in_train if missing_in_train else "None")
    if dtype_problems:
        st.warning(f"Dtype mismatches detected! {', '.join([f'{k}: {v}' for k, v in dtype_problems.items()])}")

    st.write(f"**Final features used ({len(features_to_use)}):**")
    st.write(features_to_use)
    st.write("Null Fraction (train):")
    st.write(df_train[features_to_use].isnull().mean())
    st.write("Null Fraction (live):")
    st.write(df_live[features_to_use].isnull().mean())
    
    # ---- Fit Model on Train, Predict on Live ----
    X_train = df_train[features_to_use].copy()
    y_train = df_train["hr_outcome"].astype(int) if "hr_outcome" in df_train.columns else None
    X_live = df_live[features_to_use].copy()

    # Label encode categoricals
    label_encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        # Fit on train, map live. Unseen in live -> -1
        X_train[c] = le.fit_transform(X_train[c].astype(str))
        label_encoders[c] = le
        live_vals = X_live[c].astype(str)
        X_live[c] = live_vals.map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Impute any NaN with median for numeric, -1 for categoricals
    for c in features_to_use:
        if c in cat_feats:
            X_train[c] = X_train[c].fillna(-1)
            X_live[c] = X_live[c].fillna(-1)
        else:
            X_train[c] = X_train[c].fillna(X_train[c].median())
            X_live[c] = X_live[c].fillna(X_train[c].median())

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred_live = model.predict_proba(X_live)[:, 1]
    df_live["hr_prob"] = y_pred_live

    st.success(f"Model fit OK with {len(features_to_use)} features.")

    # ---- Threshold Table: Bot Picks by Threshold ----
    st.markdown("## 🔎 Bot Picks by Threshold")
    threshold_grid = np.arange(min_thresh, max_thresh + thresh_step/2, thresh_step)
    picks_by_thresh = []
    for t in threshold_grid:
        picks = df_live[df_live["hr_prob"] >= t]
        picks_by_thresh.append({"Threshold": t, "Num Picks": len(picks)})
        st.write(f"**Threshold {t:.2f}**: {len(picks)} picks")
        if not picks.empty:
            st.dataframe(picks[["player_name", "game_date", "hr_prob"]].sort_values("hr_prob", ascending=False).reset_index(drop=True))
    
    # ---- Feature Audit and Distribution (detailed, bug-proof) ----
    st.markdown("### 🧾 Bot Audit Report & Performance Readiness")
    with st.expander("🔍 Feature Presence and Type Audit"):
        st.write("**Train features missing in live:**")
        st.write(missing_in_live if missing_in_live else "None")
        st.write("**Live features missing in train:**")
        st.write(missing_in_train if missing_in_train else "None")
        st.write("**Features with dtype mismatches:**")
        st.write(dtype_problems if dtype_problems else "None")
        st.write("**Null fraction in train (top 20):**")
        st.write(df_train[features_to_use].isnull().mean().sort_values(ascending=False).head(20))
        st.write("**Null fraction in live (top 20):**")
        st.write(df_live[features_to_use].isnull().mean().sort_values(ascending=False).head(20))
    with st.expander("📊 Feature Distribution Snapshot (Train vs Live)"):
        audit_rows = []
        for feat in features_to_use[:25]:
            if feat in df_train.columns and feat in df_live.columns:
                train_vals = pd.to_numeric(df_train[feat], errors="coerce")
                live_vals = pd.to_numeric(df_live[feat], errors="coerce")
                train_mean, train_std = train_vals.mean(), train_vals.std()
                live_mean, live_std = live_vals.mean(), live_vals.std()
                st.write(f"**{feat}** | Train: mean={train_mean:.3f}, std={train_std:.3f} | Live: mean={live_mean:.3f}, std={live_std:.3f}")
                audit_rows.append({
                    "feature": feat,
                    "train_mean": train_mean, "train_std": train_std,
                    "live_mean": live_mean, "live_std": live_std,
                })
    with st.expander("🧬 Categorical Variable Coverage"):
        for c in cat_feats:
            train_cats = set(X_train[c].astype(str).unique())
            live_cats = set(X_live[c].astype(str).unique())
            unseen = live_cats - train_cats
            st.write(f"Feature: {c} | Unseen categories in live: {unseen if unseen else 'None'}")

    # ---- Download Audit CSV Button ----
    if 'audit_rows' in locals() and audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        csv_buffer = io.StringIO()
        audit_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Audit CSV",
            data=csv_buffer.getvalue(),
            file_name="bot_feature_audit_report.csv",
            mime="text/csv"
        )

else:
    st.info("Upload both a training and a live event-level CSV to run the bot and see diagnostics.")
