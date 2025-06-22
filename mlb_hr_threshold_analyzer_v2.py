import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# --- CSV Uploaders
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_csv")
live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_csv")

min_threshold = st.number_input("Min HR Prob Threshold", 0.0, 1.0, 0.05, 0.01)
max_threshold = st.number_input("Max HR Prob Threshold", 0.0, 1.0, 0.20, 0.01)
threshold_step = st.number_input("Threshold Step", 0.001, 0.2, 0.01, 0.001)

if train_file is not None and live_file is not None:
    # --- Load Data
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.success(f"Training file loaded! {len(df_train):,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {len(df_live):,} rows, {df_live.shape[1]} columns.")
    
    # --- Standardize column names (lowercase, underscores)
    def clean_col(c):
        return c.lower().replace("-", "_").replace(" ", "_")
    df_train.columns = [clean_col(c) for c in df_train.columns]
    df_live.columns = [clean_col(c) for c in df_live.columns]
    
    # --- Feature Mapping Helper
    def match_col(col, columns):
        """Find best-matching col (case/underscore-insensitive)"""
        col = col.lower().replace("-", "_")
        matches = [c for c in columns if c.lower().replace("-", "_") == col]
        return matches[0] if matches else None

    # --- Feature Diagnostics
    train_cols = set(df_train.columns)
    live_cols = set(df_live.columns)
    missing_from_live = sorted(list(train_cols - live_cols))
    missing_from_train = sorted(list(live_cols - train_cols))
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    st.write("**Features in train missing from live:**", missing_from_live)
    st.write("**Features in live missing from train:**", missing_from_train)
    
    # Dtype checks (show only if mismatch)
    dtype_debug = []
    for col in (train_cols & live_cols):
        if df_train[col].dtype != df_live[col].dtype:
            dtype_debug.append(f"{col}: train={df_train[col].dtype}, live={df_live[col].dtype}")
    if dtype_debug:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_debug))
    
    # --- HR Outcome col
    hr_col = match_col("hr_outcome", df_train.columns)
    if not hr_col:
        st.error("No `hr_outcome` column found in training data.")
        st.stop()

    # --- Numeric/Categorical Split
    num_feats = []
    cat_feats = []
    for c in df_train.columns:
        if c == hr_col:
            continue
        if df_train[c].dtype in [np.float32, np.float64, np.int32, np.int64]:
            num_feats.append(c)
        else:
            if len(df_train[c].unique()) <= 30:  # treat as categorical if small cardinality
                cat_feats.append(c)
            else:
                num_feats.append(c)

    # --- Strict: use only features present in both AND non-null
    features_to_use = []
    for c in df_train.columns:
        if c == hr_col: continue
        c_live = match_col(c, df_live.columns)
        if c_live is None:
            continue
        if df_train[c].notnull().sum() == 0: continue
        if df_live[c_live].notnull().sum() == 0: continue
        features_to_use.append(c)

    # Final feature map
    st.markdown("**Final features used ({}):**".format(len(features_to_use)))
    st.write(features_to_use)

    # Null fraction audit
    null_report = pd.DataFrame({
        'feature': features_to_use,
        'null_frac_train': [df_train[c].isnull().mean() for c in features_to_use],
        'null_frac_live': [df_live[match_col(c, df_live.columns)].isnull().mean() for c in features_to_use]
    })
    st.dataframe(null_report, height=250)

    # Dtype Fixes
    if 'batter_id' in df_train.columns:
        df_train['batter_id'] = pd.to_numeric(df_train['batter_id'], errors='coerce').astype('Int64')
    if 'batter_id' in df_live.columns:
        df_live['batter_id'] = pd.to_numeric(df_live['batter_id'], errors='coerce').astype('Int64')
    # force "park" to str if it exists in both
    if 'park' in df_train.columns and 'park' in df_live.columns:
        df_train['park'] = df_train['park'].astype(str)
        df_live['park'] = df_live['park'].astype(str)

    # --- Build X/y for train/live
    X_train = pd.DataFrame({c: df_train[c] for c in features_to_use})
    X_live = pd.DataFrame({c: df_live[match_col(c, df_live.columns)] for c in features_to_use})
    y_train = df_train[hr_col].fillna(0).astype(int)

    # Label Encoding
    for c in features_to_use:
        if X_train[c].dtype == object or str(X_train[c].dtype).startswith("category"):
            # Fit on TRAIN, transform both, with fallback for unknowns in LIVE
            le = LabelEncoder()
            X_train[c] = X_train[c].astype(str).fillna("unknown")
            le.fit(list(X_train[c].unique()))
            X_train[c] = le.transform(X_train[c])
            # Apply to live, unknowns to special index
            X_live[c] = X_live[c].astype(str).fillna("unknown")
            known_labels = set(le.classes_)
            X_live[c] = [le.transform([x])[0] if x in known_labels else -1 for x in X_live[c]]

    # --- Model Fit
    try:
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_live)[:,1]
        st.success(f"Model fit OK with {len(features_to_use)} features.")
    except Exception as e:
        st.error(f"Model fitting or prediction failed: {e}")
        st.stop()

    # --- Show sliders again for clarity
    st.markdown("#### Threshold Controls")
    min_threshold = st.number_input("Min HR Prob Threshold", 0.0, 1.0, float(min_threshold), 0.01, key="min_thr_show")
    max_threshold = st.number_input("Max HR Prob Threshold", 0.0, 1.0, float(max_threshold), 0.01, key="max_thr_show")
    threshold_step = st.number_input("Threshold Step", 0.001, 0.2, float(threshold_step), 0.001, key="step_thr_show")

    # --- HR Bot Picks For Each Threshold
    st.subheader("Bot Picks For Each HR Prob Threshold")
    thresholds = np.arange(min_threshold, max_threshold + threshold_step/2, threshold_step)
    df_live_disp = df_live.copy()
    df_live_disp["HR_Prob"] = y_pred_proba

    for th in thresholds:
        picks = df_live_disp[df_live_disp["HR_Prob"] >= th].copy()
        picks = picks.sort_values("HR_Prob", ascending=False)
        st.markdown(f"**Threshold ≥ {th:.3f}** &mdash; {len(picks)} picks")
        if not picks.empty:
            show_cols = ["player_name", "team_code", "batter_id", "HR_Prob"] + \
                [c for c in ["batting_order", "position", "stadium", "game_date", "hard_hit_rate_20", "sweet_spot_rate_20"]
                if c in picks.columns]
            st.dataframe(picks[show_cols].reset_index(drop=True), use_container_width=True)
        else:
            st.info("No picks at this threshold.")

else:
    st.info("Please upload both the training and today's CSV files above to run the analyzer.")
