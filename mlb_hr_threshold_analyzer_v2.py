import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")

st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

def clean_id(x):
    try:
        if pd.isna(x): return None
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

def robust_numeric_columns(df):
    cols = []
    for c in df.columns:
        try:
            dt = pd.api.types.pandas_dtype(df[c].dtype)
            if (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c])) and not pd.api.types.is_bool_dtype(df[c]):
                cols.append(c)
        except Exception:
            continue
    return cols

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def get_all_stat_rolling_cols():
    roll_base = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'release_speed',
                 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
    windows = [3, 5, 7, 14]
    cols = []
    for prefix in ['B_', 'P_']:
        for base in roll_base:
            for w in windows:
                cols.append(f"{prefix}{base}_{w}")
    for typ in ['B_vsP_hand_HR_', 'P_vsB_hand_HR_', 'B_pitchtype_HR_', 'P_pitchtype_HR_']:
        for w in windows:
            cols.append(f"{typ}{w}")
    for w in [7, 14, 30]:
        cols.append(f"park_hand_HR_{w}")
    cols += [
        'hard_hit_rate_20', 'sweet_spot_rate_20', 'relative_wind_angle',
        'relative_wind_sin', 'relative_wind_cos'
    ]
    return cols

with tab2:
    st.header("Upload Event-Level CSVs & Run Model")
    uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
    uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_ev")

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold_min = st.number_input("Min HR Prob Threshold", value=0.01, min_value=0.01, max_value=0.5, step=0.01)
    with col2:
        threshold_max = st.number_input("Max HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
    with col3:
        threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.01, max_value=0.10, step=0.01)

    predict_btn = st.button("🚀 Run HR Bot Picks and Audit")

    if predict_btn:
        if uploaded_train is None or uploaded_live is None:
            st.warning("Please upload BOTH files (historical and today's event-level CSV).")
            st.stop()

        train_df = pd.read_csv(uploaded_train)
        live_df = pd.read_csv(uploaded_live)

        # Prep IDs if needed
        for df in [train_df, live_df]:
            if 'batter_id' in df.columns:
                df['batter_id'] = df['batter_id'].apply(clean_id)
            elif 'batter' in df.columns:
                df['batter_id'] = df['batter'].apply(clean_id)

        # --- MODEL FEATURES: Use all shared numeric features except hr_outcome ---
        train_numeric = set(robust_numeric_columns(train_df))
        live_numeric = set(robust_numeric_columns(live_df))
        if 'hr_outcome' in train_numeric:
            train_numeric.remove('hr_outcome')
        model_features = sorted(list(train_numeric & live_numeric))

        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)

        # Train XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
        xgb_clf.fit(train_X, train_y)
        live_X = live_df[model_features].fillna(0)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # Picks per threshold
        results = []
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            display_name_col = next((col for col in ['player_name', 'batter_name', 'batter_id'] if col in picked.columns), picked.columns[0])
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked[display_name_col])
            })
        picks_df = pd.DataFrame(results)
        st.header("Results: HR Bot Picks by Threshold")
        st.dataframe(picks_df)
        st.download_button(
            "⬇️ Download Picks by Threshold (CSV)",
            data=picks_df.to_csv(index=False),
            file_name="today_hr_bot_picks_by_threshold.csv"
        )

        # ---- FULL AUDIT REPORT ----
        st.markdown("---")
        st.header("🔍 Full Audit Report: Model, Data, Features, and Nulls")

        hist_cols = set(train_df.columns)
        live_cols = set(live_df.columns)
        missing_from_live = [c for c in model_features if c not in live_cols]
        missing_from_train = [c for c in live_cols if c not in hist_cols]

        # Features with only one unique value
        one_val_train = {c: train_df[c].nunique() for c in model_features if train_df[c].nunique() == 1}
        one_val_live = {c: live_df[c].nunique() for c in model_features if live_df[c].nunique() == 1}

        # Null counts for model features
        null_train = train_df[model_features].isnull().sum()
        null_live = live_df[model_features].isnull().sum()

        # Feature range summary
        feature_ranges_train = train_df[model_features].agg(['min', 'max', 'mean']).T
        feature_ranges_live = live_df[model_features].agg(['min', 'max', 'mean']).T

        # Compose the audit DataFrame for download
        audit_df = pd.DataFrame({
            'feature': model_features,
            'in_train': [f in hist_cols for f in model_features],
            'in_live': [f in live_cols for f in model_features],
            'nulls_train': [null_train[f] if f in null_train else np.nan for f in model_features],
            'nulls_live': [null_live[f] if f in null_live else np.nan for f in model_features],
            'unique_train': [train_df[f].nunique() if f in train_df else np.nan for f in model_features],
            'unique_live': [live_df[f].nunique() if f in live_df else np.nan for f in model_features],
            'min_train': [feature_ranges_train.loc[f, 'min'] if f in feature_ranges_train.index else np.nan for f in model_features],
            'max_train': [feature_ranges_train.loc[f, 'max'] if f in feature_ranges_train.index else np.nan for f in model_features],
            'mean_train': [feature_ranges_train.loc[f, 'mean'] if f in feature_ranges_train.index else np.nan for f in model_features],
            'min_live': [feature_ranges_live.loc[f, 'min'] if f in feature_ranges_live.index else np.nan for f in model_features],
            'max_live': [feature_ranges_live.loc[f, 'max'] if f in feature_ranges_live.index else np.nan for f in model_features],
            'mean_live': [feature_ranges_live.loc[f, 'mean'] if f in feature_ranges_live.index else np.nan for f in model_features],
        })

        st.write(f"Model features used ({len(model_features)}): {model_features}")
        st.write("**Features in history but missing from live:**", missing_from_live)
        st.write("**Features in live but missing from history:**", missing_from_train)
        st.write("**Features with only one unique value (train):**", one_val_train)
        st.write("**Features with only one unique value (live):**", one_val_live)
        st.write("**Null count for model features (train):**")
        st.write(null_train.sort_values(ascending=False))
        st.write("**Null count for model features (live):**")
        st.write(null_live.sort_values(ascending=False))
        st.write("**Feature ranges (train):**")
        st.dataframe(feature_ranges_train)
        st.write("**Feature ranges (live):**")
        st.dataframe(feature_ranges_live)
        st.write("**Sample (train):**")
        st.dataframe(train_df[model_features].head(15))
        st.write("**Sample (live):**")
        st.dataframe(live_df[model_features].head(15))

        # DOWNLOAD AUDIT REPORT CSV
        st.download_button(
            "⬇️ Download Full Audit Report CSV",
            data=audit_df.to_csv(index=False),
            file_name="mlb_hr_bot_full_audit_report.csv"
        )

        st.success("Audit complete! Download and review the full CSV above for detailed inspection.")
