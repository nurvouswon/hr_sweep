import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")

st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

# ---- UTILITY FUNCTIONS ----

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def robust_numeric_columns(df):
    cols = []
    for c in df.columns:
        try:
            dt = pd.api.types.pandas_dtype(df[c].dtype)
            if (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c])) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1:
                cols.append(c)
        except Exception:
            continue
    return cols

def clean_id(x):
    try:
        if pd.isna(x): return None
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

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

# ------------------------- TAB 1: GENERATE TODAY CSV -------------------------

with tab1:
    st.header("🔄 Generate Today's Batter Event-Level CSV (with merged rolling/stat features from history)")

    uploaded_today_lineups = st.file_uploader(
        "Upload Today's Lineups/Matchups CSV (Required: must contain mlb_id/batter_id, player_name, etc)",
        type="csv", key="todaylineup"
    )
    uploaded_sample_hist = st.file_uploader(
        "Upload Historical Event-Level CSV (must have all rolling/stat features, 'batter_id', and 'game_date')",
        type="csv", key="samplehist"
    )

    if uploaded_today_lineups and uploaded_sample_hist:
        today_lineups = pd.read_csv(uploaded_today_lineups)
        hist = pd.read_csv(uploaded_sample_hist)
        st.success(f"Loaded {len(today_lineups)} lineups, {len(hist)} historical events.")

        # Normalize col names
        today_lineups.columns = [c.strip().lower().replace(" ", "_") for c in today_lineups.columns]
        hist.columns = [c.strip().lower().replace(" ", "_") for c in hist.columns]

        id_col = "mlb_id" if "mlb_id" in today_lineups.columns else "batter_id"
        if id_col not in today_lineups.columns or "batter_id" not in hist.columns:
            st.error("Both files must have 'mlb_id' or 'batter_id'.")
            st.stop()

        today_lineups[id_col] = today_lineups[id_col].astype(str)
        hist['batter_id'] = hist['batter_id'].astype(str)

        # Get last rolling/stat features for each batter
        if 'game_date' in hist.columns:
            hist['game_date'] = pd.to_datetime(hist['game_date'], errors='coerce')
        rolling_cols = [c for c in hist.columns if (
            c.startswith('b_') or c.startswith('p_') or 
            c.startswith('park_hand_hr_') or 
            c.endswith('_rate_20') or 
            c.startswith('b_vsp_hand_hr_') or 
            c.startswith('p_vsb_hand_hr_') or 
            c.startswith('b_pitchtype_hr_') or 
            c.startswith('p_pitchtype_hr_')
        )]

        last_feats = hist.sort_values('game_date').groupby('batter_id').tail(1)
        last_feats = last_feats[['batter_id'] + rolling_cols].copy()
        merged = today_lineups.merge(last_feats, left_on=id_col, right_on='batter_id', how='left', suffixes=('', '_roll'))

        st.write("Sample of Today's Event-Level CSV (with merged rolling/stat features):")
        st.dataframe(merged.head(30))
        null_report = merged[rolling_cols].isnull().sum().sort_values(ascending=False)
        st.markdown("#### Null report for rolling/stat features in output:")
        st.text(null_report.to_string())

        # Download merged output
        merged_out_cols = [c for c in merged.columns if not c.startswith("unnamed")]
        st.download_button(
            "⬇️ Download Today's Event-Level CSV (for Prediction App, with merged features)",
            data=merged[merged_out_cols].to_csv(index=False),
            file_name=f"event_level_today_full_{datetime.now().strftime('%Y_%m_%d')}.csv"
        )
        st.info("Tab 1 complete. Use this file in Tab 2 below.")

# ---------------------- TAB 2: PREDICT & AUDIT -------------------------------

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

        # Features
        numeric_features = robust_numeric_columns(train_df)
        if 'hr_outcome' in numeric_features:
            numeric_features.remove('hr_outcome')
        model_features = [f for f in numeric_features if f in live_df.columns]

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
        st.markdown("#### All Picks (Threshold Sweep):")
        for _, row in picks_df.iterrows():
            st.write(f"**Threshold {row['threshold']}**: {row['picked_players']}")

        # ============= FULL AUDIT REPORT ==============
        st.markdown("---")
        st.header("🔍 Full Audit Report: Model, Data, Features, and Nulls")

        # Features Used
        st.write(f"Model features used ({len(model_features)}): {model_features}")

        # Missing features diagnostics
        hist_cols = set(train_df.columns)
        live_cols = set(live_df.columns)
        missing_from_live = [c for c in model_features if c not in live_cols]
        missing_from_train = [c for c in live_cols if c not in hist_cols]
        st.write("**Features in history but missing from live:**")
        st.write(missing_from_live)
        st.write("**Features in live but missing from history:**")
        st.write(missing_from_train)

        # Features with only one unique value
        st.write("**Features with only one unique value (train):**")
        st.write({c: train_df[c].nunique() for c in train_df.columns if train_df[c].nunique() == 1})
        st.write("**Features with only one unique value (live):**")
        st.write({c: live_df[c].nunique() for c in live_df.columns if live_df[c].nunique() == 1})

        # Features with nulls
        st.write("**Null count for model features (train):**")
        st.write(train_df[model_features].isnull().sum().sort_values(ascending=False))
        st.write("**Null count for model features (live):**")
        st.write(live_df[model_features].isnull().sum().sort_values(ascending=False))

        # Stats: unique values, sample, value range
        st.write("**Sample (train):**")
        st.dataframe(train_df[model_features].head(20))
        st.write("**Sample (live):**")
        st.dataframe(live_df[model_features].head(20))

        # Distributions/ranges
        st.write("**Feature ranges (train):**")
        st.write(train_df[model_features].agg(['min', 'max', 'mean']).T)
        st.write("**Feature ranges (live):**")
        st.write(live_df[model_features].agg(['min', 'max', 'mean']).T)

        st.success("Audit complete! Check all above sections for detailed pipeline integrity.")

        st.markdown("""
---
**Instructions:**
- Tab 1: Upload BOTH today's lineups/matchups and historical event-level file, download merged file for prediction.
- Tab 2: Upload event-level files, run bot, check full diagnostics if features are missing!
""")
