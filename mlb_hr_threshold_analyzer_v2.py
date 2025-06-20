import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------- Utility functions ---------------- #
def clean_id(x):
    try:
        if pd.isna(x): return None
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

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

def get_all_stat_rolling_cols():
    roll_base = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                 'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
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

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")
st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("🔄 Generate Today's Batter Event-Level CSV (with merged rolling/stat features from history)")
    st.markdown("""
    - **Step 1:** Upload Today's Lineups/Matchups CSV (must have at least `mlb_id`/`batter_id`, `player_name`, `team_code`, and `game_date`).
    - **Step 2:** Upload *Historical Event-Level CSV* (all rolling/stat columns, `batter_id`, and `game_date` required).
    - The app will merge in the latest rolling/stat features for each batter.
    """)
    
    uploaded_today_lineups = st.file_uploader(
        "Upload Today's Lineups/Matchups CSV (Required)", type="csv", key="todaylineup"
    )
    uploaded_sample_hist = st.file_uploader(
        "Upload Historical Event-Level CSV (Required)", type="csv", key="samplehist"
    )

    merge_btn = st.button("🔗 Generate Today's Batter Event-Level CSV (with merged rolling/stat features)")

    if merge_btn:
        if not uploaded_today_lineups or not uploaded_sample_hist:
            st.warning("Please upload BOTH today’s lineup/matchups and a sample historical event-level CSV.")
            st.stop()
        today_lineups = pd.read_csv(uploaded_today_lineups)
        hist = pd.read_csv(uploaded_sample_hist)
        # Normalize columns
        tcols = [c.strip().lower().replace(" ", "_") for c in today_lineups.columns]
        today_lineups.columns = tcols

        # Key for join is MLB ID (batter_id, mlb_id, etc)
        id_col = None
        if "mlb_id" in today_lineups.columns:
            id_col = "mlb_id"
        elif "batter_id" in today_lineups.columns:
            id_col = "batter_id"
        else:
            st.error("Could not find mlb_id/batter_id in lineups file.")
            st.stop()
        # Key for hist = 'batter_id'
        if 'batter_id' not in hist.columns:
            st.error("No 'batter_id' column in historical CSV.")
            st.stop()
        today_lineups[id_col] = today_lineups[id_col].astype(str)
        hist['batter_id'] = hist['batter_id'].astype(str)

        # Latest rolling/stat features for each batter from historical data
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
        st.info(f"Found {len(last_feats)} batters with latest rolling/stat features.")

        merged = today_lineups.merge(last_feats, left_on=id_col, right_on='batter_id', how='left', suffixes=('', '_roll'))
        # Fill required columns if missing
        context_cols = ['city', 'park', 'stadium', 'time', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        for col in context_cols:
            if col not in merged.columns:
                merged[col] = None

        st.markdown("#### Sample of Today's Event-Level CSV (with merged rolling/stat features):")
        st.dataframe(merged.head(20))
        null_report = merged[rolling_cols].isnull().sum().sort_values(ascending=False)
        st.markdown("#### Null report for rolling/stat features in output:")
        st.text(null_report.to_string())
        st.markdown("#### Weather/context columns in output:")
        st.dataframe(merged[context_cols].drop_duplicates())

        # Column order
        hist_cols = [c for c in hist.columns if not c.startswith("unnamed")]
        extra_cols = [c for c in merged.columns if c not in hist_cols]
        merged = merged.reindex(columns=hist_cols + extra_cols)
        merged_out_cols = [c for c in merged.columns if not c.startswith("unnamed")]
        st.success(f"Created today's event-level file: {len(merged)} batters. All rolling/stat columns now present (where available).")
        st.download_button(
            "⬇️ Download Today's Event-Level CSV (for Prediction App, with merged features)",
            data=merged[merged_out_cols].to_csv(index=False),
            file_name=f"event_level_today_full_{datetime.now().strftime('%Y_%m_%d')}.csv"
        )
        merged['rolling_stat_count'] = merged[rolling_cols].notnull().sum(axis=1)
        st.markdown("#### Non-null rolling/stat feature count per batter:")
        pname = 'player_name' if 'player_name' in merged.columns else merged.columns[0]
        st.dataframe(merged[[pname, 'rolling_stat_count']].sort_values('rolling_stat_count', ascending=False).head(25))

with tab2:
    st.header("Step 2: Upload Event-Level Files and Predict HRs")
    uploaded_train = st.file_uploader("Upload Historical Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
    uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, no hr_outcome)", type="csv", key="live_ev")
    col1, col2, col3 = st.columns(3)
    with col1:
        threshold_min = st.number_input("Min HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
    with col2:
        threshold_max = st.number_input("Max HR Prob Threshold", value=0.20, min_value=0.01, max_value=0.5, step=0.01)
    with col3:
        threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.01, max_value=0.10, step=0.01)

    predict_btn = st.button("🚀 Generate Today's HR Bot Picks")

    if predict_btn:
        if uploaded_train is None or uploaded_live is None:
            st.warning("Please upload BOTH files (historical and today's event-level CSV).")
            st.stop()
        train_df = pd.read_csv(uploaded_train)
        live_df = pd.read_csv(uploaded_live)

        for df in [train_df, live_df]:
            if 'batter_id' in df.columns:
                df['batter_id'] = df['batter_id'].apply(clean_id)
            elif 'batter' in df.columns:
                df['batter_id'] = df['batter'].apply(clean_id)

        # Strict feature match: only use columns present in both, excluding hr_outcome
        numeric_features = robust_numeric_columns(train_df)
        if 'hr_outcome' in numeric_features:
            numeric_features.remove('hr_outcome')
        model_features = [f for f in numeric_features if f in live_df.columns]
        # Enforce same order!
        model_features = [f for f in model_features if f in live_df.columns]
        X_train = train_df[model_features].fillna(0)
        y_train = train_df['hr_outcome'].astype(int)
        X_live = live_df[model_features].fillna(0)

        st.info(f"Model features used ({len(model_features)}): {model_features}")

        # Null report diagnostics
        st.markdown("#### Null/zero feature report (live data):")
        st.text(X_live.isnull().sum().sort_values(ascending=False).to_string())
        st.text((X_live == 0).sum().sort_values(ascending=False).to_string())

        # Train XGBoost and predict
        xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
        xgb_clf.fit(X_train, y_train)
        live_df['xgb_prob'] = xgb_clf.predict_proba(X_live)[:, 1]

        # Diagnostics: plot output histogram
        st.markdown("#### Histogram of predicted HR probabilities for today:")
        fig, ax = plt.subplots()
        ax.hist(live_df['xgb_prob'], bins=30)
        ax.set_xlabel('HR Probability')
        ax.set_ylabel('Number of Batters')
        st.pyplot(fig)

        # Table of results by threshold
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        results = []
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked.get('player_name', picked.get('batter_id')))
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

        st.success("Done! These are the official HR bot picks for today at each threshold.")

    st.markdown("""
    ---
    **Instructions for daily use:**  
    - Use Tab 1 to generate your merged one-row-per-batter CSV for today.
    - Upload with your historical event-level file (must have rolling/stat features and `hr_outcome`).
    - Bot selects *exactly* like your backtests: no hindsight, no leaderboard bias!
    """)
