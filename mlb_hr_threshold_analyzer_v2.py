import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")
st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

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

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

# -- TAB SETUP --
tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

# ===================== TAB 1 =====================
with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.markdown("##### Upload Today's Lineups/Matchups CSV (Required: mlb_id/batter_id, player_name, etc)")
    uploaded_lineups = st.file_uploader("Upload Today's Lineups/Matchups CSV", type="csv", key="todaylineup")
    st.markdown("##### Upload Historical Event-Level CSV (must have all rolling/stat features, 'batter_id', and 'game_date')")
    uploaded_hist_ev = st.file_uploader("Upload Historical Event-Level CSV", type="csv", key="eventlevel")

    if uploaded_lineups is not None and uploaded_hist_ev is not None:
        # -- Load data --
        today_lineups = pd.read_csv(uploaded_lineups)
        hist = pd.read_csv(uploaded_hist_ev)
        # Standardize column names
        today_lineups.columns = [c.strip().lower().replace(" ", "_") for c in today_lineups.columns]
        hist.columns = [c.strip().lower().replace(" ", "_") for c in hist.columns]
        # Find merge id
        id_col = "mlb_id" if "mlb_id" in today_lineups.columns else "batter_id"
        if id_col not in today_lineups.columns:
            st.error("No mlb_id or batter_id column in today's lineups.")
            st.stop()
        today_lineups[id_col] = today_lineups[id_col].astype(str)
        if "batter_id" not in hist.columns:
            st.error("No batter_id column in event-level history.")
            st.stop()
        hist["batter_id"] = hist["batter_id"].astype(str)
        # Rolling/stat feature columns in history (include all B_, P_, *_rate_20, park_hand_HR, hand splits, pitchtype splits)
        rolling_cols = [c for c in hist.columns if (
            c.startswith('b_') or c.startswith('p_') or c.endswith('_rate_20') or
            c.startswith('park_hand_hr_') or c.startswith('b_vsp_hand_hr_') or c.startswith('p_vsb_hand_hr_') or
            c.startswith('b_pitchtype_hr_') or c.startswith('p_pitchtype_hr_')
        )]
        # Use latest event per batter
        if "game_date" in hist.columns:
            hist['game_date'] = pd.to_datetime(hist['game_date'], errors='coerce')
            last_feats = hist.sort_values('game_date').groupby('batter_id').tail(1)
        else:
            last_feats = hist.groupby('batter_id').tail(1)
        last_feats = last_feats[['batter_id'] + rolling_cols].copy()
        st.info(f"Found {len(last_feats)} batters with latest rolling/stat features.")
        # Merge to lineups
        merged = today_lineups.merge(last_feats, left_on=id_col, right_on='batter_id', how='left', suffixes=('', '_roll'))
        # --- FORCE all rolling/stat columns from history to exist in today's merged, fill with NaN if missing
        for c in rolling_cols:
            if c not in merged.columns:
                merged[c] = np.nan
        # Optionally fill missing statcast context columns
        context_cols = ['city', 'park', 'stadium', 'time', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        for col in context_cols:
            if col not in merged.columns:
                merged[col] = None
        st.write("Sample of Today's Event-Level CSV (with merged rolling/stat features):")
        st.dataframe(merged.head(20))
        # --- Diagnostics ---
        # Null report for rolling/stat features
        st.markdown("#### Null report for rolling/stat features in output:")
        roll_diag = merged[rolling_cols].isnull().sum().sort_values(ascending=False)
        st.text(roll_diag.to_string())
        # Weather/context diagnostics
        st.markdown("#### Weather/context columns in output after merge:")
        st.dataframe(merged[context_cols].drop_duplicates())
        # Column order diagnostics (optional)
        sample_ev_file = st.file_uploader("Upload a Sample Historical Event-Level CSV (to align column order, optional)", type="csv", key="samplecolalign")
        if sample_ev_file:
            ref_ev = pd.read_csv(sample_ev_file, nrows=1)
            hist_cols = [c for c in ref_ev.columns if not c.lower().startswith("unnamed")]
            extra_cols = [c for c in merged.columns if c not in hist_cols]
            merged = merged.reindex(columns=hist_cols + extra_cols)
        # Download today's event-level CSV
        st.success(f"Created today's event-level file: {len(merged)} batters. All rolling/stat columns now present (where available).")
        st.download_button(
            "⬇️ Download Today's Event-Level CSV (for Prediction App, with merged features)",
            data=merged.to_csv(index=False),
            file_name=f"event_level_today_full_{datetime.now().strftime('%Y_%m_%d')}.csv"
        )
        # Extra diagnostics: show which features will be used for modeling
        st.markdown("#### Diagnostics: Columns in today's and historical event-level file:")
        st.write("Today's event-level columns:", merged.columns.tolist())
        st.write("History event-level columns:", hist.columns.tolist())

# ===================== TAB 2 =====================
with tab2:
    st.header("Upload Event-Level CSVs & Run Model")
    uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
    uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_ev")
    # Threshold controls
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
        with st.spinner("Processing..."):
            train_df = pd.read_csv(uploaded_train)
            live_df = pd.read_csv(uploaded_live)
            # --- Standardize col names
            train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
            live_df.columns = [c.strip().lower().replace(" ", "_") for c in live_df.columns]
            # --- Force all train numeric features into live, fill with NaN if missing
            for c in train_df.columns:
                if c not in live_df.columns:
                    live_df[c] = np.nan
            # Prep IDs
            for df in [train_df, live_df]:
                if 'batter_id' in df.columns:
                    df['batter_id'] = df['batter_id'].apply(clean_id)
                elif 'batter' in df.columns:
                    df['batter_id'] = df['batter'].apply(clean_id)
            # Model features: only intersection of robust numerics in both, >1 unique
            numeric_features = robust_numeric_columns(train_df)
            if 'hr_outcome' in numeric_features:
                numeric_features.remove('hr_outcome')
            model_features = [f for f in numeric_features if f in live_df.columns and live_df[f].nunique() > 1 and train_df[f].nunique() > 1]
            st.info(f"Model features used ({len(model_features)}): {model_features}")
            if len(model_features) == 0:
                st.error("No usable features found in BOTH files. Check diagnostics in Tab 1.")
                st.stop()
            # Diagnostics: show columns missing from either file
            st.write("Features in history but missing from live:", [c for c in numeric_features if c not in live_df.columns])
            st.write("Features in live but missing from history:", [c for c in live_df.columns if c not in numeric_features])
            # Show columns with only 1 unique value (single-valued)
            st.write("Features with only one unique value (train):", {c: train_df[c].nunique() for c in train_df.columns if train_df[c].nunique() == 1})
            st.write("Features with only one unique value (live):", {c: live_df[c].nunique() for c in live_df.columns if live_df[c].nunique() == 1})
            # --- Train XGBoost Model
            train_X = train_df[model_features].fillna(0)
            train_y = train_df['hr_outcome'].astype(int)
            xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
            xgb_clf.fit(train_X, train_y)
            live_X = live_df[model_features].fillna(0)
            live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]
            # Picks Per Threshold
            results = []
            thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
            for thresh in thresholds:
                mask = live_df['xgb_prob'] >= thresh
                picked = live_df.loc[mask]
                player_col = 'player_name' if 'player_name' in picked.columns else ('batter_id' if 'batter_id' in picked.columns else picked.columns[0])
                results.append({
                    'threshold': round(thresh, 3),
                    'num_picks': int(mask.sum()),
                    'picked_players': list(picked[player_col])
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
**Instructions:**  
- Tab 1: Upload BOTH today's lineups/matchups and historical event-level file, download merged file for prediction.
- Tab 2: Upload event-level files, run bot, check full diagnostics if features are missing!
""")
