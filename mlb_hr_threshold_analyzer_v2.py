import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import gc
from datetime import datetime, timedelta

# ========== CONTEXT MAPS ==========
park_hr_rate_map = {...}  # (same as above)
park_altitude_map = {...}
roof_status_map = {...}
team_code_to_park = {...}
mlb_team_city_map = {...}
park_hand_hr_rate_map = {...}

# ========== UTILS ==========
def dedup_columns(df): return df.loc[:, ~df.columns.duplicated()]

def parse_custom_weather_string_v2(s):
    # ... (use exact function as above for weather parsing)

def downcast_numeric(df):
    for col in df.select_dtypes(include=['float']): df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']): df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

@st.cache_data(show_spinner=True)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    # ... (use optimized version as above)

def preprocess_inputs(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]

# ========== ML MODEL UTILS ==========
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

@st.cache_resource
def load_ml_models(X_train, y_train):
    # For reproducibility and efficiency, train all models on the same train set (or load if you have pickles)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=120, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=120, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=120, verbose=0, random_seed=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def predict_hr_proba(models, X):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(X)[:, 1]
    preds['HR_Prob_Avg'] = np.mean([preds[m] for m in models], axis=0)
    return preds

# ========== STREAMLIT UI ==========
st.set_page_config("MLB HR Predictor", layout="wide")
tab1, tab2 = st.tabs(["1️⃣ Feature Engineering", "2️⃣ HR Prediction"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    uploaded_lineups = st.file_uploader("Upload Today's Matchups/Lineups CSV", type="csv", key="lineupsup")
    fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
    progress = st.empty()

    if fetch_btn and uploaded_lineups is not None:
        import pybaseball
        from pybaseball import statcast

        progress.progress(3, "Fetching Statcast data...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, "Loaded Statcast")
        st.write(f"Loaded {len(df)} raw Statcast events.")
        if len(df) == 0:
            st.error("No data! Try different dates.")
            st.stop()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        progress.progress(12, "Loaded and formatted Statcast columns.")

        # --- Read and clean lineups ---
        try:
            lineup_df = pd.read_csv(uploaded_lineups)
        except Exception as e:
            st.error(f"Could not read lineup CSV: {e}")
            st.stop()
        lineup_df.columns = [str(c).strip().lower().replace(" ", "_") for c in lineup_df.columns]
        if "park" in lineup_df.columns:
            lineup_df["park"] = lineup_df["park"].astype(str).str.lower().str.replace(" ", "_")
        for col in ['mlb_id', 'batter_id']:
            if col in lineup_df.columns and 'batter_id' not in lineup_df.columns:
                lineup_df['batter_id'] = lineup_df[col]
        for col in ['player_name', 'player name', 'name']:
            if col in lineup_df.columns and 'player_name' not in lineup_df.columns:
                lineup_df['player_name'] = lineup_df[col]
        if 'game_date' not in lineup_df.columns:
            for date_col in ['game_date', 'game date']:
                if date_col in lineup_df.columns:
                    lineup_df['game_date'] = lineup_df[date_col]
        if 'game_date' in lineup_df.columns:
            lineup_df['game_date'] = pd.to_datetime(lineup_df['game_date'], errors='coerce').dt.strftime("%Y-%m-%d")
        if 'batting_order' in lineup_df.columns:
            lineup_df['batting_order'] = lineup_df['batting_order'].astype(str).str.upper().str.strip()
        if 'team_code' in lineup_df.columns:
            lineup_df['team_code'] = lineup_df['team_code'].astype(str).str.strip().str.upper()
        if 'game_number' in lineup_df.columns:
            lineup_df['game_number'] = lineup_df['game_number'].astype(str).str.strip()
        for col in ['batter_id', 'mlb_id']:
            if col in lineup_df.columns:
                lineup_df[col] = lineup_df[col].astype(str).str.replace('.0','',regex=False).str.strip()
        if 'weather' in lineup_df.columns:
            wx_parsed = lineup_df['weather'].apply(parse_custom_weather_string_v2)
            lineup_df = pd.concat([lineup_df, wx_parsed], axis=1)

        # ==== Assign Opposing SP for Each Batter (robust, fallback for missing game_number) ====
        lineup_df['pitcher_id'] = np.nan
        grouped = lineup_df.groupby(['game_date', 'park', 'time'])
        for (gdate, park, time_), group in grouped:
            if 'team_code' not in group.columns: continue
            teams = group['team_code'].unique()
            if len(teams) < 2: continue
            team_sps = {}
            for team in teams:
                sp_row = group[(group['team_code'] == team) & (group['batting_order'] == "SP")]
                if not sp_row.empty:
                    team_sps[team] = str(sp_row.iloc[0]['batter_id'])
            for team in teams:
                opp_teams = [t for t in teams if t != team]
                if not opp_teams: continue
                opp_sp = team_sps.get(opp_teams[0], np.nan)
                idx = group[group['team_code'] == team].index
                lineup_df.loc[idx, 'pitcher_id'] = opp_sp

        # ==== STATCAST ENGINEERING ====
        for col in ['batter_id', 'mlb_id', 'pitcher_id', 'team_code']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.0','',regex=False).str.strip()

        # Park/city/context
        if 'home_team_code' in df.columns:
            df['team_code'] = df['home_team_code'].str.upper()
            df['park'] = df['home_team_code'].str.lower().str.replace(' ', '_')
        if 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')
        if 'team_code' not in df.columns and 'park' in df.columns:
            park_to_team = {v:k for k,v in team_code_to_park.items()}
            df['team_code'] = df['park'].map(park_to_team).str.upper()
        df['team_code'] = df['team_code'].astype(str).str.upper()
        df['park'] = df['team_code'].map(team_code_to_park).str.lower()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        df['city'] = df['team_code'].map(mlb_team_city_map).fillna("")
        if 'events' in df.columns:
            df['events_clean'] = df['events'].astype(str).str.lower().str.replace(' ', '')
        else:
            df['events_clean'] = ""
        if 'hr_outcome' not in df.columns:
            df['hr_outcome'] = df['events_clean'].isin(['homerun', 'home_run']).astype(int)
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        df = df[df['events_clean'].isin(valid_events)].copy()
        roll_windows = [3, 5, 7, 14, 20]
        main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]
        for col in ['batter', 'batter_id']:
            if col in df.columns:
                df['batter_id'] = df[col]
        for col in ['pitcher', 'pitcher_id']:
            if col in df.columns:
                df['pitcher_id'] = df[col]

        batter_event = fast_rolling_stats(df, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="b_")
        if not batter_event.empty:
            batter_event = batter_event.set_index('batter_id')
        df_for_pitchers = df.copy()
        if 'batter_id' in df_for_pitchers.columns:
            df_for_pitchers = df_for_pitchers.drop(columns=['batter_id'])
        df_for_pitchers = df_for_pitchers.rename(columns={"pitcher_id": "batter_id"})
        pitcher_event = fast_rolling_stats(
            df_for_pitchers, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="p_"
        )
        if not pitcher_event.empty:
            pitcher_event = pitcher_event.set_index('batter_id')
        df = pd.merge(df, batter_event.reset_index(), how="left", left_on="batter_id", right_on="batter_id")
        df = pd.merge(df, pitcher_event.reset_index(), how="left", left_on="pitcher_id", right_on="batter_id", suffixes=('', '_pitcherstat'))
        if 'batter_id_pitcherstat' in df.columns:
            df = df.drop(columns=['batter_id_pitcherstat'])
        df = dedup_columns(df)

        if 'stand' in df.columns and 'park' in df.columns:
            df['park_hand_hr_rate'] = [
                park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(stand).upper(), 1.0)
                for park, stand in zip(df['park'], df['stand'])
            ]
        else:
            df['park_hand_hr_rate'] = 1.0
        df = downcast_numeric(df)

        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV / Parquet (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download Event-Level CSV",
            data=df.to_csv(index=False),
            file_name="event_level_hr_features.csv",
            key="download_event_level"
        )
        event_parquet = io.BytesIO()
        df.to_parquet(event_parquet, index=False)
        st.download_button(
            "⬇️ Download Event-Level Parquet",
            data=event_parquet.getvalue(),
            file_name="event_level_hr_features.parquet",
            mime="application/octet-stream",
            key="download_event_level_parquet"
        )

        progress.progress(100, "All complete.")
        del df, batter_event, pitcher_event, event_parquet
        gc.collect()
    else:
        st.info("Upload a Matchups/Lineups CSV and select a date range to generate the event-level CSVs.")

with tab2:
    st.header("MLB HR Probability Predictor (6 Model Ensemble)")
    st.markdown("Upload **event-level feature CSV** with outcome label `hr_outcome`.")
    data_file = st.file_uploader("Upload Engineered Event-Level CSV", type=['csv'], key="upload_csv2")
    run_btn = st.button("Run HR Probability Prediction", type="primary")

    if run_btn and data_file is not None:
        df = pd.read_csv(data_file)
        feature_cols = [c for c in df.columns if (c.startswith('b_') or c.startswith('p_')) and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]]
        # Basic checks
        if 'hr_outcome' in df.columns:
            y = df['hr_outcome']
        else:
            st.error("CSV must include 'hr_outcome' column for training.")
            st.stop()
        X = preprocess_inputs(df, feature_cols)
        # Label encode any object/categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean(numeric_only=True))

        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        models = load_ml_models(X_train, y_train)
        # Predict
        preds = predict_hr_proba(models, X_test)
        out_df = X_test.copy()
        out_df['HR_Prob_Avg'] = preds['HR_Prob_Avg']
        for name in models:
            out_df[f'HR_Prob_{name}'] = preds[name]
        out_df['hr_outcome'] = y_test.values
        st.markdown("#### Sample Output")
        st.dataframe(out_df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download HR Probability Predictions",
            data=out_df.to_csv(index=False),
            file_name="hr_probability_predictions.csv",
            key="download_hr_preds"
        )
