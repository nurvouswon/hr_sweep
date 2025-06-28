import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
import warnings

warnings.filterwarnings("ignore")

# ========== WEATHER SCORING FUNCTION ==========
def score_weather(row):
    """Turn weather columns into a weather HR boost/reduce score"""
    # Example rules (tune as you collect more outcome data!)
    temp_score = (row.get('temp', 70) - 70) * 0.01  # every 10F above 70 = +0.1
    wind_score = 0
    if 'wind_dir_string' in row and isinstance(row['wind_dir_string'], str):
        if 'O CF' in row['wind_dir_string']:  # Out to center
            wind_score += row.get('wind_mph', 0) * 0.03
        elif 'O RF' in row['wind_dir_string'] or 'O LF' in row['wind_dir_string']:
            wind_score += row.get('wind_mph', 0) * 0.02
        elif 'I' in row['wind_dir_string']:
            wind_score -= row.get('wind_mph', 0) * 0.03
    humidity_score = -0.02 * (row.get('humidity', 50) - 50) / 10
    # You can further calibrate by studying outcome data.
    return temp_score + wind_score + humidity_score

# ========== DATA CLEANING ==========
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def cast_string_numbers(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass
    return df

def clean_ids(df):
    # Remove .0, convert to string for ID columns
    for id_col in ['batter_id', 'pitcher_id']:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str).str.replace('.0','',regex=False).str.strip()
    return df

# ========== APP LAYOUT ==========
st.set_page_config("MLB HR Predictor (Tab 2)", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

uploaded_event = st.file_uploader("Upload Event-Level CSV for Training (required)", type="csv", key="eventcsv")
uploaded_today = st.file_uploader("Upload TODAY CSV for Prediction (required)", type="csv", key="todaycsv")
run_btn = st.button("Train + Predict", type="primary")

if run_btn and uploaded_event and uploaded_today:
    # === LOAD AND CLEAN DATA ===
    st.markdown("#### Loading & cleaning data...")
    event_df = pd.read_csv(uploaded_event, low_memory=False)
    today_df = pd.read_csv(uploaded_today, low_memory=False)
    event_df = dedup_columns(event_df)
    today_df = dedup_columns(today_df)
    event_df = cast_string_numbers(event_df)
    today_df = cast_string_numbers(today_df)
    event_df = clean_ids(event_df)
    today_df = clean_ids(today_df)

    # ========== WEATHER SCORE ==========
    st.markdown("#### Scoring weather for training and today rows...")
    event_df['weather_score'] = event_df.apply(score_weather, axis=1)
    today_df['weather_score'] = today_df.apply(score_weather, axis=1)

    # ========== SELECT FEATURES ==========
    st.markdown("#### Engineering features...")
    # Drop IDs, names, direct HR target (except y)
    non_feature_cols = [
        'game_date', 'batter_id', 'player_name', 'pitcher_id',
        'hr_outcome', 'events_clean'
    ]
    # Find all rolling features and park/context features
    rolling_features = [c for c in event_df.columns if (
        c.startswith('b_') or c.startswith('p_')
    )]
    park_context = ['park_hr_rate','park_altitude','roof_status','city','park_hand_hr_rate',
                    'pitchtype_hr_rate','pitchtype_hr_rate_hand','platoon_hr_rate']
    weather_feats = ['temp','humidity','wind_mph','wind_dir_string','condition','weather_score']

    # Make sure we only use numerical features for ML
    ml_features = []
    for col in rolling_features + park_context + weather_feats:
        if col in event_df.columns and pd.api.types.is_numeric_dtype(event_df[col]):
            ml_features.append(col)
        elif col in event_df.columns and event_df[col].dtype == object:
            # Try to label encode for ML
            try:
                event_df[col], uniques = pd.factorize(event_df[col])
                today_df[col] = today_df[col].map({v: k for k, v in enumerate(uniques)})
                ml_features.append(col)
            except:
                pass

    # Ensure we have the HR outcome
    if "hr_outcome" not in event_df.columns:
        st.error("Event-level CSV must have `hr_outcome` column for training.")
        st.stop()
    X = event_df[ml_features].fillna(0)
    y = event_df["hr_outcome"].fillna(0).astype(int)

    # ========== SPLIT DATA ==========
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # ========== DEFINE MODELS ==========
    st.markdown("#### Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
    models = [
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)),
        ('lgbm', LGBMClassifier(n_jobs=-1, random_state=42)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1, weights=[5,5,5,2,2,1])

    # ========== FIT & VALIDATE ==========
    ensemble.fit(X_train, y_train)
    val_preds = ensemble.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, val_preds)
    ll = log_loss(y_val, val_preds)
    st.markdown(f"**Validation AUC:** {auc:.4f} — **LogLoss:** {ll:.4f}")

    # ========== PREDICT TODAY ==========
    st.markdown("#### Predicting HR probability for today...")
    X_today = today_df[ml_features].fillna(0)
    today_df['hr_pred_proba'] = ensemble.predict_proba(X_today)[:,1]

    # ========== SHOW + DOWNLOAD ==========
    out_cols = [
        'game_date', 'batter_id', 'player_name', 'pitcher_id', 'park', 'temp',
        'humidity', 'wind_mph', 'condition', 'wind_dir_string', 'weather_score',
        'hr_pred_proba'
    ]
    extra_out_cols = [c for c in today_df.columns if c not in out_cols]
    today_out = today_df[out_cols + extra_out_cols]
    st.dataframe(today_out.sort_values("hr_pred_proba", ascending=False).head(20))

    st.download_button(
        "⬇️ Download Full HR Predictions (TODAY)",
        data=today_out.to_csv(index=False),
        file_name="today_hr_predictions.csv",
        key="download_today_preds"
    )

    # ========== EXPLAINABILITY (SHAP) ==========
    st.markdown("#### Model explainability (feature importance):")
    explainer = shap.Explainer(models[0][1], X_train)  # SHAP for XGBoost
    shap_vals = explainer(X_val[:200])
    st.pyplot(shap.summary_plot(shap_vals, X_val[:200], show=False, plot_size=(12,6)))

    st.success("Done! You can now download predictions and review feature impacts.")

else:
    st.info("Upload event-level and TODAY CSVs, then click Train + Predict to generate predictions.")
