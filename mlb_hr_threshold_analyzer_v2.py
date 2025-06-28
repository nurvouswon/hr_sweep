import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

st.set_page_config("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score", layout="wide")

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    drop_cols = []
    for c in df.columns:
        # Remove columns with dict/list or non-numeric types
        if df[c].dtype == 'O' or df[c].dtype.name == 'category':
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                drop_cols.append(c)
        if df[c].apply(lambda x: isinstance(x, (dict, list))).any():
            drop_cols.append(c)
    df = df.drop(columns=drop_cols)
    if train_cols is not None:
        df = df.reindex(columns=train_cols, fill_value=0)
    return df

def basic_weather_score(row):
    """Sample weather scoring:  
    Favorable HR: warm (75+), moderate wind out (10+ mph), low humidity.  
    This can be replaced with data-driven or ML-imputed scoring."""
    score = 0
    # Temp
    if not pd.isna(row.get("temp", np.nan)):
        if row["temp"] >= 90:
            score += 1.2
        elif row["temp"] >= 80:
            score += 1
        elif row["temp"] >= 70:
            score += 0.7
        elif row["temp"] >= 60:
            score += 0.3
        else:
            score -= 0.2
    # Wind
    if not pd.isna(row.get("wind_mph", np.nan)):
        if row["wind_mph"] >= 15:
            score += 0.8
        elif row["wind_mph"] >= 10:
            score += 0.5
        elif row["wind_mph"] >= 5:
            score += 0.2
        else:
            score += 0
    # Humidity
    if not pd.isna(row.get("humidity", np.nan)):
        if row["humidity"] <= 30:
            score += 0.6
        elif row["humidity"] <= 50:
            score += 0.4
        elif row["humidity"] <= 70:
            score += 0.2
        else:
            score -= 0.1
    # Condition (favor 'outdoor', sunny, etc)
    cond = str(row.get("condition", "")).lower()
    if "outdoor" in cond or "sunny" in cond:
        score += 0.25
    if "indoor" in cond or "rain" in cond:
        score -= 0.25
    # Wind dir (favor Out, CF, RF for lefties, etc. -- here: if outfield wind, boost)
    wind_dir = str(row.get("wind_dir_string", "")).lower()
    if any(x in wind_dir for x in ["o cf", "out", "cf"]):
        score += 0.2
    return np.round(score, 3)

# UI
st.title("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score")
st.write("Upload your **event-level CSV for training** and **TODAY CSV for prediction**. This app will automatically clean, train a powerful ensemble (XGBoost, LightGBM, CatBoost, RF, GB, LR), and predict today's home run probabilities. A weather adjustment is applied to each row for extra accuracy.")

event_file = st.file_uploader("Upload Event-Level CSV for Training (required)", type="csv", key="eventcsvup")
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type="csv", key="todaycsvup")

if event_file and today_file:
    progress = st.progress(0, "Loading & cleaning data...")
    # Load
    event_df = pd.read_csv(event_file, low_memory=False)
    today_df = pd.read_csv(today_file, low_memory=False)

    progress.progress(5, "Scoring weather for training and today rows...")
    # Score weather
    for df in [event_df, today_df]:
        df['weather_score'] = df.apply(basic_weather_score, axis=1)
    # Dedup, clean ids/names
    for df in [event_df, today_df]:
        df = dedup_columns(df)
        for c in ['batter_id', 'pitcher_id']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace('.0','',regex=False).str.strip()
        for c in ['game_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce').dt.strftime("%Y-%m-%d")
    progress.progress(15, "Engineering features...")
    # Set up features
    target_col = "hr_outcome"
    drop_feats = [
        "hr_outcome", "events", "events_clean", "game_date",
        "batter_id", "player_name", "pitcher_id", "team_code", "city", "park"
    ]
    # Only use numeric columns, plus weather_score
    feature_cols = [c for c in event_df.columns if c not in drop_feats and
                    (event_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int, bool] or c == "weather_score")]

    # X/y and today features
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    progress.progress(25, "Splitting for validation...")
    # Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.16, random_state=42, stratify=y)
    progress.progress(35, "Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")

    # Silence model warnings for demo UX
    warnings.filterwarnings("ignore")
    models = [
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=42)),
        ("lgbm", LGBMClassifier(verbose=-1, n_jobs=-1, random_state=42)),
        ("cat", CatBoostClassifier(verbose=0, thread_count=-1, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("lr", LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1, random_state=42))
    ]
    # Voting ensemble (soft = proba average)
    ensemble = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
    ensemble.fit(X_train, y_train)
    y_val_pred = ensemble.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    progress.progress(75, f"Validation AUC: {auc:.4f} — LogLoss: {ll:.4f}")

    st.success(f"Validation AUC: {auc:.4f} — LogLoss: {ll:.4f}")

    # Predict today's
    progress.progress(90, "Predicting HR probability for today...")
    today_df['hr_pred_proba'] = ensemble.predict_proba(X_today)[:,1]
    # Optionally apply weather adjustment to prob (e.g., simple linear scale — tune as desired)
    today_df['hr_pred_proba_wx'] = np.clip(today_df['hr_pred_proba'] * (1 + today_df['weather_score'] * 0.25), 0, 1)

    progress.progress(100, "All Done! Download predictions below.")

    # Output table
    show_cols = [
        "game_date", "batter_id", "player_name", "pitcher_id", "hr_pred_proba", "weather_score", "hr_pred_proba_wx"
    ] + [c for c in today_df.columns if c.startswith("b_")][:10]  # limit batted ball stats for view
    st.markdown("### Today's Home Run Probability Predictions")
    st.dataframe(today_df[show_cols].sort_values("hr_pred_proba_wx", ascending=False).head(30))
    st.download_button(
        "⬇️ Download Full Predictions CSV",
        data=today_df.to_csv(index=False),
        file_name="today_hr_predictions.csv",
        key="download_preds"
    )
else:
    st.info("Upload both the event-level CSV (for training) and TODAY CSV (for prediction).")

st.markdown("""
---
**Notes**  
- Weather scoring is an *additional* adjustment. You can tune `basic_weather_score` as you gather outcome data.
- The ensemble uses XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, and Logistic Regression.
- If you want feature importances, validation charts, or SHAP explainability, let me know!
""")
