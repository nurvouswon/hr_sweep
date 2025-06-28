import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# For boosting models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

# --- HELPER FUNCTIONS ---
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def force_numeric(df):
    # Converts every column that can be to float, else keeps original
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
    return df

def clean_X(df, train_cols=None):
    # Remove columns with all nans, force numeric, sort
    df = dedup_columns(df)
    df = force_numeric(df)
    df = df.fillna(0)
    if train_cols is not None:
        # Align columns to match train_cols, fill missing with 0
        for c in train_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[train_cols]
    return df

def compute_weather_score(row):
    # Simple interpretable 1-10 scale based on temp, wind_mph, humidity, condition, wind_dir_string
    score = 5.0  # start neutral
    # Temp bonus (ideal: 75-90)
    if not np.isnan(row['temp']):
        if row['temp'] >= 85:
            score += 1.5
        elif row['temp'] >= 75:
            score += 1
        elif row['temp'] >= 65:
            score += 0.5
        elif row['temp'] < 55:
            score -= 1
    # Wind bonus
    if not np.isnan(row['wind_mph']):
        if row['wind_mph'] >= 15:
            score += 1
        elif row['wind_mph'] >= 8:
            score += 0.5
    # Humidity bonus (higher = more carry)
    if not np.isnan(row['humidity']):
        if row['humidity'] >= 60:
            score += 0.5
        elif row['humidity'] < 40:
            score -= 0.5
    # Condition: outdoor > indoor
    if isinstance(row['condition'], str):
        if "out" in row['condition'].lower():
            score += 0.5
    # Wind direction: if "O CF" or contains "CF" boost (out to center)
    if isinstance(row['wind_dir_string'], str) and "CF" in row['wind_dir_string']:
        score += 0.5
    # Clamp to 1–10 for sanity
    return np.clip(score, 1, 10)

# --- UPLOAD DATA ---
st.markdown("### Upload Event-Level CSV for Training (required)")
event_csv = st.file_uploader("Upload event-level HR features CSV", type="csv", key="eventcsv")
st.markdown("### Upload TODAY CSV for Prediction (required)")
today_csv = st.file_uploader("Upload TODAY HR features CSV", type="csv", key="todaycsv")

progress = st.empty()

if event_csv and today_csv:
    progress.progress(2, "Loading & cleaning data...")

    event_df = pd.read_csv(event_csv)
    today_df = pd.read_csv(today_csv)

    # Dedup/clean columns, handle string numerics
    event_df = dedup_columns(event_df)
    today_df = dedup_columns(today_df)
    # String id normalization
    for idcol in ['batter_id', 'pitcher_id']:
        for df in [event_df, today_df]:
            if idcol in df.columns:
                df[idcol] = df[idcol].astype(str).str.replace('.0','',regex=False).str.strip()

    # --- Compute weather scores for both sets ---
    progress.progress(10, "Scoring weather for training and today rows...")
    for df in [event_df, today_df]:
        for col in ['temp', 'wind_mph', 'humidity']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    for df in [event_df, today_df]:
        df['weather_score'] = df.apply(compute_weather_score, axis=1)

    # --- Select Feature Columns ---
    progress.progress(15, "Engineering features...")

    # Target column
    target_col = "hr_outcome" if "hr_outcome" in event_df.columns else "hr_flag"

    # Drop string/object columns (except id/player)
    drop_obj = []
    for c in event_df.columns:
        if event_df[c].dtype == 'O' and c not in ['batter_id', 'pitcher_id', 'player_name']:
            drop_obj.append(c)
    feature_cols = [c for c in event_df.columns if (c not in drop_obj) and c not in [
        'game_date', 'player_name', 'batter_id', 'pitcher_id', 'hr_outcome', 'hr_flag', 'events_clean'
    ]]

    # Filter to those also present in TODAY
    feature_cols = [c for c in feature_cols if c in today_df.columns]

    # --- X/y and today features ---
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    progress.progress(25, "Splitting for validation...")

    # --- Train/Test Split ---
    X, y = shuffle(X, y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

    progress.progress(40, "Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
    # --- Ensemble Model Definitions ---
    # XGBoost
    xgb_clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.11, max_depth=5, subsample=0.82, colsample_bytree=0.91, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.12, max_depth=6, subsample=0.80, colsample_bytree=0.93, random_state=42, n_jobs=-1)
    # CatBoost
    cat_clf = cb.CatBoostClassifier(iterations=230, learning_rate=0.13, depth=6, verbose=0, random_state=42)
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1, random_state=42)
    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=80, learning_rate=0.17, max_depth=3, random_state=42)
    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=400, C=1.3, solver='lbfgs', random_state=42, n_jobs=-1)

    # --- Fit all models ---
    xgb_clf.fit(X_train, y_train)
    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    gb_clf.fit(X_train, y_train)
    lr_clf.fit(X_train, y_train)

    # --- Voting Ensemble ---
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
            ("cat", cat_clf),
            ("rf", rf_clf),
            ("gb", gb_clf),
            ("lr", lr_clf)
        ],
        voting='soft', n_jobs=-1, flatten_transform=True, weights=[2,2,2,1,1,1]
    )
    ensemble.fit(X_train, y_train)

    # --- Validation ---
    progress.progress(70, "Validating...")
    val_preds = ensemble.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    loss = log_loss(y_val, val_preds)
    st.success(f"Validation AUC: {auc:.4f} — LogLoss: {loss:.4f}")

    # --- Feature Importances (XGBoost, as primary reference) ---
    progress.progress(75, "Computing top feature importances...")
    imp_vals = xgb_clf.feature_importances_
    top_imp = pd.Series(imp_vals, index=X.columns).sort_values(ascending=False).head(30)

    st.markdown("#### 🎯 **Top 30 Feature Importances (XGBoost):**")
    fig, ax = plt.subplots(figsize=(6, 12))
    top_imp[::-1].plot(kind='barh', ax=ax, color='darkblue')
    ax.set_xlabel("Importance (XGBoost)")
    st.pyplot(fig)

    # --- Predict for today ---
    progress.progress(90, "Predicting HR probability for today...")
    today_df['pred_hr_prob'] = ensemble.predict_proba(X_today)[:, 1]

    # --- Show output ---
    display_cols = [
        'batter_id', 'player_name', 'pred_hr_prob', 'weather_score',
        'park', 'pitcher_id', 'temp', 'humidity', 'wind_mph', 'condition'
    ]
    display_cols = [c for c in display_cols if c in today_df.columns]
    out_df = today_df[display_cols].sort_values("pred_hr_prob", ascending=False).reset_index(drop=True)
    st.markdown("### 📈 **Top Predicted HR Probabilities — With Weather Score**")
    st.dataframe(out_df.head(30).style.format({'pred_hr_prob': "{:.3f}", 'weather_score': "{:.1f}"}))

    st.markdown("##### Download full predictions as CSV:")
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=today_df.sort_values('pred_hr_prob', ascending=False).to_csv(index=False),
        file_name="today_hr_predictions.csv",
        key="download_today_preds"
    )

    st.success("All done! Predictions, feature importances, and weather scores are ready.")

else:
    st.info("Upload both an event-level training CSV and a today CSV to run predictions.")
