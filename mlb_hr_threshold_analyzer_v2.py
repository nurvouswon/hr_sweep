import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# ------------------- Helper Functions -------------------

def normalize_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def compute_weather_score(row):
    # Weather score for interpretability: higher is more HR friendly
    # 1. Temp > 85 is good, <65 is bad
    # 2. Wind out >10 mph is good, in >10 mph is bad
    # 3. Humidity 45-65 optimal, <30 or >80 bad
    score = 0
    temp = row.get('temp', np.nan)
    humidity = row.get('humidity', np.nan)
    wind_mph = row.get('wind_mph', np.nan)
    wind_dir = str(row.get('wind_dir_string', '')).lower()
    condition = str(row.get('condition', '')).lower()
    
    # Temperature scoring
    if not pd.isna(temp):
        if temp >= 85: score += 2
        elif temp >= 75: score += 1
        elif temp <= 60: score -= 2
        elif temp <= 70: score -= 1
    
    # Humidity scoring
    if not pd.isna(humidity):
        if 45 <= humidity <= 65: score += 1
        elif humidity < 30 or humidity > 80: score -= 1

    # Wind scoring: O CF = out to center, I CF = in from center
    if not pd.isna(wind_mph) and wind_dir:
        if 'o cf' in wind_dir or 'out' in wind_dir:
            if wind_mph >= 10: score += 2
            elif wind_mph >= 5: score += 1
        elif 'i cf' in wind_dir or 'in' in wind_dir:
            if wind_mph >= 10: score -= 2
            elif wind_mph >= 5: score -= 1

    # Condition (outdoor better for HR than indoor, but weather is key)
    if 'indoor' in condition: score -= 1
    elif 'outdoor' in condition: score += 0.5

    # Scale to 0-10
    score = min(max(score + 5, 0), 10)
    return score

def clean_X(df, train_cols=None):
    # Make all columns numeric where possible, fill missing, align columns
    for c in df.columns:
        if hasattr(df[c], 'dtype') and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)
    if train_cols is not None:
        # Align columns exactly
        df = df.reindex(columns=train_cols, fill_value=0)
    return df

# ------------------- Streamlit App -------------------

st.set_page_config("MLB HR Predictor — Deep Ensemble", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

st.markdown("**Upload Event-Level CSV for Training (required)**")
event_file = st.file_uploader("Upload event-level HR features CSV", type="csv", key="ev")
st.markdown("**Upload TODAY CSV for Prediction (required)**")
today_file = st.file_uploader("Upload TODAY HR features CSV", type="csv", key="td")

if event_file and today_file:
    progress = st.progress(0, "Loading & cleaning data...")
    # Load data
    event_df = pd.read_csv(event_file)
    today_df = pd.read_csv(today_file)
    event_df = normalize_cols(event_df)
    today_df = normalize_cols(today_df)
    event_df = dedup_columns(event_df)
    today_df = dedup_columns(today_df)

    progress.progress(10, "Scoring weather for training and today rows...")
    # Weather Score
    event_df['weather_score'] = event_df.apply(compute_weather_score, axis=1)
    today_df['weather_score'] = today_df.apply(compute_weather_score, axis=1)

    progress.progress(20, "Engineering features...")
    # Select features
    drop_cols = [
        'game_date','batter_id','player_name','pitcher_id','park','city','roof_status','condition',
        'wind_dir_string','events','events_clean','description','team_code',
        'stand','p_throws','player_name','park','city','inning_topbot','inning','batting_order'
    ]
    # Only keep numeric/stat columns + weather_score
    feature_cols = [
        c for c in event_df.columns
        if c not in drop_cols and event_df[c].dtype in [np.float64, np.float32, np.int64, np.int32] and not c.endswith("_id")
    ] + ['weather_score']

    # Check for missing feature columns before proceeding
    missing_cols = [c for c in feature_cols if c not in event_df.columns]
    if missing_cols:
        st.warning(f"Missing columns in event_df: {missing_cols}")
        st.stop()

    target_col = 'hr_outcome' if 'hr_outcome' in event_df.columns else None
    if target_col is None:
        st.error("No 'hr_outcome' column in event-level CSV!")
        st.stop()

    # X/y and today features
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]

    # For today, check missing feature columns too:
    missing_cols_today = [c for c in feature_cols if c not in today_df.columns]
    if missing_cols_today:
        st.warning(f"Missing columns in TODAY CSV: {missing_cols_today}")
        st.stop()
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    progress.progress(25, "Splitting for validation...")
    # Validation
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    progress.progress(35, "Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
    # Classifiers
    xgb_clf = xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, n_jobs=-1, verbosity=0, random_state=1, use_label_encoder=False, eval_metric="logloss")
    lgb_clf = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.1, n_jobs=-1, random_state=2)
    cb_clf = cb.CatBoostClassifier(iterations=120, learning_rate=0.1, verbose=0, random_state=3)
    rf_clf = RandomForestClassifier(n_estimators=70, n_jobs=-1, random_state=4)
    gb_clf = GradientBoostingClassifier(n_estimators=70, learning_rate=0.07, random_state=5)
    lr_clf = LogisticRegression(max_iter=350, random_state=6)
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf), ("lgb", lgb_clf), ("cb", cb_clf),
            ("rf", rf_clf), ("gb", gb_clf), ("lr", lr_clf)
        ],
        voting="soft", weights=[2,2,2,1,1,1]
    )

    # Train
    for clf in [xgb_clf, lgb_clf, cb_clf, rf_clf, gb_clf, lr_clf]:
        clf.fit(X_tr, y_tr)
    ensemble.fit(X_tr, y_tr)

    # Validation metrics
    val_preds = ensemble.predict_proba(X_val)[:,1]
    val_auc = roc_auc_score(y_val, val_preds)
    val_logloss = log_loss(y_val, val_preds)
    st.success(f"Validation AUC: **{val_auc:.4f}** — LogLoss: **{val_logloss:.4f}**")

    progress.progress(70, "Predicting HR probability for today...")

    # Predict for today
    today_df['hr_probability'] = ensemble.predict_proba(X_today)[:,1]

    # Feature importances (from main tree models)
    all_feature_importances = {}
    for name, clf in [
        ("XGBoost", xgb_clf), ("LightGBM", lgb_clf), ("CatBoost", cb_clf),
        ("RandomForest", rf_clf), ("GB", gb_clf)
    ]:
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            all_feature_importances[name] = pd.Series(importances, index=X.columns)
    # Mean importance over ensemble
    if all_feature_importances:
        importances_df = pd.DataFrame(all_feature_importances)
        importances_df["mean"] = importances_df.mean(axis=1)
        importances_df = importances_df.sort_values("mean", ascending=False)
        top_feats = importances_df.head(30)
    else:
        top_feats = pd.DataFrame()

    progress.progress(90, "Formatting output leaderboard...")

    # Output: leaderboard
    out_cols = [
        "player_name", "batter_id", "pitcher_id", "game_date",
        "park", "city", "temp", "humidity", "wind_mph", "wind_dir_string",
        "weather_score", "hr_probability"
    ]
    out_cols += [c for c in today_df.columns if c.startswith("b_avg_exit_velo")][:2]  # Example: add some Statcast features
    leaderboard = today_df.copy()
    for c in out_cols:
        if c not in leaderboard.columns:
            leaderboard[c] = np.nan
    leaderboard = leaderboard[out_cols]
    leaderboard = leaderboard.sort_values("hr_probability", ascending=False)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["weather_score"] = leaderboard["weather_score"].round(1)
    leaderboard = leaderboard.reset_index(drop=True)

    st.markdown("### 🏆 **Today's HR Leaderboard**")
    st.dataframe(leaderboard.head(30), use_container_width=True)
    st.download_button(
        "⬇️ Download Full Predictions CSV",
        data=leaderboard.to_csv(index=False),
        file_name="today_hr_predictions.csv",
        key="dl_out"
    )

    st.markdown("### 🌤️ **Top 30 Feature Importances (Ensemble Mean)**")
    if not top_feats.empty:
        st.dataframe(top_feats[["mean"]])
        fig, ax = plt.subplots(figsize=(6,12))
        top_feats["mean"].iloc[::-1].plot(kind='barh', ax=ax)
        ax.set_xlabel("Mean Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 30 Features (Ensemble)")
        st.pyplot(fig)
    else:
        st.info("No feature importance could be calculated.")

    progress.progress(100, "All done!")

else:
    st.info("Please upload both an event-level CSV (for training) and a TODAY CSV (for predictions).")
