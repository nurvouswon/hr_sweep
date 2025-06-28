import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

@st.cache_data(show_spinner=False)
def dedup_columns(df):
    # Remove duplicate columns (keep first)
    return df.loc[:, ~df.columns.duplicated()]

def fix_types(df):
    # Robust float/int/str merge (one-liner for all)
    for col in df.columns:
        # If all nan, skip
        if df[col].isnull().all(): continue
        # If any string that looks numeric, convert
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except: pass
        # If float but all whole numbers, convert to int
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def score_weather(row):
    # Weather scoring: more HRs at high temp, lower humidity, wind out, outdoor
    temp = row.get('temp', np.nan)
    humidity = row.get('humidity', np.nan)
    wind_mph = row.get('wind_mph', np.nan)
    wind_dir = str(row.get('wind_dir_string', '')).lower()
    condition = str(row.get('condition', '')).lower()

    score = 0
    # temp: +0.2 per 10F over 70, -0.2 per 10F below 70
    if not pd.isna(temp):
        score += (temp - 70) * 0.02
    # humidity: -0.15 per 10% above 50, +0.15 per 10% below 50
    if not pd.isna(humidity):
        score -= (humidity - 50) * 0.015
    # wind: +0.15 per 5 mph if "O" in wind_dir ("out"), -0.10 per 5 if "I"/"in"
    if not pd.isna(wind_mph):
        if "o" in wind_dir or "out" in wind_dir:
            score += wind_mph * 0.03
        elif "i" in wind_dir or "in" in wind_dir:
            score -= wind_mph * 0.02
    # Outdoor: +0.1, Indoor: -0.05
    if "outdoor" in condition:
        score += 0.1
    elif "indoor" in condition:
        score -= 0.05
    # Clamp to [-1, 1]
    return max(-1, min(1, score))

def clean_X(df, train_cols=None):
    # Dedup columns, fix types, fill nans, align cols with train if provided
    df = dedup_columns(df)
    df = fix_types(df)
    # Drop all string/object columns except explicitly allowed
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    # Fill nans with -1 (safe default for tree models)
    df = df.fillna(-1)
    if train_cols is not None:
        # Add any missing cols as -1, and ensure order
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    # Use all numerics except obvious IDs/context
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

# ==== Streamlit UI ====
event_file = st.file_uploader("Upload Event-Level CSV for Training (required)", type='csv', key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type='csv', key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading & cleaning data..."):
        event_df = pd.read_csv(event_file, low_memory=False)
        today_df = pd.read_csv(today_file, low_memory=False)
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    progress = st.progress(2, "Scoring weather for training and today rows...")
    # ---- Weather score feature ----
    if 'weather_score' not in event_df.columns:
        event_df['weather_score'] = event_df.apply(score_weather, axis=1)
    if 'weather_score' not in today_df.columns:
        today_df['weather_score'] = today_df.apply(score_weather, axis=1)
    progress.progress(5, "Engineering features...")

    # ==== ML Features ====
    # Find all features in both
    target_col = 'hr_outcome'
    # Use intersection of columns between event_df and today_df, numerics only
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    # Add weather_score if not present
    if 'weather_score' not in feature_cols:
        feature_cols.append('weather_score')

    # X/y and today features
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    progress.progress(15, "Splitting for validation...")

    # ==== Split/Scale ====
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    progress.progress(22, "Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")

    # ==== Ensemble Models ====
    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=125, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, n_jobs=-1)
    # CatBoost
    cat_clf = cb.CatBoostClassifier(
        iterations=120, depth=5, learning_rate=0.09, verbose=0
    )
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1)
    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.09)
    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000)

    models = [
        ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf),
        ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1, weights=[2,2,2,1,1,1])
    ensemble.fit(X_train_scaled, y_train)
    progress.progress(45, "Validating...")

    # Validation
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    progress.progress(60, "Predicting HR probability for today...")

    # ==== Predict Today ====
    today_df['hr_pred_proba'] = ensemble.predict_proba(X_today_scaled)[:,1]
    today_df['hr_pred_label'] = (today_df['hr_pred_proba'] >= 0.5).astype(int)
    st.markdown("#### Prediction Results (top 30):")
    st.dataframe(today_df.sort_values('hr_pred_proba', ascending=False).head(30))
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")
    progress.progress(100, "Done!")
else:
    st.warning("Upload both event-level and today CSVs to begin.")
