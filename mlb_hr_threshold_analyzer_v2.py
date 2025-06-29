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
import matplotlib.pyplot as plt
import io

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

# ------------- DEDUP AND TYPE FIX -------------
@st.cache_data(show_spinner=True)
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

@st.cache_data(show_spinner=True)
def fix_types(df):
    # Downcast floats and ints for memory, robust to all col types
    for col in df.columns:
        try:
            if df[col].isnull().all():
                continue
            # Downcast numerics
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'O':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        except Exception as e:
            pass
    return df

# --------- VECTORIZED WEATHER SCORE -----------
def score_weather_vectorized(df):
    temp = df.get('temp', pd.Series(np.nan, index=df.index)).astype(float)
    humidity = df.get('humidity', pd.Series(np.nan, index=df.index)).astype(float)
    wind_mph = df.get('wind_mph', pd.Series(np.nan, index=df.index)).astype(float)
    wind_dir = df.get('wind_dir_string', pd.Series("", index=df.index)).astype(str).str.lower()
    condition = df.get('condition', pd.Series("", index=df.index)).astype(str).str.lower()
    score = pd.Series(0, index=df.index, dtype=float)
    score += np.where(~temp.isna(), (temp - 70) * 0.02, 0)
    score -= np.where(~humidity.isna(), (humidity - 50) * 0.015, 0)
    score += np.where((~wind_mph.isna()) & (wind_dir.str.contains("o|out")), wind_mph * 0.03, 0)
    score -= np.where((~wind_mph.isna()) & (wind_dir.str.contains("i|in")), wind_mph * 0.02, 0)
    score += np.where(condition.str.contains("outdoor"), 0.1, 0)
    score -= np.where(condition.str.contains("indoor"), 0.05, 0)
    score = 1 + 4.5 * (score + 1)
    score = score.clip(1, 10).round().astype(int)
    return score

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

# ----------- FILE UPLOADERS -----------
event_file = st.file_uploader("Upload Event-Level Training File (CSV or Parquet)", type=['csv','parquet'], key='eventfile')
today_file = st.file_uploader("Upload TODAY CSV for Prediction", type='csv', key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and cleaning data..."):
        # Parquet or CSV support for event-level
        try:
            if event_file.name.endswith(".parquet"):
                event_df = pd.read_parquet(event_file)
            else:
                event_df = pd.read_csv(event_file, low_memory=False)
        except Exception as e:
            st.error(f"Could not load event-level file: {e}")
            st.stop()

        try:
            today_df = pd.read_csv(today_file, low_memory=False)
        except Exception as e:
            st.error(f"Could not load TODAY CSV: {e}")
            st.stop()

        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

        st.write(f"Loaded event-level: {event_df.shape}")
        st.write(f"Loaded today: {today_df.shape}")

    # --------- WEATHER SCORING -----------
    if 'weather_score' not in event_df.columns:
        event_df['weather_score'] = score_weather_vectorized(event_df)
    if 'weather_score' not in today_df.columns:
        today_df['weather_score'] = score_weather_vectorized(today_df)

    # ==== ML Features ====
    target_col = 'hr_outcome'
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    if 'weather_score' not in feature_cols:
        feature_cols.append('weather_score')

    # Prepare X/y
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    # --------- MODELING ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    xgb_clf = xgb.XGBClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    lgb_clf = lgb.LGBMClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, n_jobs=-1)
    cat_clf = cb.CatBoostClassifier(iterations=120, depth=5, learning_rate=0.09, verbose=0)
    rf_clf = RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1)
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.09)
    lr_clf = LogisticRegression(max_iter=1000)

    models = [
        ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf),
        ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1, weights=[2,2,2,1,1,1])
    ensemble.fit(X_train_scaled, y_train)

    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # --------- PREDICT -----------
    today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]
    today_df['weather_score_1_10'] = today_df['weather_score']

    # --------- OUTPUT -----------
    out_cols = []
    if "player_name" in today_df.columns:
        out_cols.append("player_name")
    out_cols += ["hr_probability", "weather_score_1_10"]

    leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["weather_score_1_10"] = leaderboard["weather_score_1_10"].round(0).astype(int)

    st.markdown("### 🏆 **Today's HR Probability Leaderboard**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")

    st.success("Prediction complete. App optimized for your data size.")
else:
    st.warning("Upload both event-level (CSV/Parquet) and today CSV to begin.")
