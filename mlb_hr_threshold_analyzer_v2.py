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

@st.cache_data(show_spinner=True)
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def score_weather(row):
    temp = row.get('temp', np.nan)
    humidity = row.get('humidity', np.nan)
    wind_mph = row.get('wind_mph', np.nan)
    wind_dir = str(row.get('wind_dir_string', '')).lower()
    condition = str(row.get('condition', '')).lower()

    score = 0
    if not pd.isna(temp):
        score += (temp - 70) * 0.02
    if not pd.isna(humidity):
        score -= (humidity - 50) * 0.015
    if not pd.isna(wind_mph):
        if "o" in wind_dir or "out" in wind_dir:
            score += wind_mph * 0.03
        elif "i" in wind_dir or "in" in wind_dir:
            score -= wind_mph * 0.02
    if "outdoor" in condition:
        score += 0.1
    elif "indoor" in condition:
        score -= 0.05
    return int(np.round(1 + 4.5 * (score + 1)))  # -1->1 range to 1->10

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

def read_event_file(uploaded_file):
    if uploaded_file.name.lower().endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, low_memory=False)
    return df

try:
    event_file = st.file_uploader("Upload Event-Level CSV or Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
    today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type='csv', key='todaycsv')

    if event_file is not None and today_file is not None:
        with st.spinner("Loading & cleaning data..."):
            event_df = read_event_file(event_file)
            today_df = pd.read_csv(today_file, low_memory=False)
            event_df = dedup_columns(event_df)
            today_df = dedup_columns(today_df)
            event_df = fix_types(event_df)
            today_df = fix_types(today_df)
            st.success(f"Event-Level Data Loaded: {event_df.shape[0]:,} rows")

        progress = st.progress(2, "Scoring weather for training and today rows...")
        if 'weather_score' not in event_df.columns:
            event_df['weather_score'] = event_df.apply(score_weather, axis=1)
        if 'weather_score' not in today_df.columns:
            today_df['weather_score'] = today_df.apply(score_weather, axis=1)
        progress.progress(5, "Engineering features...")

        target_col = 'hr_outcome'
        feat_cols_train = set(get_valid_feature_cols(event_df))
        feat_cols_today = set(get_valid_feature_cols(today_df))
        feature_cols = sorted(list(feat_cols_train & feat_cols_today))
        if 'weather_score' not in feature_cols:
            feature_cols.append('weather_score')

        X = clean_X(event_df[feature_cols])
        y = event_df[target_col]
        X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

        progress.progress(15, "Splitting for validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_today_scaled = scaler.transform(X_today)

        progress.progress(22, "Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")

        xgb_clf = xgb.XGBClassifier(
            n_estimators=125, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )
        lgb_clf = lgb.LGBMClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, n_jobs=-1)
        cat_clf = cb.CatBoostClassifier(
            iterations=120, depth=5, learning_rate=0.09, verbose=0
        )
        rf_clf = RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1)
        gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.09)
        lr_clf = LogisticRegression(max_iter=1000)

        models = [
            ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf),
            ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1, weights=[2,2,2,1,1,1])
        ensemble.fit(X_train_scaled, y_train)
        progress.progress(45, "Validating...")

        y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
        auc = roc_auc_score(y_val, y_val_pred)
        ll = log_loss(y_val, y_val_pred)
        st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

        progress.progress(60, "Predicting HR probability for today...")

        today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]
        today_df['weather_score_1_10'] = today_df['weather_score']

        # ==== Leaderboard: Top 30 Only ====
        out_cols = []
        if "player_name" in today_df.columns:
            out_cols.append("player_name")
        out_cols += ["hr_probability", "weather_score_1_10"]
        leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True).head(30)
        leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
        leaderboard["weather_score_1_10"] = leaderboard["weather_score_1_10"].round(0).astype(int)

        st.markdown("### 🏆 **Today's HR Probability — Top 30**")
        st.dataframe(leaderboard, use_container_width=True)
        st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")

        progress.progress(100, "Done!")
    else:
        st.warning("Upload both event-level file (CSV or Parquet) and today CSV to begin.")

except Exception as e:
    import traceback
    st.error("🚨 App Crashed! See details below:")
    st.code(traceback.format_exc())
