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
import gc

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score")

# ========== HELPER FUNCTIONS ==========

@st.cache_data(show_spinner=False)
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except: pass
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
    if "outdoor" in condition:
        score += 0.1
    elif "indoor" in condition:
        score -= 0.05
    # Clamp to [1, 10] (normalized from original -1 to 1)
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

def drop_low_variance_and_high_na(df, na_thresh=0.98, var_thresh=1e-10):
    # Drop columns with >98% NA or <= 1 unique value or very low variance
    n_rows = df.shape[0]
    dropped = []
    for c in list(df.columns):
        na_ratio = df[c].isna().mean()
        nunique = df[c].nunique(dropna=True)
        if na_ratio > na_thresh or nunique <= 1:
            dropped.append(c)
            df = df.drop(columns=[c])
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            var = df[c].var(ddof=0)
            if var is not None and var < var_thresh:
                dropped.append(c)
                df = df.drop(columns=[c])
    return df, dropped

# ========== UI ==========
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type='csv', key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading & cleaning data..."):
        if event_file.name.endswith(".csv"):
            event_df = pd.read_csv(event_file, low_memory=False)
        else:
            event_df = pd.read_parquet(event_file)
        today_df = pd.read_csv(today_file, low_memory=False)
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    # Drop low-variance and high-na columns (in each, separately)
    st.write("Dropping columns from event-level data:")
    event_df, dropped_event = drop_low_variance_and_high_na(event_df)
    for i in range(0, len(dropped_event), 100):
        st.write(dropped_event[i:i+100])
    st.write("Dropped columns from today data:")
    today_df, dropped_today = drop_low_variance_and_high_na(today_df)
    for i in range(0, len(dropped_today), 100):
        st.write(dropped_today[i:i+100])

    st.write("Remaining columns event-level:")
    for i in range(0, len(event_df.columns), 100):
        st.write(event_df.columns[i:i+100].to_list())
    st.write("Remaining columns today:")
    for i in range(0, len(today_df.columns), 100):
        st.write(today_df.columns[i:i+100].to_list())

    st.write("Engineering features...")

    # ========== WEATHER SCORE ==========
    if 'weather_score' not in event_df.columns:
        event_df['weather_score'] = event_df.apply(score_weather, axis=1)
    if 'weather_score' not in today_df.columns:
        today_df['weather_score'] = today_df.apply(score_weather, axis=1)

    target_col = 'hr_outcome'
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    if 'weather_score' not in feature_cols:
        feature_cols.append('weather_score')

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    # ================= DEBUG ==================
    st.write("DEBUG: event_df shape", event_df.shape)
    st.write("DEBUG: today_df shape", today_df.shape)
    st.write("DEBUG: Feature columns:", feature_cols)
    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)
    st.write("DEBUG: event_df hr_outcome unique:", event_df[target_col].unique())
    st.write("DEBUG: event_df hr_outcome value counts:", event_df[target_col].value_counts().to_dict())
    # ============== END DEBUG =================

    # ========== TRAIN/VAL SPLIT ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    st.write("Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")

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

    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]
    today_df['weather_score_1_10'] = today_df['weather_score']  # already 1-10

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

    # ==== Feature Importances: Top 30 ====
    # Aggregate from all tree models
    importance_dict = {}
    for name, clf in [
        ('XGBoost', xgb_clf), ('LightGBM', lgb_clf), ('CatBoost', cat_clf),
        ('RandomForest', rf_clf), ('GradientBoosting', gb_clf)
    ]:
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            importance_dict[name] = pd.Series(imp, index=X.columns)
    if importance_dict:
        imp_df = pd.DataFrame(importance_dict)
        imp_df['mean_importance'] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean_importance", ascending=False).head(30)
        st.markdown("### 🔑 **Top 30 Feature Importances (Averaged, All Trees)**")
        st.dataframe(imp_df[["mean_importance"]])
        fig, ax = plt.subplots(figsize=(6, 12))
        imp_df["mean_importance"].iloc[::-1].plot(kind='barh', ax=ax)
        ax.set_xlabel("Mean Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 30 Features (Ensemble Trees)")
        st.pyplot(fig)
    else:
        st.info("No feature importances available.")

    gc.collect()
else:
    st.warning("Upload both event-level and today CSVs to begin.")
