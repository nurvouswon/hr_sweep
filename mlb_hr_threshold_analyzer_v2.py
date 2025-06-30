import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEBUG BOOTSTRAP]")

# ========== Utility Functions ==========

def safe_read_csv_or_parquet(f):
    """Reads CSV or Parquet, returns DataFrame and file type."""
    try:
        if f.name.lower().endswith(".parquet"):
            df = pd.read_parquet(f)
            return df, 'parquet'
        else:
            df = pd.read_csv(f, low_memory=False)
            return df, 'csv'
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def drop_high_na_low_variance(df, thresh_na=0.97):
    """Drops columns with > thresh_na NA or only one unique value."""
    orig_cols = df.columns
    high_na = [c for c in df.columns if df[c].isnull().mean() > thresh_na]
    low_var = [c for c in df.columns if df[c].nunique(dropna=True) < 2]
    to_drop = list(set(high_na + low_var))
    df2 = df.drop(columns=to_drop, errors='ignore')
    return df2, to_drop

def score_weather(row):
    temp = row.get('temp', np.nan)
    humidity = row.get('humidity', np.nan)
    wind_mph = row.get('wind_mph', np.nan)
    wind_dir = str(row.get('wind_dir_string', '')).lower()
    condition = str(row.get('condition', '')).lower()
    score = 0
    if not pd.isna(temp): score += (temp - 70) * 0.02
    if not pd.isna(humidity): score -= (humidity - 50) * 0.015
    if not pd.isna(wind_mph):
        if "o" in wind_dir or "out" in wind_dir: score += wind_mph * 0.03
        elif "i" in wind_dir or "in" in wind_dir: score -= wind_mph * 0.02
    if "outdoor" in condition: score += 0.1
    elif "indoor" in condition: score -= 0.05
    return int(np.round(1 + 4.5 * (score + 1)))

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
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

def show_col_chunks(lst, label, chunk=100):
    st.write(f"**{label}:**")
    for i in range(0, len(lst), chunk):
        st.write(f"[{i} - {min(i+chunk-1,len(lst)-1)}]", lst[i:i+chunk])

# ========== UPLOAD & DEBUG BOOTSTRAP ==========

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv','parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type='csv', key='todaycsv')

if event_file is not None and today_file is not None:
    # 1. Read and display file info
    event_df, event_filetype = safe_read_csv_or_parquet(event_file)
    today_df, _ = safe_read_csv_or_parquet(today_file)
    if event_df is None or today_df is None:
        st.stop()
    st.success(f"DEBUG: Successfully loaded file: {event_file.name} with shape {event_df.shape}")
    st.success(f"DEBUG: Successfully loaded file: {today_file.name} with shape {today_df.shape}")

    show_col_chunks(list(event_df.columns), "Columns in event_df")
    show_col_chunks(list(today_df.columns), "Columns in today_df")

    st.write("DEBUG: event_df head:")
    st.dataframe(event_df.head(2))
    st.write("DEBUG: today_df head:")
    st.dataframe(today_df.head(2))
    st.info("👆 Check the above for shape, columns, and heads.\n\nIf you see both tables, upload is OK.\nIf not, copy the error here and STOP.")
    # ——— CONTINUE TO MODELING BELOW THIS LINE ———

    # ========== PREPROCESSING & FEATURE ENGINEERING ==========

    # Remove non-unique columns in both event_df and today_df
    st.info("Dropping high-NA and low-variance columns from event-level and today data (for stability & speed)...")
    event_df, drop1 = drop_high_na_low_variance(event_df, thresh_na=0.97)
    today_df, drop2 = drop_high_na_low_variance(today_df, thresh_na=0.97)
    st.write("Dropped columns from event-level data:")
    show_col_chunks(drop1, "Dropped Event Columns")
    st.write("Dropped columns from today data:")
    show_col_chunks(drop2, "Dropped Today Columns")
    st.write("Remaining columns event-level:")
    show_col_chunks(list(event_df.columns), "Event Columns")
    st.write("Remaining columns today:")
    show_col_chunks(list(today_df.columns), "Today Columns")

    # Sanity check: HR OUTCOME
    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found!")
    st.write("Value counts for hr_outcome:")
    st.write(event_df[target_col].value_counts().reset_index().rename(columns={'index':'hr_outcome','hr_outcome':'count'}))

    # Weather scoring
    if 'weather_score' not in event_df.columns:
        event_df['weather_score'] = event_df.apply(score_weather, axis=1)
    if 'weather_score' not in today_df.columns:
        today_df['weather_score'] = today_df.apply(score_weather, axis=1)

    # Feature set intersection
    base_drop = {'game_date','batter_id','player_name','pitcher_id','city','park','roof_status'}
    feature_cols = [c for c in event_df.columns if c in today_df.columns and c not in base_drop and c != target_col]
    # Only keep columns that are numeric or int/bool
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(event_df[c])]
    st.write("DEBUG: Feature columns:")
    show_col_chunks(feature_cols, "Feature Columns")
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    # ========== TRAIN/VALIDATE SPLIT & SCALING ==========
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.18, random_state=42, stratify=y
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # ========== MODEL TRAINING ==========
    st.info("Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
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
    st.success("Ensemble trained.")

    # ========== VALIDATION ==========
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # ========== TODAY PREDICTION ==========
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
    today_parquet = io.BytesIO()
    today_df.to_parquet(today_parquet, index=False)
    st.download_button(
        "⬇️ Download Full Prediction Parquet",
        data=today_parquet.getvalue(),
        file_name="today_hr_predictions.parquet",
        mime="application/octet-stream"
    )

    # ==== Feature Importances: Top 30 ====
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

    st.success("✅ All done! Predictions and leaderboard complete.")
else:
    st.warning("🔎 Deep HR Predictor — Modeling, Leaderboard, and Feature Importances\nYou must first upload files and run the initial upload/debug block.")
