import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import io

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Research Robust Edition")

# ========== UTILS ==========

@st.cache_data(show_spinner=False)
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

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
        if "o" in wind_dir or "out" in wind_dir:
            score += wind_mph * 0.03
        elif "i" in wind_dir or "in" in wind_dir:
            score -= wind_mph * 0.02
    if "outdoor" in condition:
        score += 0.1
    elif "indoor" in condition:
        score -= 0.05
    return int(np.round(1 + 4.5 * (score + 1)))  # -1~1 → 1~10

# ========== FILE LOADERS ==========

def load_any_file(upload):
    if upload.name.endswith('.parquet'):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)

# ========== UI ==========

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv','parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading & cleaning data..."):
        event_df = load_any_file(event_file)
        today_df = load_any_file(today_file)
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)

    st.write(f"DEBUG: event_df shape {event_df.shape}")
    st.write(f"DEBUG: today_df shape {today_df.shape}")

    # Standardize column names
    event_df.columns = [c.lower() for c in event_df.columns]
    today_df.columns = [c.lower() for c in today_df.columns]

    # Add weather_score if missing
    if 'weather_score' not in event_df.columns:
        event_df['weather_score'] = event_df.apply(score_weather, axis=1)
    if 'weather_score' not in today_df.columns:
        today_df['weather_score'] = today_df.apply(score_weather, axis=1)

    # Feature alignment and cleaning
    target_col = 'hr_outcome'
    candidate_feats = [c for c in event_df.columns if c in today_df.columns and c != target_col]
    # Drop columns that are all null or all constant in either
    features_to_use = []
    for c in candidate_feats:
        if (event_df[c].notnull().sum() > 0) and (today_df[c].notnull().sum() > 0):
            if event_df[c].nunique(dropna=True) > 1 or today_df[c].nunique(dropna=True) > 1:
                features_to_use.append(c)

    # Show dtype mismatches
    dtype_problems = []
    for c in features_to_use:
        if str(event_df[c].dtype) != str(today_df[c].dtype):
            dtype_problems.append(f"{c}: train={event_df[c].dtype}, today={today_df[c].dtype}")
    if dtype_problems:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_problems))

    st.write(f"Dropped columns from event-level data:\n{[c for c in event_df.columns if c not in features_to_use and c != target_col]}")
    st.write(f"Dropped columns from today data:\n{[c for c in today_df.columns if c not in features_to_use]}")

    st.write(f"Remaining columns event-level:\n{features_to_use[:100]}")
    st.write(f"Remaining columns today:\n{features_to_use[:100]}")

    # Preprocess: nulls, constant, and categorical encoding
    X_train = event_df[features_to_use].copy()
    X_today = today_df[features_to_use].copy()

    # Identify categorical features
    cat_feats = [c for c in features_to_use if str(X_train[c].dtype) in ("object", "category", "string")]

    # Impute numerics and categoricals
    for c in features_to_use:
        if c in cat_feats:
            X_train[c] = X_train[c].astype(str).fillna("NA")
            X_today[c] = X_today[c].astype(str).fillna("NA")
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_train[c] = X_train[c].fillna(X_train[c].mean())
            X_today[c] = pd.to_numeric(X_today[c], errors="coerce")
            X_today[c] = X_today[c].fillna(X_train[c].mean())

    # Encode categoricals
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c])
        live_vals = pd.Series(X_today[c].unique())
        new_vals = live_vals[~live_vals.isin(le.classes_)]
        if not new_vals.empty:
            le_classes = np.concatenate([le.classes_, new_vals])
            le.classes_ = le_classes
        X_today[c] = le.transform(X_today[c])
        encoders[c] = le

    # Load target
    y = event_df[target_col].astype(int)
    st.write("DEBUG: Feature columns:\n", features_to_use)
    st.write("DEBUG: X shape:", X_train.shape)
    st.write("DEBUG: y shape:", y.shape)
    st.write("DEBUG: event_df hr_outcome unique:\n", event_df[target_col].unique())
    st.write("DEBUG: event_df hr_outcome value counts:\n", event_df[target_col].value_counts())

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_today_scaled = scaler.transform(X_today)

    # Fit models
    st.write("Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=125, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
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
    ensemble.fit(X_train_scaled, y)

    # Validation
    X_train_, X_val, y_train_, y_val = train_test_split(X_train_scaled, y, test_size=0.2, random_state=42, stratify=y)
    y_val_pred = ensemble.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # Predict today
    today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]
    today_df['weather_score_1_10'] = today_df['weather_score']

    # Leaderboard
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
    # Parquet support
    parquet_buffer = io.BytesIO()
    today_df.to_parquet(parquet_buffer, index=False)
    st.download_button("⬇️ Download Full Prediction Parquet", data=parquet_buffer.getvalue(), file_name="today_hr_predictions.parquet", mime="application/octet-stream")

    # Feature importances
    importance_dict = {}
    for name, clf in [
        ('XGBoost', xgb_clf), ('LightGBM', lgb_clf), ('CatBoost', cat_clf),
        ('RandomForest', rf_clf), ('GradientBoosting', gb_clf)
    ]:
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            importance_dict[name] = pd.Series(imp, index=X_train.columns)
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

    st.success("All complete!")

else:
    st.warning("Upload both event-level and today CSVs to begin.")
