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

st.header("🔎 Deep HR Predictor — Modeling, Leaderboard, and Feature Importances")

# --- ENSURE DATAFRAMES ARE IN MEMORY ---
if 'event_df' not in locals() or 'today_df' not in locals():
    st.error("You must first upload files and run the initial upload/debug block.")
    st.stop()

# --- ENSURE NO DUPLICATE COLUMNS ---
event_df = event_df.loc[:, ~event_df.columns.duplicated()]
today_df = today_df.loc[:, ~today_df.columns.duplicated()]

# --- CHECK hr_outcome ---
if "hr_outcome" not in event_df.columns:
    st.error("No valid hr_outcome column found in event-level file. STOP.")
    st.stop()

# --- Ensure y is binary integer ---
event_df["hr_outcome"] = pd.to_numeric(event_df["hr_outcome"], errors='coerce').astype(int)

# --- Feature Selection: Drop all cols with >80% NA, low variance, or not in both dfs ---
na_thresh = 0.8
lowvar_thresh = 1e-6

# Only keep intersection (except for hr_outcome)
common_cols = list(set(event_df.columns) & set(today_df.columns))
feature_cols = [c for c in common_cols if c not in ["hr_outcome", "game_date", "batter_id", "player_name", "pitcher_id"]]

# Drop columns with too many NAs or nearly constant
cols_to_drop = []
for c in feature_cols:
    if event_df[c].isnull().mean() > na_thresh or today_df[c].isnull().mean() > na_thresh:
        cols_to_drop.append(c)
    elif event_df[c].dtype in [np.float64, np.float32, np.int64, np.int32] and event_df[c].std() < lowvar_thresh:
        cols_to_drop.append(c)
feature_cols = [c for c in feature_cols if c not in cols_to_drop]

st.write("⚡️ Feature columns after NA and variance drop:", len(feature_cols))
st.write(feature_cols)

# --- Clean up dfs ---
X = event_df[feature_cols].copy().fillna(-1)
X_today = today_df[feature_cols].copy().fillna(-1)
y = event_df["hr_outcome"]

# --- Optional: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_today_scaled = scaler.transform(X_today)

# --- Split for validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- ML Models ---
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

st.write("Training ensemble models (XGBoost, LightGBM, CatBoost, RF, GB, LR)...")
ensemble.fit(X_train, y_train)

# --- Validation ---
y_val_pred = ensemble.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_val_pred)
ll = log_loss(y_val, y_val_pred)
st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

# --- Predict for today ---
today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]

# --- Leaderboard (Top 30) ---
out_cols = []
if "player_name" in today_df.columns:
    out_cols.append("player_name")
out_cols += ["hr_probability"]
leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True).head(30)
leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)

st.markdown("### 🏆 **Today's HR Probability — Top 30**")
st.dataframe(leaderboard, use_container_width=True)
st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")

# --- Feature Importances (all trees) ---
importance_dict = {}
for name, clf in [
    ('XGBoost', xgb_clf), ('LightGBM', lgb_clf), ('CatBoost', cat_clf),
    ('RandomForest', rf_clf), ('GradientBoosting', gb_clf)
]:
    imp = getattr(clf, "feature_importances_", None)
    if imp is not None:
        importance_dict[name] = pd.Series(imp, index=feature_cols)
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

st.success("All done! If this ran without error, you are good to go.")
