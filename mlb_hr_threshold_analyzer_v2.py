import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

st.title("⚾ MLB HR Rolling Backtest — XGBoost, .13 to .20 Threshold Sweep")

# ---- Upload ----
st.markdown("### Upload your full event-level HR CSV (must include date, HR label, player, and all features):")
uploaded = st.file_uploader("Upload event-level features CSV", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
if 'date' not in df.columns:
    st.error("Your CSV must include a 'date' column (YYYY-MM-DD)!")
    st.stop()
if 'hr_outcome' not in df.columns:
    st.error("Your CSV must include an 'hr_outcome' column (1=HR, 0=not HR)!")
    st.stop()
if 'player_name' not in df.columns:
    st.error("Your CSV must include a 'player_name' column!")
    st.stop()

date_col = 'date'
hr_col = 'hr_outcome'
player_col = 'player_name'
prob_col = 'xgb_prob' if 'xgb_prob' in df.columns else None

# Features for XGBoost: Use only numeric columns, drop identifiers/labels
id_cols = [date_col, hr_col, player_col]
ignore_cols = id_cols + ['game_pk', 'game_date']  # add other identifier cols if needed
X_all = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')
X_num = X_all.select_dtypes(include=[np.number])

df[date_col] = pd.to_datetime(df[date_col]).dt.date

# All thresholds
thresholds = np.arange(0.13, 0.21, 0.01)
thresholds = np.round(thresholds, 2)

results = []
dates = sorted(df[date_col].unique())

progress = st.progress(0, text="Initializing rolling backtest...")

# Main rolling window loop
for i, this_date in enumerate(dates):
    train_mask = df[date_col] < this_date
    test_mask = df[date_col] == this_date
    if train_mask.sum() < 50 or test_mask.sum() == 0:
        continue

    X_train, y_train = X_num[train_mask], df.loc[train_mask, hr_col]
    X_test, y_test = X_num[test_mask], df.loc[test_mask, hr_col]
    names_test = df.loc[test_mask, player_col].values

    # XGBoost setup
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    pred_prob = xgb_model.predict_proba(X_test)[:, 1]

    for thresh in thresholds:
        picks_mask = pred_prob >= thresh
        picks = np.where(picks_mask)[0]
        picked_names = list(names_test[picks])
        picked_hrs = list(np.array(picked_names)[y_test.values[picks]==1]) if len(picks) else []
        missed_hrs = list(names_test[(~picks_mask) & (y_test.values == 1)])
        TP = sum(y_test.values[picks] == 1)
        FP = sum(y_test.values[picks] == 0)
        FN = sum((~picks_mask) & (y_test.values == 1))
        TN = sum((~picks_mask) & (y_test.values == 0))
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        results.append({
            'date': this_date,
            'threshold': thresh,
            'picks': len(picks),
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision': precision, 'recall': recall, 'f1_score': f1,
            'picked_players': picked_names,
            'picked_hr_players': picked_hrs,
            'missed_hr_players': missed_hrs
        })
    progress.progress((i+1)/len(dates), f"Rolling backtest: {i+1}/{len(dates)} days complete")

results_df = pd.DataFrame(results)
st.success(f"Backtest finished for {len(dates)} days, {len(thresholds)} thresholds.")

st.markdown("### Download full rolling backtest results (all thresholds, all days):")
st.download_button("Download CSV", results_df.to_csv(index=False), file_name="rolling_backtest_xgb.csv", mime="text/csv")

# Optional: show summary table in-app
show_table = st.checkbox("Show daily results table")
if show_table:
    st.dataframe(results_df)

st.markdown("""
---
**Instructions:**  
- Use this app to generate a full rolling out-of-sample backtest, sweeping all thresholds from .13 to .20.
- Download the CSV and send it to me (ChatGPT) for recap, analysis, or further breakdown by player, team, date, etc.
""")
