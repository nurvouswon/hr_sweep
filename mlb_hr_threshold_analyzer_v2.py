import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score

def clean_id(x):
    try:
        if pd.isna(x): return None
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def robust_numeric_columns(df):
    cols = []
    for c in df.columns:
        try:
            dt = pd.api.types.pandas_dtype(df[c].dtype)
            if (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c])) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1:
                cols.append(c)
        except Exception:
            continue
    return cols

st.set_page_config(page_title="True OOS HR Simulator", layout="wide")
st.title("🔬 Out-of-Sample MLB HR Backtest (No Hindsight Bias)")

st.markdown("""
Upload two CSVs:  
1. **Full event-level MLB data (March 19–June 16)**: all features, must have `game_date` and `hr_outcome`  
2. **Blind event-level data (June 17–18)**: all features, must have `game_date`—**NO HR outcome!**

Run a rolling "true out-of-sample" test: for each date, only data up to *before* that day is used for training.  
Results are shown for all thresholds 0.13–0.20.
""")

csv1 = st.file_uploader("1️⃣ Upload March 19–June 16 Event-Level CSV", type="csv", key="main")
csv2 = st.file_uploader("2️⃣ Upload Blind June 17–18 Event-Level CSV", type="csv", key="future")

thresholds = np.round(np.arange(0.13, 0.201, 0.01), 3)

if csv1 and csv2 and st.button("Run OOS Simulator"):
    df_main = pd.read_csv(csv1)
    df_future = pd.read_csv(csv2)

    # Clean up and combine
    for df in [df_main, df_future]:
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime("%Y-%m-%d")
        else:
            st.error("Missing game_date in at least one CSV!")
            st.stop()
        if 'batter_id' not in df.columns and 'batter' in df.columns:
            df['batter_id'] = df['batter']
        df['batter_id'] = df['batter_id'].apply(clean_id)

    if 'hr_outcome' not in df_main.columns:
        st.error("Main (historical) CSV missing hr_outcome column!")
        st.stop()

    # Combine, mark known and unknown outcome
    df_main['set'] = 'main'
    df_future['set'] = 'future'
    df_future['hr_outcome'] = np.nan  # explicitly make unknown
    df_all = pd.concat([df_main, df_future], ignore_index=True)
    df_all = dedup_columns(df_all)
    df_all['game_date'] = pd.to_datetime(df_all['game_date']).dt.strftime("%Y-%m-%d")

    all_dates = sorted(df_all['game_date'].unique())
    summary_rows = []

    for threshold in thresholds:
        for test_date in all_dates:
            # Only predict for June 17 and 18 (can expand later)
            if test_date not in ["2025-06-17", "2025-06-18"]:
                continue

            # Split OOS
            train = df_all[df_all['game_date'] < test_date].copy()
            test = df_all[df_all['game_date'] == test_date].copy()

            # Filter for events with valid hr_outcome for training
            train = train[train['hr_outcome'].notnull()]
            if len(train) < 20 or len(test) < 1:
                continue

            model_features = robust_numeric_columns(train)
            model_features = [c for c in model_features if c != 'hr_outcome']

            # XGBoost model (change to LogisticRegression if you want both)
            X_train = train[model_features].fillna(0)
            y_train = train['hr_outcome'].astype(int)
            X_test = test[model_features].fillna(0)

            xgb_model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.08, eval_metric='logloss', use_label_encoder=False)
            xgb_model.fit(X_train, y_train)
            test['xgb_prob'] = xgb_model.predict_proba(X_test)[:, 1]
            test['xgb_hr_pred'] = (test['xgb_prob'] > threshold).astype(int)

            # Aggregate picks for day
            picked = test.loc[test['xgb_hr_pred'] == 1, :]
            picked_names = picked['batter_name'].tolist() if 'batter_name' in picked.columns else picked['batter_id'].tolist()
            num_picks = len(picked)
            # Count actual HRs, if available
            if test['hr_outcome'].notnull().any():
                hr_hit = picked[picked['hr_outcome'] == 1]
                actual_hrs = int(hr_hit['hr_outcome'].sum())
                # Precision, recall for picked set (if outcomes available)
                if num_picks > 0:
                    precision = actual_hrs / num_picks
                else:
                    precision = None
            else:
                actual_hrs = None
                precision = None

            # Record summary
            summary_rows.append({
                "date": test_date,
                "threshold": threshold,
                "num_picks": num_picks,
                "picked_players": picked_names,
                "actual_hrs": actual_hrs,
                "precision": precision
            })

    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary)
    st.download_button("⬇️ Download Full OOS Backtest Results CSV", data=summary.to_csv(index=False), file_name="oos_backtest_summary.csv")
    st.success("Simulation complete. Analyze results above! 🚀")
