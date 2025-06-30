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

st.set_page_config("MLB HR Predictor", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEBUG BOOTSTRAP]")

# ========== Utility functions ==========
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def load_any_file(upload):
    try:
        if upload.name.endswith('.parquet'):
            df = pd.read_parquet(upload)
        else:
            df = pd.read_csv(upload)
        st.write(f"DEBUG: Successfully loaded file: {upload.name} with shape {df.shape}")
        return df
    except Exception as e:
        st.error(f"File load failed: {e}")
        st.stop()

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
    return int(np.round(1 + 4.5 * (score + 1)))

# ========== Streamlit UI ==========
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv','parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv'], key='todaycsv')

if event_file is not None and today_file is not None:
    try:
        event_df = load_any_file(event_file)
        today_df = load_any_file(today_file)
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        st.write("DEBUG: Columns in event_df:", list(event_df.columns))
        st.write("DEBUG: Columns in today_df:", list(today_df.columns))
        st.write("DEBUG: event_df head:")
        st.dataframe(event_df.head(2))
        st.write("DEBUG: today_df head:")
        st.dataframe(today_df.head(2))
    except Exception as e:
        st.error(f"CRASH during file load or initial preview: {e}")
        st.stop()

    # ========== Stop here for confirmation ==========
    st.warning("""
        👆 **Check the above for shape, columns, and heads.**
        - If you see both tables, upload is OK.
        - If not, copy the error here and STOP.
        - If OK, tell me, and I'll give you the next code chunk to run the modeling and leaderboard.
    """)

else:
    st.warning("Upload both event-level and today CSVs to begin.")
if event_file is not None and today_file is not None:
    # Files already loaded: event_df, today_df

    # --- 1. Drop columns with >90% NaN or constant value in both datasets ---
    def drop_low_info_cols(df1, df2, na_thresh=0.90):
        dropped = []
        keep = []
        for col in df1.columns:
            if col not in df2.columns:
                continue
            na1 = df1[col].isna().mean()
            na2 = df2[col].isna().mean()
            nunique1 = df1[col].nunique(dropna=True)
            nunique2 = df2[col].nunique(dropna=True)
            if (na1 > na_thresh and na2 > na_thresh) or (nunique1 <= 1 and nunique2 <= 1):
                dropped.append(col)
            else:
                keep.append(col)
        return keep, dropped

    keep_cols, dropped_cols = drop_low_info_cols(event_df, today_df)
    event_df = event_df[keep_cols]
    today_df = today_df[keep_cols]

    st.write(f"Dropped columns (low info or high NA): {len(dropped_cols)}")
    if dropped_cols:
        st.write(dropped_cols[:50])
        if len(dropped_cols) > 50:
            st.write(dropped_cols[50:100])
            if len(dropped_cols) > 100:
                st.write(dropped_cols[100:150])
    st.write(f"Kept columns: {len(keep_cols)}")

    # --- 2. Only keep intersection of columns, remove anything not in both ---
    feat_cols = [c for c in keep_cols if c in event_df.columns and c in today_df.columns and c != 'hr_outcome']
    st.write("Feature columns after intersection:", feat_cols[:100])
    if len(feat_cols) > 100:
        st.write(feat_cols[100:200])
        if len(feat_cols) > 200:
            st.write(feat_cols[200:300])

    # --- 3. Diagnostics: Nulls, datatypes ---
    st.write("Null fraction (train):")
    st.write(event_df[feat_cols].isnull().mean())
    st.write("Null fraction (today):")
    st.write(today_df[feat_cols].isnull().mean())

    # --- 4. Prepare X, y ---
    X = event_df[feat_cols].copy()
    y = event_df['hr_outcome'] if 'hr_outcome' in event_df.columns else None
    X_today = today_df[feat_cols].copy()

    # --- 5. Quick check for y ---
    st.write("DEBUG: hr_outcome unique:", pd.Series(y.unique()).tolist() if y is not None else "NOT FOUND")
    if y is None or y.isnull().all():
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()

    st.write("DEBUG: X shape", X.shape)
    st.write("DEBUG: X_today shape", X_today.shape)

    # --- 6. Pause before model ---
    st.success("✅ Data loaded and cleaned. Feature set established. Ready to proceed with modeling.")
    st.warning("Check all outputs above for any issues. If all looks good, reply OK and I will give you the next modeling chunk.")
