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
