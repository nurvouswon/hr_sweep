import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")

st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

st.markdown("""
This standalone app lets you generate **live, unbiased HR picks** for any day, just like your backtests.
- **Upload two files:**  
    1️⃣ *Historical Event-Level CSV* (**with** `hr_outcome` for training)  
    2️⃣ *Today's Event-Level CSV* (**no** `hr_outcome`, all features/lineup for today's slate)
- The model is trained only on history, and scores today with *zero lookahead*.
- Picks are displayed and downloadable for a sweep of thresholds (.13 to .20 by default).

---
""")

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

# === File Uploads ===
st.header("Step 1: Upload Data")
uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (NO hr_outcome)", type="csv", key="live_ev")

# === Threshold Controls ===
st.header("Step 2: Set HR Probability Thresholds")
col1, col2, col3 = st.columns(3)
with col1:
    threshold_min = st.number_input("Min HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
with col2:
    threshold_max = st.number_input("Max HR Prob Threshold", value=0.20, min_value=0.01, max_value=0.5, step=0.01)
with col3:
    threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.01, max_value=0.10, step=0.01)

predict_btn = st.button("🚀 Generate Today's HR Bot Picks")

if predict_btn:
    if uploaded_train is None or uploaded_live is None:
        st.warning("Please upload BOTH files (historical and today's event-level CSV).")
        st.stop()

    with st.spinner("Processing..."):

        train_df = pd.read_csv(uploaded_train)
        live_df = pd.read_csv(uploaded_live)

        # === Prep IDs ===
        for df in [train_df, live_df]:
            if 'batter_id' in df.columns:
                df['batter_id'] = df['batter_id'].apply(clean_id)
            elif 'batter' in df.columns:
                df['batter_id'] = df['batter'].apply(clean_id)

        # === Model Features ===
        numeric_features = robust_numeric_columns(train_df)
        if 'hr_outcome' in numeric_features:
            numeric_features.remove('hr_outcome')
        model_features = [f for f in numeric_features if f in live_df.columns]

        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)

        # === Train XGBoost Model ===
        xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
        xgb_clf.fit(train_X, train_y)
        live_X = live_df[model_features].fillna(0)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # === Picks Per Threshold ===
        results = []
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked.get('batter_name', picked.get('player name', picked.get('batter_id'))))
            })

        # === Output Table ===
        picks_df = pd.DataFrame(results)
        st.header("Results: HR Bot Picks by Threshold")
        st.dataframe(picks_df)
        st.download_button(
            "⬇️ Download Picks by Threshold (CSV)",
            data=picks_df.to_csv(index=False),
            file_name="today_hr_bot_picks_by_threshold.csv"
        )

        st.markdown("#### All Picks (Threshold Sweep):")
        for _, row in picks_df.iterrows():
            st.write(f"**Threshold {row['threshold']}**: {row['picked_players']}")

        st.success("Done! These are the official HR bot picks for today at each threshold.")

st.markdown("""
---
**Instructions for daily use:**  
- Use Statcast/lineup tools to generate today's event-level features for all batters (no `hr_outcome`).  
- Upload with your up-to-date historical event file.  
- Bot will select exactly like backtests: *no hindsight, no leaderboard bias, only pure picks!*  
""")
