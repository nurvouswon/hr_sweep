import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

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

predict_btn = st.button("🚀 Generate Today's HR Bot Picks & Diagnostics")

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

        # === Diagnostics Panel: Show Features Used ===
        st.markdown("### 🔍 Diagnostics & Audit Panel")
        st.write(f"**Number of features used:** {len(model_features)}")
        st.write(f"**Model features:** {model_features}")

        # Missing values check (for audit)
        nulls_report = pd.DataFrame({
            'nulls_train': train_df[model_features].isnull().sum(),
            'nulls_live': live_df[model_features].isnull().sum()
        })
        st.markdown("#### Null Values in Features")
        st.dataframe(nulls_report)

        # Value distributions (train/live) for each feature
        st.markdown("#### Feature Value Ranges (Train/Live)")
        feat_ranges = []
        for col in model_features:
            feat_ranges.append({
                "feature": col,
                "train_min": np.nanmin(train_df[col]),
                "train_max": np.nanmax(train_df[col]),
                "train_mean": np.nanmean(train_df[col]),
                "live_min": np.nanmin(live_df[col]),
                "live_max": np.nanmax(live_df[col]),
                "live_mean": np.nanmean(live_df[col]),
            })
        feat_ranges_df = pd.DataFrame(feat_ranges)
        st.dataframe(feat_ranges_df)

        # === XGBoost Model Fit ===
        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False
        )
        xgb_clf.fit(train_X, train_y)
        live_X = live_df[model_features].fillna(0)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # === Feature Importance Audit ===
        importances = xgb_clf.feature_importances_
        fi_df = pd.DataFrame({'feature': model_features, 'importance': importances}).sort_values('importance', ascending=False)
        st.markdown("#### XGBoost Feature Importances")
        st.dataframe(fi_df.head(25))
        st.bar_chart(fi_df.set_index('feature').head(15))

        # === Score Distribution ===
        st.markdown("#### Predicted HR Probability Distribution (All Batters Today)")
        fig, ax = plt.subplots()
        ax.hist(live_df['xgb_prob'], bins=30, edgecolor='black')
        ax.set_xlabel("Predicted HR Probability")
        ax.set_ylabel("Number of Batters")
        st.pyplot(fig)

        st.write(f"**Summary Stats (Today's Probabilities):**")
        st.write(live_df['xgb_prob'].describe())

        # === Picks Per Threshold ===
        results = []
        audit_rows = []
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask].copy()
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked.get('batter_name', picked.get('player name', picked.get('batter_id'))))
            })
            # For audit: Store all rows and scores for this threshold
            for _, row in picked.iterrows():
                audit_row = row.to_dict()
                audit_row['threshold'] = thresh
                audit_row['picked'] = True
                audit_rows.append(audit_row)
        picks_df = pd.DataFrame(results)

        st.header("Results: HR Bot Picks by Threshold")
        st.dataframe(picks_df)
        st.download_button(
            "⬇️ Download Picks by Threshold (CSV)",
            data=picks_df.to_csv(index=False),
            file_name="today_hr_bot_picks_by_threshold.csv"
        )

        # === Per-batter audit: Downloadable full file of all batters, scores, features, and pick flag
        st.markdown("### 📝 Downloadable Full Audit/Diagnostics CSV")
        full_audit = live_df.copy()
        full_audit['picked_any_threshold'] = False
        for thresh in thresholds:
            full_audit[f"pick_{round(thresh,3)}"] = full_audit['xgb_prob'] >= thresh
            full_audit['picked_any_threshold'] |= full_audit['xgb_prob'] >= thresh
        st.dataframe(full_audit.head(20))
        st.download_button(
            "⬇️ Download Full Audit CSV (All Batters/All Features/All Scores)",
            data=full_audit.to_csv(index=False),
            file_name="full_batter_scoring_audit.csv"
        )

        st.markdown("#### All Picks (Threshold Sweep):")
        for _, row in picks_df.iterrows():
            st.write(f"**Threshold {row['threshold']}**: {row['picked_players']}")

        st.success("Done! These are the official HR bot picks for today at each threshold. **Audit and diagnostics now available.**")

st.markdown("""
---
**Instructions for daily use:**  
- Use Statcast/lineup tools to generate today's event-level features for all batters (no `hr_outcome`).  
- Upload with your up-to-date historical event file.  
- Bot will select exactly like backtests: *no hindsight, no leaderboard bias, only pure picks!*  
""")
