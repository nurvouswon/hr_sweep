import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import io

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")
st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

st.markdown("""
This app generates **live, unbiased HR picks** for any day, just like your backtests.
1. **Upload two files:**
   - 🏋️ *Historical Event-Level CSV* (**with** `hr_outcome` for training)
   - ⚡ *Today's Event-Level CSV* (**no** `hr_outcome`, all features/lineup for today's slate)
2. The model is trained only on history, and scores today with **zero lookahead**.
3. Picks are displayed and downloadable for a sweep of thresholds (default: .01 to .13).
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
            if (
                (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c]))
                and not pd.api.types.is_bool_dtype(df[c])
                and df[c].nunique(dropna=True) > 1
                and not c.lower().startswith(('mlb_id', 'batter_id', 'pitcher_id'))
            ):
                cols.append(c)
        except Exception:
            continue
    return cols

def audit_report(train_df, live_df, model_features):
    # Features in train but not live, and vice versa
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    missing_from_live = pd.DataFrame({'feature': list(train_cols - live_cols)})
    missing_from_train = pd.DataFrame({'feature': list(live_cols - train_cols)})
    # Null count
    train_null = train_df[model_features].isnull().sum().reset_index()
    train_null.columns = ['feature', 'null_count']
    live_null = live_df[model_features].isnull().sum().reset_index()
    live_null.columns = ['feature', 'null_count']
    # Dtypes
    train_types = pd.DataFrame({'feature': train_df[model_features].columns, 'dtype': [str(train_df[c].dtype) for c in model_features]})
    live_types = pd.DataFrame({'feature': live_df[model_features].columns, 'dtype': [str(live_df[c].dtype) for c in model_features]})
    # Value counts
    unique_train = pd.DataFrame({'feature': model_features, 'nunique': [train_df[c].nunique(dropna=True) for c in model_features]})
    unique_live = pd.DataFrame({'feature': model_features, 'nunique': [live_df[c].nunique(dropna=True) for c in model_features]})
    return {
        'missing_from_live': missing_from_live,
        'missing_from_train': missing_from_train,
        'train_null_count': train_null,
        'live_null_count': live_null,
        'train_types': train_types,
        'live_types': live_types,
        'unique_train': unique_train,
        'unique_live': unique_live,
    }

def download_audit_report(audit, model_features):
    output = io.StringIO()
    output.write("=== Model Features Used ===\n")
    output.write(f"{model_features}\n\n")
    output.write("=== Features missing from live file ===\n")
    output.write(audit['missing_from_live'].to_csv(index=False))
    output.write("\n=== Features missing from history file ===\n")
    output.write(audit['missing_from_train'].to_csv(index=False))
    output.write("\n=== Null count (history/train file) ===\n")
    output.write(audit['train_null_count'].to_csv(index=False))
    output.write("\n=== Null count (live/today file) ===\n")
    output.write(audit['live_null_count'].to_csv(index=False))
    output.write("\n=== Dtypes (history/train) ===\n")
    output.write(audit['train_types'].to_csv(index=False))
    output.write("\n=== Dtypes (live/today) ===\n")
    output.write(audit['live_types'].to_csv(index=False))
    output.write("\n=== Unique value count (history/train) ===\n")
    output.write(audit['unique_train'].to_csv(index=False))
    output.write("\n=== Unique value count (live/today) ===\n")
    output.write(audit['unique_live'].to_csv(index=False))
    return output.getvalue().encode("utf-8")

# === File Uploads ===
st.header("Step 1: Upload Data")
uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (NO hr_outcome)", type="csv", key="live_ev")

# === Threshold Controls ===
st.header("Step 2: Set HR Probability Thresholds")
col1, col2, col3 = st.columns(3)
with col1:
    threshold_min = st.number_input("Min HR Prob Threshold", value=0.01, min_value=0.01, max_value=0.5, step=0.01)
with col2:
    threshold_max = st.number_input("Max HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
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

        for df in [train_df, live_df]:
            if 'batter_id' in df.columns:
                df['batter_id'] = df['batter_id'].apply(clean_id)
            elif 'batter' in df.columns:
                df['batter_id'] = df['batter'].apply(clean_id)

        # === Model Features ===
        numeric_features = robust_numeric_columns(train_df)
        if 'hr_outcome' in numeric_features:
            numeric_features.remove('hr_outcome')
        # **NEW: Only keep features that are in BOTH train and live**
        model_features = [f for f in numeric_features if f in live_df.columns]
        # Remove features with only one unique value (not predictive)
        for c in model_features.copy():
            try:
                if train_df[c].nunique(dropna=True) <= 1:
                    model_features.remove(c)
            except Exception:
                pass

        # === Diagnostics & Audit Report ===
        audit = audit_report(train_df, live_df, model_features)
        st.markdown("### 📝 Audit Report Summary (Preview)")
        st.write(f"Model features used ({len(model_features)}):", model_features)
        st.write("Features in history but missing from live:", audit['missing_from_live']['feature'].tolist())
        st.write("Features in live but missing from history:", audit['missing_from_train']['feature'].tolist())
        lnull = audit['live_null_count']
        st.dataframe(lnull.sort_values('null_count', ascending=False))
        tnull = audit['train_null_count']
        st.dataframe(tnull.sort_values('null_count', ascending=False))

        st.markdown("#### 🔽 Download Full Audit CSV")
        st.download_button(
            "Download Full Model/Feature Audit",
            data=download_audit_report(audit, model_features),
            file_name="mlb_hr_model_audit_report.csv"
        )

        # Show dtypes and unique value count
        with st.expander("⚡ Feature Dtypes (Train/Live):"):
            st.dataframe(audit['train_types'])
            st.dataframe(audit['live_types'])
        with st.expander("⚡ Unique value count (Train/Live):"):
            st.dataframe(audit['unique_train'].sort_values('nunique', ascending=True))
            st.dataframe(audit['unique_live'].sort_values('nunique', ascending=True))

        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)
        live_X = live_df[model_features].fillna(0)

        # === Train XGBoost Model ===
        xgb_clf = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=-1,
            use_label_encoder=False
        )
        xgb_clf.fit(train_X, train_y)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # === Picks Per Threshold ===
        results = []
        thresholds = np.arange(threshold_min, threshold_max + 0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            if 'batter_name' in picked.columns:
                player_col = 'batter_name'
            elif 'player_name' in picked.columns:
                player_col = 'player_name'
            else:
                player_col = 'batter_id'
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked[player_col])
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
- Upload your up-to-date historical event file.
- Bot will select exactly like backtests: *no hindsight, no leaderboard bias, only pure picks!*
""")
