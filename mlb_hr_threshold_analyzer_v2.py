import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

st.set_page_config(page_title="MLB HR Bot Live Predictor", layout="wide")

st.title("MLB HR Bot Live Predictor (No Hindsight Bias)")

st.markdown("""
This app produces live, unbiased HR picks—just like your backtests.

**Workflow:**
1. Download/merge your 'today' event-level CSV (with all rolling/stat features).
2. Upload BOTH the historical event-level file (with `hr_outcome`) and today's (with merged features, no outcome).
3. Review full diagnostics before and after prediction.
""")

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

def full_feature_alignment(train_df, live_df):
    # Add any missing columns as all-NaN, in both directions (but do not fill 'hr_outcome' in live)
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    missing_in_train = list(live_cols - train_cols)
    missing_in_live = [c for c in (train_cols - live_cols) if c != 'hr_outcome']
    for c in missing_in_train:
        train_df[c] = np.nan
    for c in missing_in_live:
        live_df[c] = np.nan
    # Reorder columns to train order + extras
    final_cols = [c for c in train_df.columns if c != 'hr_outcome'] + ['hr_outcome']
    train_df = train_df.reindex(columns=final_cols)
    live_df = live_df.reindex(columns=[c for c in final_cols if c != 'hr_outcome'])
    return train_df, live_df

def audit_report(train_df, live_df, model_features):
    # Build the report as a dict of DataFrames for easy CSV export
    rep = {}
    # Features missing in each
    rep['missing_from_train'] = pd.DataFrame({'feature': [c for c in live_df.columns if c not in train_df.columns]})
    rep['missing_from_live'] = pd.DataFrame({'feature': [c for c in train_df.columns if c not in live_df.columns]})
    # All-NaN in each
    rep['all_nan_in_train'] = pd.DataFrame({'feature': [c for c in train_df.columns if train_df[c].isna().all()]})
    rep['all_nan_in_live'] = pd.DataFrame({'feature': [c for c in live_df.columns if live_df[c].isna().all()]})
    # One-unique in each
    rep['one_unique_train'] = pd.DataFrame({'feature': [c for c in train_df.columns if train_df[c].nunique(dropna=True)==1]})
    rep['one_unique_live'] = pd.DataFrame({'feature': [c for c in live_df.columns if live_df[c].nunique(dropna=True)==1]})
    # Null count
    rep['train_null_count'] = train_df.isnull().sum().reset_index().rename(columns={'index':'feature',0:'train_nulls'})
    rep['live_null_count'] = live_df.isnull().sum().reset_index().rename(columns={'index':'feature',0:'live_nulls'})
    # Features used
    rep['model_features'] = pd.DataFrame({'feature':model_features})
    # Shapes
    rep['shape'] = pd.DataFrame({'dataset':['train','live'], 'rows':[len(train_df),len(live_df)], 'cols':[len(train_df.columns),len(live_df.columns)]})
    return rep

def audit_report_to_csv(report_dict):
    # Collate all sheets into one CSV for download
    from io import StringIO
    buffer = StringIO()
    for name, df in report_dict.items():
        buffer.write(f"## {name}\n")
        df.to_csv(buffer, index=False)
        buffer.write("\n")
    return buffer.getvalue()

# =================== UI ======================
st.header("2️⃣ Upload Event-Level CSVs & Run Model")

uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_ev")

col1, col2, col3 = st.columns(3)
with col1:
    threshold_min = st.number_input("Min HR Prob Threshold", value=0.01, min_value=0.01, max_value=0.5, step=0.01)
with col2:
    threshold_max = st.number_input("Max HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
with col3:
    threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.01, max_value=0.10, step=0.01)

predict_btn = st.button("🚀 Generate Today's HR Bot Picks & Audit")

if predict_btn:
    if uploaded_train is None or uploaded_live is None:
        st.warning("Please upload BOTH files (historical and today's event-level CSV).")
        st.stop()

    with st.spinner("Processing..."):
        train_df = pd.read_csv(uploaded_train)
        live_df = pd.read_csv(uploaded_live)

        # ====== Feature Alignment: Guarantee all columns present in both files =====
        train_df, live_df = full_feature_alignment(train_df, live_df)

        # Model features: robust, present in BOTH, numeric, not hr_outcome
        all_numeric = set(robust_numeric_columns(train_df)) | set(robust_numeric_columns(live_df))
        model_features = [c for c in all_numeric if c in train_df.columns and c in live_df.columns and c != 'hr_outcome']

        # ==== AUDIT REPORT: Preview diagnostics, then export ====
        audit = audit_report(train_df, live_df, model_features)
        st.markdown("### 📝 Audit Report Summary (Preview)")
        st.write(f"Model features used ({len(model_features)}):", model_features)
        st.write(f"Features in history but missing from live: {audit['missing_from_live']['feature'].tolist()}")
        st.write(f"Features in live but missing from history: {audit['missing_from_train']['feature'].tolist()}")
        st.write("Null count (live):")
        st.dataframe(audit['live_null_count'].sort_values(1,ascending=False))
        st.write("Null count (train):")
        st.dataframe(audit['train_null_count'].sort_values(1,ascending=False))

        audit_csv = audit_report_to_csv(audit)
        st.download_button(
            "⬇️ Download Full Audit Report (CSV)",
            data=audit_csv,
            file_name=f"audit_report_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv"
        )

        # === Model Prep ===
        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)
        live_X = live_df[model_features].fillna(0)

        # === Model Train ===
        xgb_clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, use_label_encoder=False
        )
        xgb_clf.fit(train_X, train_y)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # === Picks Table ===
        results = []
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            name_col = 'player_name' if 'player_name' in picked.columns else (picked.columns[0] if len(picked) else 'batter_id')
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': list(picked[name_col]) if name_col in picked.columns else []
            })

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
**Instructions:**  
- In Tab 1, generate today's event-level CSV by merging the latest history with today’s lineups (as you’ve been doing).
- In Tab 2, upload both files here to predict and audit.
- **Check the audit report (above and CSV)** for missing/NaN features or mismatches.
""")
