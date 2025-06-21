import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# --- Upload CSVs ---
st.markdown("#### Upload Training Event-Level CSV (with hr_outcome)")
train_file = st.file_uploader("Upload Training Event-Level CSV", type="csv", key="trainfile")
st.markdown("#### Upload Today's Event-Level CSV (with merged features, NO hr_outcome)")
live_file = st.file_uploader("Upload Today's Event-Level CSV", type="csv", key="livefile")

min_thr = st.number_input("Min HR Prob Threshold", 0.00, 0.5, 0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.20, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)
    
    # ===========
    # DEEP FEATURE DIAGNOSTICS
    # ===========
    st.markdown("### 🩺 Deep Feature Diagnostics")
    st.write("Live file columns:", list(live_df.columns))
    st.write("Train file columns:", list(train_df.columns))

    st.write("Live columns with non-null values:", [col for col in live_df.columns if live_df[col].notnull().sum() > 0])
    st.write("Train columns with non-null values:", [col for col in train_df.columns if train_df[col].notnull().sum() > 0])
    st.write("Live columns ALL null:", [col for col in live_df.columns if live_df[col].isnull().all()])
    st.write("Train columns ALL null:", [col for col in train_df.columns if train_df[col].isnull().all()])

    st.write("Live sample (non-null cols only):")
    not_null_live = live_df[[col for col in live_df.columns if live_df[col].notnull().sum() > 0]]
    st.dataframe(not_null_live.head(2))

    # Print dtypes for common join keys
    for key in ['batter_id', 'mlb_id', 'player_name']:
        if key in train_df.columns and key in live_df.columns:
            st.write(f"{key} dtype in train:", train_df[key].dtype)
            st.write(f"{key} dtype in live:", live_df[key].dtype)

    # Display top null columns with counts/fractions
    audit_df = pd.DataFrame({
        'feature': live_df.columns,
        'live_notnull': [live_df[c].notnull().sum() for c in live_df.columns],
        'live_null_frac': [live_df[c].isnull().mean() for c in live_df.columns],
        'live_dtype': [str(live_df[c].dtype) for c in live_df.columns]
    })
    st.write("Feature audit of live data:")
    st.dataframe(audit_df.sort_values("live_notnull", ascending=False).head(40))

    # ===========
    # COERCE DTYPE PATCH FOR COMMON KEYS
    # ===========
    if 'batter_id' in train_df.columns and 'batter_id' in live_df.columns:
        train_df['batter_id'] = train_df['batter_id'].astype(str)
        live_df['batter_id'] = live_df['batter_id'].astype(str)
    if 'mlb_id' in train_df.columns and 'mlb_id' in live_df.columns:
        train_df['mlb_id'] = train_df['mlb_id'].astype(str)
        live_df['mlb_id'] = live_df['mlb_id'].astype(str)

    # ===========
    # Feature filter debug (PRESERVE!)
    # ===========
    st.markdown("### 🔍 Audit Report:")
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]
    st.write("Intersect columns:")
    st.write(intersect)

    # For each intersected column, show dtype, unique, and nulls
    feat_diag = []
    for c in intersect:
        diag = {
            'feature': c,
            'train_dtype': str(train_df[c].dtype),
            'live_dtype': str(live_df[c].dtype),
            'train_unique': train_df[c].nunique(dropna=False),
            'live_unique': live_df[c].nunique(dropna=False),
            'train_null_frac': np.round(train_df[c].isnull().mean(), 4),
            'live_null_frac': np.round(live_df[c].isnull().mean(), 4)
        }
        feat_diag.append(diag)
    st.dataframe(pd.DataFrame(feat_diag))

    # Robust model feature filtering (ALLOW all numeric and float columns with >1 unique and not all null)
    def is_feature_valid(col):
        # Allow numeric and float features with at least 2 unique non-null, and NOT all-null in either file
        if col == "hr_outcome":
            return False
        tdt = train_df[col].dtype
        ldt = live_df[col].dtype
        # Allow both int, float, or bool, or if both are object but <20 unique (likely category)
        if pd.api.types.is_numeric_dtype(tdt) or pd.api.types.is_numeric_dtype(ldt):
            tu = train_df[col].nunique(dropna=True)
            lu = live_df[col].nunique(dropna=True)
            return (tu > 1) and (lu > 1) and (not train_df[col].isnull().all()) and (not live_df[col].isnull().all())
        # For object (category-like), must match and be low-cardinality
        if (tdt == 'object' and ldt == 'object'):
            tu = train_df[col].nunique(dropna=True)
            lu = live_df[col].nunique(dropna=True)
            return (tu > 1) and (lu > 1) and (tu < 20) and (lu < 20) and (not train_df[col].isnull().all()) and (not live_df[col].isnull().all())
        return False

    model_features = [c for c in intersect if is_feature_valid(c)]
    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # ---- DIAG DROP LISTS ----
    dropped_all_null = [c for c in intersect if train_df[c].isnull().all() and live_df[c].isnull().all()]
    dropped_constant = [c for c in intersect if train_df[c].nunique(dropna=True) <= 1 and live_df[c].nunique(dropna=True) <= 1]
    st.write("Dropped (all null both):")
    st.write(dropped_all_null)
    st.write("Dropped (constant in both):")
    st.write(dropped_constant)

    # ========== MODEL FIT ========== #
    from sklearn.linear_model import LogisticRegression

    if len(model_features) > 0:
        model = LogisticRegression(max_iter=200, solver="liblinear")
        X = train_df[model_features].fillna(0)
        y = train_df['hr_outcome'].astype(int)
        model.fit(X, y)
        X_live = live_df[model_features].fillna(0)
        hr_probs = model.predict_proba(X_live)[:, 1]
        live_df["hr_pred_prob"] = hr_probs

        # Output sweep
        st.markdown("### Results: HR Bot Picks by Threshold")
        out_report = {}
        thresholds = np.arange(min_thr, max_thr + step_thr, step_thr)
        for thr in thresholds:
            picks = live_df.loc[live_df["hr_pred_prob"] >= thr, "player_name"].tolist()
            out_report[thr] = picks
            st.write(f"Threshold {thr:.2f}: {picks}")

        st.markdown("Done! These are the official HR bot picks for today at each threshold.")
    else:
        st.error("No usable model features found. Check diagnostics above.")

    # ========== AUDIT DOWNLOADS ========== #
    audit_text = StringIO()
    print("Model features used:", model_features, file=audit_text)
    print("Dropped (all null both):", dropped_all_null, file=audit_text)
    print("Dropped (constant in both):", dropped_constant, file=audit_text)
    audit_text = audit_text.getvalue()
    st.markdown("### 📋 Model Feature Audit Summary")
    st.code(audit_text)
    st.download_button("⬇️ Download Audit Text (.txt)", audit_text, file_name="mlb_hr_model_feature_audit.txt")

    # Table for detailed feature audit
    st.markdown("#### Features/Stats Diagnostic Table")
    st.dataframe(pd.DataFrame(feat_diag))
