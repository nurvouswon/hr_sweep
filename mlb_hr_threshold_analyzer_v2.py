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

min_thr = st.number_input("Min HR Prob Threshold", 0.00, 0.5, 0.01, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.15, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # --- Deep Feature Diagnostics ---
    st.subheader("🩺 Deep Feature Diagnostics")

    st.markdown("**Live file columns:**")
    for i in range(0, len(live_df.columns), 100):
        st.write(f"[{i} - {min(i+100, len(live_df.columns))}]")
        st.write(list(live_df.columns[i:i+100]))

    st.markdown("**Train file columns:**")
    for i in range(0, len(train_df.columns), 100):
        st.write(f"[{i} - {min(i+100, len(train_df.columns))}]")
        st.write(list(train_df.columns[i:i+100]))

    # Non-null cols
    st.markdown("**Live columns with non-null values:**")
    st.write(list(live_df.columns[live_df.notnull().any()]))
    st.markdown("**Train columns with non-null values:**")
    st.write(list(train_df.columns[train_df.notnull().any()]))

    # All-null cols
    st.markdown("**Live columns ALL null:**")
    st.write(list(live_df.columns[live_df.isnull().all()]))
    st.markdown("**Train columns ALL null:**")
    st.write(list(train_df.columns[train_df.isnull().all()]))

    # Sample dtype check
    if 'batter_id' in train_df and 'batter_id' in live_df:
        st.write(f"batter_id dtype in train: {train_df['batter_id'].dtype}")
        st.write(f"batter_id dtype in live: {live_df['batter_id'].dtype}")

    if 'player_name' in train_df and 'player_name' in live_df:
        st.write(f"player_name dtype in train: {train_df['player_name'].dtype}")
        st.write(f"player_name dtype in live: {live_df['player_name'].dtype}")

    # --- Feature-by-feature diagnostics ---
    st.markdown("### 🔍 Feature Audit Table (train/live overlap)")
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]
    feat_diag = []
    for c in intersect:
        diag = {
            'feature': c,
            'train_dtype': str(train_df[c].dtype),
            'live_dtype': str(live_df[c].dtype),
            'train_unique': train_df[c].nunique(dropna=False),
            'live_unique': live_df[c].nunique(dropna=False),
            'train_null_frac': np.round(train_df[c].isnull().mean(), 4),
            'live_null_frac': np.round(live_df[c].isnull().mean(), 4),
            'train_all_null': train_df[c].isnull().all(),
            'live_all_null': live_df[c].isnull().all(),
            'train_constant': train_df[c].nunique(dropna=False) <= 1,
            'live_constant': live_df[c].nunique(dropna=False) <= 1,
        }
        feat_diag.append(diag)
    feat_diag_df = pd.DataFrame(feat_diag)
    st.dataframe(feat_diag_df, use_container_width=True)

    # RELAXED filter: all numeric/categorical features that are not all null in both files
    candidate_features = [
        c for c in intersect
        if (pd.api.types.is_numeric_dtype(train_df[c]) or pd.api.types.is_numeric_dtype(live_df[c]))
        and (not train_df[c].isnull().all())
        and (not live_df[c].isnull().all())
        and c != 'hr_outcome'
    ]
    st.markdown("#### 🟡 Candidate Numeric Features (relaxed):")
    st.write(candidate_features)

    # STRICT: Model features must be numeric, at least 2 unique, not all-null in either file
    def is_feature_valid(col):
        if col == "hr_outcome":
            return False
        tdt = train_df[col].dtype
        ldt = live_df[col].dtype
        if pd.api.types.is_numeric_dtype(tdt) or pd.api.types.is_numeric_dtype(ldt):
            tu = train_df[col].nunique(dropna=True)
            lu = live_df[col].nunique(dropna=True)
            return (tu > 1) and (lu > 1) and (not train_df[col].isnull().all()) and (not live_df[col].isnull().all())
        # If object/categorical, allow only if both files have low-cardinality, at least 2 unique non-null, and not all-null
        if (tdt == 'object' and ldt == 'object'):
            tu = train_df[col].nunique(dropna=True)
            lu = live_df[col].nunique(dropna=True)
            return (tu > 1) and (lu > 1) and (tu < 20) and (lu < 20) and (not train_df[col].isnull().all()) and (not live_df[col].isnull().all())
        return False

    model_features = [c for c in intersect if is_feature_valid(c)]
    st.markdown("#### 🟢 Model Features Used (strict):")
    st.write(model_features)

    # --- Model Fit & Predict ---
    if len(model_features) < 2:
        st.error("❗ Less than 2 usable features detected for modeling! Check feature audit above or relax filter.")
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, solver="liblinear")
        X = train_df[model_features].fillna(0)
        y = train_df['hr_outcome'].astype(int)
        model.fit(X, y)

        # Score live
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

    st.markdown("---")
    st.markdown("**Full feature diagnostics and audit shown above. If features are missing, check null/constant counts in train/live or relax the filter.**")
