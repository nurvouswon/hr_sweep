import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# Upload training data
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train")
if train_file:
    train_df = pd.read_csv(train_file)
    st.success(f"Training file loaded! {train_df.shape[0]:,} rows, {train_df.shape[1]} columns.")

# Upload today's data
live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live")
if live_file:
    live_df = pd.read_csv(live_file)
    st.success(f"Today's file loaded! {live_df.shape[0]:,} rows, {live_df.shape[1]} columns.")

# --- Show feature columns diagnostic ---
if train_file and live_file:
    st.markdown("### 🩺 Deep Feature Diagnostics")
    st.markdown("**Live file columns:**")
    st.code(str(list(live_df.columns)))
    st.markdown("**Train file columns:**")
    st.code(str(list(train_df.columns)))

    # Null diagnostics
    st.markdown("**Live columns with non-null values:**")
    st.code(str([c for c in live_df.columns if live_df[c].notnull().sum() > 0]))
    st.markdown("**Train columns with non-null values:**")
    st.code(str([c for c in train_df.columns if train_df[c].notnull().sum() > 0]))
    st.markdown("**Live columns ALL null:**")
    st.code(str([c for c in live_df.columns if live_df[c].notnull().sum() == 0]))
    st.markdown("**Train columns ALL null:**")
    st.code(str([c for c in train_df.columns if train_df[c].notnull().sum() == 0]))

    # Dtype diagnostics for sample critical columns
    for col in ["batter_id", "player_name"]:
        if col in train_df.columns and col in live_df.columns:
            st.write(f"`{col}` dtype in train: {train_df[col].dtype}")
            st.write(f"`{col}` dtype in live: {live_df[col].dtype}")

    # Get intersection features (case-insensitive)
    train_lower = {c.lower():c for c in train_df.columns}
    live_lower = {c.lower():c for c in live_df.columns}
    shared_cols_lower = sorted(list(set(train_lower.keys()) & set(live_lower.keys())))

    # Exclude non-feature columns
    exclude_cols = ["hr_outcome"]
    candidate_features = [
        c for c in shared_cols_lower
        if c not in [e.lower() for e in exclude_cols]
        and train_df[train_lower[c]].notnull().sum() > 0
        and live_df[live_lower[c]].notnull().sum() > 0
        and (str(train_df[train_lower[c]].dtype).startswith("float") or str(train_df[train_lower[c]].dtype).startswith("int"))
    ]
    st.markdown("#### 🟡 Candidate Numeric Features (relaxed):")
    st.code(str([train_lower[c] for c in candidate_features]))

    # --- Robust feature mapping ---
    def match_col(colname, columns):
        for c in columns:
            if c.lower() == colname.lower():
                return c
        raise ValueError(f"Column {colname} not found in DataFrame.")

    train_feature_map = {f: match_col(f, train_df.columns) for f in [train_lower[c] for c in candidate_features]}
    live_feature_map = {f: match_col(f, live_df.columns) for f in [live_lower[c] for c in candidate_features]}
    model_features = list(train_feature_map.keys())

    st.markdown("#### 🟢 Model Features Used (strict):")
    st.code(str(model_features))
    st.write("Model will use this many features:", len(model_features))

    # Model training and prediction
    try:
        X = train_df[[train_feature_map[f] for f in model_features]].fillna(0)
        y = train_df['hr_outcome'].astype(int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        st.success(f"Model fit OK with {len(model_features)} features.")

        # Predict
        live_pred_X = live_df[[live_feature_map[f] for f in model_features]].fillna(0)
        probs = model.predict_proba(live_pred_X)[:,1]
        live_df['HR_Prob'] = probs

        st.write("Sample predictions:")
        st.dataframe(live_df[['player_name', 'HR_Prob']].head(10))

        # Threshold slider & output
        min_thr = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        max_thr = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        thr_step = st.number_input("Threshold Step", min_value=0.001, max_value=0.1, value=0.01, step=0.01)
        st.markdown("### Results: HR Bot Picks by Threshold")
        for t in np.arange(min_thr, max_thr+thr_step, thr_step):
            picks = live_df[live_df['HR_Prob'] >= t]['player_name'].tolist()
            st.write(f"Threshold {t:.2f}: {picks}")

    except Exception as e:
        st.error(f"Model fitting or prediction failed: {str(e)}")
        st.code(f"Features used: {model_features}")
        st.code(f"train_feature_map: {train_feature_map}")
        st.code(f"live_feature_map: {live_feature_map}")
