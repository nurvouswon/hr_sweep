import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config("MLB HR Predictor", layout="wide")
st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# --- Uploaders
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)", type=["csv"], key="train_csv"
)
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type=["csv"], key="live_csv"
)

# --- Threshold controls
min_thr = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
step = st.number_input("Threshold Step", min_value=0.001, max_value=0.2, value=0.01, step=0.01)

if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")

    # --- Feature Diagnostics
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    missing_in_live = sorted([c for c in train_cols if c not in live_cols])
    missing_in_train = sorted([c for c in live_cols if c not in train_cols])
    st.write("**Features in train missing from live:**")
    st.write(missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**")
    st.write(missing_in_train if missing_in_train else "None")

    # --- Standardize columns to lower case
    df_train.columns = [c.lower() for c in df_train.columns]
    df_live.columns = [c.lower() for c in df_live.columns]

    # --- Get intersection for modeling
    candidate_feats = [c for c in df_train.columns if c in df_live.columns and c not in ("hr_outcome",)]
    features_to_use = []
    for c in candidate_feats:
        if (df_train[c].notnull().sum() > 0) and (df_live[c].notnull().sum() > 0):
            if df_train[c].nunique(dropna=True) > 1 or df_live[c].nunique(dropna=True) > 1:
                features_to_use.append(c)

    # --- Dtype mismatch warnings
    dtype_problems = []
    for c in features_to_use:
        if c in df_train.columns and c in df_live.columns:
            if str(df_train[c].dtype) != str(df_live[c].dtype):
                dtype_problems.append(f"{c}: train={df_train[c].dtype}, live={df_live[c].dtype}")
    if dtype_problems:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_problems))

    st.write(f"**Final features used ({len(features_to_use)}):**")
    st.write(features_to_use)

    st.write("Null Fraction (train):")
    st.write(df_train[features_to_use].isnull().mean())
    st.write("Null Fraction (live):")
    st.write(df_live[features_to_use].isnull().mean())

    # --- Preprocess
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()
    cat_feats = [c for c in features_to_use if str(X_train[c].dtype) in ("object", "category", "string")]

    # --- Impute missing numerics with mean, cats with "NA"
    for c in features_to_use:
        if c in cat_feats:
            X_train[c] = X_train[c].astype(str).fillna("NA")
            X_live[c] = X_live[c].astype(str).fillna("NA")
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_train[c] = X_train[c].fillna(X_train[c].mean())
            X_live[c] = pd.to_numeric(X_live[c], errors="coerce")
            X_live[c] = X_live[c].fillna(X_train[c].mean())

    # --- Encode categoricals
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c])
        live_vals = pd.Series(X_live[c].unique())
        new_vals = live_vals[~live_vals.isin(le.classes_)]
        if not new_vals.empty:
            le_classes = np.concatenate([le.classes_, new_vals])
            le.classes_ = le_classes
        X_live[c] = le.transform(X_live[c])
        encoders[c] = le

    # --- Model Fit & Predict
    try:
        y_train = df_train["hr_outcome"].astype(int)
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_live)[:, 1]
        df_live_out = df_live.copy()
        df_live_out["hr_prob"] = y_pred_proba
        st.success(f"Model fit OK with {len(features_to_use)} features (XGBoost).")
    except Exception as e:
        st.error(f"Model fitting or prediction failed: {e}")
        st.stop()

    # --- Threshold controls and picks
    st.markdown("### Threshold Controls")
    st.write(f"Min HR Prob Threshold: {min_thr}")
    st.write(f"Max HR Prob Threshold: {max_thr}")
    st.write(f"Threshold Step: {step}")

    st.header("🔎 Bot Picks by Threshold")

    # Try to use player name, fallback to batter_id, then anything available
    if 'player_name' in df_live_out.columns:
        player_col = 'player_name'
    elif 'batter' in df_live_out.columns:
        player_col = 'batter'
    elif 'mlb_id' in df_live_out.columns:
        player_col = 'mlb_id'
    else:
        player_col = df_live_out.columns[0]

    for thr in np.arange(min_thr, max_thr + step, step):
        picks = df_live_out[df_live_out["hr_prob"] >= thr].copy()
        if not picks.empty:
            st.subheader(f"Players with HR Prob ≥ {thr:.2f} ({len(picks)})")
            want_cols = [player_col, "batter_id", "hr_prob"]
            display_cols = [c for c in want_cols if c in picks.columns]
            rest_cols = [c for c in picks.columns if c not in display_cols]
            show_cols = display_cols + rest_cols
            st.dataframe(
                picks[show_cols].sort_values("hr_prob", ascending=False).reset_index(drop=True)
            )
        else:
            st.write(f"No players at threshold ≥ {thr:.2f}")

else:
    st.info("Please upload both training and today's event-level CSVs to continue.")
