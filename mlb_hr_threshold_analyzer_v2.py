import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# Uploaders
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)", type=["csv"], key="train_csv"
)
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type=["csv"], key="live_csv"
)

# Threshold controls
min_thr = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
step = st.number_input("Threshold Step", min_value=0.001, max_value=0.2, value=0.01, step=0.01)

if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")

    # Diagnostic table
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    missing_in_live = sorted([c for c in train_cols if c not in live_cols])
    missing_in_train = sorted([c for c in live_cols if c not in train_cols])
    st.write("**Features in train missing from live:**")
    st.write(missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**")
    st.write(missing_in_train if missing_in_train else "None")

    # Standardize column names
    df_train.columns = [c.lower() for c in df_train.columns]
    df_live.columns = [c.lower() for c in df_live.columns]

    # Intersecting features
    candidate_feats = [c for c in df_train.columns if c in df_live.columns and c not in ("hr_outcome",)]
    # Remove all-null/all-constant columns
    features_to_use = []
    for c in candidate_feats:
        if (df_train[c].notnull().sum() > 0) and (df_live[c].notnull().sum() > 0):
            if df_train[c].nunique(dropna=True) > 1 or df_live[c].nunique(dropna=True) > 1:
                features_to_use.append(c)

    # Dtype harmonization
    cat_feats = [c for c in features_to_use if str(df_train[c].dtype) in ("object", "category", "string")]
    num_feats = [c for c in features_to_use if c not in cat_feats]

    dtype_problems = []
    for c in features_to_use:
        if c in df_train.columns and c in df_live.columns:
            # Only care if not both categorical or both numeric
            dt1, dt2 = str(df_train[c].dtype), str(df_live[c].dtype)
            if (dt1 != dt2) and not (
                dt1 in ("object", "string", "category") and dt2 in ("object", "string", "category")
            ):
                dtype_problems.append(f"{c}: train={dt1}, live={dt2}")
    if dtype_problems:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_problems))

    st.write(f"**Final features used ({len(features_to_use)}):**")
    st.write(features_to_use)
    st.write("Null Fraction (train):")
    st.write(df_train[features_to_use].isnull().mean())
    st.write("Null Fraction (live):")
    st.write(df_live[features_to_use].isnull().mean())

    # Impute/convert numeric features to float32
    for c in num_feats:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce").astype(np.float32)
        df_train[c] = df_train[c].fillna(df_train[c].mean())
        df_live[c] = pd.to_numeric(df_live[c], errors="coerce").astype(np.float32)
        df_live[c] = df_live[c].fillna(df_train[c].mean())

    # Categorical features: string with NA
    for c in cat_feats:
        df_train[c] = df_train[c].astype(str).fillna("NA")
        df_live[c] = df_live[c].astype(str).fillna("NA")

    # LabelEncode categoricals (train on train set; expand for new live values)
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        df_train[c] = le.fit_transform(df_train[c])
        # For live: add any unseen categories to encoder
        live_vals = pd.Series(df_live[c].unique())
        new_vals = live_vals[~live_vals.isin(le.classes_)]
        if not new_vals.empty:
            le_classes = np.concatenate([le.classes_, new_vals])
            le.classes_ = le_classes
        df_live[c] = le.transform(df_live[c])
        encoders[c] = le

    # Strict feature order for both sets
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()
    X_live = X_live[X_train.columns.tolist()]

    # Train XGBoost
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

    # Show picks at each threshold
    st.markdown("### Threshold Controls")
    st.write(f"Min HR Prob Threshold: {min_thr}")
    st.write(f"Max HR Prob Threshold: {max_thr}")
    st.write(f"Threshold Step: {step}")

    st.header("🔎 Bot Picks by Threshold")
    if 'player_name' in df_live_out.columns:
        player_col = 'player_name'
    elif 'batter' in df_live_out.columns:
        player_col = 'batter'
    else:
        player_col = features_to_use[0]  # fallback

    for thr in np.arange(min_thr, max_thr + step, step):
        picks = df_live_out[df_live_out["hr_prob"] >= thr].copy()
        if not picks.empty:
            st.subheader(f"Players with HR Prob ≥ {thr:.2f} ({len(picks)} picks)")
            display_cols = (
                [player_col, "batter_id"]
                + [c for c in picks.columns if c not in [player_col, "batter_id", "hr_prob"]]
                + ["hr_prob"]
            )
            st.dataframe(picks[display_cols].sort_values("hr_prob", ascending=False).reset_index(drop=True))
        else:
            st.write(f"No players at threshold ≥ {thr:.2f}")

else:
    st.info("Please upload both training and today's event-level CSVs to continue.")
