import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

    # Display feature diagnostics
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    missing_in_live = sorted([c for c in train_cols if c not in live_cols])
    missing_in_train = sorted([c for c in live_cols if c not in train_cols])
    st.write("**Features in train missing from live:**")
    st.write(missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**")
    st.write(missing_in_train if missing_in_train else "None")

    # Standardize column names to lower case for easier matching
    df_train.columns = [c.lower() for c in df_train.columns]
    df_live.columns = [c.lower() for c in df_live.columns]

    # Get intersection for modeling
    candidate_feats = [c for c in df_train.columns if c in df_live.columns and c not in ("hr_outcome",)]
    # Drop columns that are all null or all constant in either
    features_to_use = []
    for c in candidate_feats:
        if (df_train[c].notnull().sum() > 0) and (df_live[c].notnull().sum() > 0):
            # Remove columns that are constant in both (e.g., always 0 or always same string)
            if df_train[c].nunique(dropna=True) > 1 or df_live[c].nunique(dropna=True) > 1:
                features_to_use.append(c)

    # Show dtype mismatches
    dtype_problems = []
    for c in features_to_use:
        if c in df_train.columns and c in df_live.columns:
            if str(df_train[c].dtype) != str(df_live[c].dtype):
                dtype_problems.append(f"{c}: train={df_train[c].dtype}, live={df_live[c].dtype}")
    if dtype_problems:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_problems))

    st.write(f"**Final features used ({len(features_to_use)}):**")
    st.write(features_to_use)

    # Show null fraction
    st.write("Null Fraction (train):")
    st.write(df_train[features_to_use].isnull().mean())
    st.write("Null Fraction (live):")
    st.write(df_live[features_to_use].isnull().mean())

    # Preprocess for model
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()

    # Find categorical features (object or string)
    cat_feats = [c for c in features_to_use if str(X_train[c].dtype) in ("object", "category", "string")]

    # Impute missing numerics with mean; categorical with "NA"
    for c in features_to_use:
        if c in cat_feats:
            X_train[c] = X_train[c].astype(str).fillna("NA")
            X_live[c] = X_live[c].astype(str).fillna("NA")
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_train[c] = X_train[c].fillna(X_train[c].mean())
            X_live[c] = pd.to_numeric(X_live[c], errors="coerce")
            X_live[c] = X_live[c].fillna(X_train[c].mean())  # use train mean

    # Encode categorical
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c])
        # If live has unseen values, assign them a special code
        live_vals = pd.Series(X_live[c].unique())
        new_vals = live_vals[~live_vals.isin(le.classes_)]
        if not new_vals.empty:
            le_classes = np.concatenate([le.classes_, new_vals])
            le.classes_ = le_classes
        X_live[c] = le.transform(X_live[c])
        encoders[c] = le

    # Fit model
    try:
        y_train = df_train["hr_outcome"].astype(int)
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_live)[:, 1]
        df_live_out = df_live.copy()
        df_live_out["hr_prob"] = y_pred_proba
        st.success(f"Model fit OK with {len(features_to_use)} features.")
    except Exception as e:
        st.error(f"Model fitting or prediction failed: {e}")
        st.stop()

    # Show threshold controls and picks
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

    st.markdown("### 🧾 Bot Audit Report & Performance Readiness")

    # Feature presence audit
    with st.expander("🔍 Feature Presence and Type Audit"):
        st.write("**Train features missing in live:**")
        st.write(missing_in_live if missing_in_live else "None (perfect feature overlap)")
        st.write("**Live features missing in train:**")
        st.write(missing_in_train if missing_in_train else "None (perfect feature overlap)")
        if dtype_problems:
            st.write("**Features with dtype mismatches:**")
            st.write(dtype_problems)
        else:
            st.write("No dtype mismatches detected.")
        st.write("**Null fraction in train (key features):**")
        st.write(df_train[features_to_use].isnull().mean().sort_values(ascending=False).head(20))
        st.write("**Null fraction in live (key features):**")
        st.write(df_live[features_to_use].isnull().mean().sort_values(ascending=False).head(20))

    # Distribution comparison for top features (backtest vs. live)
    with st.expander("📊 Feature Distribution Snapshot (Train vs Live)"):
    # Pick a few most important numeric features
        top_feats = [c for c in features_to_use if "hr" in c or "rate" in c or "launch" in c]
        for feat in top_feats[:8]:
            if feat in df_train.columns and feat in df_live.columns:
                st.write(f"#### {feat}")
                st.write(f"Train: mean={df_train[feat].mean():.3f}, std={df_train[feat].std():.3f} | Live: mean={df_live[feat].mean():.3f}, std={df_live[feat].std():.3f}")

    # Categorical coverage
    with st.expander("🧬 Categorical Variable Coverage"):
        for c in cat_feats:
            train_cats = set(df_train[c].astype(str).unique())
            live_cats = set(df_live[c].astype(str).unique())
            unseen = live_cats - train_cats
            st.write(f"Feature: {c} | Unseen categories in live: {unseen if unseen else 'None'}")
 
    st.info(
        """
        **What does this mean?**  
        - If your features are matching, not all-null, and not full of unseen categories, your bot is running in a “backtest-like” environment and you can compare picks, hit rates, and precision just like your historical tests.
        - If you see mismatches or nulls, performance may differ and you can use this report to fine-tune your pipeline or model.
        - For *true production*, track the bot’s picks daily and compare realized HRs with predicted probabilities at each threshold—**that is your ultimate real-world validation**.
        """
    )

    # Show bot picks for each threshold
    for thr in np.arange(min_thr, max_thr + step, step):
        picks = df_live_out[df_live_out["hr_prob"] >= thr].copy()
        if not picks.empty:
            st.subheader(f"Players with HR Prob ≥ {thr:.2f} ({len(picks)} picks)")
            st.dataframe(
                picks[[player_col, "batter_id"] + [c for c in picks.columns if c not in [player_col, "batter_id", "hr_prob"]] + ["hr_prob"]]
                .sort_values("hr_prob", ascending=False).reset_index(drop=True)
            )
        else:
            st.write(f"No players at threshold ≥ {thr:.2f}")

else:
    st.info("Please upload both training and today's event-level CSVs to continue.")
