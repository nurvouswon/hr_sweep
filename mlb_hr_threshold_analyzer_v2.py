import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config("HR Prediction: Auto Feature Select + Triple Stacking", layout="wide")
st.header("2️⃣ Upload Event-Level CSVs & Run Model (Auto-FeatureSelect + Stacking)")

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

    # Clean column names
    df_train.columns = [c.lower() for c in df_train.columns]
    df_live.columns = [c.lower() for c in df_live.columns]

    # Remove hr_outcome from live if present
    if "hr_outcome" in df_live.columns:
        df_live = df_live.drop(columns=["hr_outcome"])

    # Feature diagnostics: show mismatches
    train_cols = set(df_train.columns)
    live_cols = set(df_live.columns)
    missing_in_live = sorted([c for c in train_cols if c not in live_cols])
    missing_in_train = sorted([c for c in live_cols if c not in train_cols])
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    st.write("**Features in train missing from live:**")
    st.write(missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**")
    st.write(missing_in_train if missing_in_train else "None")

    # Keep only intersection (except for outcome col)
    feature_cols = [c for c in df_train.columns if c in df_live.columns and c != "hr_outcome"]

    # Impute, encode, convert
    X_train = df_train[feature_cols].copy()
    X_live = df_live[feature_cols].copy()
    y_train = df_train["hr_outcome"].astype(int)

    # Handle categoricals
    cat_feats = [c for c in feature_cols if X_train[c].dtype == object or str(X_train[c].dtype) == "category"]
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        X_train[c] = X_train[c].astype(str).fillna("NA")
        X_live[c] = X_live[c].astype(str).fillna("NA")
        le.fit(list(X_train[c].values) + list(X_live[c].values))
        X_train[c] = le.transform(X_train[c])
        X_live[c] = le.transform(X_live[c])
        encoders[c] = le
    for c in feature_cols:
        if c not in cat_feats:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_live[c] = pd.to_numeric(X_live[c], errors="coerce")
            X_train[c] = X_train[c].fillna(X_train[c].mean())
            X_live[c] = X_live[c].fillna(X_train[c].mean())

    # === Feature Variance Diagnostics ===
    st.subheader("🧪 Feature Variance Diagnostics")
    low_var_cols = []
    for c in feature_cols:
        vals = pd.Series(X_train[c]).dropna().unique()
        if len(vals) <= 1:
            low_var_cols.append(c)
    st.write(f"Columns with 1 unique value (all-constant or all-NaN): {len(low_var_cols)}")
    if low_var_cols:
        st.code(low_var_cols)
    else:
        st.write("No all-constant columns! 👍")

    # Remove all-constant columns from modeling
    usable_feature_cols = [c for c in feature_cols if c not in low_var_cols]

    # === Feature Selection Slider ===
    st.subheader("🔬 Step 1: Select Features for Modeling")
    lgb_for_select = LGBMClassifier(n_estimators=100, random_state=42)
    lgb_for_select.fit(X_train[usable_feature_cols], y_train)
    importances = lgb_for_select.feature_importances_

    # Slider for number of features
    N = st.slider("Number of features to use (by importance)", 2, len(usable_feature_cols), min(30, len(usable_feature_cols)))
    indices = np.argsort(importances)[::-1][:N]
    kept_features = [usable_feature_cols[i] for i in indices if importances[i] > 0]
    if not kept_features:
        st.warning("No features had >0 importance! Using ALL usable features instead.")
        kept_features = usable_feature_cols

    X_train_sel = X_train[kept_features].values
    X_live_sel = X_live[kept_features].values

    st.write(f"Top {len(kept_features)} features by importance:")
    st.code(kept_features)

    # === Stacking Ensemble ===
    st.subheader("🤖 Step 2: Triple Model Stacking (XGB, LGBM, CatBoost)")
    xgb = ("xgb", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=42))
    lgb = ("lgb", LGBMClassifier(n_estimators=100, random_state=42))
    cat = ("cat", CatBoostClassifier(iterations=100, verbose=0, random_seed=42))
    stack = StackingClassifier(
        estimators=[xgb, lgb, cat],
        final_estimator=LGBMClassifier(n_estimators=100, random_state=42),
        passthrough=True,
        n_jobs=-1
    )
    stack.fit(X_train_sel, y_train)
    y_pred_proba = stack.predict_proba(X_live_sel)[:, 1]

    df_live_out = df_live.copy()
    df_live_out["hr_prob"] = y_pred_proba
    st.success(f"Stacking model fit OK with {len(kept_features)} selected features.")

    # Show feature importances for audit
    feature_imp_df = pd.DataFrame({"feature": kept_features, "importance": importances[indices][:len(kept_features)]})
    feature_imp_df = feature_imp_df.sort_values("importance", ascending=False)
    with st.expander("🔎 View Feature Importances (from selection LGBM)", expanded=False):
        st.dataframe(feature_imp_df)

    # --- Threshold controls and picks ---
    st.header("🔎 Bot Picks by Threshold")
    player_col = "player_name" if "player_name" in df_live_out.columns else ("batter" if "batter" in df_live_out.columns else kept_features[0])

    for thr in np.arange(min_thr, max_thr + step, step):
        picks = df_live_out[df_live_out["hr_prob"] >= thr].copy()
        if not picks.empty:
            pick_cols = [c for c in [player_col, "batter_id"] if c in picks.columns] + [c for c in picks.columns if c not in [player_col, "batter_id", "hr_prob"]] + ["hr_prob"]
            st.subheader(f"Players with HR Prob ≥ {thr:.2f} ({len(picks)})")
            st.dataframe(
                picks[pick_cols].sort_values("hr_prob", ascending=False).reset_index(drop=True)
            )
        else:
            st.write(f"No players at threshold ≥ {thr:.2f}")

    # --- Download full prediction CSV ---
    st.download_button(
        "⬇️ Download Full HR Prob Predictions (All Batters)",
        data=df_live_out.to_csv(index=False),
        file_name="hr_prob_predictions.csv"
    )

else:
    st.info("Please upload both training and today's event-level CSVs to continue.")
