import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import datetime
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

st.set_page_config("🟦 MLB HR Predictor: Triple-Stack + Audit", layout="wide")
st.title("🟦 MLB HR Predictor: Triple Stack (XGB/LGBM/Cat) + Full Audit Log Button")

# --- File Uploaders ---
st.header("2️⃣ Upload Event-Level CSVs & Run Model")
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)", type=["csv"], key="train_csv"
)
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type=["csv"], key="live_csv"
)

# --- Threshold controls ---
st.sidebar.markdown("### HR Prob Threshold Controls")
min_thr = st.sidebar.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
max_thr = st.sidebar.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
step = st.sidebar.number_input("Threshold Step", min_value=0.001, max_value=0.2, value=0.01, step=0.01)

# --- Internal session state for audit log ---
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = {}

# --- Main model logic ---
if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")

    # --- Feature diagnostics ---
    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    missing_in_live = sorted([c for c in train_cols if c not in live_cols])
    missing_in_train = sorted([c for c in live_cols if c not in train_cols])
    st.write("**Features in train missing from live:**", missing_in_live if missing_in_live else "None")
    st.write("**Features in live missing from train:**", missing_in_train if missing_in_train else "None")

    # --- Standardize column names ---
    df_train.columns = [c.lower() for c in df_train.columns]
    df_live.columns = [c.lower() for c in df_live.columns]

    # --- Candidate features ---
    candidate_feats = [c for c in df_train.columns if c in df_live.columns and c not in ("hr_outcome",)]
    features_to_use = []
    for c in candidate_feats:
        if (df_train[c].notnull().sum() > 0) and (df_live[c].notnull().sum() > 0):
            if df_train[c].nunique(dropna=True) > 1 or df_live[c].nunique(dropna=True) > 1:
                features_to_use.append(c)

    # --- Dtype audit ---
    dtype_problems = []
    for c in features_to_use:
        if str(df_train[c].dtype) != str(df_live[c].dtype):
            dtype_problems.append(f"{c}: train={df_train[c].dtype}, live={df_live[c].dtype}")
    if dtype_problems:
        st.warning("⚠️ Dtype mismatches detected! " + "; ".join(dtype_problems))

    st.write(f"**Final features used ({len(features_to_use)}):**")
    st.code(features_to_use)

    # --- Null fraction audit ---
    null_train = df_train[features_to_use].isnull().mean()
    null_live = df_live[features_to_use].isnull().mean()
    st.markdown("**Null Fraction (train):**")
    st.code(null_train.to_string())
    st.markdown("**Null Fraction (live):**")
    st.code(null_live.to_string())

    # --- Preprocess ---
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()
    cat_feats = [c for c in features_to_use if str(X_train[c].dtype) in ("object", "category", "string")]

    for c in features_to_use:
        if c in cat_feats:
            X_train[c] = X_train[c].astype(str).fillna("NA")
            X_live[c] = X_live[c].astype(str).fillna("NA")
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(X_train[c].mean())
            X_live[c] = pd.to_numeric(X_live[c], errors="coerce").fillna(X_train[c].mean())

    # --- Encode categoricals ---
    encoders = {}
    for c in cat_feats:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c])
        live_vals = pd.Series(X_live[c].unique())
        new_vals = live_vals[~live_vals.isin(le.classes_)]
        if not new_vals.empty:
            le.classes_ = np.concatenate([le.classes_, new_vals])
        X_live[c] = le.transform(X_live[c])
        encoders[c] = le

    y_train = df_train["hr_outcome"].astype(int)

    # --- MODEL 1: XGBoost ---
    model_xgb = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=42
    )
    model_xgb.fit(X_train, y_train)
    proba_xgb = model_xgb.predict_proba(X_live)[:, 1]

    # --- MODEL 2: LightGBM ---
    model_lgbm = LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_lgbm.fit(X_train, y_train)
    proba_lgbm = model_lgbm.predict_proba(X_live)[:, 1]

    # --- MODEL 3: CatBoost ---
    model_cat = CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.1,
        verbose=0, random_seed=42
    )
    model_cat.fit(X_train, y_train)
    proba_cat = model_cat.predict_proba(X_live)[:, 1]

    # --- Ensemble/stacked prediction ---
    proba_ensemble = (proba_xgb + proba_lgbm + proba_cat) / 3

    df_live_out = df_live.copy()
    df_live_out["hr_prob_xgb"] = proba_xgb
    df_live_out["hr_prob_lgbm"] = proba_lgbm
    df_live_out["hr_prob_cat"] = proba_cat
    df_live_out["hr_prob"] = proba_ensemble

    st.success(f"Model fit OK with {len(features_to_use)} features (XGB/LGBM/CatBoost).")

    # --- Feature importance for audit ---
    def feat_importance_df(model, cols, name):
        try:
            imp = model.feature_importances_
            return pd.DataFrame({"feature": cols, f"importance_{name}": imp})
        except Exception:
            return pd.DataFrame({"feature": cols, f"importance_{name}": np.nan})

    feat_imp_xgb = feat_importance_df(model_xgb, features_to_use, "xgb")
    feat_imp_lgbm = feat_importance_df(model_lgbm, features_to_use, "lgbm")
    feat_imp_cat = feat_importance_df(model_cat, features_to_use, "cat")

    # --- Leaderboards ---
    st.header("🔎 Bot Picks by Threshold (Ensemble)")
    player_col = next((c for c in ["player_name", "batter", "mlb_id"] if c in df_live_out.columns), features_to_use[0])

    leaderboards = {}
    for thr in np.arange(min_thr, max_thr + step, step):
        picks = df_live_out[df_live_out["hr_prob"] >= thr].copy()
        picks = picks.sort_values("hr_prob", ascending=False)
        leaderboards[f"thr_{thr:.2f}"] = picks
        if not picks.empty:
            st.subheader(f"Players with HR Prob ≥ {thr:.2f} ({len(picks)})")
            st.dataframe(
                picks[[player_col, "batter_id"] + [c for c in picks.columns if c not in [player_col, "batter_id", "hr_prob"]] + ["hr_prob"]]
                .reset_index(drop=True)
            )
        else:
            st.write(f"No players at threshold ≥ {thr:.2f}")

    # --- Audit log objects to package ---
    audit_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_folder = tempfile.mkdtemp(prefix="audit_")
    logs = {
        "config.txt": [
            f"Run time: {audit_time}",
            f"Features used: {features_to_use}",
            f"Categorical features: {cat_feats}",
            f"Model thresholds: min={min_thr}, max={max_thr}, step={step}"
        ],
        "feature_audit_train.csv": null_train.to_csv(),
        "feature_audit_live.csv": null_live.to_csv(),
        "feature_importance_xgb.csv": feat_imp_xgb.to_csv(index=False),
        "feature_importance_lgbm.csv": feat_imp_lgbm.to_csv(index=False),
        "feature_importance_cat.csv": feat_imp_cat.to_csv(index=False),
        "predictions.csv": df_live_out.to_csv(index=False),
        "leaderboards.txt": "\n\n".join([f"=== {k} ===\n{v[[player_col, 'batter_id', 'hr_prob']].head(20).to_string()}" for k, v in leaderboards.items()]),
    }
    # Write files to temp audit folder
    for fname, content in logs.items():
        with open(os.path.join(audit_folder, fname), "w", encoding="utf-8") as f:
            if isinstance(content, list):
                f.write("\n".join(content))
            else:
                f.write(content)

    # Zip the folder
    audit_zip_path = os.path.join(audit_folder, f"audit_log_{audit_time}.zip")
    with zipfile.ZipFile(audit_zip_path, "w") as zf:
        for fname in logs:
            zf.write(os.path.join(audit_folder, fname), fname)

    with open(audit_zip_path, "rb") as f:
        st.download_button(
            "⬇️ Download Full Audit Log (ZIP)",
            data=f,
            file_name=f"audit_log_{audit_time}.zip"
        )

    # Cleanup temp folder on rerun/exit
    shutil.rmtree(audit_folder, ignore_errors=True)

else:
    st.info("Please upload both training and today's event-level CSVs to continue.")
