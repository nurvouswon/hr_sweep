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
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.20, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

def safe_to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass
    return df

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # --- Feature Intersect + Type Diagnostic ---
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]

    # Candidate features: everything except clear meta/string columns
    meta_cols = [
        'player_name', 'position', 'p_throws', 'game_date', 'team_code',
        'mlb_id', 'city', 'stadium', 'time', 'weather'
    ]
    candidate_numeric = [c for c in intersect if c not in meta_cols]

    # Coerce to numeric everywhere for these columns
    train_df = safe_to_numeric(train_df, candidate_numeric)
    live_df = safe_to_numeric(live_df, candidate_numeric)

    st.markdown("#### 🔬 Feature Diagnostic Table")
    diagnostics = []
    for c in intersect:
        train_dtype = str(train_df[c].dtype) if c in train_df else "MISSING"
        live_dtype = str(live_df[c].dtype) if c in live_df else "MISSING"
        train_unique = train_df[c].nunique(dropna=False) if c in train_df else np.nan
        live_unique = live_df[c].nunique(dropna=False) if c in live_df else np.nan
        train_null = train_df[c].isnull().mean() if c in train_df else np.nan
        live_null = live_df[c].isnull().mean() if c in live_df else np.nan
        diagnostics.append({
            "feature": c,
            "train_dtype": train_dtype,
            "live_dtype": live_dtype,
            "train_unique": train_unique,
            "live_unique": live_unique,
            "train_null_frac": round(train_null, 4),
            "live_null_frac": round(live_null, 4),
        })
    diag_df = pd.DataFrame(diagnostics)
    st.dataframe(diag_df)

    # --- Feature Filtering (Optimized) ---
    usable_features = []
    dropped_allnull_both = []
    dropped_constant_both = []
    dropped_allnull_either = []
    dropped_constant_either = []
    kept_nonmeta = []
    for c in candidate_numeric:
        is_numeric = pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(live_df[c])
        all_null_train = train_df[c].isnull().all()
        all_null_live = live_df[c].isnull().all()
        constant_train = train_df[c].nunique(dropna=False) <= 1
        constant_live = live_df[c].nunique(dropna=False) <= 1

        if all_null_train and all_null_live:
            dropped_allnull_both.append(c)
        elif constant_train and constant_live:
            dropped_constant_both.append(c)
        elif all_null_train or all_null_live:
            dropped_allnull_either.append(c)
        elif constant_train or constant_live:
            dropped_constant_either.append(c)
        elif is_numeric:
            usable_features.append(c)

    # For model: require present, numeric, not all null, not constant on both
    model_features = [c for c in usable_features if c not in dropped_allnull_both and c not in dropped_constant_both]

    st.markdown("#### Intersect columns:")
    st.write(intersect)
    st.markdown(f"#### Model features used ({len(model_features)}): {model_features}")

    st.markdown("#### Dropped (all null both):")
    st.write(dropped_allnull_both)
    st.markdown("#### Dropped (constant in both):")
    st.write(dropped_constant_both)
    st.markdown("#### Dropped (all null in train/live only):")
    st.write(dropped_allnull_either)
    st.markdown("#### Dropped (constant in train/live only):")
    st.write(dropped_constant_either)

    st.write(f"Train events: {len(train_df)}, Live events: {len(live_df)}")

    # --- Model Fit & Predict ---
    from sklearn.linear_model import LogisticRegression

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

    # --- Download & Audit Output ---
    st.markdown("### 📋 Detailed Feature Audit Table")
    st.dataframe(diag_df)
    st.download_button("⬇️ Download Feature Diagnostic (.csv)", diag_df.to_csv(index=False), file_name="mlb_hr_feature_diagnostic.csv")
    st.download_button("⬇️ Download Bot Results (.csv)", live_df[["player_name", "hr_pred_prob"]].to_csv(index=False), file_name="mlb_hr_bot_results.csv")

    # Optional: Full feature summary
    feature_summary = pd.DataFrame({
        'feature': model_features,
        'train_null_frac': [train_df[c].isnull().mean() for c in model_features],
        'live_null_frac': [live_df[c].isnull().mean() for c in model_features],
        'train_unique': [train_df[c].nunique(dropna=False) for c in model_features],
        'live_unique': [live_df[c].nunique(dropna=False) for c in model_features],
    })
    st.dataframe(feature_summary)

    # End of Tab 2
