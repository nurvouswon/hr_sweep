import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# ---- Uploaders ----
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_csv")
live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_csv")

min_thr = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
step_thr = st.number_input("Threshold Step", min_value=0.001, max_value=0.2, value=0.01, step=0.001)

if train_file and live_file:
    df_train = pd.read_csv(train_file)
    df_live = pd.read_csv(live_file)
    st.write(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    st.write(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")

    # --- Diagnostics: Feature Intersections ---
    train_cols = set(df_train.columns.str.lower())
    live_cols = set(df_live.columns.str.lower())
    train_only = sorted(train_cols - live_cols)
    live_only = sorted(live_cols - train_cols)
    both = sorted(train_cols & live_cols)

    st.markdown("### 🩺 Feature Diagnostics Table (train/live overlap)")
    if train_only:
        st.write("**Features in train missing from live:**", train_only)
    if live_only:
        st.write("**Features in live missing from train:**", live_only)

    # Dtype audit (optional, advanced debug)
    dtype_mismatches = []
    for c in both:
        if c in df_train.columns and c in df_live.columns:
            if str(df_train[c].dtype) != str(df_live[c].dtype):
                dtype_mismatches.append((c, str(df_train[c].dtype), str(df_live[c].dtype)))
    if dtype_mismatches:
        st.warning("⚠️ Dtype mismatches detected!\n" + "\n".join([f"{x[0]}: train={x[1]}, live={x[2]}" for x in dtype_mismatches]))

    # Normalize column names to lowercase for matching
    df_train.columns = df_train.columns.str.lower()
    df_live.columns = df_live.columns.str.lower()

    # Use only features present in BOTH train and live, and not all-null in either
    candidate_feats = [c for c in both if df_train[c].notnull().sum() > 0 and df_live[c].notnull().sum() > 0]
    # Remove obvious metadata/object columns not suitable for regression
    not_allowed = {'game_date','player_name','description','events','stadium','city','team_code','mlb_id','weather','position','batting_order','time','game_number','home_team','away_team','pitch_type'}
    features_to_use = [c for c in candidate_feats if c not in not_allowed]
    # Exclude object columns not categorical
    object_cols = [c for c in features_to_use if df_train[c].dtype == "object" or df_live[c].dtype == "object"]
    # Only keep categorical object columns with <20 unique values in both
    cat_feats = [c for c in object_cols if max(df_train[c].nunique(), df_live[c].nunique()) < 20]
    # Remove non-categorical objects from features to use
    features_to_use = [c for c in features_to_use if c not in object_cols or c in cat_feats]

    st.write(f"Final features used ({len(features_to_use)}):")
    st.write(features_to_use)

    # Audit for nulls
    st.write("#### Null Fraction (train):")
    st.write(df_train[features_to_use].isnull().mean())
    st.write("#### Null Fraction (live):")
    st.write(df_live[features_to_use].isnull().mean())

    # Prepare features
    X_train = df_train[features_to_use].copy()
    X_live = df_live[features_to_use].copy()

    # Handle categoricals safely (fit on union, fillna "NA")
    for c in cat_feats:
        train_vals = X_train[c].fillna("NA").astype(str)
        live_vals = X_live[c].fillna("NA").astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train_vals, live_vals]).unique())
        X_train[c] = le.transform(train_vals)
        X_live[c] = le.transform(live_vals)

    # Fill numeric nulls with train median
    num_feats = [c for c in features_to_use if c not in cat_feats]
    for c in num_feats:
        X_train[c] = X_train[c].fillna(X_train[c].median())
        X_live[c] = X_live[c].fillna(X_train[c].median())

    # Ready for model
    if "hr_outcome" in df_train.columns:
        y_train = df_train["hr_outcome"]
        try:
            model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)
            st.success(f"Model fit OK with {len(features_to_use)} features.")

            # Predict live probs
            probs = model.predict_proba(X_live)[:,1]
            df_preds = df_live.copy()
            df_preds["HR_Prob"] = probs

            # Threshold controls
            st.write("### Threshold Controls")
            thrs = np.arange(min_thr, max_thr + step_thr/2, step_thr)
            out = []
            for t in thrs:
                n = (df_preds["HR_Prob"] >= t).sum()
                out.append({"Threshold": round(t, 3), "Num_Predicted": int(n)})
            st.dataframe(pd.DataFrame(out))

            st.write("### Live HR Probabilities")
            st.dataframe(df_preds[["player_name","HR_Prob"] + [c for c in features_to_use if c in df_preds.columns]].sort_values("HR_Prob", ascending=False).head(25))

        except Exception as e:
            st.error(f"Model fitting or prediction failed: {e}")

    else:
        st.warning("Training CSV does not contain 'hr_outcome' column.")

else:
    st.info("Please upload BOTH training and today's CSV files.")
