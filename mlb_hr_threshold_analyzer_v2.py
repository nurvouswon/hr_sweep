import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# === Upload Files ===
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_file")
if train_file:
    train_df = pd.read_csv(train_file)
    st.success(f"Training file loaded! {train_df.shape[0]:,} rows, {train_df.shape[1]} columns.")
    st.write("Train columns:", list(train_df.columns))

live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_file")
if live_file:
    live_df = pd.read_csv(live_file)
    st.success(f"Today's file loaded! {live_df.shape[0]:,} rows, {live_df.shape[1]} columns.")
    st.write("Live columns:", list(live_df.columns))

# === Proceed if both files are loaded ===
if train_file and live_file:
    st.markdown("### 🩺 Deep Feature Diagnostics")

    # -- Map columns: case-insensitive intersection
    train_cols = list(train_df.columns)
    live_cols = list(live_df.columns)
    train_col_map = {c.lower(): c for c in train_cols}
    live_col_map = {c.lower(): c for c in live_cols}

    # Exclude label column and any known non-feature columns
    label_col = 'hr_outcome'
    exclude_cols = set([label_col, 'player_name', 'game_date', 'team_code'])

    # Compute case-insensitive intersection for candidate features
    feature_keys = [c for c in train_col_map if c in live_col_map and c not in exclude_cols]
    st.write(f"Columns present in BOTH train/live (case-insensitive): {len(feature_keys)}")

    # Model will use these features, mapped to actual DataFrame column names
    model_features_train = [train_col_map[c] for c in feature_keys]
    model_features_live  = [live_col_map[c]  for c in feature_keys]

    # Show diagnostic
    st.markdown("**Train Features Used:**")
    st.code(model_features_train)
    st.markdown("**Live Features Used:**")
    st.code(model_features_live)
    st.write(f"Model will use {len(model_features_train)} features.")

    # Check missing in either side (diagnostic)
    missing_in_train = [c for c in live_col_map if c not in train_col_map and c not in exclude_cols]
    missing_in_live  = [c for c in train_col_map if c not in live_col_map and c not in exclude_cols]
    st.markdown("**Features in live but not train (case-insensitive):**")
    st.code(missing_in_train)
    st.markdown("**Features in train but not live (case-insensitive):**")
    st.code(missing_in_live)

    # Confirm label present in train
    if label_col not in train_cols:
        st.error(f"Training file is missing '{label_col}' column. Check your data!")
    else:
        # === Prepare model inputs ===
        try:
            X_train = train_df[model_features_train].fillna(0)
            y_train = train_df[label_col].astype(int)
            X_live  = live_df[model_features_live].fillna(0)

            # == Fit Model ==
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            st.success(f"Model fit OK with {len(model_features_train)} features.")

            # == Predict ==
            preds = model.predict_proba(X_live)[:, 1]
            live_df['HR_Prob'] = preds

            # == Threshold controls ==
            min_thr = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            max_thr = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
            thr_step = st.number_input("Threshold Step", min_value=0.001, max_value=0.1, value=0.01, step=0.01)

            # == Display picks for each threshold ==
            st.markdown("### Results: HR Bot Picks by Threshold")
            for t in np.arange(min_thr, max_thr + thr_step, thr_step):
                picks = live_df[live_df['HR_Prob'] >= t]['player_name'].tolist()
                st.write(f"Threshold {t:.2f}: {', '.join(picks)}")

            # == Show full leaderboard ==
            st.markdown("### 🔝 Full HR Leaderboard")
            st.dataframe(live_df[['player_name', 'HR_Prob']].sort_values('HR_Prob', ascending=False).reset_index(drop=True).head(30))

        except Exception as e:
            st.error(f"Model fitting or prediction failed: {str(e)}")
            st.code(f"Train features: {model_features_train}")
            st.code(f"Live features: {model_features_live}")
            st.code(f"Exception details: {repr(e)}")
            # Show sample rows if possible
            st.write("Sample train row:", train_df.head(1).to_dict())
            st.write("Sample live row:", live_df.head(1).to_dict())

else:
    st.info("Upload both a training and a live event-level CSV to continue.")
