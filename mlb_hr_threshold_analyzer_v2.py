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

# Known non-numeric/categorical columns (expand if needed)
non_numeric_cols = [
    "player_name", "pitch_type", "stadium", "city", "park", "weather", "condition",
    "game_date", "team_code", "events", "description", "stand", "home_team",
    "away_team", "type", "bb_type", "inning_topbot", "if_fielding_alignment",
    "of_fielding_alignment", "pitch_name"
]

def convert_numeric(df):
    for c in df.columns:
        if c not in non_numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # Convert all possible columns to numeric
    train_df = convert_numeric(train_df)
    live_df = convert_numeric(live_df)

    # --- Feature Audit and Filtering ---
    st.markdown("### 🔍 Audit Report:")

    # Intersect columns
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]

    # Build detailed audit table
    audit_rows = []
    for c in intersect:
        train_dtype = train_df[c].dtype
        live_dtype = live_df[c].dtype
        train_unique = train_df[c].nunique(dropna=False)
        live_unique = live_df[c].nunique(dropna=False)
        train_null = train_df[c].isnull().mean()
        live_null = live_df[c].isnull().mean()
        train_zero = (train_df[c] == 0).mean() if pd.api.types.is_numeric_dtype(train_df[c]) else None
        live_zero = (live_df[c] == 0).mean() if pd.api.types.is_numeric_dtype(live_df[c]) else None
        audit_rows.append({
            'feature': c,
            'train dtype': str(train_dtype),
            'live dtype': str(live_dtype),
            'train unique': train_unique,
            'live unique': live_unique,
            'train % null': round(train_null, 4),
            'live % null': round(live_null, 4),
            'train % zero': round(train_zero, 4) if train_zero is not None else None,
            'live % zero': round(live_zero, 4) if live_zero is not None else None,
        })
    audit_df = pd.DataFrame(audit_rows)

    # Model features filter (now looser: allow all numeric features present in both, NOT all-null in both)
    potential_numeric = [
        c for c in intersect
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(live_df[c])
    ]
    model_features = [
        c for c in potential_numeric
        if not (train_df[c].isnull().all() and live_df[c].isnull().all())
        and c != "hr_outcome"
    ]
    audit_df["used_in_model"] = audit_df["feature"].isin(model_features)

    st.write(f"Intersect columns:\n\n{list(intersect)}")
    for row in audit_df.itertuples():
        st.write(
            f"Col: {row.feature} | train dtype: {row._2} | live dtype: {row._3} | "
            f"train unique: {row._4} | live unique: {row._5} | "
            f"train null: {row._6} | live null: {row._7}"
        )

    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # Diagnostic feature drop lists
    dropped_constant_train = [c for c in model_features if train_df[c].nunique(dropna=False) <= 1]
    dropped_constant_live = [c for c in model_features if live_df[c].nunique(dropna=False) <= 1]
    dropped_null_train = [c for c in model_features if train_df[c].isnull().all()]
    dropped_null_live = [c for c in model_features if live_df[c].isnull().all()]
    st.write("Features dropped (constant in train):", dropped_constant_train)
    st.write("Features dropped (constant in live):", dropped_constant_live)
    st.write("Features dropped (all-null in train):", dropped_null_train)
    st.write("Features dropped (all-null in live):", dropped_null_live)
    st.write("Null count in live file (top 20):")
    st.write(live_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write("Null count in train file (top 20):")
    st.write(train_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write(f"Train events: {len(train_df)}, Live events: {len(live_df)}")

    # --- Model Fit & Predict ---
    from sklearn.linear_model import LogisticRegression

    X = train_df[model_features].fillna(0)
    y = train_df['hr_outcome'].astype(int)
    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X, y)

    # Score live
    X_live = live_df[model_features].fillna(0)
    hr_probs = model.predict_proba(X_live)[:, 1]
    live_df["hr_pred_prob"] = hr_probs

    # Output sweep
    st.markdown("### Results: HR Bot Picks by Threshold")
    thresholds = np.arange(min_thr, max_thr + step_thr, step_thr)
    for thr in thresholds:
        picks = live_df.loc[live_df["hr_pred_prob"] >= thr, "player_name"].tolist()
        st.write(f"Threshold {thr:.2f}: {picks}")

    st.markdown("Done! These are the official HR bot picks for today at each threshold.")

    # --- ENHANCED AUDIT DOWNLOADS ---
    audit_csv = audit_df.to_csv(index=False)
    audit_txt = StringIO()
    print("===== MLB HR Model Feature Audit =====", file=audit_txt)
    for row in audit_df.itertuples():
        print(row, file=audit_txt)
    audit_txt = audit_txt.getvalue()
    st.markdown("### 📋 Detailed Model Audit Table")
    st.dataframe(audit_df)
    st.download_button("⬇️ Download Audit Table (.csv)", audit_csv, file_name="mlb_hr_feature_audit.csv")
    st.download_button("⬇️ Download Audit Table (.txt)", audit_txt, file_name="mlb_hr_feature_audit.txt")

    # Download feature summary with model usage
    st.download_button(
        "⬇️ Download Feature Model Usage (.csv)",
        audit_df[["feature", "used_in_model"]].to_csv(index=False),
        file_name="mlb_hr_feature_model_usage.csv"
    )
