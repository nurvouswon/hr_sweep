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

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)
    
    # --- Diagnostic: Print intersecting columns and dtypes ---
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]
    
    st.write("#### Intersecting Columns & Dtypes")
    diag = []
    for col in intersect:
        t_dtype = train_df[col].dtype
        l_dtype = live_df[col].dtype
        t_unique = train_df[col].nunique(dropna=True)
        l_unique = live_df[col].nunique(dropna=True)
        t_null = train_df[col].isnull().mean()
        l_null = live_df[col].isnull().mean()
        diag.append([col, str(t_dtype), str(l_dtype), t_unique, l_unique, t_null, l_null])
    diag_df = pd.DataFrame(diag, columns=["col","train_dtype","live_dtype","train_unique","live_unique","train_null_frac","live_null_frac"])
    st.dataframe(diag_df)
    
    # --- Attempt to auto-convert all intersect columns to numeric, where possible ---
    # Create a copy before conversion for diagnostics
    train_df_orig = train_df.copy()
    live_df_orig = live_df.copy()
    for col in intersect:
        # Try to coerce to numeric if possible
        train_df[col] = pd.to_numeric(train_df[col], errors="ignore")
        live_df[col]  = pd.to_numeric(live_df[col],  errors="ignore")
        # If still object, try again (catch mixed types)
        if train_df[col].dtype == 'object':
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        if live_df[col].dtype == 'object':
            live_df[col] = pd.to_numeric(live_df[col], errors="coerce")
        # If one is int and the other is float, upcast both to float for alignment
        if (train_df[col].dtype == 'int64' and live_df[col].dtype == 'float64') or \
           (train_df[col].dtype == 'float64' and live_df[col].dtype == 'int64'):
            train_df[col] = train_df[col].astype(float)
            live_df[col]  = live_df[col].astype(float)
    
    # --- New numeric column selection after conversion ---
    numeric_cols = [c for c in intersect if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(live_df[c]) and c != "hr_outcome"]
    
    # Exclude constant or all-null features (in both train and live)
    good_cols = []
    dropped_constant = []
    dropped_null = []
    for c in numeric_cols:
        is_const = (train_df[c].nunique(dropna=False) <= 1) and (live_df[c].nunique(dropna=False) <= 1)
        is_null = (train_df[c].isnull().all()) and (live_df[c].isnull().all())
        if is_const or is_null:
            if is_const:
                dropped_constant.append(c)
            if is_null:
                dropped_null.append(c)
            continue
        good_cols.append(c)
    model_features = good_cols
    
    # --- Show Diagnostic Info ---
    st.write(f"Model features used ({len(model_features)}): {model_features}")
    st.write("Dropped (all null both):", dropped_null)
    st.write("Dropped (constant in both):", dropped_constant)
    st.write("Null count in live (top 20):")
    st.write(live_df[model_features].isnull().sum().sort_values(ascending=False).head(20))
    st.write("Null count in train (top 20):")
    st.write(train_df[model_features].isnull().sum().sort_values(ascending=False).head(20))
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

    # --- Full audit table download ---
    st.download_button("⬇️ Download Feature Diagnostic Table (.csv)", diag_df.to_csv(index=False), file_name="mlb_hr_feature_audit_table.csv")
