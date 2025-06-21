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

min_thr = st.number_input("Min HR Prob Threshold", 0.00, 0.5, 0.05, step=0.01)
max_thr = st.number_input("Max HR Prob Threshold", 0.01, 0.5, 0.20, step=0.01)
step_thr = st.number_input("Threshold Step", 0.01, 0.1, 0.01, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # Show intersect columns and basic audit
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect_cols = [c for c in train_cols if c in live_cols]

    st.subheader("🔍 Audit Report:")
    st.write("**Intersect columns:**", intersect_cols)

    # Feature-by-feature audit table
    audit_rows = []
    for col in intersect_cols:
        t_dtype = str(train_df[col].dtype)
        l_dtype = str(live_df[col].dtype)
        t_unique = train_df[col].nunique(dropna=False)
        l_unique = live_df[col].nunique(dropna=False)
        t_null = train_df[col].isnull().mean()
        l_null = live_df[col].isnull().mean()
        audit_rows.append({
            "col": col,
            "train dtype": t_dtype,
            "live dtype": l_dtype,
            "train unique": t_unique,
            "live unique": l_unique,
            "train null": round(t_null, 4),
            "live null": round(l_null, 4)
        })
    audit_table = pd.DataFrame(audit_rows)
    st.dataframe(audit_table)

    # Step 1: Numeric in BOTH, not hr_outcome
    num_both = []
    for col in intersect_cols:
        if (
            pd.api.types.is_numeric_dtype(train_df[col]) and
            pd.api.types.is_numeric_dtype(live_df[col]) and
            col != "hr_outcome"
        ):
            num_both.append(col)

    st.write("**Numeric in both train & live:**", num_both)

    # Step 2: Not all-null in either
    not_null = [c for c in num_both if not train_df[c].isnull().all() and not live_df[c].isnull().all()]
    st.write("**Not all-null in both files:**", not_null)

    # Step 3: Not constant in BOTH files
    not_constant = [c for c in not_null if train_df[c].nunique(dropna=False) > 1 and live_df[c].nunique(dropna=False) > 1]
    st.write("**Not constant in both files:**", not_constant)

    # Model features used
    model_features = not_constant
    st.write(f"**Model features used ({len(model_features)}):** {model_features}")

    # --- Diagnostics for why columns are dropped ---
    dropped_all_null = [c for c in num_both if train_df[c].isnull().all() or live_df[c].isnull().all()]
    dropped_constant = [c for c in not_null if train_df[c].nunique(dropna=False) <= 1 or live_df[c].nunique(dropna=False) <= 1]
    dropped_not_numeric = [c for c in intersect_cols if c not in num_both and c != "hr_outcome"]

    st.write("**Dropped (all null in either file):**", dropped_all_null)
    st.write("**Dropped (constant in either file):**", dropped_constant)
    st.write("**Dropped (not numeric in both):**", dropped_not_numeric)

    # Optional: show dtypes for dropped not numeric columns
    if dropped_not_numeric:
        nonnum_types = [(c, train_df[c].dtype, live_df[c].dtype) for c in dropped_not_numeric]
        st.write("**Dropped non-numeric dtypes:**", nonnum_types)

    # Final X, y, and model run
    if model_features:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, solver="liblinear")
        X = train_df[model_features].fillna(0)
        y = train_df['hr_outcome'].astype(int)
        model.fit(X, y)

        # Score live
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
    else:
        st.error("No model features passed filtering. Check audit and dropped features above.")

    st.markdown("Done! These are the official HR bot picks for today at each threshold.")

    # Download full audit table
    st.download_button(
        "⬇️ Download Audit Table (.csv)",
        audit_table.to_csv(index=False),
        file_name="mlb_hr_feature_audit_table.csv"
    )
