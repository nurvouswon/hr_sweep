import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# ---- 1. FILE UPLOADS ----
train_file = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train")
live_file = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live")

min_threshold = st.number_input("Min HR Prob Threshold", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
max_threshold = st.number_input("Max HR Prob Threshold", value=0.20, min_value=0.0, max_value=1.0, step=0.01)
threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.001, max_value=1.0, step=0.01)

if train_file and live_file:
    train_df = pd.read_csv(train_file)
    live_df = pd.read_csv(live_file)

    # --------- 2. FEATURE DIAGNOSTICS AND ROBUST SELECTION ----------
    st.subheader("🩺 Deep Feature Diagnostics")

    # Lowercase columns for intersection
    train_cols = set([c.lower() for c in train_df.columns])
    live_cols = set([c.lower() for c in live_df.columns])
    common_cols = train_cols & live_cols

    # Use original casing from train file (for better modeling)
    def find_real_case(colname, columns):
        for c in columns:
            if c.lower() == colname:
                return c
        return colname

    # Get canonical names
    canon_cols = [find_real_case(col, train_df.columns) for col in common_cols]

    feature_diagnostics = []
    candidate_features = []

    for col in canon_cols:
        t = train_df[col]
        l = live_df[col] if col in live_df.columns else live_df[[c for c in live_df.columns if c.lower() == col.lower()][0]]

        # Try to align dtypes if possible
        if t.dtype != l.dtype:
            try:
                if np.issubdtype(t.dtype, np.number) and np.issubdtype(l.dtype, np.number):
                    t = t.astype(float)
                    l = l.astype(float)
            except Exception:
                pass

        t_null_frac = t.isnull().mean()
        l_null_frac = l.isnull().mean()
        t_unique = t.nunique(dropna=True)
        l_unique = l.nunique(dropna=True)
        # Allow all but all-null columns
        if (t_null_frac < 1.0) and (l_null_frac < 1.0):
            if np.issubdtype(t.dtype, np.number) or np.issubdtype(l.dtype, np.number):
                candidate_features.append(col)
        feature_diagnostics.append({
            "feature": col,
            "train_dtype": str(t.dtype),
            "live_dtype": str(l.dtype),
            "train_unique": t_unique,
            "live_unique": l_unique,
            "train_null_frac": round(t_null_frac, 3),
            "live_null_frac": round(l_null_frac, 3),
        })

    # ---- Diagnostic Outputs ----
    st.markdown("**Live file columns:**")
    st.code(str(list(live_df.columns)))
    st.markdown("**Train file columns:**")
    st.code(str(list(train_df.columns)))
    st.markdown("**Live columns with non-null values:**")
    st.code(str([c for c in live_df.columns if live_df[c].notnull().any()]))
    st.markdown("**Train columns with non-null values:**")
    st.code(str([c for c in train_df.columns if train_df[c].notnull().any()]))
    st.markdown("**Live columns ALL null:**")
    st.code(str([c for c in live_df.columns if live_df[c].isnull().all()]))
    st.markdown("**Train columns ALL null:**")
    st.code(str([c for c in train_df.columns if train_df[c].isnull().all()]))
    st.markdown("**Feature audit of live data:**")
    st.dataframe(pd.DataFrame(feature_diagnostics))

    st.markdown("#### 🟡 Candidate Numeric Features (relaxed):")
    st.write(candidate_features)

    # -------------- 3. MODELING AND PREDICTIONS -----------------
    # Check minimum features for model fit
    usable_features = candidate_features.copy()
    if "hr_outcome" in train_df.columns:
        st.markdown("#### 🟢 Model Features Used (strict):")
        st.write(usable_features)

        X = train_df[usable_features].fillna(0)
        y = train_df["hr_outcome"].astype(int)

        model = LogisticRegression(max_iter=1000)
        try:
            model.fit(X, y)
            st.success(f"Model fit OK with {len(usable_features)} features.")

            # Predict on live (fillna=0 for now)
            live_pred_df = live_df[usable_features].fillna(0)
            probs = model.predict_proba(live_pred_df)[:,1]
            live_df['HR_Prob'] = probs

            # Threshold sweep
            thresholds = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
            results = {}
            for thr in thresholds:
                picks = live_df[live_df['HR_Prob'] >= thr]
                results[round(thr, 3)] = list(picks['player_name'])

            st.markdown("### Results: HR Bot Picks by Threshold")
            for thr, picks in results.items():
                st.write(f"**Threshold {thr:.3f}**: {picks}")

            # Show table
            st.dataframe(live_df[['player_name', 'HR_Prob'] + [f for f in usable_features if f != 'player_name']])

        except Exception as e:
            st.error(f"Model fitting or prediction failed: {e}")

    else:
        st.warning("Your training file must have the 'hr_outcome' column for modeling.")
else:
    st.info("Upload both CSV files to begin.")
