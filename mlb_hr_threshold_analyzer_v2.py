import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import io

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

uploaded_train = st.file_uploader("Upload Training Event-Level CSV (with hr_outcome)", type="csv", key="train_ev")
uploaded_live = st.file_uploader("Upload Today's Event-Level CSV (with merged features, NO hr_outcome)", type="csv", key="live_ev")

col1, col2, col3 = st.columns(3)
with col1:
    threshold_min = st.number_input("Min HR Prob Threshold", value=0.01, min_value=0.01, max_value=0.5, step=0.01)
with col2:
    threshold_max = st.number_input("Max HR Prob Threshold", value=0.13, min_value=0.01, max_value=0.5, step=0.01)
with col3:
    threshold_step = st.number_input("Threshold Step", value=0.01, min_value=0.01, max_value=0.10, step=0.01)

predict_btn = st.button("🚀 Generate Today's HR Bot Picks")

def robust_numeric_columns(df):
    cols = []
    for c in df.columns:
        try:
            dt = pd.api.types.pandas_dtype(df[c].dtype)
            if (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c])) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
                cols.append(c)
        except Exception:
            continue
    return cols

def clean_id(x):
    try:
        if pd.isna(x): return None
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

if predict_btn:
    if uploaded_train is None or uploaded_live is None:
        st.warning("Please upload BOTH files (historical and today's event-level CSV).")
        st.stop()

    with st.spinner("Processing..."):
        train_df = pd.read_csv(uploaded_train)
        live_df = pd.read_csv(uploaded_live)

        # Ensure IDs standardized
        for df in [train_df, live_df]:
            if 'batter_id' in df.columns:
                df['batter_id'] = df['batter_id'].apply(clean_id)
            elif 'batter' in df.columns:
                df['batter_id'] = df['batter'].apply(clean_id)

        # === Robust Feature Intersection & Diagnostics ===
        numeric_features_train = robust_numeric_columns(train_df)
        numeric_features_live = robust_numeric_columns(live_df)
        intersect_features = [f for f in numeric_features_train if f in numeric_features_live]

        # Remove target if present
        if 'hr_outcome' in intersect_features:
            intersect_features.remove('hr_outcome')

        # Remove constant or all-null columns in either train or live
        final_features = []
        features_dropped = {'constant_in_train': [], 'constant_in_live': [], 'allnull_in_train': [], 'allnull_in_live': []}
        for col in intersect_features:
            nunq_train = train_df[col].nunique(dropna=True)
            nunq_live = live_df[col].nunique(dropna=True)
            if nunq_train <= 1:
                features_dropped['constant_in_train'].append(col)
                continue
            if nunq_live <= 1:
                features_dropped['constant_in_live'].append(col)
                continue
            if train_df[col].notnull().sum() == 0:
                features_dropped['allnull_in_train'].append(col)
                continue
            if live_df[col].notnull().sum() == 0:
                features_dropped['allnull_in_live'].append(col)
                continue
            final_features.append(col)
        model_features = final_features

        # === AUDIT REPORT ===
        missing_in_live = [f for f in numeric_features_train if f not in numeric_features_live]
        missing_in_train = [f for f in numeric_features_live if f not in numeric_features_train]
        constant_train = {f: train_df[f].unique() for f in intersect_features if train_df[f].nunique(dropna=True) == 1}
        constant_live = {f: live_df[f].unique() for f in intersect_features if live_df[f].nunique(dropna=True) == 1}

        audit = {
            "model_features_used": model_features,
            "features_in_train_but_missing_from_live": missing_in_live,
            "features_in_live_but_missing_from_train": missing_in_train,
            "features_constant_in_train": features_dropped['constant_in_train'],
            "features_constant_in_live": features_dropped['constant_in_live'],
            "features_allnull_in_train": features_dropped['allnull_in_train'],
            "features_allnull_in_live": features_dropped['allnull_in_live'],
            "constant_values_train": constant_train,
            "constant_values_live": constant_live,
            "live_null_count": live_df[model_features].isnull().sum().sort_values(ascending=False),
            "train_null_count": train_df[model_features].isnull().sum().sort_values(ascending=False),
            "train_rows": len(train_df),
            "live_rows": len(live_df),
        }

        # Display the audit to user
        st.markdown(f"### 🔍 **Audit Report:**")
        st.markdown(f"**Model features used ({len(model_features)}):** {model_features}")
        st.markdown(f"**Features in history but missing from live:** {missing_in_live}")
        st.markdown(f"**Features in live but missing from history:** {missing_in_train}")
        st.markdown(f"**Features dropped (constant in train):** {features_dropped['constant_in_train']}")
        st.markdown(f"**Features dropped (constant in live):** {features_dropped['constant_in_live']}")
        st.markdown(f"**Features dropped (all-null in train):** {features_dropped['allnull_in_train']}")
        st.markdown(f"**Features dropped (all-null in live):** {features_dropped['allnull_in_live']}")
        st.markdown("**Null count in live file (top 20):**")
        st.dataframe(audit['live_null_count'].head(20))
        st.markdown("**Null count in train file (top 20):**")
        st.dataframe(audit['train_null_count'].head(20))
        st.write(f"Train events: {audit['train_rows']}, Live events: {audit['live_rows']}")

        # AUDIT REPORT DOWNLOAD
        audit_csv = io.StringIO()
        pd.DataFrame({
            'feature': model_features,
            'null_count_live': audit['live_null_count'],
            'null_count_train': audit['train_null_count']
        }).to_csv(audit_csv, index=True)
        audit_csv.seek(0)
        st.download_button("⬇️ Download Audit Report CSV", audit_csv.getvalue(), file_name="mlb_hr_model_audit_report.csv")

        # === Model Run ===
        if len(model_features) < 2:
            st.error("Not enough features after cleaning! Check audit above for missing or all-null columns.")
            st.stop()

        train_X = train_df[model_features].fillna(0)
        train_y = train_df['hr_outcome'].astype(int)
        live_X = live_df[model_features].fillna(0)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=-1,
            use_label_encoder=False
        )
        xgb_clf.fit(train_X, train_y)
        live_df['xgb_prob'] = xgb_clf.predict_proba(live_X)[:, 1]

        # === Picks Per Threshold ===
        results = []
        thresholds = np.arange(threshold_min, threshold_max+0.001, threshold_step)
        for thresh in thresholds:
            mask = live_df['xgb_prob'] >= thresh
            picked = live_df.loc[mask]
            player_col = next((col for col in ['player_name', 'batter_name', 'mlb_id', 'batter_id'] if col in picked.columns), None)
            picked_players = list(picked[player_col]) if player_col else []
            results.append({
                'threshold': round(thresh, 3),
                'num_picks': int(mask.sum()),
                'picked_players': picked_players
            })

        picks_df = pd.DataFrame(results)
        st.header("Results: HR Bot Picks by Threshold")
        st.dataframe(picks_df)
        st.download_button(
            "⬇️ Download Picks by Threshold (CSV)",
            data=picks_df.to_csv(index=False),
            file_name="today_hr_bot_picks_by_threshold.csv"
        )

        st.markdown("#### All Picks (Threshold Sweep):")
        for _, row in picks_df.iterrows():
            st.write(f"**Threshold {row['threshold']}**: {row['picked_players']}")

        st.success("Done! These are the official HR bot picks for today at each threshold.")
