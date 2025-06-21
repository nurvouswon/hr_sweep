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

    # Feature filter debug (PRESERVE!)
    st.markdown("### 🔍 Audit Report:")
    train_cols = set(train_df.columns)
    live_cols = set(live_df.columns)
    intersect = [c for c in train_cols if c in live_cols]

    # EXTRA DEBUG: Print full intersect and column properties
    st.write("Intersect columns:", intersect)
    for c in sorted(intersect):
        st.write(f"Col: {c}  | train dtype: {train_df[c].dtype}  | live dtype: {live_df[c].dtype}  | train unique: {train_df[c].nunique(dropna=False)}  | live unique: {live_df[c].nunique(dropna=False)}  | train null: {train_df[c].isnull().mean()}  | live null: {live_df[c].isnull().mean()}")

    # Maximally permissive model feature filter: keep all intersecting columns except id/meta columns, drop only constant/all-null in either file
    meta_drop = ['hr_outcome','player_name']
    model_features = [
        c for c in intersect
        if c not in meta_drop
        and train_df[c].notnull().any() and live_df[c].notnull().any()
        and train_df[c].nunique(dropna=False)>1 and live_df[c].nunique(dropna=False)>1
    ]
    st.write(f"Model features used ({len(model_features)}): {model_features}")

    # Rest of audit unchanged
    missing_in_live = [c for c in model_features if c not in live_df.columns]
    missing_in_train = [c for c in model_features if c not in train_df.columns]
    st.write("Features in history but missing from live:", missing_in_live)
    st.write("Features in live but missing from history:", missing_in_train)

    dropped_constant_train = [c for c in intersect if train_df[c].nunique(dropna=False) <= 1]
    dropped_constant_live = [c for c in intersect if live_df[c].nunique(dropna=False) <= 1]
    st.write("Features dropped (constant in train):", dropped_constant_train)
    st.write("Features dropped (constant in live):", dropped_constant_live)
    dropped_null_train = [c for c in intersect if train_df[c].isnull().all()]
    dropped_null_live = [c for c in intersect if live_df[c].isnull().all()]
    st.write("Features dropped (all-null in train):", dropped_null_train)
    st.write("Features dropped (all-null in live):", dropped_null_live)
    st.write("Null count in live file (top 20):")
    st.write(live_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write("Null count in train file (top 20):")
    st.write(train_df.isnull().sum().sort_values(ascending=False).head(20))
    st.write(f"Train events: {len(train_df)}, Live events: {len(live_df)}")

    # --- Model Fit & Predict ---
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

    st.markdown("Done! These are the official HR bot picks for today at each threshold.")

    # --- ENHANCED AUDIT DOWNLOADS ---
    # Lowercase col compare for safety
    train_cols_lower = set([c.lower() for c in train_df.columns])
    live_cols_lower = set([c.lower() for c in live_df.columns])
    used_features = [c for c in model_features if c.lower() in live_cols_lower and c.lower() in train_cols_lower]
    missing_in_live_lower = [c for c in model_features if c.lower() not in live_cols_lower]
    missing_in_train_lower = [c for c in model_features if c.lower() not in train_cols_lower]

    train_nulls = train_df[model_features].isnull().mean().sort_values(ascending=False) if model_features else pd.Series()
    live_nulls = live_df[model_features].isnull().mean().sort_values(ascending=False) if model_features else pd.Series()

    def get_constant_cols(df, cols):
        return [c for c in cols if df[c].nunique(dropna=False) <= 1]

    train_constant = get_constant_cols(train_df, model_features)
    live_constant = get_constant_cols(live_df, model_features)

    # Column mapping/merge suggestions (fuzzy, for debugging pipeline)
    def suggest_col_map(missing, present):
        suggestions = []
        for m in missing:
            for p in present:
                if m.replace('_','').lower() == p.replace('_','').lower():
                    suggestions.append((m, p))
        return suggestions

    mapping_suggestions = suggest_col_map(missing_in_live, live_cols) + suggest_col_map(missing_in_train, train_cols)
    mapping_suggestions = list(set(mapping_suggestions))

    def group_feature_type(col):
        cl = col.lower()
        if 'hr' in cl and 'rate' in cl:
            return 'HR Rate'
        if 'launch_speed' in cl or 'exit_velocity' in cl:
            return 'Batted Ball'
        if 'pitchtype' in cl or 'pitch_' in cl or cl.startswith('p_'):
            return 'Pitch'
        if 'park' in cl or 'stadium' in cl:
            return 'Park/Env'
        if 'wind' in cl or 'temp' in cl or 'humidity' in cl:
            return 'Weather'
        if 'woba' in cl or 'xwoba' in cl:
            return 'Advanced Stat'
        if 'b_' in cl or 'p_' in cl:
            return 'Rolling'
        if cl in ['batter_id','mlb_id','game_date','player_name']:
            return 'Meta'
        return 'Other'

    feature_summary = pd.DataFrame({
        'feature': model_features,
        'type': [group_feature_type(c) for c in model_features],
        'train_null_frac': [train_df[c].isnull().mean() for c in model_features],
        'live_null_frac': [live_df[c].isnull().mean() for c in model_features],
        'constant_in_train': [c in train_constant for c in model_features],
        'constant_in_live': [c in live_constant for c in model_features]
    })

    # Feature importance from model, if possible
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_[0]
        if importances is not None:
            feature_summary = feature_summary.set_index('feature')
            feature_summary['importance'] = pd.Series(importances, index=model_features)
            feature_summary = feature_summary.reset_index()
    except Exception:
        pass

    # Audit summary text
    audit_summary = {
        "n_train_cols": len(train_cols),
        "n_live_cols": len(live_cols),
        "n_used": len(model_features),
        "n_missing_in_live": len(missing_in_live),
        "n_missing_in_train": len(missing_in_train),
        "n_constant_in_train": len(train_constant),
        "n_constant_in_live": len(live_constant)
    }
    audit_text = StringIO()
    print("===== MLB HR Model Audit Report =====", file=audit_text)
    for k,v in audit_summary.items():
        print(f"{k}: {v}", file=audit_text)
    print("\n-- Features used (model input):", file=audit_text)
    print(model_features, file=audit_text)
    print("\n-- Features in history but missing from live:", file=audit_text)
    print(missing_in_live, file=audit_text)
    print("\n-- Features in live but missing from history:", file=audit_text)
    print(missing_in_train, file=audit_text)
    print("\n-- Features with high nulls in live (top 10):", file=audit_text)
    print(live_nulls.head(10), file=audit_text)
    print("\n-- Features with high nulls in train (top 10):", file=audit_text)
    print(train_nulls.head(10), file=audit_text)
    print("\n-- Constant features in train:", file=audit_text)
    print(train_constant, file=audit_text)
    print("\n-- Constant features in live:", file=audit_text)
    print(live_constant, file=audit_text)
    print("\n-- Suggested column mapping:", file=audit_text)
    for a,b in mapping_suggestions:
        print(f"{a} ↔ {b}", file=audit_text)
    print("\n-- Feature coverage by type:", file=audit_text)
    print(feature_summary['type'].value_counts(), file=audit_text)
    audit_text = audit_text.getvalue()

    st.markdown("### 📋 Detailed Model Audit Report")
    st.code(audit_text)
    st.markdown("#### Features/Stats Table (sortable)")
    st.dataframe(feature_summary.sort_values(by="importance" if "importance" in feature_summary.columns else "feature", ascending=False))
    st.download_button("⬇️ Download Full Audit Text (.txt)", audit_text, file_name="mlb_hr_model_audit_report.txt")
    st.download_button("⬇️ Download Feature Summary (.csv)", feature_summary.to_csv(index=False), file_name="mlb_hr_feature_summary.csv")

    # Download column suggestions
    if mapping_suggestions:
        mapping_df = pd.DataFrame(mapping_suggestions, columns=["missing_column", "possible_match"])
        st.markdown("#### ⚡ Column Rename/Join Suggestions")
        st.dataframe(mapping_df)
        st.download_button("⬇️ Download Column Mapping Suggestions (.csv)", mapping_df.to_csv(index=False), file_name="mlb_hr_col_mapping_suggestions.csv")
