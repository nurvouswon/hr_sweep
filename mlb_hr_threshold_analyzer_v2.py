import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.base import clone
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH STACKED]", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH STACKED]")

# --- SAFE FILE READER (handles CSV or Parquet) ---
def safe_read(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
    # Remove duplicate columns
    return df.loc[:, ~df.columns.duplicated()]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def drop_high_na_low_var(df, thresh_na=0.25, thresh_var=1e-7):
    # Drop columns with >thresh_na fraction missing or low variance
    cols_to_drop = []
    na_frac = df.isnull().mean()
    low_var_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).std() < thresh_var]
    for c in df.columns:
        if na_frac.get(c, 0) > thresh_na:
            cols_to_drop.append(c)
        elif c in low_var_cols:
            cols_to_drop.append(c)
    df2 = df.drop(columns=cols_to_drop, errors="ignore")
    return df2, cols_to_drop

def cluster_select_features(X, threshold=0.95, show_debug=False):
    # Only use numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return list(num_cols), []
    corr = X[num_cols].corr().abs()
    # Distance = 1 - abs(correlation)
    dist = 1 - corr
    # Cluster features
    linkage_matrix = linkage(squareform(dist.values, checks=False), method='average')
    cluster_ids = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    # Pick the most "central" feature in each cluster (highest mean correlation to others)
    selected = []
    for cluster in np.unique(cluster_ids):
        members = num_cols[cluster_ids == cluster]
        if len(members) == 1:
            selected.append(members[0])
        else:
            mean_corr = corr.loc[members, members].mean().sort_values(ascending=False)
            selected.append(mean_corr.idxmax())
    dropped = [c for c in num_cols if c not in selected]
    if show_debug:
        st.write(f"Feature clusters (threshold {threshold}):")
        for cluster in np.unique(cluster_ids):
            members = list(num_cols[cluster_ids == cluster])
            st.write(f"Cluster {cluster}: {members}")
        st.write("Selected features from clusters:", selected)
        st.write("Dropped features from clusters:", dropped)
    return selected, dropped

# ==== Streamlit UI ====
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading & cleaning data..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)
        st.write(f"DEBUG: Successfully loaded file: {getattr(event_file, 'name', 'event_file')} with shape {event_df.shape}")
        st.write(f"DEBUG: Successfully loaded file: {getattr(today_file, 'name', 'today_file')} with shape {today_df.shape}")
        st.write("DEBUG: Columns in event_df:")
        st.write(list(event_df.columns))
        st.write("DEBUG: Columns in today_df:")
        st.write(list(today_df.columns))
        st.write("DEBUG: event_df head:")
        st.dataframe(event_df.head(3))
        st.write("DEBUG: today_df head:")
        st.dataframe(today_df.head(3))
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    # --- Check for hr_outcome ---
    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found!")

    # Show value counts for hr_outcome, avoid duplicate column names
    value_counts = event_df[target_col].value_counts(dropna=False)
    value_counts = value_counts.reset_index()
    value_counts.columns = ['hr_outcome', 'count']
    st.write("Value counts for hr_outcome:")
    st.dataframe(value_counts)

    # =========== DROP BAD COLS (robust for memory & NaN) ===========
    st.write("Dropping columns with >25% missing or near-zero variance...")
    event_df, event_dropped = drop_high_na_low_var(event_df, thresh_na=0.25, thresh_var=1e-7)
    today_df, today_dropped = drop_high_na_low_var(today_df, thresh_na=0.25, thresh_var=1e-7)
    st.write("Dropped columns from event-level data:")
    st.write(event_dropped)
    st.write("Dropped columns from today data:")
    st.write(today_dropped)
    st.write("Remaining columns event-level:")
    st.write(list(event_df.columns))
    st.write("Remaining columns today:")
    st.write(list(today_df.columns))

    # =========== SELECT FEATURE SET ===========
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))

    # =========== CLUSTER-BASED FEATURE SELECTION ===========
    st.write("Running cluster-based feature selection (removing highly correlated features)...")
    selected_feats, cluster_dropped = cluster_select_features(event_df[feature_cols], threshold=0.95, show_debug=True)
    st.write("Dropped features from clustering:")
    st.write(cluster_dropped)
    feature_cols = selected_feats

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    # =========== SPLIT & SCALE ===========
    st.write("Splitting for validation and scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # =========== STACKED ENSEMBLE ===========

    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=125, max_depth=5, learning_rate=0.08, n_jobs=-1)),
        ('cat', cb.CatBoostClassifier(iterations=120, depth=5, learning_rate=0.09, verbose=0)),
        ('rf', RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.09)),
        ('lr', LogisticRegression(max_iter=1000)),
    ]
    # For stacking, use meta-learner (logistic regression)
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(X_train_scaled, y_train)

    # =========== VALIDATION ===========
    st.write("Validating (Stacked ensemble)...")
    y_val_pred = stack.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC (Stacked): **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # =========== PREDICT ===========
    st.write("Predicting HR probability for today (Stacked ensemble)...")
    today_df['hr_probability'] = stack.predict_proba(X_today_scaled)[:,1]

    # ==== Leaderboard: Top 30 Only ====
    out_cols = []
    if "player_name" in today_df.columns:
        out_cols.append("player_name")
    out_cols += ["hr_probability"]
    leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True).head(30)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)

    st.markdown("### 🏆 **Today's HR Probability — Top 30 (Stacked Ensemble)**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")
else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
