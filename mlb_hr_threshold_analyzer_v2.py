import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH STACKED]", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH STACKED]")

def safe_read(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
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

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def cluster_select_features(X, threshold=0.95):
    corr = X.corr().abs()
    np.fill_diagonal(corr.values, 0)
    corr = corr.fillna(0)
    dist = 1 - corr.values
    if np.all(dist == 1):
        return X.columns.tolist(), []
    linkage_matrix = linkage(squareform(dist, checks=False), method='average')
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    cluster_to_feat = {}
    for idx, c in enumerate(clusters):
        cluster_to_feat.setdefault(c, []).append(X.columns[idx])
    selected_features = [v[0] for v in cluster_to_feat.values()]
    dropped_features = [f for v in cluster_to_feat.values() for f in v[1:]]
    return selected_features, dropped_features

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

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found!")

    value_counts = event_df[target_col].value_counts(dropna=False)
    value_counts = value_counts.reset_index()
    value_counts.columns = ['hr_outcome', 'count']
    st.write("Value counts for hr_outcome:")
    st.dataframe(value_counts)

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

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write("Running cluster-based feature selection (removing highly correlated features)...")
    cluster_selected, cluster_dropped = cluster_select_features(event_df[feature_cols], threshold=0.95)
    st.write("Selected features from clusters:")
    st.write(cluster_selected)
    st.write("Dropped features from clusters:")
    st.write(cluster_dropped)
    feature_cols = cluster_selected

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    st.write("Splitting for validation and scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.08,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, n_jobs=-1)
    cat_clf = cb.CatBoostClassifier(iterations=120, depth=5, learning_rate=0.09, verbose=0)
    rf_clf = RandomForestClassifier(n_estimators=120, max_depth=7, n_jobs=-1)
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.09)
    lr_clf = LogisticRegression(max_iter=1000)

    base_models = [
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf),
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('lr', lr_clf)
    ]
    for name, model in base_models:
        with st.spinner(f"Fitting {name.upper()}..."):
            model.fit(X_train_scaled, y_train)

    st.write("Fitting ensemble...")
    ensemble = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1, weights=[2,2,2,1,1,1])
    ensemble.fit(X_train_scaled, y_train)

    st.write("Validating...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    st.write("Predicting HR probability for today...")
    today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]

    out_cols = []
    if "player_name" in today_df.columns:
        out_cols.append("player_name")
    out_cols += ["hr_probability"]
    leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True).head(30)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)

    st.markdown("### 🏆 **Today's HR Probability — Top 30**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
