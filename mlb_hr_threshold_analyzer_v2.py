import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

st.set_page_config("2️⃣ MLB HR Predictor — Meta-Stacked Deep Ensemble", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Meta-Stacked Deep Ensemble")

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

def cluster_select_features(df, threshold=0.95):
    corr = df.corr().abs()
    clusters = []
    selected = []
    dropped = []
    visited = set()
    for col in corr.columns:
        if col in visited:
            continue
        cluster = [col]
        visited.add(col)
        for other in corr.columns:
            if other != col and other not in visited and corr.loc[col, other] >= threshold:
                cluster.append(other)
                visited.add(other)
        clusters.append(cluster)
        selected.append(cluster[0])
        dropped.extend(cluster[1:])
    return selected, clusters, dropped

def downcast_df(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int', 'int64', 'int32']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files (may take 1-2 min)..."):
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

    st.write("Running cluster-based feature selection (removing highly correlated features)...")
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    X_for_cluster = event_df[feature_cols]
    selected_features, clusters, cluster_dropped = cluster_select_features(X_for_cluster, threshold=0.95)
    st.write(f"Feature clusters (threshold 0.95):")
    for i, cluster in enumerate(clusters):
        st.write(f"Cluster {i+1}: {cluster}")
    st.write("Selected features from clusters:")
    st.write(selected_features)
    st.write("Dropped features from clusters:")
    st.write(cluster_dropped)

    # Apply selected features to X and X_today
    X = clean_X(event_df[selected_features])
    y = event_df[target_col]
    X_today = clean_X(today_df[selected_features], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

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

    # =========== META-STACKED ENSEMBLE ===========
    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    # **Speed optimized params for debug. For max accuracy, raise n_estimators to 120-200, n_jobs>1**
    xgb_clf = xgb.XGBClassifier(
        n_estimators=60, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss',
        n_jobs=1, verbosity=1, tree_method='hist'
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=60, max_depth=5, learning_rate=0.08, n_jobs=1, verbose=1)
    cat_clf = cb.CatBoostClassifier(iterations=60, depth=5, learning_rate=0.09, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=40, max_depth=7, n_jobs=1, verbose=1)
    gb_clf = GradientBoostingClassifier(n_estimators=40, max_depth=5, learning_rate=0.09, verbose=1)
    lr_clf = LogisticRegression(max_iter=400, solver='lbfgs', n_jobs=1, verbose=1)

    base_models = [
        ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf),
        ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
    ]

    # === OOF meta stacking ===
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train_scaled.shape[0], len(base_models)))
    test_preds = np.zeros((X_today_scaled.shape[0], len(base_models)))
    model_status = []
    model_names = []
    for i, (name, model) in enumerate(base_models):
        try:
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
                X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[val_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                if name in ('xgb', 'lgb', 'cat'):
                    fit_kwargs = {'verbose': False}
                    if name == 'xgb':
                        fit_kwargs['eval_set'] = [(X_va, y_va)]
                        fit_kwargs['early_stopping_rounds'] = 10
                    elif name == 'lgb':
                        fit_kwargs['eval_set'] = [(X_va, y_va)]
                        fit_kwargs['early_stopping_rounds'] = 10
                    elif name == 'cat':
                        fit_kwargs['eval_set'] = [(X_va, y_va)]
                        fit_kwargs['early_stopping_rounds'] = 10
                    model.fit(X_tr, y_tr, **fit_kwargs)
                else:
                    model.fit(X_tr, y_tr)
                oof_preds[val_idx, i] = model.predict_proba(X_va)[:, 1]
            # Full fit on all data for test set
            model.fit(X_train_scaled, y_train)
            test_preds[:, i] = model.predict_proba(X_today_scaled)[:, 1]
            model_status.append(f"{name} OK")
            model_names.append(name)
        except Exception as e:
            st.warning(f"{name} failed: {e}")

    st.info("Base model training status: " + ', '.join(model_status))
    if not model_names:
        st.error("All base models failed! Try fewer features or rows.")
        st.stop()

    # Meta model (stacker)
    st.write("Fitting meta-stacker (LogisticRegression on base model predictions)...")
    meta_model = LogisticRegression(max_iter=400, solver='lbfgs')
    meta_model.fit(oof_preds[:, :len(model_names)], y_train)
    y_val_meta = meta_model.predict_proba(oof_preds[:, :len(model_names)])[:, 1]
    auc = roc_auc_score(y_train, y_val_meta)
    ll = log_loss(y_train, y_val_meta)
    st.info(f"Meta-validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # Predict test set (today) using stacker
    today_df['hr_probability'] = meta_model.predict_proba(test_preds[:, :len(model_names)])[:, 1]

    # ==== Leaderboard: Top 30 Only ====
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
