import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

st.set_page_config("2️⃣ MLB HR Predictor — Deep Research Stacking Ensemble", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble Stacking + Weather Score [PARLAY SHARP STACK]")

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

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Meta-stacking: Fit base models, then meta-model on their outputs."""
    def __init__(self, base_models, meta_model, n_jobs=1):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.base_models_ = [clone(m) for m in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        self.base_out = np.zeros((X.shape[0], len(self.base_models_)))
        # Fit base models
        for i, model in enumerate(self.base_models_):
            model.fit(X, y)
            self.base_out[:,i] = model.predict_proba(X)[:,1]
        # Fit meta-model
        self.meta_model_.fit(self.base_out, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([m.predict_proba(X)[:,1] for m in self.base_models_])
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        meta_features = np.column_stack([m.predict_proba(X)[:,1] for m in self.base_models_])
        return self.meta_model_.predict(meta_features)

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)
        st.write(f"DEBUG: Loaded event_df {event_df.shape}, today_df {today_df.shape}")
        st.dataframe(event_df.head(3))
        st.dataframe(today_df.head(3))
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("No hr_outcome column found.")
        st.stop()
    st.success("✅ 'hr_outcome' column found!")

    # Drop NA / low variance only
    event_df, event_dropped = drop_high_na_low_var(event_df, thresh_na=0.25, thresh_var=1e-7)
    today_df, today_dropped = drop_high_na_low_var(today_df, thresh_na=0.25, thresh_var=1e-7)
    st.write("Dropped columns event-level:", event_dropped)
    st.write("Dropped columns today:", today_dropped)

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    st.write("DEBUG: Feature cols:", feature_cols)
    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    # Split/scale
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # Main base models
    st.write("Training base models for meta-stacking...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08, use_label_encoder=False,
        eval_metric='logloss', n_jobs=1, tree_method='hist', verbosity=1
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=120, depth=5, learning_rate=0.09, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=60, max_depth=7, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=60, max_depth=5, learning_rate=0.09)
    base_models = [xgb_clf, lgb_clf, cat_clf, rf_clf, gb_clf]
    meta_model = LogisticRegression(max_iter=400, solver='lbfgs', n_jobs=1)

    # Fit meta-stacking
    try:
        stacker = StackingEnsemble(base_models=base_models, meta_model=meta_model)
        stacker.fit(X_train_scaled, y_train)
        st.info("Meta-stacking ensemble trained!")
    except Exception as e:
        st.warning(f"Meta-stacking failed: {e}")
        st.write("Falling back to weighted soft-voting.")
        # Soft-voting fallback
        from sklearn.ensemble import VotingClassifier
        models = [
            ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf),
            ('rf', rf_clf), ('gb', gb_clf), 
        ]
        weights = [3,3,2,1,1]
        for name, clf in models:
            try:
                clf.fit(X_train_scaled, y_train)
            except Exception as ex:
                st.warning(f"{name} failed: {ex}")
        stacker = VotingClassifier(estimators=models, voting='soft', n_jobs=1, weights=weights)
        stacker.fit(X_train_scaled, y_train)

    # =========== VALIDATION ===========
    st.write("Validating stacking ensemble...")
    y_val_pred = stacker.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # =========== PREDICT ===========
    st.write("Predicting HR probability for today...")
    today_df['hr_probability'] = stacker.predict_proba(X_today_scaled)[:,1]

    # ==== Leaderboard: Top 10 Only (max sharpness for parlay) ====
    out_cols = []
    if "player_name" in today_df.columns:
        out_cols.append("player_name")
    out_cols += ["hr_probability"]
    leaderboard = today_df[out_cols].sort_values("hr_probability", ascending=False).reset_index(drop=True).head(10)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)

    st.markdown("### 🏆 **Today's HR Probability — Top 10 Parlay Picks**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
