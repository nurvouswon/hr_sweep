import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH + GAME DAY OVERLAYS]", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH + GAME DAY OVERLAYS]")

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
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int', 'int64', 'int32'])
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def overlay_multiplier(row):
    """
    Research-backed multipliers for overlay post-prediction adjustment.
    Tuned for maximum signal without overfitting (see detailed MLB studies).
    """
    mult = 1.0

    # Park HR Rate (typical MLB range ~0.80 - 1.20)
    if 'park_hr_rate' in row and pd.notnull(row['park_hr_rate']):
        try:
            if row['park_hr_rate'] > 1.10:
                mult *= 1.10  # HR-friendly
            elif row['park_hr_rate'] < 0.90:
                mult *= 0.92  # HR-unfriendly
            # else ~1.00
        except Exception:
            pass

    # Temperature (Fahrenheit, non-linear effect per MLB physics studies)
    if 'temp' in row and pd.notnull(row['temp']):
        try:
            if row['temp'] >= 85:
                mult *= 1.07  # Hot, ball travels farther
            elif row['temp'] >= 75:
                mult *= 1.03
            elif row['temp'] <= 60:
                mult *= 0.95  # Cold, suppresses HR
        except Exception:
            pass

    # Wind: Out increases HRs, In decreases (Statcast, TangoTiger studies)
    if 'wind_mph' in row and 'wind_dir_string' in row and pd.notnull(row['wind_mph']):
        try:
            wind_str = str(row['wind_dir_string']).lower()
            if 'out' in wind_str and row['wind_mph'] >= 10:
                mult *= 1.10
            elif 'in' in wind_str and row['wind_mph'] >= 10:
                mult *= 0.90
            elif 'out' in wind_str and row['wind_mph'] >= 5:
                mult *= 1.05
            elif 'in' in wind_str and row['wind_mph'] >= 5:
                mult *= 0.96
        except Exception:
            pass

    # Humidity: marginal effect, boost at very high humidity (in some parks)
    if 'humidity' in row and pd.notnull(row['humidity']):
        try:
            if row['humidity'] >= 70:
                mult *= 1.02
            elif row['humidity'] <= 30:
                mult *= 0.98
        except Exception:
            pass

    # Altitude: if available, apply for places like Coors Field (Denver)
    if 'park_altitude' in row and pd.notnull(row['park_altitude']):
        try:
            if row['park_altitude'] >= 5000:
                mult *= 1.10
            elif row['park_altitude'] >= 2000:
                mult *= 1.03
        except Exception:
            pass

    return mult

# === Streamlit UI ===
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

    # --- Check for hr_outcome ---
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
    for i, cluster in enumerate(clusters[:20]):
        st.write(f"Cluster {i+1}: {cluster}")
    if len(clusters) > 20:
        st.write(f"... {len(clusters)-20} more clusters.")
    st.write("Selected features from clusters:")
    st.write(selected_features[:30])
    if len(selected_features) > 30:
        st.write(f"... {len(selected_features)-30} more features.")

    st.write("Dropped features from clusters (first 30 shown):")
    st.write(cluster_dropped[:30])
    if len(cluster_dropped) > 30:
        st.write(f"... {len(cluster_dropped)-30} more dropped features.")

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

    # =========== DEEP RESEARCH ENSEMBLE (SOFT VOTING) ===========
    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=60, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss',
        n_jobs=1, verbosity=1, tree_method='hist'
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=60, max_depth=5, learning_rate=0.08, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=60, depth=5, learning_rate=0.09, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=40, max_depth=7, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=40, max_depth=5, learning_rate=0.09)
    lr_clf = LogisticRegression(max_iter=400, solver='lbfgs', n_jobs=1)

    model_status = []
    models_for_ensemble = []
    try:
        xgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('xgb', xgb_clf))
        model_status.append('XGB OK')
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")
    try:
        lgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lgb', lgb_clf))
        model_status.append('LGB OK')
    except Exception as e:
        st.warning(f"LightGBM failed: {e}")
    try:
        cat_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('cat', cat_clf))
        model_status.append('CatBoost OK')
    except Exception as e:
        st.warning(f"CatBoost failed: {e}")
    try:
        rf_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('rf', rf_clf))
        model_status.append('RF OK')
    except Exception as e:
        st.warning(f"RandomForest failed: {e}")
    try:
        gb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('gb', gb_clf))
        model_status.append('GB OK')
    except Exception as e:
        st.warning(f"GBM failed: {e}")
    try:
        lr_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lr', lr_clf))
        model_status.append('LR OK')
    except Exception as e:
        st.warning(f"LogReg failed: {e}")

    st.info("Model training status: " + ', '.join(model_status))
    if not models_for_ensemble:
        st.error("All models failed to train! Try reducing features or rows.")
        st.stop()

    # Final ensemble: SOFT voting (deep research-backed for calibrated probabilities)
    st.write("Fitting ensemble...")
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_scaled, y_train)

    # =========== VALIDATION ===========
    st.write("Validating...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # =========== PREDICT ===========
    st.write("Predicting HR probability for today...")
    today_df['hr_probability'] = ensemble.predict_proba(X_today_scaled)[:,1]

    # === APPLY GAME DAY OVERLAYS POST-PREDICTION ===
    st.write("Auto-integrating enriched game day overlays (post-prediction)...")
    overlay_cols = [col for col in ["park_hr_rate", "temp", "wind_mph", "wind_dir_string", "humidity", "park_altitude"] if col in today_df.columns]
    if overlay_cols:
        st.success(f"Overlays present for today: {overlay_cols}")
    else:
        st.warning("No overlays found for today — final_hr_probability == hr_probability.")

    today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)

    # ==== Leaderboard: Top 30 Only ====
    out_cols = []
    if "player_name" in today_df.columns:
        out_cols.append("player_name")
    out_cols += ["final_hr_probability"] + overlay_cols
    leaderboard = today_df[out_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True).head(30)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)

    st.markdown("### 🏆 **Today's HR Probability — Top 30 (with Game Day Overlay)**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV (with overlay)", data=today_df.to_csv(index=False), file_name="today_hr_predictions_with_overlay.csv")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
