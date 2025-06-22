import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("2️⃣ Upload Event-Level CSVs & Run Model")

# Upload Training CSV
train_file = st.file_uploader(
    "Upload Training Event-Level CSV (with hr_outcome)",
    type=["csv"], key="train"
)
if train_file:
    df_train = pd.read_csv(train_file)
    st.success(f"Training file loaded! {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
else:
    st.stop()

# Upload Today's CSV
live_file = st.file_uploader(
    "Upload Today's Event-Level CSV (with merged features, NO hr_outcome)",
    type=["csv"], key="live"
)
if live_file:
    df_live = pd.read_csv(live_file)
    st.success(f"Today's file loaded! {df_live.shape[0]:,} rows, {df_live.shape[1]} columns.")
else:
    st.stop()

# Normalize column case
df_train.columns = [c.lower() for c in df_train.columns]
df_live.columns = [c.lower() for c in df_live.columns]

st.subheader("🩺 Feature Diagnostics Table (train/live overlap)")

# Features in each
features_in_train = set(df_train.columns)
features_in_live = set(df_live.columns)

missing_from_live = list(features_in_train - features_in_live)
missing_from_train = list(features_in_live - features_in_train)

if missing_from_live:
    st.write("**Features in train missing from live:**")
    st.write(missing_from_live)
if missing_from_train:
    st.write("**Features in live missing from train:**")
    st.write(missing_from_train)

# Show dtype mismatches for overlapping columns
overlap_cols = sorted(list((features_in_train & features_in_live) - {'hr_outcome'}))
dtype_mismatches = []
for c in overlap_cols:
    if df_train[c].dtype != df_live[c].dtype:
        dtype_mismatches.append((c, str(df_train[c].dtype), str(df_live[c].dtype)))
if dtype_mismatches:
    st.write("⚠️ **Dtype mismatches detected!**")
    st.dataframe(pd.DataFrame(dtype_mismatches, columns=["feature", "train_dtype", "live_dtype"]))

# Show null fraction for both train and live
def null_frac(series):
    return np.mean(series.isnull())

diagnostic_table = []
for c in overlap_cols:
    diagnostic_table.append({
        "feature": c,
        "train_dtype": str(df_train[c].dtype),
        "live_dtype": str(df_live[c].dtype),
        "train_null_frac": null_frac(df_train[c]),
        "live_null_frac": null_frac(df_live[c]),
    })
st.dataframe(pd.DataFrame(diagnostic_table))

# Filter out columns with all null in either train or live
features_to_use = [c for c in overlap_cols if (df_train[c].notnull().sum() > 0 and df_live[c].notnull().sum() > 0)]

# Optional: Remove high-null columns (e.g. > 99% null)
features_to_use = [c for c in features_to_use if null_frac(df_train[c]) < 0.99 and null_frac(df_live[c]) < 0.99]

# Remove non-numeric and non-binary categorical (object) columns
numeric_feats = []
cat_feats = []
for c in features_to_use:
    if pd.api.types.is_numeric_dtype(df_train[c]) and pd.api.types.is_numeric_dtype(df_live[c]):
        numeric_feats.append(c)
    elif df_train[c].dtype == "object" or df_live[c].dtype == "object":
        # Only include if two or fewer unique values in both
        if (df_train[c].nunique() <= 2) and (df_live[c].nunique() <= 2):
            cat_feats.append(c)
features_to_use = numeric_feats + cat_feats

st.write(f"**Final features used ({len(features_to_use)}):**")
st.write(features_to_use)

# Threshold controls
st.subheader("Min HR Prob Threshold")
min_thresh = st.number_input("Min HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
st.subheader("Max HR Prob Threshold")
max_thresh = st.number_input("Max HR Prob Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
st.subheader("Threshold Step")
step_thresh = st.number_input("Threshold Step", min_value=0.001, max_value=1.0, value=0.01, step=0.01)

# Modeling
st.subheader("🔍 Model Results")

# Prepare X, y
X_train = df_train[features_to_use].copy()
X_live = df_live[features_to_use].copy()
y_train = df_train["hr_outcome"] if "hr_outcome" in df_train else None

# Label encode binary categoricals
for c in cat_feats:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    X_live[c] = le.transform(X_live[c].fillna("NA").astype(str))  # Fill NA for unseen

try:
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    st.success(f"Model fit OK with {len(features_to_use)} features.")
    preds = model.predict_proba(X_live)[:, 1]
    df_live["hr_prob"] = preds

    # Threshold slider loop
    st.subheader("Thresholded Predictions Table")
    threshold_results = []
    for thresh in np.arange(min_thresh, max_thresh + step_thresh, step_thresh):
        n_above = (df_live["hr_prob"] >= thresh).sum()
        threshold_results.append({"threshold": round(thresh, 4), "n_above": int(n_above)})
    st.dataframe(pd.DataFrame(threshold_results))

    st.write("**Sample Predictions:**")
    st.dataframe(df_live[["player_name", "hr_prob"] + features_to_use].head(20))
except Exception as e:
    st.error(f"Model fitting or prediction failed: {e}")
