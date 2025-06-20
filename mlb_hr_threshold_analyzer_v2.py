import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MLB HR Backtest Analyzer", layout="wide")

st.title("⚾ MLB HR Model Backtester & Threshold Sweeper")

# ---- UPLOAD ----
uploaded = st.file_uploader("Upload event-level CSV (with game_date or date column)", type=["csv"])
if not uploaded:
    st.stop()

# ---- LOAD & AUTO-FIX DATE COLUMN ----
df = pd.read_csv(uploaded, low_memory=False)

# Fix: Accept `game_date`, or attempt to parse other date-like columns
date_col = None
for candidate in ["date", "game_date", "Date", "GAME_DATE"]:
    if candidate in df.columns:
        date_col = candidate
        break

if not date_col:
    # Try to auto-detect a date column
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        st.error("No date or game_date column found! Please add one to your CSV.")
        st.stop()

# Standardize to 'date' column (YYYY-MM-DD)
df['date'] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
if df['date'].isna().all():
    st.error(f"Failed to parse {date_col} into dates! Please check your file.")
    st.stop()

# ---- CHOOSE THRESHOLDS ----
st.markdown("**Choose thresholds to sweep (inclusive):**")
col1, col2, col3 = st.columns(3)
thresh_start = col1.number_input("Threshold start", value=0.13, step=0.01, min_value=0.01, max_value=1.0)
thresh_end = col2.number_input("Threshold end", value=0.20, step=0.01, min_value=0.01, max_value=1.0)
thresh_step = col3.number_input("Threshold step", value=0.01, step=0.01, min_value=0.001, max_value=1.0)
if thresh_end < thresh_start:
    st.warning("Threshold end must be >= start.")
    st.stop()

# ---- SELECT MODEL PROB COLUMN ----
prob_cols = [c for c in df.columns if 'prob' in c]
if not prob_cols:
    st.error("No model probability columns found! (looked for columns with 'prob' in name, e.g. xgb_prob or logit_prob)")
    st.stop()
prob_col = st.selectbox("Model probability column", prob_cols, index=0)

# ---- SET HR OUTCOME COLUMN ----
hr_outcome_col = None
for col in ['hr_outcome', 'HR', 'is_hr', 'hr']:
    if col in df.columns:
        hr_outcome_col = col
        break
if not hr_outcome_col:
    st.warning("No HR outcome column found, using 'events'=='home_run' if available.")
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower() == 'home_run'
        hr_outcome_col = 'hr_outcome'
    else:
        st.error("No HR outcome or events column found in CSV. Can't score results.")
        st.stop()

df[hr_outcome_col] = df[hr_outcome_col].astype(int)

# ---- THRESHOLD SWEEP BACKTEST ----
results = []
for t in np.arange(thresh_start, thresh_end + thresh_step, thresh_step):
    t = np.round(t, 6)  # to avoid floating point drift
    preds = (df[prob_col] >= t).astype(int)
    TP = ((preds == 1) & (df[hr_outcome_col] == 1)).sum()
    FP = ((preds == 1) & (df[hr_outcome_col] == 0)).sum()
    FN = ((preds == 0) & (df[hr_outcome_col] == 1)).sum()
    TN = ((preds == 0) & (df[hr_outcome_col] == 0)).sum()
    picks = preds.sum()
    precision = TP / picks if picks > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
    results.append(dict(threshold=t, picks=picks, TP=TP, FP=FP, FN=FN, TN=TN,
                       precision=precision, recall=recall, f1_score=f1))
sweep_df = pd.DataFrame(results)
st.subheader(f"Threshold Sweep Results ({prob_col})")
st.dataframe(sweep_df, use_container_width=True)
st.line_chart(sweep_df.set_index("threshold")[["precision", "recall", "f1_score"]])

# ---- DAILY BACKTEST AT TOP 3 THRESHOLDS ----
top_thresholds = sweep_df.sort_values("f1_score", ascending=False).head(3)['threshold'].tolist()
st.markdown("### Daily Backtest at Top 3 Thresholds (by F1)")
daily_results = []
for t in top_thresholds:
    t = np.round(t, 6)
    group = []
    for date, g in df.groupby("date"):
        picked = g[g[prob_col] >= t]
        tp = picked[hr_outcome_col].sum()
        fp = (picked[hr_outcome_col] == 0).sum()
        fn = g[(g[prob_col] < t) & (g[hr_outcome_col] == 1)].shape[0]
        tn = (g[prob_col] < t).sum() - fn
        picks = picked.shape[0]
        precision = tp / picks if picks else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        group.append(dict(date=date, threshold=t, picks=picks, TP=tp, FP=fp, FN=fn, TN=tn,
                          precision=precision, recall=recall, f1_score=f1,
                          picked_players=";".join(picked['player name'].astype(str).unique()[:30]) if 'player name' in picked else "N/A"
        ))
    daily_results.extend(group)
daily_df = pd.DataFrame(daily_results)
st.dataframe(daily_df, use_container_width=True)

# ---- EXPORT ALL RESULTS ----
merged = df[["date", "player name", prob_col, hr_outcome_col]]
for t in np.arange(thresh_start, thresh_end + thresh_step, thresh_step):
    merged[f"pick_at_{t:.3f}"] = (merged[prob_col] >= t).astype(int)
csv = sweep_df.to_csv(index=False)
st.download_button("Download Threshold Sweep CSV", csv, file_name="threshold_sweep_results.csv")
csv2 = daily_df.to_csv(index=False)
st.download_button("Download Daily Backtest Results", csv2, file_name="daily_backtest_results.csv")
csv3 = merged.to_csv(index=False)
st.download_button("Download All Picks by Threshold", csv3, file_name="all_picks_by_threshold.csv")

st.success("✅ Done! This app auto-handles your date column and makes all backtests easy.")
