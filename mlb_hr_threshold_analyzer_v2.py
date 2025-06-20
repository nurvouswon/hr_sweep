import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config("MLB HR Model Analyzer", layout="wide")
st.title("⚾ MLB HR Model Threshold Analyzer & Leaderboard App")
st.caption("Upload event-level CSV with `hr_outcome`, `logit_prob`, `xgb_prob`. Get optimal threshold, daily & player leaderboards, plots, downloads.")

def threshold_sweep(df, prob_col, true_col='hr_outcome', t_min=0.01, t_max=0.20, step=0.01):
    thresholds = np.arange(t_min, t_max+step, step)
    results = []
    for thresh in thresholds:
        pred = (df[prob_col] > thresh).astype(int)
        tp = ((pred == 1) & (df[true_col] == 1)).sum()
        fp = ((pred == 1) & (df[true_col] == 0)).sum()
        fn = ((pred == 0) & (df[true_col] == 1)).sum()
        tn = ((pred == 0) & (df[true_col] == 0)).sum()
        precision = precision_score(df[true_col], pred, zero_division=0)
        recall = recall_score(df[true_col], pred, zero_division=0)
        f1 = f1_score(df[true_col], pred, zero_division=0)
        results.append({
            "threshold": round(thresh, 3),
            "picks": int(pred.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
    df_sweep = pd.DataFrame(results)
    best_row = df_sweep.loc[df_sweep['f1_score'].idxmax()]
    return df_sweep, float(best_row['threshold']), best_row.to_dict()

def daily_backtest(df, prob_col, threshold, date_col='game_date', true_col='hr_outcome', name_col='batter_name'):
    backtest = []
    for day, group in df.groupby(date_col):
        pred = (group[prob_col] > threshold).astype(int)
        actual = group[true_col]
        picked = group.loc[pred == 1, name_col].tolist() if name_col in group.columns else []
        picked_hr = group.loc[(pred == 1) & (actual == 1), name_col].tolist() if name_col in group.columns else []
        missed_hr = group.loc[(pred == 0) & (actual == 1), name_col].tolist() if name_col in group.columns else []
        tp = ((pred == 1) & (actual == 1)).sum()
        fp = ((pred == 1) & (actual == 0)).sum()
        fn = ((pred == 0) & (actual == 1)).sum()
        tn = ((pred == 0) & (actual == 0)).sum()
        precision = precision_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)
        backtest.append({
            "date": day,
            "picks": int(pred.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "picked_players": picked,
            "picked_hr_players": picked_hr,
            "missed_hr_players": missed_hr,
        })
    return pd.DataFrame(backtest).sort_values("date")

def player_leaderboard(df, prob_col, threshold, name_col='batter_name', true_col='hr_outcome'):
    pred = (df[prob_col] > threshold).astype(int)
    leaderboard = df.copy()
    leaderboard['picked'] = pred
    leaderboard['hit'] = (pred == 1) & (df[true_col] == 1)
    leaderboard = leaderboard.groupby(name_col).agg(
        total_events = (true_col, "count"),
        picked_count = ('picked', 'sum'),
        hr_count = (true_col, "sum"),
        picked_hr = ('hit', 'sum')
    ).reset_index()
    leaderboard['pick_precision'] = leaderboard['picked_hr'] / leaderboard['picked_count']
    leaderboard['pick_precision'] = leaderboard['pick_precision'].fillna(0)
    leaderboard = leaderboard.sort_values('picked_hr', ascending=False)
    return leaderboard

uploaded = st.file_uploader("Upload Scored Event-Level CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    st.success(f"Loaded {len(df)} events, columns: {', '.join(df.columns[:20])}...")
    available_models = [c for c in ['logit_prob', 'xgb_prob'] if c in df.columns]
    if not available_models:
        st.error("Your file must contain at least one model probability column named 'logit_prob' or 'xgb_prob'.")
        st.stop()

    # Sweep and show results for each available model
    for prob_col in available_models:
        st.header(f"Model: `{prob_col}`")
        sweep_df, best_thr, best_row = threshold_sweep(df, prob_col)
        st.write("#### Threshold Sweep Table (click to expand)")
        st.dataframe(sweep_df.style.format(precision=3), use_container_width=True)
        st.write(f"**Optimal threshold (F1):** `{best_thr}` — F1 = {best_row['f1_score']:.3f}, Precision = {best_row['precision']:.3f}, Recall = {best_row['recall']:.3f}, Picks = {best_row['picks']}")
        st.download_button(f"⬇️ Download Sweep Results for {prob_col}", sweep_df.to_csv(index=False), file_name=f"sweep_{prob_col}.csv")

        # Plots
        st.subheader("Threshold Sweep Plots")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sweep_df['threshold'], sweep_df['precision'], label="Precision")
        ax.plot(sweep_df['threshold'], sweep_df['recall'], label="Recall")
        ax.plot(sweep_df['threshold'], sweep_df['f1_score'], label="F1")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_title(f"Threshold vs. Precision/Recall/F1 for {prob_col}")
        st.pyplot(fig)

        # Pick thresholds to run backtest
        thresholds_to_run = [best_thr, 0.10, 0.13]
        thresholds_to_run = sorted(set(thresholds_to_run))
        st.write(f"#### Full-Season Backtest at Thresholds: {', '.join([str(t) for t in thresholds_to_run])}")

        for thr in thresholds_to_run:
            st.subheader(f"Backtest & Leaderboards at threshold = {thr:.3f} (model `{prob_col}`)")
            daily_bt = daily_backtest(df, prob_col, thr)
            st.markdown("##### Daily Summary")
            st.dataframe(daily_bt[['date', 'picks', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1_score']].style.format(precision=3), use_container_width=True)
            st.download_button(f"⬇️ Download Daily Backtest {prob_col} @ {thr:.3f}", daily_bt.to_csv(index=False), file_name=f"backtest_{prob_col}_thr{thr:.3f}.csv")

            st.markdown("##### Per-Player HR Leaderboard")
            p_lb = player_leaderboard(df, prob_col, thr)
            st.dataframe(p_lb.style.format(precision=3), use_container_width=True)
            st.download_button(f"⬇️ Download Player Leaderboard {prob_col} @ {thr:.3f}", p_lb.to_csv(index=False), file_name=f"player_leaderboard_{prob_col}_thr{thr:.3f}.csv")

            # Player summary plot
            top_picks = p_lb.head(20)
            fig2, ax2 = plt.subplots(figsize=(10,5))
            ax2.bar(top_picks['batter_name'], top_picks['picked_hr'], label="HRs as Picked", color='g')
            ax2.bar(top_picks['batter_name'], top_picks['picked_count'], alpha=0.3, label="Total Picks", color='b')
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.legend()
            plt.title(f"Top 20 Players: Picked HRs & Total Picks at thr={thr:.2f}")
            plt.tight_layout()
            st.pyplot(fig2)

    st.info("Done! Download sweep, daily and player leaderboard CSVs, then upload here for expert analysis and insights.")

else:
    st.info("Please upload a CSV with columns like `hr_outcome`, `logit_prob`, and/or `xgb_prob`.")
