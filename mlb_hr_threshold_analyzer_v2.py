import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MLB XGBoost Backtest/Blind Future Picks", layout="wide")

st.title("MLB XGBoost Backtester & Blind Future Picker")
st.write("""
- Upload your main event-level scored CSV for the season (must have `game_date`, `xgb_prob`, and `batter_name` or similar).
- Optionally upload a **blind/future** CSV (e.g., for June 17–19, with the same columns, but no HR outcome known).
- The app will show, for each threshold (0.13–0.20), the model's picks *per day*.
""")

# --- Main Season Upload ---
st.header("1️⃣ Upload Main Scored Event-Level CSV (Season)")
main_csv = st.file_uploader("Main Event-Level CSV", type="csv", key="main_csv")

# --- Blind/Future Upload ---
st.header("2️⃣ (Optional) Upload Blind/Future Scored Event-Level CSV (e.g., June 17–19 only)")
blind_csv = st.file_uploader("Blind/Future Event-Level CSV", type="csv", key="blind_csv")

def prepare_event_df(df):
    # Checks and column normalizations
    if "game_date" not in df.columns:
        raise ValueError("Your CSV must include a 'game_date' column (YYYY-MM-DD)!")
    if "xgb_prob" not in df.columns:
        raise ValueError("Your CSV must include an 'xgb_prob' column (model probabilities)!")
    if "batter_name" not in df.columns:
        if "player_name" in df.columns:
            df["batter_name"] = df["player_name"]
        elif "batter" in df.columns:
            df["batter_name"] = df["batter"]
        elif "batter_id" in df.columns:
            df["batter_name"] = df["batter_id"]
        else:
            df["batter_name"] = "Unknown"
    # Always ensure game_date is str (not datetime)
    df["game_date"] = df["game_date"].astype(str)
    return df

def sweep_thresholds_and_display(df, sweep_name=""):
    # Run threshold sweep, output picks by day/threshold
    rows = []
    for threshold in np.arange(0.13, 0.201, 0.01):
        picks = df[df["xgb_prob"] >= round(threshold, 4)].copy()
        if picks.empty:
            continue
        picks_day = (
            picks.groupby("game_date")["batter_name"]
            .apply(list)
            .reset_index(name="picked_players")
        )
        picks_day["threshold"] = round(threshold, 3)
        picks_day["num_picks"] = picks_day["picked_players"].apply(len)
        picks_day["sweep"] = sweep_name
        rows.append(picks_day)
    if not rows:
        return pd.DataFrame()
    result_df = pd.concat(rows, ignore_index=True)
    return result_df

main_results = None
blind_results = None

# --- Main Picks Analysis ---
if main_csv is not None:
    st.subheader("Main Season: Threshold Sweep (.13–.20)")
    main_df = pd.read_csv(main_csv)
    try:
        main_df = prepare_event_df(main_df)
        main_results = sweep_thresholds_and_display(main_df, sweep_name="main")
        st.dataframe(main_results, use_container_width=True)
        st.download_button(
            "⬇️ Download Main Season Picks by Threshold",
            data=main_results.to_csv(index=False),
            file_name="main_season_picks_by_threshold.csv",
        )
    except Exception as e:
        st.error(f"Error with main CSV: {e}")

# --- Blind/Future Picks Analysis ---
if blind_csv is not None:
    st.subheader("Blind/Future: Threshold Sweep (.13–.20)")
    blind_df = pd.read_csv(blind_csv)
    try:
        blind_df = prepare_event_df(blind_df)
        blind_results = sweep_thresholds_and_display(blind_df, sweep_name="blind")
        st.dataframe(blind_results, use_container_width=True)
        st.download_button(
            "⬇️ Download Blind/Future Picks by Threshold",
            data=blind_results.to_csv(index=False),
            file_name="blind_future_picks_by_threshold.csv",
        )
    except Exception as e:
        st.error(f"Error with blind/future CSV: {e}")

# --- Comparison/Instructions ---
st.header("How to Use These Results")
st.markdown("""
- **Main picks**: What your XGBoost model would have picked for each day (and threshold) over the historical period.
- **Blind/future picks**: Upload an event-level CSV for dates (e.g., June 17–19) where you have no HR outcomes, to get unbiased "future" predictions by the model.
- For both, download the results and analyze how your model would have performed at each threshold. Give the pick CSVs to your Analyzer Bot for a full breakdown.
""")

st.success("App ready! Upload your CSVs and run your threshold sweeps above.")

# -------------- END OF APP --------------
