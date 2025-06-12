import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# --- LOGISTIC WEIGHTS ---
LOGISTIC_WEIGHTS = {
    'iso_value': 5.757820079, 'hit_distance_sc': 0.6411852127, 'pull_side': 0.5569402386,
    'launch_speed_angle': 0.5280235471, 'B_pitch_pct_CH_5': 0.3858783912, 'park_handed_hr_rate': 0.3438658641,
    'B_median_ev_7': 0.33462617, 'B_pitch_pct_CU_3': 0.3280395666, 'P_max_ev_5': 0.3113203434,
    'P_pitch_pct_SV_3': 0.2241205438, 'B_pitch_pct_EP_5': 0.2163322514, 'P_pitch_pct_ST_14': 0.2052831283,
    'P_rolling_hr_rate_7': 0.1877664166, 'P_pitch_pct_FF_5': 0.1783978536, 'P_median_ev_3': 0.1752142738,
    'groundball': 0.1719989086, 'B_pitch_pct_KC_5': 0.1615036223, 'B_pitch_pct_FS_3': 0.1595644445,
    'P_pitch_pct_FC_14': 0.1591148241, 'B_pitch_pct_SI_14': 0.1570044892, 'B_max_ev_5': 0.1540596514,
    'P_pitch_pct_CU_7': 0.1524371468, 'P_pitch_pct_SL_3': 0.1429928993, 'P_pitch_pct_FO_14': 0.1332430394,
    'B_pitch_pct_SV_5': 0.1257929016, 'P_hit_distance_sc_7': 0.1236586016, 'B_iso_value_14': 0.1199768939,
    'P_woba_value_5': 0.1175567692, 'B_pitch_pct_CS_14': 0.1137568069, 'pitch_pct_FO': 0.1124543401,
    'B_pitch_pct_FF_7': 0.105404093, 'is_barrel': 0.1044204311, 'B_pitch_pct_FA_7': 0.1041956255,
    'pitch_pct_FF': 0.1041947265, 'B_pitch_pct_ST_3': 0.1016502344, 'pitch_pct_ST': 0.09809980426,
    'pitch_pct_CH': 0.09588455603,
    # ...rest of weights...
}
INTERCEPT = 0

progress = st.progress(0, text="Waiting for required CSV uploads...")

# --- 1. UPLOAD ALL CSVs (ALL 3 ARE REQUIRED) ---
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"], key="player")
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"], key="event")
lineup_csv = st.sidebar.file_uploader("Lineup/Matchup CSV (required, all confirmed)", type=["csv"], key="lineup")

if not player_csv or not event_csv or not lineup_csv:
    st.warning("⬆️ Upload **player-level**, **event-level**, and **lineup/matchup** CSVs to begin!")
    st.stop()
progress.progress(10, text="CSVs uploaded. Loading files...")

player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)
lineup_df = pd.read_csv(lineup_csv)
progress.progress(20, text="CSV dataframes loaded...")

# --- CLEAN LINEUP/MATCHUP HEADERS ---
lineup_df.columns = lineup_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")

# --- PITCHER MATCHUP LOGIC ---
if "game_number" in lineup_df.columns:
    unique_games = lineup_df["game_number"].unique()
else:
    unique_games = [1]
pitcher_map = {}
for g in unique_games:
    sub = lineup_df[lineup_df.get("game_number", 1) == g]
    teams = sub["team_code"].unique()
    for t in teams:
        sp_row = sub[(sub["team_code"] == t) & (sub["position"].str.upper() == "SP")]
        if not sp_row.empty:
            pitcher_map[(g, t)] = sp_row.iloc[0]["mlb_id"]
def get_opposing_pitcher(row):
    g = row.get("game_number", 1)
    t = row["team_code"]
    teams = lineup_df[lineup_df.get("game_number", 1) == g]["team_code"].unique()
    opposing = [x for x in teams if x != t]
    return pitcher_map.get((g, opposing[0])) if opposing else None
lineup_df["pitcher_id"] = lineup_df.apply(get_opposing_pitcher, axis=1)

progress.progress(35, text="Pitcher matchups assigned for each batter...")

# --- MERGE PLAYER-LEVEL FEATURES ---
lineup_df["batter_id"] = lineup_df["mlb_id"].astype(str)
player_df["batter_id"] = player_df["batter_id"].astype(str)
merged = lineup_df.merge(player_df, on="batter_id", how="left", suffixes=('', '_feat'))
progress.progress(60, text="Merged lineup and player-level features...")

# --- WEATHER LOGIC (by city from lineup CSV) ---
def fetch_weather(city, date_str=None, time_str=None):
    try:
        api_key = st.secrets["weather"]["api_key"]
        query = city if pd.notna(city) else "New York"
        # If date_str is None just use today
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={query}&dt="
        from datetime import datetime
        dt_str = datetime.now().strftime('%Y-%m-%d')
        url = url + dt_str
        resp = requests.get(url, timeout=7)
        data = resp.json()
        day = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = day.get("avgtemp_f", 72)
        humidity = day.get("avghumidity", 55)
        wind_mph = day.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception:
        return 72, 55, 7

if not merged.empty:
    city = merged.iloc[0].get("city", "New York")
    time_str = merged.iloc[0].get("time", None)
    temp, humidity, wind_mph = fetch_weather(city)
    merged['weather_temp'] = temp
    merged['weather_humidity'] = humidity
    merged['weather_wind_mph'] = wind_mph
progress.progress(70, text="Weather API data attached...")

# --- FILL MISSING LOGIT FEATURES ---
for col in LOGISTIC_WEIGHTS:
    if col not in merged.columns:
        merged[col] = 0

# --- CALCULATE HR LOGIT SCORE & PROBABILITY ---
def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

merged['HR Logit Score'] = merged.apply(calc_hr_logit, axis=1)
merged['HR Probability'] = 1 / (1 + np.exp(-merged['HR Logit Score']))
progress.progress(85, text="Logistic model scoring complete...")

# --- RANK LEADERBOARD ---
merged = merged.sort_values("HR Logit Score", ascending=False).reset_index(drop=True)
merged["Rank"] = np.arange(1, len(merged) + 1)

# --- FINAL LEADERBOARD DISPLAY ---
leaderboard_cols = [
    "Rank", "player_name", "batter_id", "team_code", "batting_order", "pitcher_id",
    "stadium", "city", "weather_temp", "weather_humidity", "weather_wind_mph",
    "HR Logit Score", "HR Probability"
]
# Add your top features for context
for col in ["iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5", "park_handed_hr_rate"]:
    if col in merged.columns and col not in leaderboard_cols:
        leaderboard_cols.append(col)

progress.progress(100, text="Leaderboard ready! 🎉")

st.success(f"Done! Leaderboard generated for uploaded confirmed hitters ({len(merged)} hitters)")

st.header("Predicted Home Run Leaderboard (Today's Confirmed Hitters)")
st.dataframe(merged[leaderboard_cols].head(15), use_container_width=True)
st.caption("Scored by HR Logit Score (one row per actual confirmed hitter, with daily pitcher matchup, park/city/weather features).")

with st.expander("See all columns for today's lineup batters:"):
    st.dataframe(merged, use_container_width=True)
