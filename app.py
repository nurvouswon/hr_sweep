import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB HR Predictor", layout="wide")

# --- HARDCODED LOGISTIC WEIGHTS ---
LOGIT_WEIGHTS = {
    "iso_value": 5.757820079, "hit_distance_sc": 0.6411852127, "pull_side": 0.5569402386,
    "launch_speed_angle": 0.5280235471, "B_pitch_pct_CH_5": 0.3858783912, "park_handed_hr_rate": 0.3438658641,
    "B_median_ev_7": 0.33462617, "B_pitch_pct_CU_3": 0.3280395666, "P_max_ev_5": 0.3113203434,
    "P_pitch_pct_SV_3": 0.2241205438, "B_pitch_pct_EP_5": 0.2163322514, "P_pitch_pct_ST_14": 0.2052831283,
    "P_rolling_hr_rate_7": 0.1877664166, "P_pitch_pct_FF_5": 0.1783978536, "P_median_ev_3": 0.1752142738,
    "groundball": 0.1719989086, "B_pitch_pct_KC_5": 0.1615036223, "B_pitch_pct_FS_3": 0.1595644445,
    "P_pitch_pct_FC_14": 0.1591148241, "B_pitch_pct_SI_14": 0.1570044892, "B_max_ev_5": 0.1540596514,
    # ...add as needed
}

# --- MLB ID → NAME LOOKUP ---
@st.cache_data(show_spinner=False)
def get_mlb_id_name_dict():
    try:
        url = "https://statsapi.mlb.com/api/v1/people?sportId=1&season=2024"
        resp = requests.get(url, timeout=8)
        data = resp.json()
        id_map = {}
        for person in data.get("people", []):
            id_map[str(person["id"])] = f"{person.get('lastName','')}, {person.get('firstName','')}"
        return id_map
    except Exception:
        return {}

ID2NAME = get_mlb_id_name_dict()

# --- WEATHER LOOKUP FUNCTION ---
def fetch_weather(park, date_str):
    try:
        api_key = st.secrets["weather"]["api_key"]
    except Exception:
        st.warning("Weather API key missing! Add to secrets.toml as [weather] api_key = \"...\"")
        return {"weather_temp": np.nan, "weather_humidity": np.nan, "weather_wind_mph": np.nan}
    park_map = {
        "camden_yards": "Baltimore", "yankee_stadium": "Bronx", "dodger_stadium": "Los Angeles",
    }
    city = park_map.get(park, park.replace("_", " ").title())
    params = {"key": api_key, "q": city, "dt": date_str}
    try:
        r = requests.get("https://api.weatherapi.com/v1/history.json", params=params, timeout=10)
        if not r.ok:
            return {"weather_temp": np.nan, "weather_humidity": np.nan, "weather_wind_mph": np.nan}
        j = r.json()
        day = j["forecast"]["forecastday"][0]["day"]
        return {
            "weather_temp": day.get("avgtemp_f", np.nan),
            "weather_humidity": day.get("avghumidity", np.nan),
            "weather_wind_mph": day.get("maxwind_mph", np.nan),
        }
    except Exception:
        return {"weather_temp": np.nan, "weather_humidity": np.nan, "weather_wind_mph": np.nan}

# --- UI: FILE UPLOADERS & DATE PICKER ---
st.title("MLB Home Run Predictor ⚾️")
st.markdown("""
**1. Upload required event-level and player-level CSVs (Analyzer output format).**  
**2. Select a date (default: today).**  
**3. Press 'Run Predictor' to get the leaderboard.**
""")
col1, col2 = st.columns(2)
with col1:
    event_csv = st.file_uploader("Event-level CSV (required)", type="csv", key="event_csv")
with col2:
    player_csv = st.file_uploader("Player-level CSV (required)", type="csv", key="player_csv")
sel_date = st.date_input("Prediction/Test Date", value=datetime.today(), key="date_select")
run_btn = st.button("Run Predictor", type="primary")

if not event_csv or not player_csv:
    st.info("⬆️ Upload both event and player CSVs to continue.")
    st.stop()

if run_btn:
    # --- LOAD DATA ---
    try:
        events = pd.read_csv(event_csv)
        players = pd.read_csv(player_csv)
    except Exception as e:
        st.error(f"CSV parsing error: {e}")
        st.stop()

    # --- FILTER EVENTS BY DATE ---
    date_str = sel_date.strftime("%Y-%m-%d")
    if "game_date" in events.columns:
        day_events = events[events['game_date'] == date_str].copy()
        if day_events.empty:
            st.warning(f"No event data for {date_str}. Check your file and date selection.")
            st.stop()
    else:
        st.warning("'game_date' column missing in event CSV.")
        st.stop()

    # --- GET ONLY CONFIRMED LINEUP BATTERS ---
    if "batter_id" in day_events.columns and "bat_score" in day_events.columns:
        # Assume presence in day_events == confirmed lineup; add other logic if needed.
        confirmed_batters = day_events["batter_id"].unique()
        players = players[players["batter_id"].isin(confirmed_batters)]
    else:
        st.warning("Could not filter to confirmed lineup batters. All players will be scored.")

    # --- WEATHER FOR PARK/DATE ---
    park = day_events["park"].mode().iloc[0] if "park" in day_events.columns else "camden_yards"
    wx = fetch_weather(park, date_str)

    # --- MATCH PITCHER & HAND FOR EACH BATTER ---
    matchup_cols = ["batter_id", "pitcher_id", "p_throws", "park", "park_handed_hr_rate"]
    if all(col in day_events.columns for col in matchup_cols):
        matchups = day_events[matchup_cols].drop_duplicates("batter_id")
        players = pd.merge(players, matchups, how="left", on="batter_id", suffixes=('', '_matchup'))
    else:
        st.warning("Pitcher/batter matchup columns missing. Only using player-level data.")

    # --- WEATHER DATA ADD ---
    for k, v in wx.items():
        players[k] = v

    # --- ADD PLAYER NAME & TEAM ---
    players["Player"] = players["batter_id"].astype(str).map(ID2NAME).fillna(players["batter_id"].astype(str))
    players["MLB ID"] = players["batter_id"]
    players["Team"] = day_events["home_team"].mode().iloc[0] if "home_team" in day_events.columns else ""

    # --- LOGIT SCORE & HR PROBABILITY ---
    def logit_row(row):
        score = 0
        for feat, wt in LOGIT_WEIGHTS.items():
            if feat in row and not pd.isnull(row[feat]):
                score += row[feat] * wt
        return score

    players["HR Logit Score"] = players.apply(logit_row, axis=1)
    players["HR Probability"] = 1 / (1 + np.exp(-players["HR Logit Score"]))

    # --- TOP HR FEATURES ---
    show_feats = [
        "Player", "MLB ID", "Team", "pitcher_id", "p_throws", "HR Probability", "HR Logit Score",
        "iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5",
        "park_handed_hr_rate", "B_median_ev_7", "B_pitch_pct_CU_3", "P_max_ev_5", "P_pitch_pct_SV_3",
        "weather_temp", "weather_humidity", "weather_wind_mph"
    ]
    present_cols = [c for c in show_feats if c in players.columns]

    leaderboard = players.sort_values("HR Probability", ascending=False)

    st.header(f"Predicted HR Leaderboard for {date_str}")
    st.dataframe(leaderboard[present_cols].head(15), use_container_width=True)
    st.caption("Leaderboard: filtered to confirmed lineup batters, scored vs. probable starters with weather, matchup & stat features.")

    # --- OPTION: SEE ALL DATA IF WANTED ---
    with st.expander("See All Features for Top Batters"):
        st.dataframe(leaderboard.head(15), use_container_width=True)
