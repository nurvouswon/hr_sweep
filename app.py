import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ------------------ CONFIG -------------------
API_KEY = st.secrets["weather"]["api_key"] if "weather" in st.secrets else ""
WEATHER_URL = "http://api.weatherapi.com/v1/history.json"

# ------------------ LOGIT WEIGHTS -------------------
LOGIT_WEIGHTS = {
    "iso_value": 5.757820079, "hit_distance_sc": 0.6411852127, "pull_side": 0.5569402386, "launch_speed_angle": 0.5280235471,
    "B_pitch_pct_CH_5": 0.3858783912, "park_handed_hr_rate": 0.3438658641, "B_median_ev_7": 0.33462617, "B_pitch_pct_CU_3": 0.3280395666,
    "P_max_ev_5": 0.3113203434, "P_pitch_pct_SV_3": 0.2241205438, "B_pitch_pct_EP_5": 0.2163322514, "P_pitch_pct_ST_14": 0.2052831283,
    "P_rolling_hr_rate_7": 0.1877664166, "P_pitch_pct_FF_5": 0.1783978536, "P_median_ev_3": 0.1752142738, "groundball": 0.1719989086,
    "B_pitch_pct_KC_5": 0.1615036223, "B_pitch_pct_FS_3": 0.1595644445, "P_pitch_pct_FC_14": 0.1591148241, "B_pitch_pct_SI_14": 0.1570044892,
    "B_max_ev_5": 0.1540596514, "P_pitch_pct_CU_7": 0.1524371468, "P_pitch_pct_SL_3": 0.1429928993, "P_pitch_pct_FO_14": 0.1332430394,
    "B_pitch_pct_SV_5": 0.1257929016, "P_hit_distance_sc_7": 0.1236586016, "B_iso_value_14": 0.1199768939, "P_woba_value_5": 0.1175567692,
    "B_pitch_pct_CS_14": 0.1137568069, "pitch_pct_FO": 0.1124543401, "B_pitch_pct_FF_7": 0.105404093, "is_barrel": 0.1044204311,
    "B_pitch_pct_FA_7": 0.1041956255, "pitch_pct_FF": 0.1041947265, "B_pitch_pct_ST_3": 0.1016502344, "pitch_pct_ST": 0.09809980426,
    "pitch_pct_CH": 0.09588455603, "B_pitch_pct_SL_3": 0.09395294235, "P_rolling_hr_rate_5": 0.09176055559, "B_pitch_pct_SC_14": 0.08671517652,
    "platoon": 0.08601459992, "P_pitch_pct_FS_3": 0.08464192523, "B_iso_value_7": 0.08090866123, "B_pitch_pct_KC_7": 0.08079362526,
    "B_median_ev_14": 0.07898600411, "B_pitch_pct_KN_7": 0.07368063279, "B_pitch_pct_SL_14": 0.07334392117, "P_pitch_pct_SV_5": 0.06890378686,
    "P_pitch_pct_CH_3": 0.06804529698, "P_woba_value_7": 0.0674790282, "B_launch_angle_7": 0.06733255236, "P_pitch_pct_ST_7": 0.06545350898,
    "B_pitch_pct_FF_14": 0.06491620372, "P_max_ev_7": 0.06116445719, "P_max_ev_3": 0.05980174448, "B_pitch_pct_FC_7": 0.05788952516,
    "B_pitch_pct_FA_3": 0.05587337787, "pitch_pct_FC": 0.05483038609, "P_pitch_pct_KC_7": 0.05350923671, "B_max_ev_3": 0.05203847819,
    "P_launch_angle_5": 0.05141139562, "P_pitch_pct_CS_14": 0.05139024478, "B_pitch_pct_FA_14": 0.05021331706, "P_pitch_pct_CU_14": 0.05020601371,
    "P_rolling_hr_rate_3": 0.04837416267, "P_pitch_pct_EP_3": 0.04716192902, "B_pitch_pct_EP_7": 0.04703265604, "P_iso_value_7": 0.04279584322,
    "P_pitch_pct_CS_7": 0.04223520154, "B_hit_distance_sc_7": 0.04213173751, "P_hit_distance_sc_14": 0.04051098632, "pitch_pct_EP": 0.04016871102,
    # ... [truncate here for brevity, but you can include all weights as above]
    "line_drive": -0.7114540736
}

# ------------------ WEATHER FETCH -------------------
@st.cache_data(show_spinner=False)
def get_weather(city, date, api_key=API_KEY):
    """Fetch weather data for city and date using weatherapi.com."""
    try:
        params = {"key": api_key, "q": city, "dt": date}
        resp = requests.get(WEATHER_URL, params=params, timeout=10)
        data = resp.json()
        # Default: take the 1st hour with temp
        for h in data.get("forecast", {}).get("forecastday", [{}])[0].get("hour", []):
            if "temp_f" in h:
                return {
                    "temp": h["temp_f"], "wind_mph": h["wind_mph"], "humidity": h["humidity"]
                }
        return {"temp": None, "wind_mph": None, "humidity": None}
    except Exception as e:
        st.warning(f"Weather API error: {e}")
        return {"temp": None, "wind_mph": None, "humidity": None}

# ------------------ HR LOGIT SCORE -------------------
def calc_hr_logit_score(row, weights=LOGIT_WEIGHTS):
    """Calculate logit score using your weights, treating missing as 0."""
    return sum(float(row.get(f, 0) or 0) * w for f, w in weights.items())

# ------------------ STREAMLIT APP -------------------
st.title("⚾ MLB HR Predictor: Leaderboard & Probability")

st.markdown("Upload your **event-level** and **player-level** Analyzer CSVs. Select a game date for prediction/testing.")

event_csv = st.file_uploader("Event-level CSV (Analyzer)", type="csv")
player_csv = st.file_uploader("Player-level CSV (Analyzer)", type="csv")

selected_date = st.date_input("Game Date for Leaderboard", value=datetime.now().date())

if event_csv and player_csv:
    # Read & Clean
    events = pd.read_csv(event_csv, low_memory=False)
    players = pd.read_csv(player_csv, low_memory=False)
    # Drop duplicate cols by keeping first
    events = events.loc[:,~events.columns.duplicated()]
    players = players.loc[:,~players.columns.duplicated()]

    # Filter to selected date
    if "game_date" in events.columns:
        df = events[events["game_date"].astype(str) == str(selected_date)]
        if df.empty:
            st.warning("No events for selected date in CSV.")
            st.stop()
    else:
        st.error("Missing 'game_date' in event CSV.")
        st.stop()

    # Merge player stats onto event rows by batter_id
    if "batter_id" in df.columns and "batter_id" in players.columns:
        merged = df.merge(players, on="batter_id", how="left", suffixes=("", "_pl"))
    else:
        st.error("batter_id column missing in one or both CSVs.")
        st.stop()

    # Try to add weather columns (optionally, per row)
    weather_cols = ["temp", "wind_mph", "humidity"]
    if all(c in merged.columns for c in ["city", "game_date"]):
        for city in merged["city"].unique():
            city_date = merged[merged["city"] == city]["game_date"].iloc[0]
            weather = get_weather(city, city_date)
            for c in weather_cols:
                merged.loc[merged["city"] == city, c] = weather[c]
    else:
        # If you want to require city/date, change this block
        merged["temp"] = merged["wind_mph"] = merged["humidity"] = np.nan

    # Calculate logit scores
    merged["hr_logit_score"] = merged.apply(calc_hr_logit_score, axis=1)
    # Optional: probability (sigmoid)
    merged["prob_hr"] = 1 / (1 + np.exp(-merged["hr_logit_score"]))

    # Sort and deduplicate columns for output
    leaderboard = merged.copy()
    leaderboard = leaderboard.loc[:, ~leaderboard.columns.duplicated()]

    # Show leaderboard
    sort_cols = ["prob_hr", "hr_logit_score"]
    show_cols = ["batter", "pitcher", "team_code", "prob_hr", "hr_logit_score", "park_handed_hr_rate", "game_date", "temp", "wind_mph", "humidity"]
    show_cols = [c for c in show_cols if c in leaderboard.columns]
    leaderboard = leaderboard.sort_values(by="prob_hr", ascending=False)
    st.dataframe(leaderboard[show_cols].head(25).style.format({"prob_hr": "{:.3f}", "hr_logit_score": "{:.2f}"}), use_container_width=True)

    # Download
    st.download_button(
        "Download Full Leaderboard CSV",
        data=leaderboard.to_csv(index=False),
        file_name=f"hr_leaderboard_{selected_date}.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload both CSVs to generate your leaderboard.")
