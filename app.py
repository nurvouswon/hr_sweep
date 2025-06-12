import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# ------------------------ LOGISTIC WEIGHTS ------------------------
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
    # ...add more as needed...
}
INTERCEPT = 0

# ------------------------ PROGRESS BAR SETUP ------------------------
progress_text = "Ready to go..."
progress = st.progress(0, text=progress_text)

# ------------------------ 1. FILE UPLOADERS ------------------------
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"], key="player")
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"], key="event")
progress.progress(10, text="Waiting for both player-level and event-level CSVs...")

if not player_csv or not event_csv:
    st.warning("⬆️ Upload **both** player-level and event-level CSVs to begin!")
    st.stop()

player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)
progress.progress(20, text="Files uploaded, reading player/event CSVs...")

# ------------------------ 2. DATE SELECTION ------------------------
st.sidebar.header("Step 2: Prediction Date")
sel_date = st.sidebar.date_input("Select Date for Prediction", value=datetime.now())
sel_date_str = sel_date.strftime("%Y-%m-%d")
progress.progress(30, text="Prediction date set...")

# ------------------------ 3. WEATHER LOOKUP ------------------------
def fetch_weather(city, date_str):
    try:
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"
        resp = requests.get(url, timeout=7)
        data = resp.json()
        day = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = day.get("avgtemp_f", 72)
        humidity = day.get("avghumidity", 55)
        wind_mph = day.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception:
        return 72, 55, 7  # fallback

# Use mode park/city if available, else fallback input
if "park" in event_df.columns and not event_df.empty:
    park = event_df["park"].mode()[0]
else:
    park = st.sidebar.text_input("Ballpark for Weather Lookup", "camden_yards")

weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park, sel_date_str)
player_df["weather_temp"] = weather_temp
player_df["weather_humidity"] = weather_humidity
player_df["weather_wind_mph"] = weather_wind_mph
progress.progress(40, text="Weather data loaded and merged...")

# ------------------------ 4. FILL MISSING FEATURES WITH ZEROS ------------------------
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0
progress.progress(50, text="Features filled/prepped...")

# ------------------------ 5. CALCULATE LOGIT SCORE & PROB ------------------------
def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

player_df["HR Logit Score"] = player_df.apply(calc_hr_logit, axis=1)
player_df["HR Probability"] = 1 / (1 + np.exp(-player_df["HR Logit Score"]))
progress.progress(60, text="Scoring model applied to all players...")

# ------------------------ 6. LOOKUP PLAYER NAMES USING MLB API ------------------------
@st.cache_data(ttl=86400)
def get_mlb_player_names(mlb_id_list):
    out_map = {}
    ids = [int(i) for i in mlb_id_list if not pd.isna(i)]
    CHUNK = 100
    for i in range(0, len(ids), CHUNK):
        id_str = ",".join(str(j) for j in ids[i:i+CHUNK])
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={id_str}"
        try:
            resp = requests.get(url, timeout=7)
            data = resp.json().get("people", [])
            for p in data:
                out_map[str(p["id"])] = p.get("fullName", "")
        except Exception:
            continue
    return out_map

unique_ids = player_df["batter_id"].dropna().astype(int).unique().tolist()
name_map = get_mlb_player_names(unique_ids)
player_df["Player"] = player_df["batter_id"].astype(int).astype(str).map(name_map).fillna(player_df["batter_id"])
progress.progress(80, text="Player names resolved using MLB API...")

# ------------------------ 7. LEADERBOARD BUILD & RANK ------------------------
player_df = player_df.drop_duplicates(subset="batter_id")  # One row per batter
player_df = player_df.sort_values("HR Logit Score", ascending=False).reset_index(drop=True)
player_df["Rank"] = np.arange(1, len(player_df)+1)
progress.progress(90, text="Leaderboard sorted and ranked...")

leaderboard_cols = [
    "Rank", "Player", "batter_id", "HR Logit Score", "HR Probability",
    "weather_temp", "weather_humidity", "weather_wind_mph"
]
for col in ["iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5", "park_handed_hr_rate"]:
    if col in player_df.columns and col not in leaderboard_cols:
        leaderboard_cols.append(col)

progress.progress(100, text="Leaderboard complete! 🎉")

st.success(f"Done! Leaderboard generated for {sel_date_str} with {len(player_df)} batters.")

# ------------------------ 8. DISPLAY LEADERBOARD ------------------------
st.header("Predicted Home Run Leaderboard")
st.dataframe(player_df[leaderboard_cols].head(15), use_container_width=True)
st.caption("Sorted by HR Logit Score (most predictive for home runs). Player names are from MLB.com API.")

with st.expander("See all feature columns for leaderboard batters"):
    st.dataframe(player_df, use_container_width=True)
