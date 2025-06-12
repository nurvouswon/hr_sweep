import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB HR Predictor", layout="wide")

# 1. -- FILE UPLOADERS (REQUIRED) --
st.title("MLB Home Run Predictor ⚾️")
st.sidebar.header("Step 1: Upload CSVs")
player_csv = st.sidebar.file_uploader("Player-Level CSV (REQUIRED)", type=["csv"])
event_csv = st.sidebar.file_uploader("Event-Level CSV (REQUIRED)", type=["csv"])
if not player_csv or not event_csv:
    st.warning("⬆️ Please upload BOTH player-level and event-level CSVs to begin!")
    st.stop()
player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)

# 2. -- DATE SELECTOR --
today_str = datetime.now().strftime("%Y-%m-%d")
sel_date = st.sidebar.date_input("Select Prediction Date", value=datetime.now())
sel_date_str = sel_date.strftime("%Y-%m-%d")

# 3. -- LOGISTIC WEIGHTS (CUSTOMIZE YOUR LIST) --
LOGISTIC_WEIGHTS = {
    'iso_value': 5.757820079,
    'hit_distance_sc': 0.6411852127,
    'pull_side': 0.5569402386,
    'launch_speed_angle': 0.5280235471,
    'B_pitch_pct_CH_5': 0.3858783912,
    'park_handed_hr_rate': 0.3438658641,
    'B_median_ev_7': 0.33462617,
    'B_pitch_pct_CU_3': 0.3280395666,
    # ...add more as you wish
}
INTERCEPT = 0

# 4. -- MLB PLAYER ID TO NAME LOOKUP (Chadwick Bureau, cached) --
@st.cache_data(ttl=86400)
def get_player_id_map():
    url = "https://raw.githubusercontent.com/chadwickbureau/register/master/people.csv"
    try:
        df = pd.read_csv(url, usecols=['key_mlbam', 'name_display_last_first'])
        id_map = {int(row["key_mlbam"]): row["name_display_last_first"] for _, row in df.iterrows()}
        return id_map
    except Exception:
        return {}

player_id_map = get_player_id_map()

def resolve_batter_name(bid):
    """Try MLB ID, then fallback to CSV batter name, then string of ID."""
    try:
        bid_int = int(float(bid))
        name = player_id_map.get(bid_int)
        if name and isinstance(name, str) and name.strip():
            return name
    except Exception:
        pass
    # Fallback: Use 'batter' column in player_df if present
    try:
        if "batter" in player_df.columns:
            row = player_df[player_df["batter_id"] == bid]
            if not row.empty:
                possible_name = row.iloc[0]["batter"]
                if isinstance(possible_name, str) and len(possible_name) > 1:
                    return possible_name
    except Exception:
        pass
    return str(bid)

# 5. -- WEATHER LOOKUP BY BALLPARK --
def fetch_weather(park, date_str):
    park_locations = {
        "camden_yards": "Baltimore,MD",
        "target_field": "Minneapolis,MN",
        "wrigley_field": "Chicago,IL",
        # Add more mappings as needed!
    }
    location = park_locations.get(str(park).lower(), park)
    try:
        api_key = st.secrets["weather"]["api_key"]  # WeatherAPI.com
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}"
        resp = requests.get(url, timeout=7)
        data = resp.json()
        weather = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = weather.get("avgtemp_f", 72)
        humidity = weather.get("avghumidity", 55)
        wind_mph = weather.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception:
        st.warning(f"Could not fetch weather for {park} on {date_str}. Using defaults.")
        return 72, 55, 7

# 6. -- GET BALLPARK FOR WEATHER LOOKUP --
if "park" in event_df.columns and not event_df.empty:
    park = event_df["park"].mode()[0]
else:
    park = st.sidebar.text_input("Ballpark for Weather Lookup", "camden_yards")

weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park, sel_date_str)

# 7. -- FILL MISSING FEATURES (so model doesn't error)
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0

# 8. -- CALCULATE LOGIT SCORE & PROBABILITY FOR EACH BATTER --
def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

player_df["HR Logit Score"] = player_df.apply(calc_hr_logit, axis=1)
player_df["HR Probability"] = 1 / (1 + np.exp(-player_df["HR Logit Score"]))

# 9. -- ADD WEATHER TO DISPLAY --
player_df["Weather Temp (F)"] = weather_temp
player_df["Weather Humidity"] = weather_humidity
player_df["Weather Wind (mph)"] = weather_wind_mph

# 10. -- ADD REAL PLAYER NAMES (not just IDs) --
player_df["Player"] = player_df["batter_id"].apply(resolve_batter_name)

# 11. -- DISPLAY LEADERBOARD (top 15 by default) --
st.header("Predicted Home Run Leaderboard")
cols_to_show = [
    "Player", "batter_id", "HR Probability", "HR Logit Score",
    "Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"
]
for c in LOGISTIC_WEIGHTS.keys():
    if c in player_df.columns:
        cols_to_show.append(c)
leaderboard = player_df[cols_to_show].sort_values("HR Probability", ascending=False)
st.dataframe(leaderboard.head(15), use_container_width=True)

st.caption(
    "Leaderboard = One row per batter, using logistic weights, weather, player ID to name mapping, and all core features. "
    "If player name can't be resolved, shows batter_id. If you want to see more columns or raw data, unhide extra columns in the leaderboard."
)

# Optional: show debug info
with st.expander("Debug: Show Batter Name Mapping for first 10 IDs"):
    st.write(player_df[["batter_id", "Player"]].head(10))
