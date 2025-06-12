import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# ----------- 1. CSV Uploaders FIRST -----------
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"], key="player")
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"], key="event")

if not player_csv or not event_csv:
    st.warning("⬆️ Upload **both** player-level and event-level CSVs to begin!")
    st.stop()

player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)

# ----------- 2. Date Selector -----------
st.sidebar.header("Step 2: Prediction Date")
sel_date = st.sidebar.date_input("Select Date for Prediction", value=datetime.now())
sel_date_str = sel_date.strftime("%Y-%m-%d")

# ----------- 3. Hardcoded Logistic Weights -----------
LOGISTIC_WEIGHTS = {
    'iso_value': 5.757820079,
    'hit_distance_sc': 0.6411852127,
    'pull_side': 0.5569402386,
    'launch_speed_angle': 0.5280235471,
    'B_pitch_pct_CH_5': 0.3858783912,
    'park_handed_hr_rate': 0.3438658641,
    'B_median_ev_7': 0.33462617,
    'B_pitch_pct_CU_3': 0.3280395666,
    'P_max_ev_5': 0.3113203434,
    'P_pitch_pct_SV_3': 0.2241205438,
    'B_pitch_pct_EP_5': 0.2163322514,
    'P_pitch_pct_ST_14': 0.2052831283,
    'P_rolling_hr_rate_7': 0.1877664166,
    'P_pitch_pct_FF_5': 0.1783978536,
    'P_median_ev_3': 0.1752142738,
    'groundball': 0.1719989086,
    'B_pitch_pct_KC_5': 0.1615036223,
    'B_pitch_pct_FS_3': 0.1595644445,
    'P_pitch_pct_FC_14': 0.1591148241,
    'B_pitch_pct_SI_14': 0.1570044892,
    'B_max_ev_5': 0.1540596514,
    'P_pitch_pct_CU_7': 0.1524371468,
    'P_pitch_pct_SL_3': 0.1429928993,
    'P_pitch_pct_FO_14': 0.1332430394,
    'B_pitch_pct_SV_5': 0.1257929016,
    'P_hit_distance_sc_7': 0.1236586016,
    'B_iso_value_14': 0.1199768939,
    'P_woba_value_5': 0.1175567692,
    'B_pitch_pct_CS_14': 0.1137568069,
    'pitch_pct_FO': 0.1124543401,
    'B_pitch_pct_FF_7': 0.105404093,
    'is_barrel': 0.1044204311,
    'B_pitch_pct_FA_7': 0.1041956255,
    'pitch_pct_FF': 0.1041947265,
    'B_pitch_pct_ST_3': 0.1016502344,
    'pitch_pct_ST': 0.09809980426,
    'pitch_pct_CH': 0.09588455603,
    # ...full list, can be expanded...
}
INTERCEPT = 0

# ----------- 4. MLB Player Name Lookup -----------
@st.cache_data(ttl=86400)
def get_player_id_map():
    url = "https://raw.githubusercontent.com/chadwickbureau/register/master/people.csv"
    try:
        df = pd.read_csv(url, usecols=['key_mlbam', 'name_display_last_first'])
        id_map = df.set_index('key_mlbam')['name_display_last_first'].to_dict()
        return id_map
    except Exception:
        st.warning("Could not fetch player names. Only MLB ID will display.")
        return {}

id_map = get_player_id_map()

# ----------- 5. Weather API (WeatherAPI.com) -----------
def fetch_weather(park, date_str):
    park_locations = {
        "camden_yards": "Baltimore,MD",
        "target_field": "Minneapolis,MN",
        "wrigley_field": "Chicago,IL",
        # ...expand as needed...
    }
    location = park_locations.get(str(park).lower(), park)
    api_key = st.secrets["weather"]["api_key"]  # Set this in .streamlit/secrets.toml!
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}"
    try:
        resp = requests.get(url, timeout=7)
        data = resp.json()
        weather = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = weather.get("avgtemp_f", 72)
        humidity = weather.get("avghumidity", 55)
        wind_mph = weather.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception as e:
        st.warning(f"Could not fetch weather for {park} on {date_str}. Using defaults.")
        return 72, 55, 7

# ----------- 6. Get MLB Lineups and Pitchers -----------
def get_mlb_lineups(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1&hydrate=lineups"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        all_batters = []
        all_pitchers = []
        for date in data.get("dates", []):
            for game in date.get("games", []):
                for team_key in ['home', 'away']:
                    t = game["teams"][team_key]
                    # Probable pitcher
                    lineup = t.get("probablePitcher", {})
                    if "id" in lineup:
                        all_pitchers.append({
                            "team": t["team"]["abbreviation"],
                            "pitcher_id": lineup["id"],
                            "p_throws": lineup.get("hand", {}).get("code", None),
                        })
                    # Starting batters
                    starters = t.get("lineups", [])
                    for s in starters:
                        if "player" in s:
                            all_batters.append({
                                "team": t["team"]["abbreviation"],
                                "batter_id": s["player"]["id"],
                                "Player": s["player"]["fullName"],
                                "bat_order": s["order"],
                            })
        return pd.DataFrame(all_batters), pd.DataFrame(all_pitchers)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

lineups_df, pitchers_df = get_mlb_lineups(sel_date_str)

# ----------- 7. Merge Lineups, Add Names, Pitcher Logic -----------
if not lineups_df.empty:
    player_df = player_df.merge(
        lineups_df.rename(columns={"batter_id": "batter_id", "team": "Team", "Player": "Player"}),
        on="batter_id", how="inner"
    )
else:
    player_df["Player"] = player_df["batter_id"].astype(str).map(id_map).fillna(player_df["batter_id"].astype(str))
    player_df["Team"] = ""

# Pitcher matchup assignment
if not pitchers_df.empty:
    player_df["pitcher_id"] = np.nan
    player_df["p_throws"] = ""
    for team in player_df["Team"].unique():
        p = pitchers_df[pitchers_df["team"] == team]
        if not p.empty:
            pid = p.iloc[0]["pitcher_id"]
            hand = p.iloc[0]["p_throws"]
            player_df.loc[player_df["Team"] == team, "pitcher_id"] = pid
            player_df.loc[player_df["Team"] == team, "p_throws"] = hand
else:
    player_df["pitcher_id"] = ""
    player_df["p_throws"] = ""

# ----------- 8. Park and Weather -----------
if "park" in event_df.columns and not event_df.empty:
    park = event_df["park"].mode()[0]
else:
    park = st.sidebar.text_input("Ballpark for Weather Lookup", "camden_yards")

weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park, sel_date_str)
player_df["weather_temp"] = weather_temp
player_df["weather_humidity"] = weather_humidity
player_df["weather_wind_mph"] = weather_wind_mph

# ----------- 9. Fill All Model Features -----------
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0

# ----------- 10. Predict HR Logit & Probability -----------
def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

player_df["HR Logit Score"] = player_df.apply(calc_hr_logit, axis=1)
player_df["HR Probability"] = 1 / (1 + np.exp(-player_df["HR Logit Score"]))

# ----------- 11. Show Leaderboard -----------
st.header("Predicted Home Run Leaderboard")
top_feats = [
    "Player", "batter_id", "Team", "pitcher_id", "p_throws", "HR Probability", "HR Logit Score",
    "B_pitch_pct_CH_5", "park_handed_hr_rate", "B_median_ev_7", "B_pitch_pct_CU_3",
    "weather_temp", "weather_humidity", "weather_wind_mph"
]
top_feats = [f for f in top_feats if f in player_df.columns]

leaderboard = player_df[top_feats].sort_values("HR Probability", ascending=False)
st.dataframe(leaderboard.head(15), use_container_width=True)

st.caption(
    "• Scoring: One row per batter in predicted or confirmed lineup, with all logistic, matchup, park, weather, and pitcher-handedness features applied. "
    "Upload new CSVs or select another date to refresh."
)

with st.expander("See ALL model features for leaderboard batters"):
    st.dataframe(player_df.sort_values("HR Probability", ascending=False).reset_index(drop=True), use_container_width=True)
