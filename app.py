import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# --- 1. CSV Uploaders ---
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"], key="player")
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"], key="event")

if not player_csv or not event_csv:
    st.warning("⬆️ Upload **both** player-level and event-level CSVs to begin!")
    st.stop()

player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)

st.sidebar.header("Step 2: Prediction Date")
today_str = datetime.now().strftime("%Y-%m-%d")
sel_date = st.sidebar.date_input("Select Date for Prediction", value=datetime.now())
sel_date_str = sel_date.strftime("%Y-%m-%d")

st.success(f"✅ CSVs uploaded! Player-level rows: {len(player_df)} | Event-level rows: {len(event_df)}")

# --- 2. Hardcoded Logistic Weights ---
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
    # ... (add more weights as needed) ...
}
INTERCEPT = 0

# --- 3. WeatherAPI Function ---
def fetch_weather(park, date_str):
    park_locations = {
        "camden_yards": "Baltimore,MD",
        "target_field": "Minneapolis,MN",
        "wrigley_field": "Chicago,IL",
        # Add more mappings as needed
    }
    location = park_locations.get(str(park).lower(), park)
    try:
        api_key = st.secrets["weather"]["api_key"]  # WeatherAPI key
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

# --- 4. MLB Lineups + Pitcher API ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_lineups_and_pitchers(date_str):
    """Return (batters_df, pitchers_dict) for date_str from MLB API."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1&hydrate=lineups"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        all_batters = []
        pitchers_dict = {}
        for date in data.get("dates", []):
            for game in date.get("games", []):
                for team_key in ["home", "away"]:
                    tinfo = game["teams"][team_key]
                    team_code = tinfo["team"].get("abbreviation", None)
                    # Pitcher
                    pinfo = tinfo.get("probablePitcher", {})
                    if "id" in pinfo and team_code:
                        pitchers_dict[team_code] = {
                            "pitcher_id": pinfo["id"],
                            "p_throws": pinfo.get("hand", {}).get("code", None),
                            "pitcher_name": pinfo.get("fullName", "")
                        }
                    # Batters
                    starters = tinfo.get("lineups", [])
                    for s in starters:
                        if "player" in s:
                            all_batters.append({
                                "Team": team_code,
                                "batter_id": s["player"]["id"],
                                "Player": s["player"]["fullName"],
                                "bat_order": s["order"],
                            })
        return pd.DataFrame(all_batters), pitchers_dict
    except Exception as e:
        st.warning("Could not fetch MLB.com lineups for this date. Showing all player-level CSV rows instead.")
        return pd.DataFrame(), {}

# --- 5. Map Player Names from Player-Level CSV if present ---
player_name_map = {}
if "batter_id" in player_df.columns and "batter" in player_df.columns:
    player_name_map = dict(zip(player_df["batter_id"], player_df["batter"]))

# --- 6. Get today's lineups and probable pitchers ---
lineups_df, pitchers_dict = get_lineups_and_pitchers(sel_date_str)

# --- 7. Filter player_df for today's batters or fallback ---
if not lineups_df.empty:
    # Merge lineup with player-level data by MLB ID, preserving player names
    player_df = player_df.merge(
        lineups_df[["batter_id", "Team", "Player"]],
        on="batter_id", how="inner"
    )
else:
    # If no lineup, just use player-level data and map names from CSV
    player_df["Player"] = player_df["batter_id"].map(player_name_map)
    player_df["Team"] = player_df.get("Team", "")

# --- 8. Pitcher matchup assignment ---
player_df["pitcher_id"] = np.nan
player_df["p_throws"] = ""
player_df["pitcher_name"] = ""
for team in player_df["Team"].unique():
    pinfo = pitchers_dict.get(team, None)
    if pinfo:
        player_df.loc[player_df["Team"] == team, "pitcher_id"] = pinfo["pitcher_id"]
        player_df.loc[player_df["Team"] == team, "p_throws"] = pinfo.get("p_throws", "")
        player_df.loc[player_df["Team"] == team, "pitcher_name"] = pinfo.get("pitcher_name", "")

# --- 9. Weather for main ballpark of the event csv ---
if "park" in event_df.columns and not event_df.empty:
    park = event_df["park"].mode()[0]
else:
    park = st.sidebar.text_input("Ballpark for Weather Lookup", "camden_yards")
weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park, sel_date_str)
player_df["weather_temp"] = weather_temp
player_df["weather_humidity"] = weather_humidity
player_df["weather_wind_mph"] = weather_wind_mph

# --- 10. Fill missing features with zeros for scoring ---
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0

# --- 11. Progress Bar Setup ---
progress_bar = st.progress(0, text="Calculating HR leaderboard...")

def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

# --- 12. Compute HR Scores (with progress bar) ---
scores = []
n_players = len(player_df)
for i, row in player_df.iterrows():
    logit = calc_hr_logit(row)
    scores.append(logit)
    if n_players > 1:
        progress_bar.progress(int(100 * (i + 1) / n_players) / 100.0, text=f"Processing ({i+1}/{n_players}) players...")

player_df["HR Logit Score"] = scores
player_df["HR Probability"] = 1 / (1 + np.exp(-player_df["HR Logit Score"]))

# --- 13. NaN-fix for ranking ---
player_df["HR Logit Score"] = pd.to_numeric(player_df["HR Logit Score"], errors="coerce")
player_df["HR Logit Score"] = player_df["HR Logit Score"].fillna(-9999)
player_df["Rank"] = player_df["HR Logit Score"].rank(method="first", ascending=False).astype(int)
player_df = player_df.sort_values("HR Logit Score", ascending=False)

progress_bar.empty()  # Remove progress bar

# --- 14. Show Leaderboard (top 15, by logit) ---
st.header("Predicted Home Run Leaderboard")
top_feats = [
    "Rank", "Player", "batter_id", "Team", "pitcher_id", "pitcher_name",
    "p_throws", "HR Probability", "HR Logit Score", "weather_temp", "weather_humidity", "weather_wind_mph"
]
top_feats = [f for f in top_feats if f in player_df.columns]

leaderboard = player_df[top_feats].head(15)
st.dataframe(leaderboard, use_container_width=True)

st.caption(
    "• Leaderboard: 1 row per batter from today's expected lineups (or all in CSV if not available), with all features: pitcher, park, weather, park HR rate, park handed HR rate, etc. "
    "Upload new CSVs or select another date to refresh."
)

# --- 15. Option to show full features for top batters ---
with st.expander("See all model features for leaderboard batters"):
    st.dataframe(player_df.sort_values("Rank").reset_index(drop=True), use_container_width=True)
