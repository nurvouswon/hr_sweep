import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.special import expit

st.set_page_config(page_title="HR Predictor", layout="wide")

# --- 1. MLB ID -> Name Map (from public Baseball Savant dump, update this url as needed) ---
@st.cache_data(show_spinner=False)
def get_player_name_map():
    url = "https://gist.githubusercontent.com/steve0hh/1aafcbeab0bb0e90ef9a/raw/players.csv"
    try:
        df = pd.read_csv(url)
        df["mlb_id"] = df["mlb_id"].astype(str)
        return dict(zip(df["mlb_id"], df["player_name"]))
    except Exception as e:
        st.warning("Could not fetch player names. Only MLB ID will display.")
        return {}

player_map = get_player_name_map()

# --- 2. Hardcoded LOGISTIC WEIGHTS (add or trim as needed) ---
logistic_weights = {
    'iso_value': 5.757820079, 'hit_distance_sc': 0.6411852127, 'pull_side': 0.5569402386,
    'launch_speed_angle': 0.5280235471, 'B_pitch_pct_CH_5': 0.3858783912, 'park_handed_hr_rate': 0.3438658641,
    'B_median_ev_7': 0.33462617, 'B_pitch_pct_CU_3': 0.3280395666, 'P_max_ev_5': 0.3113203434,
    'P_pitch_pct_SV_3': 0.2241205438, 'B_pitch_pct_EP_5': 0.2163322514, 'P_pitch_pct_ST_14': 0.2052831283,
    'P_rolling_hr_rate_7': 0.1877664166, 'P_pitch_pct_FF_5': 0.1783978536, 'P_median_ev_3': 0.1752142738,
    'groundball': 0.1719989086, 'B_pitch_pct_KC_5': 0.1615036223, 'B_pitch_pct_FS_3': 0.1595644445,
    'P_pitch_pct_FC_14': 0.1591148241, 'B_pitch_pct_SI_14': 0.1570044892, 'B_max_ev_5': 0.1540596514,
    # ... (snip: add more features from your list as needed)
    'is_barrel': 0.1044204311, 'platoon': 0.08601459992,
    # You can extend this dictionary with all weights from your list
}

# --- 3. WEATHER FUNCTION ---
def fetch_weather(park, date_str):
    """Query weatherapi.com for game weather. Requires [weather] api_key in secrets."""
    api_key = st.secrets["weather"]["api_key"]
    # Basic park lookup, update with real park locations as needed
    park_locations = {
        "camden_yards": "Baltimore,MD",
        # Add more mappings for other parks as needed
    }
    loc = park_locations.get(park, "Baltimore,MD")
    url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={loc}&dt={date_str}"
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        cond = j['forecast']['forecastday'][0]['day']
        return {
            "Weather Temp (F)": cond['avgtemp_f'],
            "Weather Humidity": cond['avghumidity'],
            "Weather Wind (mph)": cond['maxwind_mph']
        }
    except Exception as e:
        return {
            "Weather Temp (F)": np.nan,
            "Weather Humidity": np.nan,
            "Weather Wind (mph)": np.nan
        }

# --- 4. SIDEBAR: Date & CSV Uploads ---
st.sidebar.title("Upload Data")
today_str = datetime.now().strftime("%Y-%m-%d")
sel_date = st.sidebar.date_input("Select date", value=datetime.now())
sel_date_str = sel_date.strftime("%Y-%m-%d")

player_level_file = st.sidebar.file_uploader("Upload PLAYER-LEVEL CSV (from Analyzer)", type=["csv"], key="playercsv")
event_level_file = st.sidebar.file_uploader("Upload EVENT-LEVEL CSV (from Analyzer)", type=["csv"], key="eventcsv")

# --- 5. DATA LOADING ---
player_df = None
if player_level_file:
    player_df = pd.read_csv(player_level_file)
    # Normalize MLB ID as string
    if "batter_id" in player_df.columns:
        player_df["batter_id"] = player_df["batter_id"].astype(str)
    # Add player name using map
    player_df["Player"] = player_df["batter_id"].map(player_map).fillna(player_df["batter_id"])
    # Add "MLB ID" and "Team" columns
    player_df["MLB ID"] = player_df["batter_id"]
    # Try to get team if present (not always in player-level CSV)
    if "team" in player_df.columns:
        player_df["Team"] = player_df["team"]
    elif "Team" not in player_df.columns:
        player_df["Team"] = ""

# --- 6. WEATHER: Try to infer park from player_df or ask user ---
if player_df is not None and "park" in player_df.columns and player_df["park"].notnull().any():
    park = player_df["park"].mode()[0]
else:
    park = st.sidebar.text_input("Ballpark (for weather lookup)", "camden_yards")
weather_data = fetch_weather(park, sel_date_str)

# --- 7. FILL MISSING FEATURES WITH ZEROES ---
if player_df is not None:
    for feat in logistic_weights:
        if feat not in player_df.columns:
            player_df[feat] = 0

# --- 8. CALCULATE LOGIT SCORE AND HR PROBABILITY ---
def calc_logit(row):
    score = 0
    for feat, weight in logistic_weights.items():
        val = row.get(feat, 0)
        try:
            val = float(val)
        except:
            val = 0
        score += val * weight
    return score

if player_df is not None:
    player_df["HR Logit Score"] = player_df.apply(calc_logit, axis=1)
    player_df["HR Probability"] = expit(player_df["HR Logit Score"])
    # Add weather columns (same value for all, or update per row as needed)
    for k, v in weather_data.items():
        player_df[k] = v

# --- 9. SELECT TOP N FEATURES TO DISPLAY ---
top_feats = [
    'iso_value', 'hit_distance_sc', 'pull_side', 'launch_speed_angle',
    'B_pitch_pct_CH_5', 'park_handed_hr_rate', 'B_median_ev_7', 'B_pitch_pct_CU_3', 'P_max_ev_5'
    # Add/trim as preferred
]
top_feats = [f for f in top_feats if f in player_df.columns]

# --- 10. DISPLAY LEADERBOARD ---
if player_df is not None and not player_df.empty:
    st.header("Predicted Home Run Leaderboard")
    show_all_feats = st.checkbox("Show all feature columns", value=False)
    base_cols = ["Player", "MLB ID", "Team", "HR Probability", "HR Logit Score"]
    weather_cols = ["Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"]
    display_cols = base_cols + (list(player_df.columns) if show_all_feats else top_feats) + weather_cols
    display_cols = [c for c in display_cols if c in player_df.columns]

    leaderboard = player_df.sort_values("HR Probability", ascending=False)
    st.dataframe(leaderboard[display_cols].head(15), use_container_width=True)

else:
    st.warning("Please upload a valid player-level CSV from Analyzer.")

st.caption("Logistic weights and features are hardcoded. Weather pulled from WeatherAPI by ballpark & date.")
