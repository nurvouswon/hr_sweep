import streamlit as st
import pandas as pd
import requests
import datetime

st.set_page_config(page_title="HR Predictor", layout="wide")

st.title("Home Run Predictor")

# --- 1. Upload CSVs ---
player_file = st.file_uploader("Upload Player-Level HR Features CSV", type=["csv"])
event_file = st.file_uploader("Upload Event-Level Data CSV", type=["csv (optional)"])

player_df = None
event_df = None

if player_file:
    try:
        player_df = pd.read_csv(player_file)
    except Exception as e:
        st.error(f"Error loading player-level CSV: {e}")

if event_file:
    try:
        event_df = pd.read_csv(event_file)
    except Exception as e:
        st.error(f"Error loading event-level CSV: {e}")

if player_df is None or player_df.empty:
    st.info("Please upload a valid Player-Level CSV to view leaderboard.")
    st.stop()

# --- 2. MLB PlayerID → Name Lookup Function (for batter_id) ---
@st.cache_data(show_spinner=False)
def get_player_name_from_id(mlb_id):
    try:
        # MLB Stats API: https://statsapi.mlb.com/api/v1/people/{personId}
        url = f"https://statsapi.mlb.com/api/v1/people/{int(mlb_id)}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            info = r.json()
            name = info["people"][0]["fullName"]
            return name
    except Exception:
        pass
    return str(mlb_id)

# --- 3. Weather Lookup ---
def fetch_weather(park, date_str):
    api_key = st.secrets["weather"]["api_key"]
    # Park location mapping (expand as needed)
    park_locations = {
        "camden_yards": "Baltimore,MD",
        "oracle_park": "San Francisco,CA",
        # Add more mappings!
    }
    city_state = park_locations.get(park, park)
    url = (
        f"http://api.weatherapi.com/v1/history.json?key={api_key}"
        f"&q={city_state}&dt={date_str}"
    )
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if "forecast" in data:
            d = data["forecast"]["forecastday"][0]["day"]
            temp = d.get("avgtemp_f", None)
            humidity = d.get("avghumidity", None)
            wind = d.get("maxwind_mph", None)
            return {"temp": temp, "humidity": humidity, "wind": wind}
    except Exception:
        pass
    return {"temp": None, "humidity": None, "wind": None}

# --- 4. Date Selector (today by default) ---
today = datetime.date.today()
sel_date = st.date_input("Prediction Date", value=today)
sel_date_str = sel_date.strftime("%Y-%m-%d")

# --- 5. Get Park for Weather ---
if "park" in player_df.columns and player_df["park"].notnull().any():
    park = player_df["park"].mode()[0]
else:
    park = st.text_input("Ballpark (for weather lookup)", "camden_yards")

weather_data = fetch_weather(park, sel_date_str)

# --- 6. Logistic Weights (hardcoded) ---
logit_weights = {
    "iso_value": 5.757820079, "hit_distance_sc": 0.6411852127, "pull_side": 0.5569402386, "launch_speed_angle": 0.5280235471,
    "B_pitch_pct_CH_5": 0.3858783912, "park_handed_hr_rate": 0.3438658641, "B_median_ev_7": 0.33462617, "B_pitch_pct_CU_3": 0.3280395666,
    "P_max_ev_5": 0.3113203434, "P_pitch_pct_SV_3": 0.2241205438, "B_pitch_pct_EP_5": 0.2163322514, "P_pitch_pct_ST_14": 0.2052831283,
    "P_rolling_hr_rate_7": 0.1877664166, "P_pitch_pct_FF_5": 0.1783978536, "P_median_ev_3": 0.1752142738, "groundball": 0.1719989086,
    "B_pitch_pct_KC_5": 0.1615036223, "B_pitch_pct_FS_3": 0.1595644445, "P_pitch_pct_FC_14": 0.1591148241, "B_pitch_pct_SI_14": 0.1570044892,
    "B_max_ev_5": 0.1540596514,
    # add as needed... this is a shortened set for demo
}

# --- 7. Fill Missing Features with 0 (only for features in weights) ---
for feat in logit_weights:
    if feat not in player_df.columns:
        player_df[feat] = 0

# --- 8. HR Logit Score & Probability ---
def compute_logit(row):
    s = 0
    for feat, weight in logit_weights.items():
        s += row[feat] * weight
    return s

player_df["hr_logit_score"] = player_df.apply(compute_logit, axis=1)
# Probability with sigmoid (for logistic regression style output)
import numpy as np
player_df["prob_hr"] = 1 / (1 + np.exp(-player_df["hr_logit_score"]))

# --- 9. Add Player Names via MLB API ---
@st.cache_data(show_spinner=False)
def id_to_name_dict(ids):
    names = {}
    for i in ids:
        names[i] = get_player_name_from_id(i)
    return names

all_ids = player_df["batter_id"].unique().tolist()
id_to_name = id_to_name_dict(all_ids)
player_df["Player"] = player_df["batter_id"].map(id_to_name)
player_df["MLB ID"] = player_df["batter_id"]
player_df["Team"] = player_df["park"] if "park" in player_df.columns else ""

# --- 10. Insert Weather Columns ---
player_df["Weather Temp (F)"] = weather_data["temp"]
player_df["Weather Humidity"] = weather_data["humidity"]
player_df["Weather Wind (mph)"] = weather_data["wind"]

# --- 11. Leaderboard Columns to Show ---
top_feats = [
    "iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5", "park_handed_hr_rate",
    "B_median_ev_7", "B_pitch_pct_CU_3", "P_max_ev_5", "P_pitch_pct_SV_3", "B_pitch_pct_EP_5", "P_pitch_pct_ST_14"
]
# Only keep top_feats actually in DataFrame
top_feats = [f for f in top_feats if f in player_df.columns]

cols_to_show = (
    ["Player", "MLB ID", "Team", "prob_hr", "hr_logit_score"] +
    top_feats + ["Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"]
)
cols_rename = {
    "prob_hr": "HR Probability",
    "hr_logit_score": "HR Logit Score"
}

# --- 12. Display Leaderboard ---
leaderboard = player_df.sort_values("prob_hr", ascending=False).reset_index(drop=True)
display_cols = [c for c in cols_to_show if c in leaderboard.columns]
leaderboard_disp = leaderboard[display_cols].rename(columns=cols_rename)

st.header("Predicted Home Run Leaderboard")
show_all_feats = st.checkbox("Show all feature columns", value=False)
if show_all_feats:
    st.dataframe(leaderboard.rename(columns=cols_rename), use_container_width=True)
else:
    st.dataframe(leaderboard_disp.head(15), use_container_width=True)

# --- 13. Event-Level Details (optional, NOT displayed by default) ---
if False and event_df is not None and not event_df.empty:
    # This block is skipped per your latest request (no dropdown/detail)
    st.subheader("Underlying Event-Level Details (disabled)")
