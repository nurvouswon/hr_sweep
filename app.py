import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB HR Predictor", layout="wide")

# --- 1. HARDCODED LOGISTIC WEIGHTS (edit as needed) ---
LOGIT_WEIGHTS = {
    "iso_value": 5.757820079,
    "hit_distance_sc": 0.6411852127,
    "pull_side": 0.5569402386,
    "launch_speed_angle": 0.5280235471,
    "B_pitch_pct_CH_5": 0.3858783912,
    "park_handed_hr_rate": 0.3438658641,
    "B_median_ev_7": 0.33462617,
    "B_pitch_pct_CU_3": 0.3280395666,
    "P_max_ev_5": 0.3113203434,
    "P_pitch_pct_SV_3": 0.2241205438,
    # ... include more weights as needed ...
}

# --- 2. MLB.com PLAYER NAME LOOKUP ---
@st.cache_data(show_spinner=False)
def get_mlb_id_name_dict():
    url = "https://statsapi.mlb.com/api/v1/sports/1/players?fields=people,id,fullName"
    try:
        r = requests.get(url)
        r.raise_for_status()
        players = r.json().get("people", [])
        return {str(p["id"]): p["fullName"] for p in players}
    except Exception:
        return {}

MLB_ID_NAME = get_mlb_id_name_dict()

def lookup_name(mlb_id):
    if pd.isna(mlb_id): return ""
    return MLB_ID_NAME.get(str(int(mlb_id)), str(mlb_id))

# --- 3. MLB.com LINEUPS FETCH (by date) ---
@st.cache_data(show_spinner=True)
def get_mlb_lineups(date_str):
    """Returns dict: {team: [batter_id, ...]}, {team: (starter_id, starter_name)}"""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&expand=schedule.lineups"
    try:
        r = requests.get(url)
        r.raise_for_status()
        games = r.json().get("dates", [{}])[0].get("games", [])
        lineups = {}
        starters = {}
        for g in games:
            home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
            away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
            if "lineups" not in g: continue
            for l in g["lineups"]:
                t = l.get("team", {}).get("abbreviation")
                batters = [int(p["person"]["id"]) for p in l.get("battingOrder", [])]
                sp = l.get("pitchers", [{}])[0].get("person", {})
                sp_id = sp.get("id")
                sp_name = sp.get("fullName", "")
                if batters: lineups[t] = batters
                if sp_id: starters[t] = (sp_id, sp_name)
        return lineups, starters
    except Exception:
        return {}, {}

# --- 4. WEATHER LOOKUP ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_weather(city, date_str):
    try:
        api_key = st.secrets["weather"]["api_key"]
    except Exception:
        st.warning("Missing weatherapi.com API key in Streamlit secrets.")
        return None
    # We use history API for past/future
    q = city.replace(" ", "%20")
    url = (
        f"http://api.weatherapi.com/v1/history.json?key={api_key}"
        f"&q={q}&dt={date_str}"
    )
    try:
        r = requests.get(url)
        r.raise_for_status()
        day = r.json()["forecast"]["forecastday"][0]["day"]
        return {
            "temp": day["avgtemp_f"],
            "humidity": day["avghumidity"],
            "wind_mph": day["maxwind_mph"],
        }
    except Exception:
        return None

# --- 5. BALLPARK TO CITY MAPPING ---
PARK_LOOKUP = {
    "camden_yards": "Baltimore, MD",
    "dodger_stadium": "Los Angeles, CA",
    # Extend as needed!
}

# --- 6. SIDEBAR UPLOADS + DATE ---
st.title("MLB Home Run Predictor Leaderboard")

with st.sidebar:
    st.header("1. Upload Required Data")
    event_csv = st.file_uploader("Event-Level CSV (Analyzer)", type="csv", key="ev")
    player_csv = st.file_uploader("Player-Level CSV (Analyzer)", type="csv", key="pl")

    st.header("2. Select Game Date")
    today = datetime.today().date()
    sel_date = st.date_input("Game date", value=today, min_value=datetime(2023, 3, 1).date(), max_value=today+timedelta(days=7))
    sel_date_str = sel_date.strftime("%Y-%m-%d")

    st.header("3. Show Columns")
    show_all_feats = st.checkbox("Show all feature columns", value=False)

if not event_csv or not player_csv:
    st.warning("Please upload BOTH event-level and player-level CSVs to continue.")
    st.stop()

event_df = pd.read_csv(event_csv)
player_df = pd.read_csv(player_csv)

# --- 7. LINEUPS & MATCHUPS FETCH ---
with st.spinner("Fetching MLB lineups..."):
    lineups, starters = get_mlb_lineups(sel_date_str)
if not lineups:
    st.error(f"Could not find MLB.com lineups for {sel_date_str}.")
    st.stop()

# --- 8. BUILD PREDICTION LEADERBOARD ---
results = []
# Index player-level by MLB ID for fast lookup
player_df["batter_id"] = player_df["batter_id"].astype(str)
player_df.set_index("batter_id", inplace=True)

for team, batters in lineups.items():
    if team not in starters:
        continue
    sp_id, sp_name = starters[team]
    # Weather: use park from event CSV (use most common park for this team)
    team_events = event_df[event_df["home_team"] == team]
    park = team_events["park"].mode()[0] if not team_events.empty else ""
    city = PARK_LOOKUP.get(park, park)
    weather = fetch_weather(city, sel_date_str) if city else {}
    for batter_id in batters:
        str_batter = str(batter_id)
        # Pull all features for this batter
        if str_batter not in player_df.index:
            continue
        feat = player_df.loc[str_batter].copy()
        # Build feature vector for scoring
        fv = {}
        for k, v in LOGIT_WEIGHTS.items():
            # Use player-level, fallback to event-level if needed
            val = feat[k] if k in feat else np.nan
            fv[k] = val if not pd.isna(val) else 0.0
        # Insert weather features
        if weather:
            fv["weather_temp"] = weather.get("temp", np.nan)
            fv["weather_humidity"] = weather.get("humidity", np.nan)
            fv["weather_wind"] = weather.get("wind_mph", np.nan)
        # Logistic HR score
        hr_logit = sum(LOGIT_WEIGHTS[k] * fv.get(k, 0) for k in LOGIT_WEIGHTS)
        hr_prob = 1 / (1 + np.exp(-hr_logit))
        # Store result row
        results.append({
            "Player": lookup_name(batter_id),
            "MLB ID": batter_id,
            "Team": team,
            "Opp SP": lookup_name(sp_id),
            "HR Probability": hr_prob,
            "HR Logit Score": hr_logit,
            **{f: fv[f] for f in list(LOGIT_WEIGHTS.keys())[:15] if f in fv},
            "Weather Temp (F)": fv.get("weather_temp", ""),
            "Weather Humidity": fv.get("weather_humidity", ""),
            "Weather Wind (mph)": fv.get("weather_wind", ""),
        })

# --- 9. DISPLAY LEADERBOARD ---
leaderboard = pd.DataFrame(results)
if leaderboard.empty:
    st.error("No qualified batters found for selected date/lineups.")
    st.stop()

st.header(f"Predicted Home Run Leaderboard ({sel_date_str})")
top_cols = [
    "Player", "MLB ID", "Team", "Opp SP", "HR Probability", "HR Logit Score",
    *list(LOGIT_WEIGHTS.keys())[:10],
    "Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"
]
if show_all_feats:
    st.dataframe(leaderboard, use_container_width=True)
else:
    st.dataframe(leaderboard[top_cols].sort_values("HR Probability", ascending=False).head(15), use_container_width=True)
