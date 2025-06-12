import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# --- 1. Uploaders at Top ---
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

# --- 2. MLB API: Get Confirmed Lineups and Pitchers ---
@st.cache_data(ttl=1800)
def get_lineups_and_pitchers(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=lineups,probablePitcher"
    resp = requests.get(url, timeout=15)
    games = resp.json().get("dates", [{}])[0].get("games", [])
    batters = []
    pitchers = {}
    for game in games:
        # Home/Away team, probable pitcher
        for which in ["home", "away"]:
            tinfo = game["teams"][which]
            lineup = tinfo.get("lineups", [])
            team_code = tinfo["team"]["abbreviation"]
            if tinfo.get("probablePitcher", {}):
                pitchers[team_code] = {
                    "pitcher_id": tinfo["probablePitcher"].get("id"),
                    "pitcher_name": tinfo["probablePitcher"].get("fullName"),
                    "p_throws": tinfo["probablePitcher"].get("hand", {}).get("code", None)
                }
            # Add batters (if any)
            for entry in lineup:
                if "player" in entry:
                    batters.append({
                        "batter_id": entry["player"]["id"],
                        "Player": entry["player"]["fullName"],
                        "Team": team_code,
                        "bat_order": entry.get("order", None),
                        "game_pk": game["gamePk"],
                        "park": game.get("venue", {}).get("name", ""),
                        "home_team": game["teams"]["home"]["team"]["abbreviation"]
                    })
    return pd.DataFrame(batters), pitchers

lineups_df, pitchers_dict = get_lineups_and_pitchers(sel_date_str)

if lineups_df.empty:
    st.error("Could not fetch confirmed lineups for this date. Only players in your CSV will be used.")
    filtered_df = player_df.copy()
    filtered_df["Player"] = filtered_df["batter_id"].astype(str)
    filtered_df["Team"] = ""
    filtered_df["pitcher_id"] = ""
    filtered_df["pitcher_name"] = ""
    filtered_df["p_throws"] = ""
else:
    # Merge: keep only today's starters, merge in player data
    filtered_df = lineups_df.merge(player_df, on="batter_id", how="left")
    # Map pitchers for matchup logic
    filtered_df["pitcher_id"] = filtered_df["Team"].map(lambda x: pitchers_dict.get(x, {}).get("pitcher_id"))
    filtered_df["pitcher_name"] = filtered_df["Team"].map(lambda x: pitchers_dict.get(x, {}).get("pitcher_name"))
    filtered_df["p_throws"] = filtered_df["Team"].map(lambda x: pitchers_dict.get(x, {}).get("p_throws"))

# --- 3. Weather Lookup (weatherapi.com) ---
def fetch_weather(city, date_str):
    api_key = st.secrets["weather"]["api_key"]
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"
    try:
        resp = requests.get(url, timeout=10)
        day = resp.json().get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = day.get("avgtemp_f", 72)
        humidity = day.get("avghumidity", 55)
        wind_mph = day.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception as e:
        return 72, 55, 7

if not filtered_df.empty and "park" in filtered_df.columns:
    park_city = filtered_df["park"].mode()[0] if filtered_df["park"].notnull().any() else "Baltimore,MD"
else:
    park_city = st.sidebar.text_input("Ballpark/City for Weather", "Baltimore,MD")
weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park_city, sel_date_str)
filtered_df["weather_temp"] = weather_temp
filtered_df["weather_humidity"] = weather_humidity
filtered_df["weather_wind_mph"] = weather_wind_mph

# --- 4. Logistic Weights (add your full dict here) ---
LOGISTIC_WEIGHTS = {
    'iso_value': 5.757820079,
    'hit_distance_sc': 0.6411852127,
    'pull_side': 0.5569402386,
    'launch_speed_angle': 0.5280235471,
    'B_pitch_pct_CH_5': 0.3858783912,
    'park_handed_hr_rate': 0.3438658641,
    'B_median_ev_7': 0.33462617,
    'B_pitch_pct_CU_3': 0.3280395666,
    # ... continue with all weights you want
}
INTERCEPT = 0

for col in LOGISTIC_WEIGHTS:
    if col not in filtered_df.columns:
        filtered_df[col] = 0

# --- 5. Progress Bar ---
st.subheader("Scoring Batters...")
progress = st.progress(0, text="Calculating HR Logistic Scores...")
scores = []
for i, row in filtered_df.iterrows():
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    scores.append(score)
    progress.progress(int((i + 1) / len(filtered_df) * 100), text=f"{i+1}/{len(filtered_df)} ({int((i+1)/len(filtered_df)*100)}%) batters scored")
filtered_df["HR_Logit_Score"] = scores
filtered_df["HR_Probability"] = 1 / (1 + np.exp(-filtered_df["HR_Logit_Score"]))

# --- 6. Show Leaderboard, Ranked by Logit Score ---
st.header("Predicted Home Run Leaderboard (Today’s Lineups)")
leaderboard_cols = [
    "Player", "batter_id", "Team", "pitcher_name", "pitcher_id", "p_throws",
    "HR_Probability", "HR_Logit_Score",
    "weather_temp", "weather_humidity", "weather_wind_mph"
]
# Show all available columns in leaderboard
leaderboard_cols += [c for c in LOGISTIC_WEIGHTS.keys() if c in filtered_df.columns and c not in leaderboard_cols]
leaderboard_cols = [c for c in leaderboard_cols if c in filtered_df.columns]
leaderboard = filtered_df[leaderboard_cols].sort_values("HR_Logit_Score", ascending=False).reset_index(drop=True)

st.dataframe(leaderboard.head(15), use_container_width=True)

st.caption(
    "Leaderboard: Confirmed starters only. Scores use logistic weights, pitcher matchups, weather, park, and handedness. "
    "Change the date and re-upload for tomorrow’s predictions."
)
