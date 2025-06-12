import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# =================== 1. CSV UPLOADERS ===================
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"])
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"])
if not player_csv or not event_csv:
    st.warning("⬆️ Upload **both** player-level and event-level CSVs to begin!")
    st.stop()
player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)

# =================== 2. DATE SELECTION ===================
st.sidebar.header("Step 2: Prediction Date")
today = datetime.now()
sel_date = st.sidebar.date_input("Select Date for Prediction", value=today)
sel_date_str = sel_date.strftime("%Y-%m-%d")

# =================== 3. LIVE LINEUPS/PITCHERS ===================
@st.cache_data(ttl=1800)
def get_lineups_and_pitchers(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=lineups,probablePitcher"
    resp = requests.get(url, timeout=15)
    games = resp.json().get("dates", [{}])[0].get("games", [])
    batters = []
    pitchers = {}
    for game in games:
        for which in ["home", "away"]:
            tinfo = game["teams"][which]
            team = tinfo.get("team", {})
            team_code = team.get("abbreviation", None)
            if not team_code:
                continue  # skip if not present
            if tinfo.get("probablePitcher", {}):
                pitchers[team_code] = {
                    "pitcher_id": tinfo["probablePitcher"].get("id"),
                    "pitcher_name": tinfo["probablePitcher"].get("fullName"),
                    "p_throws": tinfo["probablePitcher"].get("hand", {}).get("code", None)
                }
            lineup = tinfo.get("lineups", [])
            for entry in lineup:
                if "player" in entry:
                    batters.append({
                        "batter_id": entry["player"]["id"],
                        "Player": entry["player"]["fullName"],
                        "Team": team_code,
                        "bat_order": entry.get("order", None),
                        "game_pk": game.get("gamePk", None),
                        "park": game.get("venue", {}).get("name", ""),
                        "home_team": game["teams"]["home"]["team"].get("abbreviation", "")
                    })
    return pd.DataFrame(batters), pitchers

lineups_df, pitchers_dict = get_lineups_and_pitchers(sel_date_str)
if lineups_df.empty:
    st.warning("No MLB.com lineups available for this date. Using all uploaded players only.")

# =================== 4. WEATHER API LOOKUP ===================
def fetch_weather(park, date_str):
    park_locations = {
        "camden_yards": "Baltimore,MD",
        "target_field": "Minneapolis,MN",
        "wrigley_field": "Chicago,IL",
        # Extend with more parks as needed
    }
    location = park_locations.get(str(park).lower(), park)
    api_key = st.secrets["weather"]["api_key"]
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}"
    try:
        resp = requests.get(url, timeout=7)
        weather = resp.json().get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = weather.get("avgtemp_f", 72)
        humidity = weather.get("avghumidity", 55)
        wind_mph = weather.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception:
        return 72, 55, 7

# =================== 5. LOGISTIC WEIGHTS ===================
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
    # ...extend as needed...
}
INTERCEPT = 0

# =================== 6. FILTER TO TODAY'S BATTERS ===================
if not lineups_df.empty:
    # Only use players from today's lineups (join on MLB ID)
    player_df = player_df.merge(
        lineups_df[["batter_id", "Player", "Team", "park"]],
        on="batter_id", how="inner"
    )
else:
    # Fallback: all player_df, assign names as MLB ID string
    player_df["Player"] = player_df["batter_id"].astype(str)
    player_df["Team"] = ""

# =================== 7. PITCHER MATCHUP MERGE ===================
player_df["pitcher_id"] = ""
player_df["pitcher_name"] = ""
player_df["p_throws"] = ""
if not lineups_df.empty:
    for team_code in player_df["Team"].unique():
        pitcher = pitchers_dict.get(team_code, {})
        pid = pitcher.get("pitcher_id", "")
        pname = pitcher.get("pitcher_name", "")
        pthrows = pitcher.get("p_throws", "")
        player_df.loc[player_df["Team"] == team_code, "pitcher_id"] = pid
        player_df.loc[player_df["Team"] == team_code, "pitcher_name"] = pname
        player_df.loc[player_df["Team"] == team_code, "p_throws"] = pthrows

# =================== 8. WEATHER DATA ===================
# Use the most common park in filtered player_df
if "park" in player_df.columns and not player_df["park"].isna().all():
    park = player_df["park"].dropna().mode()[0]
else:
    park = st.sidebar.text_input("Ballpark for Weather Lookup", "camden_yards")
weather_temp, weather_humidity, weather_wind_mph = fetch_weather(park, sel_date_str)
player_df["weather_temp"] = weather_temp
player_df["weather_humidity"] = weather_humidity
player_df["weather_wind_mph"] = weather_wind_mph

# =================== 9. FILL MISSING FEATURES ===================
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0

# =================== 10. PROGRESS BAR ===================
st.header("Predicted Home Run Leaderboard")
progress_text = "Scoring batters…"
my_bar = st.progress(0, text=progress_text)
leaderboard_rows = []

def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

for idx, row in player_df.iterrows():
    logit = calc_hr_logit(row)
    prob = 1 / (1 + np.exp(-logit))
    player_df.at[idx, "HR Logit Score"] = logit
    player_df.at[idx, "HR Probability"] = prob
    my_bar.progress(min(int(100*(idx+1)/len(player_df)), 100), text=f"{progress_text} {int(100*(idx+1)/len(player_df))}%")

player_df["Rank"] = player_df["HR Logit Score"].rank(method="first", ascending=False).astype(int)
player_df = player_df.sort_values("HR Logit Score", ascending=False)

# =================== 11. DISPLAY LEADERBOARD ===================
show_all_feats = st.checkbox("Show all feature columns", value=False)
cols_to_show = ["Rank", "Player", "batter_id", "Team", "pitcher_id", "pitcher_name", "p_throws",
    "HR Probability", "HR Logit Score",
    "weather_temp", "weather_humidity", "weather_wind_mph"]
top_feats = [f for f in cols_to_show if f in player_df.columns]
if show_all_feats:
    st.dataframe(player_df.head(15), use_container_width=True)
else:
    st.dataframe(player_df[top_feats].head(15), use_container_width=True)

st.caption(
    "• Leaderboard = dynamic, daily filter, pitcher matchup, weather, park. Upload CSVs and pick a date for real-time HR prediction, **one row per batter**."
)
