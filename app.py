import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("MLB Home Run Predictor")

# ---- 1. HARDCODED LOGISTIC WEIGHTS ----
LOGIT_WEIGHTS = {
    "iso_value": 5.757820079, "hit_distance_sc": 0.6411852127, "pull_side": 0.5569402386, "launch_speed_angle": 0.5280235471,
    "B_pitch_pct_CH_5": 0.3858783912, "park_handed_hr_rate": 0.3438658641, "B_median_ev_7": 0.33462617,
    "B_pitch_pct_CU_3": 0.3280395666, "P_max_ev_5": 0.3113203434, "P_pitch_pct_SV_3": 0.2241205438,
    "B_pitch_pct_EP_5": 0.2163322514, "P_pitch_pct_ST_14": 0.2052831283, "P_rolling_hr_rate_7": 0.1877664166,
    "P_pitch_pct_FF_5": 0.1783978536, "P_median_ev_3": 0.1752142738, "groundball": 0.1719989086,
    "B_pitch_pct_KC_5": 0.1615036223, "B_pitch_pct_FS_3": 0.1595644445, "P_pitch_pct_FC_14": 0.1591148241,
    "B_pitch_pct_SI_14": 0.1570044892, "B_max_ev_5": 0.1540596514, "P_pitch_pct_CU_7": 0.1524371468,
    "P_pitch_pct_SL_3": 0.1429928993, "P_pitch_pct_FO_14": 0.1332430394, "B_pitch_pct_SV_5": 0.1257929016,
    "P_hit_distance_sc_7": 0.1236586016, "B_iso_value_14": 0.1199768939, "P_woba_value_5": 0.1175567692,
    "B_pitch_pct_CS_14": 0.1137568069, "pitch_pct_FO": 0.1124543401, "B_pitch_pct_FF_7": 0.105404093,
    "is_barrel": 0.1044204311, "B_pitch_pct_FA_7": 0.1041956255, "pitch_pct_FF": 0.1041947265,
    "B_pitch_pct_ST_3": 0.1016502344, "pitch_pct_ST": 0.09809980426, "pitch_pct_CH": 0.09588455603,
    "B_pitch_pct_SL_3": 0.09395294235, "P_rolling_hr_rate_5": 0.09176055559, "B_pitch_pct_SC_14": 0.08671517652,
    "platoon": 0.08601459992,
    # ... add more features as needed
}

# ---- 2. DATE PICKER ----
today = datetime.date.today()
sel_date = st.sidebar.date_input("Select Date", today)
sel_date_str = sel_date.strftime("%Y-%m-%d")

# ---- 3. CSV UPLOADS (REQUIRED) ----
st.sidebar.markdown("### Upload Required Data")
event_csv = st.sidebar.file_uploader("Event-level CSV (required)", type="csv", key="eventcsv")
player_csv = st.sidebar.file_uploader("Player-level CSV (required)", type="csv", key="playercsv")
if not event_csv or not player_csv:
    st.warning("Upload both event-level and player-level CSVs to proceed.")
    st.stop()
event_df = pd.read_csv(event_csv)
player_df = pd.read_csv(player_csv)

# ---- 4. MLB.com LINEUP & PLAYER LOOKUP FUNCTIONS ----
@st.cache_data(show_spinner=False)
def get_schedule_and_probables(date_str):
    """Returns games + mapping of team code → probable pitcher ID and handedness."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    r = requests.get(url)
    if not r.ok: return None, {}
    games = r.json().get("dates", [{}])[0].get("games", [])
    probables = {}
    for g in games:
        for side in ["home", "away"]:
            team = g["teams"][side]["team"]["abbreviation"]
            pitcher = g["teams"][side].get("probablePitcher")
            if pitcher:
                pid = pitcher["id"]
                handed = pitcher.get("pitchHand", {}).get("code", "R")
                probables[(team, side)] = (pid, handed)
    return games, probables

@st.cache_data(show_spinner=False)
def lookup_player_names(ids):
    # MLB.com API (batters and pitchers, full names)
    names = {}
    for pid in set(ids):
        try:
            r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{int(pid)}")
            if r.ok:
                d = r.json()["people"][0]
                names[int(pid)] = d["fullName"]
        except Exception:
            continue
    return names

# ---- 5. WEATHER LOOKUP (BY PARK, DATE) ----
def fetch_weather(park, date_str):
    api_key = st.secrets["weather"]["api_key"]
    park_locations = {
        "camden_yards": "Baltimore,MD", "target_field": "Minneapolis,MN", # Add more as needed
    }
    loc = park_locations.get(park.lower(), "New York,NY")
    url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={loc}&dt={date_str}"
    r = requests.get(url)
    if r.ok and "forecast" in r.json():
        data = r.json()["forecast"]["forecastday"][0]["day"]
        return {
            "temp": data["avgtemp_f"],
            "humidity": data["avghumidity"],
            "wind": data["maxwind_mph"],
        }
    return {"temp": np.nan, "humidity": np.nan, "wind": np.nan}

# ---- 6. MAIN LEADERBOARD BUILD ----
def get_batter_leaderboard(event_df, player_df, probables, sel_date_str):
    # Filter event_df for the right date (if game_date column exists)
    if "game_date" in event_df.columns:
        events_today = event_df[event_df["game_date"] == sel_date_str]
    else:
        events_today = event_df
    if "team" not in events_today.columns and "home_team" in events_today.columns:
        events_today["team"] = events_today["home_team"]  # fallback

    # Use first team/park in file for weather by default
    park = events_today["park"].iloc[0] if "park" in events_today.columns else "camden_yards"
    weather = fetch_weather(park, sel_date_str)

    # Merge player-level features into event-level by batter_id
    agg_dict = {col: "mean" for col in events_today.columns if col not in ["batter", "batter_id", "pitcher_id", "pitcher", "team"]}
    batter_rows = events_today.groupby("batter_id").agg(agg_dict).reset_index()
    batter_rows = pd.merge(batter_rows, player_df, how="left", left_on="batter_id", right_on="batter_id", suffixes=("_event", "_player"))
    # Merge team for each batter from events_today (mode)
    team_map = events_today.groupby("batter_id")["team"].agg(lambda x: x.mode()[0] if not x.empty else None)
    batter_rows["team"] = batter_rows["batter_id"].map(team_map)
    # Assign pitcher (probable) for matchup: home team batters get away pitcher and vice versa
    def assign_pitcher_and_handedness(row):
        team = row.get("team")
        pitcher_id = None
        pitcher_hand = "R"
        pitcher_side = None
        if team:
            # Guess home/away by team code match
            home_prob = probables.get((team, "home"))
            away_prob = probables.get((team, "away"))
            # If this batter is home team, pitcher is away, and vice versa
            if home_prob:
                pitcher_id, pitcher_hand = probables.get((team, "away"), (None, "R"))
                pitcher_side = "away"
            elif away_prob:
                pitcher_id, pitcher_hand = probables.get((team, "home"), (None, "R"))
                pitcher_side = "home"
        return pd.Series([pitcher_id, pitcher_hand, pitcher_side])
    batter_rows[["pitcher_id", "p_throws", "pitcher_side"]] = batter_rows.apply(assign_pitcher_and_handedness, axis=1)
    # Merge pitcher name (for show)
    all_pids = batter_rows["batter_id"].tolist() + batter_rows["pitcher_id"].dropna().astype(int).tolist()
    name_map = lookup_player_names(set(all_pids))
    batter_rows["Player"] = batter_rows["batter_id"].map(lambda bid: name_map.get(int(bid), str(bid)))
    batter_rows["Pitcher"] = batter_rows["pitcher_id"].map(lambda pid: name_map.get(int(pid), str(pid)) if not pd.isnull(pid) else "")
    # Add weather
    for k, v in weather.items():
        batter_rows[f"weather_{k}"] = v
    # Handedness matchup feature
    if "stand" in events_today.columns:
        batter_rows["handed_matchup"] = batter_rows.apply(lambda row:
            f"{row['stand']}{row['p_throws']}", axis=1
        )
    # Fill missing features for logistic
    for feat in LOGIT_WEIGHTS:
        if feat not in batter_rows.columns:
            batter_rows[feat] = 0
    logits = np.zeros(len(batter_rows))
    for feat, wt in LOGIT_WEIGHTS.items():
        logits += batter_rows[feat].astype(float).fillna(0) * wt
    batter_rows["HR Logit Score"] = logits
    batter_rows["HR Probability"] = 1 / (1 + np.exp(-batter_rows["HR Logit Score"]))
    # Useful columns
    keep_cols = [
        "Player", "batter_id", "team", "Pitcher", "pitcher_id", "p_throws", "handed_matchup",
        "HR Probability", "HR Logit Score",
        "weather_temp", "weather_humidity", "weather_wind",
        "park_handed_hr_rate", "park_hr_rate", "iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5", "B_median_ev_7", "B_pitch_pct_CU_3", "P_max_ev_5"
    ]
    keep_cols = [c for c in keep_cols if c in batter_rows.columns]
    leaderboard = batter_rows[keep_cols].sort_values("HR Probability", ascending=False).reset_index(drop=True)
    leaderboard.rename(columns={"batter_id": "MLB ID"}, inplace=True)
    return leaderboard

# ---- RUN ALL LOGIC ----
games, probables = get_schedule_and_probables(sel_date_str)
if not probables:
    st.error("Could not fetch MLB probables for this date from MLB.com.")
    st.stop()

leaderboard = get_batter_leaderboard(event_df, player_df, probables, sel_date_str)

st.header("Predicted Home Run Leaderboard")
st.dataframe(leaderboard.head(15), use_container_width=True)
