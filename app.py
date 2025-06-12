import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from typing import Dict

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="MLB HR Predictor", layout="wide")

# Hardcoded logistic weights (feature: weight)
LOGISTIC_WEIGHTS = {
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
    "B_pitch_pct_EP_5": 0.2163322514,
    "P_pitch_pct_ST_14": 0.2052831283,
    "P_rolling_hr_rate_7": 0.1877664166,
    "P_pitch_pct_FF_5": 0.1783978536,
    "P_median_ev_3": 0.1752142738,
    "groundball": 0.1719989086,
    "B_pitch_pct_KC_5": 0.1615036223,
    "B_pitch_pct_FS_3": 0.1595644445,
    "P_pitch_pct_FC_14": 0.1591148241,
    "B_pitch_pct_SI_14": 0.1570044892,
    "B_max_ev_5": 0.1540596514,
    "P_pitch_pct_CU_7": 0.1524371468,
    "P_pitch_pct_SL_3": 0.1429928993,
    "P_pitch_pct_FO_14": 0.1332430394,
    "B_pitch_pct_SV_5": 0.1257929016,
    "P_hit_distance_sc_7": 0.1236586016,
    "B_iso_value_14": 0.1199768939,
    "P_woba_value_5": 0.1175567692,
    "B_pitch_pct_CS_14": 0.1137568069,
    "pitch_pct_FO": 0.1124543401,
    "B_pitch_pct_FF_7": 0.105404093,
    "is_barrel": 0.1044204311,
    "B_pitch_pct_FA_7": 0.1041956255,
    "pitch_pct_FF": 0.1041947265,
    "B_pitch_pct_ST_3": 0.1016502344,
    "pitch_pct_ST": 0.09809980426,
    "pitch_pct_CH": 0.09588455603,
    "B_pitch_pct_SL_3": 0.09395294235,
    "P_rolling_hr_rate_5": 0.09176055559,
    "B_pitch_pct_SC_14": 0.08671517652,
    "platoon": 0.08601459992,
    "P_pitch_pct_FS_3": 0.08464192523,
    "B_iso_value_7": 0.08090866123,
    "B_pitch_pct_KC_7": 0.08079362526,
    "B_median_ev_14": 0.07898600411,
    "B_pitch_pct_KN_7": 0.07368063279,
    "B_pitch_pct_SL_14": 0.07334392117,
    # ... (add all remaining weights up to the number you want to use)
}

# User controls how many features to display in the leaderboard table
MAX_FEATURES_DISPLAY = 15

# ---------------------
# WEATHER LOOKUP
# ---------------------
def fetch_weather(api_key, city, date):
    # date format: YYYY-MM-DD
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            day = data["forecast"]["forecastday"][0]["day"]
            # Most important: temp, humidity, wind
            return {
                "temp": day.get("avgtemp_f"),
                "humidity": day.get("avghumidity"),
                "wind_mph": day.get("maxwind_mph"),
                "condition": day.get("condition", {}).get("text", ""),
            }
    except Exception as e:
        return None
    return None

# ---------------------
# PLAYER ID → NAME
# ---------------------
def build_id_name_dict(event_df):
    id_name = {}
    # event_df["batter_id"] and event_df["batter"] are usually both IDs, but try to find names
    # If player_name is available in event data, use that (sometimes it's for pitcher)
    if "player_name" in event_df.columns:
        for _, row in event_df.iterrows():
            try:
                pid = int(row.get("batter_id") or row.get("batter"))
                name = row.get("batter")
                # If batter column is numeric, try player_name or fallback to ID
                if isinstance(name, (int, float)) or str(name).isdigit():
                    name = row.get("player_name", pid)
                id_name[pid] = name
            except Exception:
                continue
    else:
        for _, row in event_df.iterrows():
            pid = int(row.get("batter_id") or row.get("batter"))
            name = row.get("batter")
            id_name[pid] = name
    return id_name

# ---------------------
# LOGISTIC SCORING
# ---------------------
def calc_logit_score(row: pd.Series, weights: Dict[str, float]):
    s = 0.0
    for feat, w in weights.items():
        val = row.get(feat, np.nan)
        if pd.isna(val):
            val = 0.0
        s += float(val) * w
    # Standard logit (can tweak if needed): probability = 1 / (1 + exp(-score))
    prob = 1 / (1 + np.exp(-s))
    return s, prob

# ---------------------
# APP UI
# ---------------------
st.title("⚾️ MLB Home Run Predictor - Leaderboard")

st.markdown(
    """
    **Instructions:**  
    1. Upload your *event-level* and *player-level* Analyzer CSVs.
    2. Select the target date (today by default).
    3. View the HR probability leaderboard, with detailed stats and weather.
    """
)

col1, col2 = st.columns(2)
event_file = col1.file_uploader("Upload Event-Level Analyzer CSV", type="csv")
player_file = col2.file_uploader("Upload Player-Level Analyzer CSV", type="csv")

date_selected = st.date_input("Select Game Date", value=datetime.date.today())

if not event_file or not player_file:
    st.warning("Please upload both Event-Level and Player-Level CSVs to continue.")
    st.stop()

event_df = pd.read_csv(event_file, low_memory=False)
player_df = pd.read_csv(player_file, low_memory=False)

# Build MLB ID → name dict from event-level
id_name_dict = build_id_name_dict(event_df)

# Filter event_df for only events on selected date (in case multiple days in data)
event_df = event_df[event_df["game_date"] == str(date_selected)]

# If no events, warn and stop
if len(event_df) == 0:
    st.error("No events found for the selected date. Please check your data.")
    st.stop()

# Weather handling
if {"temp", "humidity", "wind_mph"}.issubset(event_df.columns):
    weather_data = {
        "temp": float(event_df["temp"].iloc[0]),
        "humidity": float(event_df["humidity"].iloc[0]),
        "wind_mph": float(event_df["wind_mph"].iloc[0]),
        "condition": event_df["condition"].iloc[0] if "condition" in event_df.columns else "",
    }
else:
    # Use park/city field for weather lookup (try first row's park as location)
    city = event_df["park"].iloc[0] if "park" in event_df.columns else "New York"
    api_key = st.secrets["weatherapi_key"]
    weather_data = fetch_weather(api_key, city, str(date_selected))

# Merge top features from player-level with event-level for each player
# Prioritize: event-level > player-level > 0
feature_cols = list(LOGISTIC_WEIGHTS.keys())
# Sometimes column names may have been flattened: check for B_ or P_ prefixes

# Prepare leaderboard rows
rows = []
players = event_df["batter_id"].unique()
for pid in players:
    # Get name for leaderboard
    name = id_name_dict.get(pid, str(pid))
    # Team and other info (if available)
    player_rows = event_df[event_df["batter_id"] == pid]
    team = player_rows["home_team"].iloc[0] if "home_team" in player_rows.columns else ""
    # Compose player-level stats
    if "batter_id" in player_df.columns:
        player_row = player_df[player_df["batter_id"] == pid]
    else:
        player_row = player_df[player_df["batter"] == pid]
    player_row = player_row.iloc[0] if not player_row.empty else None

    # Compose feature vector
    feat_vals = {}
    for feat in feature_cols:
        if feat in player_rows.columns and not player_rows[feat].isna().all():
            # Take mean of this feature across all events (or customize as needed)
            feat_vals[feat] = player_rows[feat].mean()
        elif player_row is not None and feat in player_row.index and not pd.isna(player_row[feat]):
            feat_vals[feat] = player_row[feat]
        else:
            feat_vals[feat] = 0.0

    # Logistic score and prob
    logit, prob = calc_logit_score(pd.Series(feat_vals), LOGISTIC_WEIGHTS)

    # Save for leaderboard
    rows.append(
        {
            "Player": name,
            "MLB ID": pid,
            "Team": team,
            "HR Logit Score": logit,
            "HR Probability": prob,
            **{f: feat_vals[f] for f in feature_cols[:MAX_FEATURES_DISPLAY]},
            "Weather Temp (F)": weather_data.get("temp"),
            "Weather Humidity": weather_data.get("humidity"),
            "Weather Wind (mph)": weather_data.get("wind_mph"),
        }
    )

leaderboard = pd.DataFrame(rows)
leaderboard = leaderboard.sort_values("HR Probability", ascending=False).reset_index(drop=True)

# Display table with expand for full features and event log
st.subheader("Predicted Home Run Leaderboard")
st.dataframe(
    leaderboard[["Player", "MLB ID", "Team", "HR Probability", "HR Logit Score"] + feature_cols[:MAX_FEATURES_DISPLAY] +
                ["Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"]].style.format(
        {**{k: "{:.3f}" for k in feature_cols[:MAX_FEATURES_DISPLAY]}, "HR Probability": "{:.3f}", "HR Logit Score": "{:.2f}"}
    ),
    use_container_width=True,
)

with st.expander("See ALL Player Feature Details and Event Logs"):
    for i, row in leaderboard.iterrows():
        pid = row["MLB ID"]
        st.markdown(f"### {row['Player']} ({row['Team']}) — HR Prob: {row['HR Probability']:.3f}")
        st.write("**All Features:**")
        st.write(row)
        # Show event-level log (this date, this batter)
        batter_events = event_df[event_df["batter_id"] == pid]
        st.write("**Event-Level Data (Game Log):**")
        st.dataframe(batter_events, hide_index=True, use_container_width=True)
        st.markdown("---")

# Export
st.download_button("Download Leaderboard (CSV)", leaderboard.to_csv(index=False), file_name=f"hr_leaderboard_{date_selected}.csv")

if weather_data:
    st.info(f"**Weather for {date_selected} ({city}):** Temp: {weather_data.get('temp')}°F, Humidity: {weather_data.get('humidity')}%, Wind: {weather_data.get('wind_mph')} mph, Condition: {weather_data.get('condition','')}")
else:
    st.warning("No weather data found for this park/date.")
