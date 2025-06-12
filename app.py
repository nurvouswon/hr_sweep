import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date

# --- HARD CODED LOGISTIC WEIGHTS (top features, expand as needed) ---
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
    # ... (add more if you want, just match keys to features in CSVs)
}
FEATURES_DISPLAY = list(LOGISTIC_WEIGHTS.keys())[:15]  # Show top 15 by default
MAX_FEATURES_DISPLAY = 15

# --- APP TITLE ---
st.title("MLB Home Run Predictor Leaderboard")

# --- DATE SELECTION ---
today_str = date.today().strftime('%Y-%m-%d')
sel_date = st.date_input("Select date", date.today())
sel_date_str = sel_date.strftime('%Y-%m-%d')

# --- CSV UPLOADS ---
st.header("Upload Data")
event_csv = st.file_uploader("Upload Event-Level CSV (Analyzer Output)", type=["csv"], key="ev")
player_csv = st.file_uploader("Upload Player-Level CSV (Analyzer Output)", type=["csv"], key="pl")

if not (event_csv and player_csv):
    st.stop()

event_df = pd.read_csv(event_csv, low_memory=False)
player_df = pd.read_csv(player_csv, low_memory=False)

# --- WEATHER LOOKUP ---
def fetch_weather(park, dt):
    # Use your API key stored in Streamlit secrets
    api_key = st.secrets["weatherapi"]["key"]
    # For past or present date, use park's city or known location mapping
    park_city_map = {
        # Add mappings for all ballparks you care about!
        "camden_yards": "Baltimore",
        "wrigley_field": "Chicago",
        # ... etc ...
    }
    city = park_city_map.get(str(park).lower(), "New York")  # Fallback: NY
    # Format date: yyyy-MM-dd for WeatherAPI.com
    dt_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
    url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={dt_str}"
    try:
        resp = requests.get(url)
        js = resp.json()
        day = js["forecast"]["forecastday"][0]["day"]
        temp = day.get("avgtemp_f")
        humidity = day.get("avghumidity")
        wind_mph = day.get("maxwind_mph")
        return {"temp": temp, "humidity": humidity, "wind_mph": wind_mph}
    except Exception as e:
        st.warning(f"Weather lookup failed: {e}")
        return {"temp": None, "humidity": None, "wind_mph": None}

# --- LOGISTIC SCORE CALC ---
def calc_logit_score(row, weights):
    s = 0.0
    for feat, w in weights.items():
        v = row.get(feat, 0.0)
        try:
            s += float(v) * w
        except:
            s += 0.0
    prob = 1 / (1 + np.exp(-s))
    return s, prob

# --- BATTER EVENTS ONLY ---
BATTING_EVENTS = [
    "single", "double", "triple", "home_run", "field_out", "force_out", "strikeout", "walk",
    "hit_by_pitch", "sac_fly", "ground_out", "fly_out", "pop_out", "line_out",
    "reached_on_error", "fielder_choice"
]
event_batters = event_df[event_df["events"].str.lower().isin(BATTING_EVENTS)]
if event_batters.empty:
    st.error("No batting events found for this date. Please check your event-level CSV!")
    st.stop()

# --- MAPPING BATTER ID <-> NAME ---
id_name_dict = dict(event_batters[["batter_id", "batter"]].drop_duplicates().values)
# Park is from the first event (should all be same for a given home game)
park = event_batters["park"].iloc[0] if "park" in event_batters.columns else ""
weather_data = fetch_weather(park, sel_date_str)

# --- LEADERBOARD GENERATION ---
rows = []
for pid, name in id_name_dict.items():
    # Team (from first event for this batter)
    player_events = event_batters[event_batters["batter_id"] == pid]
    team = player_events["home_team"].iloc[0] if "home_team" in player_events.columns and not player_events["home_team"].isna().all() else ""
    # Player-level features (by MLB ID)
    player_row = player_df[player_df["batter_id"] == pid]
    player_row = player_row.iloc[0] if not player_row.empty else None
    # Compose feature values (prefer event-level, fallback to player-level, fallback 0)
    feat_vals = {}
    for feat in LOGISTIC_WEIGHTS.keys():
        if feat in player_events.columns and not player_events[feat].isna().all():
            feat_vals[feat] = player_events[feat].mean()
        elif player_row is not None and feat in player_row.index and not pd.isna(player_row[feat]):
            feat_vals[feat] = player_row[feat]
        else:
            feat_vals[feat] = 0.0
    logit, prob = calc_logit_score(feat_vals, LOGISTIC_WEIGHTS)
    row = {
        "Player": name,
        "MLB ID": pid,
        "Team": team,
        "HR Probability": prob,
        "HR Logit Score": logit,
        **{f: feat_vals[f] for f in FEATURES_DISPLAY},
        "Weather Temp (F)": weather_data.get("temp"),
        "Weather Humidity": weather_data.get("humidity"),
        "Weather Wind (mph)": weather_data.get("wind_mph"),
    }
    rows.append(row)

leaderboard = pd.DataFrame(rows)
leaderboard = leaderboard.sort_values("HR Probability", ascending=False).reset_index(drop=True)

# --- DISPLAY LEADERBOARD ---
st.header("Predicted Home Run Leaderboard")
show_all_feats = st.checkbox("Show all feature columns", value=False)
cols_to_show = ["Player", "MLB ID", "Team", "HR Probability", "HR Logit Score"] + (list(LOGISTIC_WEIGHTS.keys()) if show_all_feats else FEATURES_DISPLAY) + ["Weather Temp (F)", "Weather Humidity", "Weather Wind (mph)"]
st.dataframe(leaderboard[cols_to_show].head(15), use_container_width=True)

# --- EXPANDER FOR EVENT-LEVEL DETAILS ---
st.subheader("Click below to see underlying event-level details for each batter")
for idx, row in leaderboard.head(15).iterrows():
    batter_id = row["MLB ID"]
    with st.expander(f'{row["Player"]} ({batter_id}) - Details'):
        batter_events = event_batters[event_batters["batter_id"] == batter_id]
        st.write(batter_events.reset_index(drop=True))

# --- TROUBLESHOOTING HELP ---
if leaderboard.empty:
    st.warning("Leaderboard is empty. Double-check your event and player CSVs for data and format issues!")
