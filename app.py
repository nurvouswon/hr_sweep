import streamlit as st
import pandas as pd
import requests
from pybaseball import statcast_batter, playerid_lookup
from datetime import datetime, timedelta

st.title("⚾️ Today's MLB HR Probability Leaderboard")

# -- 1. Get today's starting lineups (free public source) --
@st.cache_data(ttl=900)
def get_todays_lineup_names():
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    resp = requests.get(url)
    games = resp.json()["events"]
    batters = set()
    for game in games:
        for competitor in game["competitions"][0]["competitors"]:
            if "lineup" in competitor:
                for player in competitor["lineup"]:
                    if player.get("battingOrder") and player.get("athlete"):
                        name = player["athlete"]["displayName"]
                        if len(name.split(" ")) == 2:  # skip Jr./Sr./III etc for simplicity
                            batters.add(name)
    return list(batters)

todays_batters = get_todays_lineup_names()
if not todays_batters:
    st.warning("Couldn't load today's MLB lineups from ESPN. Try reloading or check your connection.")
else:
    st.write(f"Found {len(todays_batters)} batters in today's MLB starting lineups.")

# -- 2. Rolling windows in days: season, 14, 7, 5, 3 --
windows = [162, 14, 7, 5, 3]
results = []

def get_player_id(name):
    try:
        first, last = name.split(" ", 1)
        player_info = playerid_lookup(last, first)
        if not player_info.empty:
            return int(player_info.iloc[0]['key_mlbam'])
    except Exception:
        return None
    return None

def get_statcast(player_id, days_back):
    try:
        if days_back == 162:
            start_date = datetime(datetime.now().year, 3, 28)
        else:
            start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=1)
        df = statcast_batter(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), player_id)
        return df
    except Exception:
        return pd.DataFrame()

# -- 3. Scan all batters --
progress = st.progress(0)
for idx, batter in enumerate(todays_batters):
    player_id = get_player_id(batter)
    player_stats = {'Batter': batter}
    for window in windows:
        data = get_statcast(player_id, window)
        if not data.empty and 'type' in data.columns:
            data = data[data['type'] == "X"]
            data = data[data['launch_speed'].notnull() & data['launch_angle'].notnull()]
            ev = data['launch_speed'].mean()
            barrels = data[(data['launch_speed'] > 95) & (data['launch_angle'].between(20, 35))].shape[0]
            total = len(data)
            barrel_rate = barrels / total if total > 0 else 0
            player_stats[f"EV_{window}"] = round(ev, 1) if ev is not None else 0
            player_stats[f"BarrelRate_{window}"] = round(barrel_rate, 3)
            player_stats[f"PA_{window}"] = total
        else:
            player_stats[f"EV_{window}"] = None
            player_stats[f"BarrelRate_{window}"] = None
            player_stats[f"PA_{window}"] = 0
    results.append(player_stats)
    progress.progress((idx + 1) / len(todays_batters))

df = pd.DataFrame(results)

# -- 4. Calculate HR Probability Score (last 7 days, or fallback to next most recent) --
def get_score(row, window):
    ev = row[f"EV_{window}"]
    br = row[f"BarrelRate_{window}"]
    if ev is None or br is None:
        return 0
    ev_norm = (ev - 80) / (105 - 80)
    ev_norm = max(0, min(ev_norm, 1))
    br_norm = min(br / 0.15, 1)
    return round((ev_norm * 0.6 + br_norm * 0.4) * 100, 1)

for window in windows:
    df[f"HR_Score_{window}"] = df.apply(lambda row: get_score(row, window), axis=1)

# -- 5. Display the Leaderboard: HR Probability (last 7 days) --
sort_window = "HR_Score_7"
if df[sort_window].sum() == 0:
    sort_window = "HR_Score_14"
st.subheader(f"Top HR Probability Batters (Sorted by {sort_window.replace('_', ' ')})")
show_cols = ["Batter"] + [f"HR_Score_{w}" for w in windows] + [f"EV_{w}" for w in windows] + [f"BarrelRate_{w}" for w in windows] + [f"PA_{w}" for w in windows]
st.dataframe(df.sort_values(sort_window, ascending=False).reset_index(drop=True)[show_cols])

st.caption("Columns ending in _162 = Season, _14 = last 14d, _7 = last 7d, _5 = last 5d, _3 = last 3d. HR_Score columns use normalized EV (60%) and barrel rate (40%).")
