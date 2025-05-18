import streamlit as st
import pandas as pd
import requests
from pybaseball import statcast_batter, playerid_lookup
from datetime import datetime, timedelta

st.title("⚾️ Today's MLB HR Probability Leaderboard")

# -- 1. Get today's lineups from MLB.com (NO name filter) --
def get_todays_mlb_lineups():
    today = datetime.now().strftime("%Y-%m-%d")
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    resp = requests.get(schedule_url)
    if resp.status_code != 200:
        return []
    games = resp.json().get("dates", [{}])[0].get("games", [])
    batters = set()
    for game in games:
        game_pk = game.get("gamePk")
        box_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        box = requests.get(box_url)
        if box.status_code != 200:
            continue
        box_data = box.json()
        for team_side in ["home", "away"]:
            try:
                lineup = box_data["liveData"]["boxscore"]["teams"][team_side]["players"]
                for player_id, pdata in lineup.items():
                    # Only take batters in starting lineup (battingOrder present)
                    if "battingOrder" in pdata and "person" in pdata:
                        full_name = pdata["person"]["fullName"]
                        batters.add(full_name)  # <-- No name length restriction
            except Exception:
                continue
    return list(batters)

@st.cache_data(ttl=900)
def load_todays_batters():
    return get_todays_mlb_lineups()

todays_batters = load_todays_batters()

if not todays_batters:
    st.warning("Couldn't load today's MLB lineups from MLB.com. Try reloading or check your connection.")
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
        if player_id is None:
            return pd.DataFrame()
        if days_back == 162:
            start_date = datetime(datetime.now().year, 3, 28)
        else:
            start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=1)
        df = statcast_batter(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), player_id)
        return df
    except Exception:
        return pd.DataFrame()

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

def get_score(row, window):
    ev_col = f"EV_{window}"
    br_col = f"BarrelRate_{window}"
    if ev_col not in row or br_col not in row:
        return 0
    ev = row[ev_col]
    br = row[br_col]
    if ev is None or br is None:
        return 0
    ev_norm = (ev - 80) / (105 - 80)
    ev_norm = max(0, min(ev_norm, 1))
    br_norm = min(br / 0.15, 1)
    return round((ev_norm * 0.6 + br_norm * 0.4) * 100, 1)

if not df.empty:
    for window in windows:
        df[f"HR_Score_{window}"] = df.apply(lambda row: get_score(row, window), axis=1)

score_cols = [f"HR_Score_{w}" for w in windows]
existing_score_cols = [col for col in score_cols if col in df.columns]

# Pick the first available HR_Score column to sort by, prefer most recent
sort_window = None
for col in score_cols[1:]:
    if col in df.columns and df[col].sum() > 0:
        sort_window = col
        break
if sort_window is None and existing_score_cols:
    sort_window = existing_score_cols[0]

if sort_window is not None:
    st.subheader(f"Top HR Probability Batters (Sorted by {sort_window.replace('_', ' ')})")
    show_cols = ["Batter"] + score_cols + [f"EV_{w}" for w in windows] + [f"BarrelRate_{w}" for w in windows] + [f"PA_{w}" for w in windows]
    show_cols = [col for col in show_cols if col in df.columns]
    st.dataframe(df.sort_values(sort_window, ascending=False).reset_index(drop=True)[show_cols])
else:
    st.warning("No HR probability data found for today's lineups. This may be due to missing Statcast data for recent days.")

st.caption("Columns ending in _162 = Season, _14 = last 14d, _7 = last 7d, _5 = last 5d, _3 = last 3d. HR_Score columns use normalized EV (60%) and barrel rate (40%).")
