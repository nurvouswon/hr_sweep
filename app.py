# app.py – MLB HR Matchup Leaderboard (Full Enhanced Version)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import unicodedata
import difflib
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from pybaseball.lahman import people

# API Key (replace this for local dev or configure in Streamlit Cloud)
API_KEY = st.secrets["weather"]["api_key"]
error_log = []

# Caching: Speeds up re-runs
@st.cache_data
def cached_statcast_batter(start, end, batter_id):
    return statcast_batter(start, end, batter_id)

@st.cache_data
def cached_statcast_pitcher(start, end, pitcher_id):
    return statcast_pitcher(start, end, pitcher_id)

@st.cache_data
def cached_playerid_lookup(last, first):
    return playerid_lookup(last, first)

@st.cache_data
def cached_weather_api(city, date, api_key):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
    resp = requests.get(url, timeout=10)
    return resp.json()

# Name normalization for matching
def normalize_name(name):
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

# Park orientations and factors
ballpark_orientations = {
    "Yankee Stadium": "N", "Fenway Park": "N", "Tropicana Field": "N",
    "Camden Yards": "NE", "Rogers Centre": "NE", "Comerica Park": "N",
    "Progressive Field": "NE", "Target Field": "N", "Kauffman Stadium": "NE",
    "Guaranteed Rate Field": "NE", "Angel Stadium": "NE", "Minute Maid Park": "N",
    "Oakland Coliseum": "N", "T-Mobile Park": "N", "Globe Life Field": "NE",
    "Dodger Stadium": "NE", "Chase Field": "N", "Coors Field": "N",
    "Oracle Park": "E", "Wrigley Field": "NE", "Great American Ball Park": "N",
    "American Family Field": "NE", "PNC Park": "NE", "Busch Stadium": "NE",
    "Truist Park": "N", "LoanDepot Park": "N", "Citi Field": "N",
    "Nationals Park": "NE", "Petco Park": "N", "Citizens Bank Park": "NE"
}

park_factors = {
    "Yankee Stadium": 1.19, "Fenway Park": 0.97, "Tropicana Field": 0.85,
    "Camden Yards": 1.13, "Rogers Centre": 1.10, "Comerica Park": 0.96,
    "Progressive Field": 1.01, "Target Field": 1.04, "Kauffman Stadium": 0.98,
    "Guaranteed Rate Field": 1.18, "Angel Stadium": 1.05, "Minute Maid Park": 1.06,
    "Oakland Coliseum": 0.82, "T-Mobile Park": 0.86, "Globe Life Field": 1.00,
    "Dodger Stadium": 1.10, "Chase Field": 1.06, "Coors Field": 1.30,
    "Oracle Park": 0.82, "Wrigley Field": 1.12, "Great American Ball Park": 1.26,
    "American Family Field": 1.17, "PNC Park": 0.87, "Busch Stadium": 0.87,
    "Truist Park": 1.06, "LoanDepot Park": 0.86, "Citi Field": 1.05,
    "Nationals Park": 1.05, "Petco Park": 0.85, "Citizens Bank Park": 1.19
}
# === Compass & Wind Logic ===
compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

def get_compass_idx(dir_str):
    dir_str = dir_str.upper()
    try: return compass.index(dir_str)
    except: return -1

def is_wind_out(wind_dir, park_orientation):
    wi = get_compass_idx(wind_dir)
    pi = get_compass_idx(park_orientation)
    if wi == -1 or pi == -1: return "unknown"
    if abs(wi - pi) <= 1 or abs(wi - pi) >= 7: return "out"
    elif abs(wi - pi) == 4: return "in"
    else: return "side"

def get_weather(city, date, park_orientation, game_time, api_key=API_KEY):
    try:
        data = cached_weather_api(city, date, api_key)
        game_hour = int(game_time.split(":")[0]) if game_time else 14
        hours = data['forecast']['forecastday'][0]['hour']
        weather_hour = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - game_hour))
        temp = weather_hour.get('temp_f', None)
        wind = weather_hour.get('wind_mph', None)
        wind_dir = weather_hour.get('wind_dir', '')[:2].upper()
        humidity = weather_hour.get('humidity', None)
        condition = weather_hour.get('condition', {}).get('text', None)
        wind_effect = is_wind_out(wind_dir, park_orientation)
        return {
            "Temp": temp, "Wind": wind, "WindDir": wind_dir, "WindEffect": wind_effect,
            "Humidity": humidity, "Condition": condition
        }
    except Exception as e:
        error_log.append(f"Weather error for {city} on {date}: {e}")
        return {
            "Temp": None, "Wind": None, "WindDir": None, "WindEffect": None,
            "Humidity": None, "Condition": None
        }

# === Player ID & Handedness ===
def get_player_id(name):
    try:
        first, last = name.split(" ", 1)
        info = cached_playerid_lookup(last, first)
        if not info.empty:
            return int(info.iloc[0]['key_mlbam'])
    except Exception as e:
        error_log.append(f"Player ID lookup failed for {name}: {e}")
    return None

MANUAL_HANDEDNESS = {
    'alexander canario': ('R', 'R'),
    'liam hicks': ('L', 'R'),
    'patrick bailey': ('B', 'R'),
}

@st.cache_data
def get_handedness(name):
    name = normalize_name(name)
    if name in MANUAL_HANDEDNESS:
        return MANUAL_HANDEDNESS[name]
    try:
        first, last = name.split(" ", 1)
        info = cached_playerid_lookup(last.capitalize(), first.capitalize())
        if not info.empty:
            mlbam_id = info.iloc[0]['key_mlbam']
            url = f'https://statsapi.mlb.com/api/v1/people/{mlbam_id}'
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                hand = data['people'][0]
                bats = hand['batSide']['code']
                throws = hand['pitchHand']['code']
                return bats, throws
    except Exception as e:
        error_log.append(f"Handedness lookup failed for {name}: {e}")
    return None, None
    # === Normalization Utilities ===
def norm_barrel(x): return min(x / 0.15, 1) if pd.notnull(x) else 0
def norm_ev(x): return max(0, min((x - 80) / (105 - 80), 1)) if pd.notnull(x) else 0
def norm_park(x): return max(0, min((x - 0.8) / (1.3 - 0.8), 1)) if pd.notnull(x) else 0

def norm_weather(temp, wind, wind_effect):
    score = 1
    if temp and temp > 80: score += 0.05
    if wind and wind > 10:
        if wind_effect == "out": score += 0.07
        elif wind_effect == "in": score -= 0.07
    return max(0.8, min(score, 1.2))

# === Custom Park/Matchup Boosts ===
def custom_2025_boost(row):
    bonus = 0
    if row.get('Park') == 'Citi Field': bonus += 0.025
    if row.get('Park') == 'Comerica Park': bonus += 0.02
    if row.get('Park') == 'Wrigley Field' and row.get('WindEffect') == 'out': bonus += 0.03
    if row.get('Park') in ['American Family Field','Citizens Bank Park'] and row.get('WindEffect') == 'out': bonus += 0.015
    if row.get('Park') == 'Dodger Stadium' and row.get('BatterHandedness') == 'R': bonus += 0.01
    if row.get('Temp') and row.get('Temp') > 80: bonus += 0.01
    if row.get('BatterHandedness') == 'R' and row.get('Park') in [
        "Yankee Stadium","Great American Ball Park","Guaranteed Rate Field"]: bonus += 0.012
    if row.get('Humidity') and row.get('Humidity') > 65 and row.get('Park') in ["Truist Park","LoanDepot Park"]: bonus += 0.01
    if row.get('Park') in ["Dodger Stadium","Petco Park","Oracle Park"]:
        game_time = row.get('Time')
        if game_time:
            try:
                hour = int(str(game_time).split(":")[0])
                if hour < 17: bonus -= 0.01
            except Exception:
                bonus -= 0.01
        else:
            bonus -= 0.01
    if row.get('PitcherHandedness') == 'L': bonus += 0.01
    return bonus

# === Pitcher Pitch Mix (% usage) ===
def get_pitcher_pitch_mix(pitcher_id, window=14):
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_pitcher(start, end, pitcher_id)
        total = len(df)
        if total == 0: return {}
        pct = df['pitch_type'].value_counts(normalize=True).to_dict()
        return {
            "FB%": round(100 * sum(pct.get(pt, 0) for pt in ['FF', 'FT', 'SI']), 1),
            "SL%": round(100 * pct.get('SL', 0), 1),
            "CU%": round(100 * pct.get('CU', 0), 1),
            "CH%": round(100 * pct.get('CH', 0), 1),
            "CT%": round(100 * pct.get('CT', 0), 1),
            "SP%": round(100 * pct.get('FS', 0), 1)
        }
    except Exception as e:
        error_log.append(f"Pitch mix error: {e}")
        return {}

# === Batter wOBA by Pitch Type
def get_batter_pitchtype_woba(batter_id, window=14):
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_batter(start, end, batter_id)
        if df.empty or 'pitch_type' not in df.columns:
            return {}
        result = {}
        for pt in df['pitch_type'].dropna().unique():
            result[pt] = round(df[df['pitch_type'] == pt]['woba_value'].mean(), 3)
        return result
    except Exception as e:
        error_log.append(f"Batter wOBA by pitch type error: {e}")
        return {}

# === Pitch Type Matchup Boost
def calc_pitchtype_boost(batter_pitch_woba, pitcher_mix):
    try:
        boost = 0
        total_weight = 0
        for pt_code, label in {
            'FF': 'FB%', 'SI': 'FB%', 'FT': 'FB%', 'SL': 'SL%', 'CU': 'CU%', 'CH': 'CH%', 'CT': 'CT%', 'FS': 'SP%'
        }.items():
            if label not in pitcher_mix: continue
            pitch_pct = pitcher_mix[label] / 100
            woba = batter_pitch_woba.get(pt_code, 0.320)
            boost += (woba - 0.320) * pitch_pct
            total_weight += pitch_pct
        return round(boost * 0.15, 3) if total_weight > 0 else 0
    except Exception as e:
        error_log.append(f"Pitch type matchup boost error: {e}")
        return 0
        # === Final HR Score ===
def calc_hr_score(row):
    batter_score = (
        norm_barrel(row.get('B_BarrelRate_14')) * 0.12 +
        norm_barrel(row.get('B_BarrelRate_7')) * 0.09 +
        norm_barrel(row.get('B_BarrelRate_5')) * 0.07 +
        norm_barrel(row.get('B_BarrelRate_3')) * 0.05 +
        norm_ev(row.get('B_EV_14')) * 0.08 +
        norm_ev(row.get('B_EV_7')) * 0.06 +
        norm_ev(row.get('B_EV_5')) * 0.04 +
        norm_ev(row.get('B_EV_3')) * 0.02 +
        (row.get('B_SLG_14') or 0) * 0.06 +
        (row.get('B_xslg_14') or 0) * 0.06 +
        (row.get('B_xwoba_14') or 0) * 0.10 +
        (row.get('B_sweet_spot_pct_14') or 0) * 0.03 +
        (row.get('B_pull_pct_14') or 0) * 0.01 +
        (row.get('B_oppo_pct_14') or 0) * 0.01 +
        (row.get('B_gbfb_14') or 0) * 0.01 +
        (row.get('B_hardhit_pct_14') or 0) * 0.02
    )

    pitcher_score = (
        norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.07 +
        norm_barrel(row.get('P_BarrelRateAllowed_7')) * 0.05 +
        norm_barrel(row.get('P_BarrelRateAllowed_5')) * 0.03 +
        norm_barrel(row.get('P_BarrelRateAllowed_3')) * 0.02 +
        norm_ev(row.get('P_EVAllowed_14')) * 0.05 +
        norm_ev(row.get('P_EVAllowed_7')) * 0.03 +
        norm_ev(row.get('P_EVAllowed_5')) * 0.02 +
        norm_ev(row.get('P_EVAllowed_3')) * 0.01 +
        -(row.get('P_SLG_14') or 0) * 0.06 +
        -(row.get('P_xslg_14') or 0) * 0.06 +
        -(row.get('P_xwoba_14') or 0) * 0.05 +
        -(row.get('P_sweet_spot_pct_14') or 0) * 0.02 +
        -(row.get('P_pull_pct_14') or 0) * 0.01 +
        -(row.get('P_oppo_pct_14') or 0) * 0.01 +
        -(row.get('P_gbfb_14') or 0) * 0.01 +
        -(row.get('P_hardhit_pct_14') or 0) * 0.02
    )

    park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.10
    weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.15
    regression_score = max(0, min((row.get('xhr_diff', 0) or 0) / 5, 0.12))
    platoon_score = ((row.get('PlatoonWoba') or 0.320) - 0.320) * 0.10
    pitchtype_boost = row.get("PitchMixBoost", 0)
    profile_score = row.get('BattedBallScore', 0) + row.get('PitcherBBScore', 0)
    custom_bonus = custom_2025_boost(row)

    return round(batter_score + pitcher_score + park_score + weather_score +
                 regression_score + platoon_score + pitchtype_boost + profile_score + custom_bonus, 3)

# === Streamlit App UI ===
st.title("⚾ MLB HR Matchup Leaderboard – Fully Enhanced")
st.markdown("Upload the 4 required CSVs to begin.")

uploaded_file = st.file_uploader("Matchups CSV", type=["csv"])
xhr_file = st.file_uploader("xHR/HR CSV", type=["csv"])
batted_file = st.file_uploader("Batter Batted-Ball CSV", type=["csv"])
pitcher_file = st.file_uploader("Pitcher Batted-Ball CSV", type=["csv"])

if uploaded_file and xhr_file and batted_file and pitcher_file:
    df = pd.read_csv(uploaded_file)
    xhr = pd.read_csv(xhr_file)
    batted = pd.read_csv(batted_file).rename(columns={"id": "batter_id"})
    pitcher = pd.read_csv(pitcher_file).rename(columns={"id": "pitcher_id", "bbe": "bbe_pbb"})
    pitcher = pitcher.rename(columns={c: f"{c}_pbb" for c in pitcher.columns if c not in ["pitcher_id", "name_pbb"]})
    df['norm_batter'] = df['Batter'].map(normalize_name)
    df['batter_id'] = df['Batter'].map(get_player_id)
    df['pitcher_id'] = df['Pitcher'].map(get_player_id)
    xhr['player_norm'] = xhr['player'].map(normalize_name)
    df = df.merge(xhr[['player_norm', 'xhr', 'hr_total', 'xhr_diff']], left_on='norm_batter', right_on='player_norm', how='left')
    df['ParkFactor'] = df['Park'].map(park_factors)
    df['ParkOrientation'] = df['Park'].map(ballpark_orientations)

    rows = []
progress = st.progress(0)

for idx, row in df_merged.iterrows():
    try:
        weather = get_weather(row['City'], row['Date'], row['ParkOrientation'], row['Time'])
        bstats = get_batter_stats_multi(row['batter_id'])
        pstats = get_pitcher_stats_multi(row['pitcher_id'])
        badv = get_batter_advanced_stats_xslg(row['batter_id'])
        padv = get_pitcher_advanced_stats_xslg(row['pitcher_id'])
        b_bats, _ = get_handedness(row['Batter'])
        _, p_throws = get_handedness(row['Pitcher'])
        platoon = get_platoon_woba(row['batter_id'], p_throws)
        pitchmix = get_pitcher_pitch_mix(row['pitcher_id'])
        pitchwoba = get_batter_pitchtype_woba(row['batter_id'])
        pitchboost = calc_pitchtype_boost(pitchwoba, pitchmix)

        r = row.to_dict()

        # Ensure IDs are carried forward
        r['batter_id'] = row.get('batter_id')
        r['pitcher_id'] = row.get('pitcher_id')

        r.update(weather)
        r.update(bstats)
        r.update(pstats)
        r.update(badv)
        r.update(padv)
        r['PlatoonWoba'] = platoon
        r['PitchMixBoost'] = pitchboost

        rows.append(r)

    except Exception as e:
        error_log.append(f"{row['Batter']} vs {row['Pitcher']}: {e}")

    # Progress percentage
    progress.progress((i + 1) / len(df_merged), text=f"Processing {int(100 * (i + 1) / len(df_merged))}%")
    df_final = pd.DataFrame(records)
    df_final = df_final.merge(batted, on="batter_id", how="left")
    df_final = df_final.merge(pitcher, on="pitcher_id", how="left")

    df_final['BattedBallScore'] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final['PitcherBBScore'] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values('HR_Score', ascending=False)

    cols = [
        'Batter','Pitcher','HR_Score','xhr_diff','xhr','hr_total',
        'B_BarrelRate_14','B_EV_14','B_SLG_14','B_xslg_14','B_xwoba_14',
        'B_sweet_spot_pct_14','B_pull_pct_14','B_oppo_pct_14','B_hardhit_pct_14',
        'P_BarrelRateAllowed_14','P_EVAllowed_14','P_SLG_14','P_xslg_14','P_xwoba_14',
        'P_sweet_spot_pct_14','P_pull_pct_14','P_oppo_pct_14','P_hardhit_pct_14',
        'PlatoonWoba','PitchMixBoost','Temp','Wind','WindEffect','ParkFactor',
        'BattedBallScore','PitcherBBScore'
    ]
    st.dataframe(df_leaderboard[cols].head(15))
    st.bar_chart(df_leaderboard.set_index('Batter')[['HR_Score']].head(5))
    st.download_button("Download CSV", df_leaderboard.to_csv(index=False), "hr_leaderboard.csv")

    if error_log:
        with st.expander("⚠️ Errors & Logs"):
            for e in error_log:
                st.text(e)
else:
    st.info("Please upload all 4 required files.")
