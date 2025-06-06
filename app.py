import streamlit as st
import pandas as pd
import numpy as np
import requests
import unicodedata
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

API_KEY = st.secrets["weather"]["api_key"]
error_log = []

def log_error(context, exception, level="ERROR"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] [{level}] {context}: {exception}"
    error_log.append(msg)
    print(msg)

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
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.text.strip():
            return resp.json()
        else:
            raise ValueError(f"Empty or bad response: {resp.status_code}")
    except Exception as e:
        log_error("Weather API call", e)
        return {}
# --- Static logit weights for HR Score (auto-generated from your CSV + 3 custom features) ---
STATIC_LOGIT_WEIGHTS = {
    "B_SLG_14": 0.5267278705,
    "P_SLG_14": 0.4898397892,
    "B_hardhit_pct_14": 0.2465825531,
    "P_hardhit_pct_14": 0.2028337498,
    "P_hardhit_pct_5": 0.2012464233,
    "P_xwoba_14": 0.1986446811,
    "B_hardhit_pct_7": 0.1982819304,
    "P_hardhit_pct_7": 0.1876102681,
    "P_SLG_7": 0.1783106545,
    "B_xwoba_14": 0.1751735811,
    "P_BarrelRateAllowed_14": 0.1746593534,
    "B_SLG_7": 0.1693330283,
    "B_hardhit_pct_3": 0.16468403,
    "B_xSLG_14": 0.1573305465,
    "B_hardhit_pct_5": 0.1568523201,
    "P_hardhit_pct_3": 0.1398760269,
    "P_SLG_5": 0.1342024796,
    "P_xwoba_7": 0.1295064101,
    "B_xISO_14": 0.1275490596,
    "P_xwoba_5": 0.1185868111,
    "B_BarrelRate_14": 0.1117798081,
    "P_BarrelRateAllowed_7": 0.1080167763,
    "B_SLG_5": 0.1076433055,
    "P_sweet_spot_pct_14": 0.09603252322,
    "B_xISO_3": 0.09308931784,
    "P_BarrelRateAllowed_5": 0.08896610738,
    "P_xSLG_14": 0.07173565634,
    "B_xwoba_5": 0.05684525149,
    "B_xwoba_7": 0.05661626699,
    "P_xSLG_3": 0.05267014337,
    "P_xSLG_5": 0.04969560583,
    "P_xSLG_7": 0.0360047754,
    "B_xSLG_3": 0.03036553773,
    "P_sweet_spot_pct_7": 0.02761455222,
    "P_sweet_spot_pct_5": 0.02185941319,
    "P_xwoba_3": 0.01645838203,
    "B_EV_3": 0.0147095849,
    "B_EV_14": 0.01296157332,
    "B_BarrelRate_7": 0.01130558273,
    "P_BarrelRateAllowed_3": 0.001490703858,
    "P_EVAllowed_3": 0.001331538993,
    "P_EVAllowed_5": -0.003756845536,
    "B_sweet_spot_pct_14": -0.004009074271,
    "P_xISO_14": -0.008865489247,
    "P_EVAllowed_14": -0.0104865996,
    "B_EV_7": -0.01484089458,
    "B_xISO_7": -0.01995787278,
    "B_EV_5": -0.02028979606,
    "B_xSLG_5": -0.02329611474,
    "B_xISO_5": -0.02491246841,
    "P_xISO_3": -0.03036961059,
    "B_xwoba_3": -0.03219370009,
    "P_sweet_spot_pct_3": -0.03474891921,
    "P_EVAllowed_7": -0.03796445805,
    "B_xSLG_7": -0.04978170743,
    "B_BarrelRate_5": -0.05432801179,
    "P_xISO_7": -0.06858784063,
    "B_sweet_spot_pct_7": -0.0740723242,
    "B_sweet_spot_pct_5": -0.09186272762,
    "P_xISO_5": -0.09250716848,
    "B_sweet_spot_pct_3": -0.09464224113,
    "B_SLG_3": -0.09583742732,
    "B_BarrelRate_3": -0.1755261804,
    "P_SLG_3": -0.2039291056,
    # --- Custom added features for HR context ---
    "HandedHRRate": 1.3,
    "ParkHRRate": 0.8,
    "PitchTypeHRRate": 0.8
}
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

def normalize_park_name(park):
    return str(park).strip().lower().replace(' ', '_')

    df_merged['park'] = df_merged['park'].apply(normalize_park_name)
    park_hr['park'] = park_hr['park'].apply(normalize_park_name)
    
def get_player_id(name):
    if not isinstance(name, str) or len(name.strip().split()) < 2:
        log_error("Player ID lookup", f"Invalid player name format: {name}")
        return None
    try:
        first, last = name.split(" ", 1)
        info = cached_playerid_lookup(last, first)
        if not info.empty:
            return int(info.iloc[0]['key_mlbam'])
    except Exception as e:
        log_error("Player ID lookup", e)
    return None

MANUAL_HANDEDNESS = {}

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
        log_error("Handedness lookup", e)
    return None, None

ballpark_orientations = {
    "sutter_health_park": "NE", "yankee_stadium": "N", "fenway_park": "N", "tropicana_field": "N",
    "camden_yards": "NE", "rogers_centre": "NE", "comerica_park": "N", "progressive_field": "NE",
    "target_field": "N", "kauffman_stadium": "NE", "guaranteed_rate_field": "NE", "angel_stadium": "NE",
    "minute_maid_park": "N", "oakland_coliseum": "N", "t-mobile_park": "N", "tmobile_park": "N", "globe_life_field": "NE",
    "dodger_stadium": "NE", "chase_field": "N", "coors_field": "N", "oracle_park": "E", "wrigley_field": "NE",
    "great_american_ball_park": "N", "american_family_field": "NE", "pnc_park": "NE", "busch_stadium": "NE",
    "truist_park": "N", "loandepot_park": "N", "citi_field": "N", "nationals_park": "NE", "petco_park": "N",
    "citizens_bank_park": "NE"
}
park_factors = {
    "sutter_health_park": 1.12, "yankee_stadium": 1.19, "fenway_park": 0.97, "tropicana_field": 0.85,
    "camden_yards": 1.13, "rogers_centre": 1.10, "comerica_park": 0.96, "progressive_field": 1.01,
    "target_field": 1.04, "kauffman_stadium": 0.98, "guaranteed_rate_field": 1.18, "angel_stadium": 1.05,
    "minute_maid_park": 1.06, "oakland_coliseum": 0.82, "t-mobile_park": 0.86, "tmobile_park": 0.86, "globe_life_field": 1.00,
    "dodger_stadium": 1.10, "chase_field": 1.06, "coors_field": 1.30, "oracle_park": 0.82, "wrigley_field": 1.12,
    "great_american_ball_park": 1.26, "american_family_field": 1.17, "pnc_park": 0.87, "busch_stadium": 0.87,
    "truist_park": 1.06, "loandepot_park": 0.86, "citi_field": 1.05, "nationals_park": 1.05, "petco_park": 0.85,
    "citizens_bank_park": 1.19
}
team_code_to_park = {
    'PHI': 'citizens_bank_park',
    'ATL': 'truist_park',
    'NYM': 'citi_field',
    'BOS': 'fenway_park',
    'NYY': 'yankee_stadium',
    'CHC': 'wrigley_field',
    'LAD': 'dodger_stadium',
    'OAK': 'sutter_health_park',
    'CIN': 'great_american_ball_park',
    'DET': 'comerica_park',
    'HOU': 'minute_maid_park',
    'MIA': 'loandepot_park',
    'TB': 'tropicana_field',
    'MIL': 'american_family_field',
    'SD': 'petco_park',
    'SF': 'oracle_park',
    'TOR': 'rogers_centre',
    'CLE': 'progressive_field',
    'MIN': 'target_field',
    'KC': 'kauffman_stadium',
    'CWS': 'guaranteed_rate_field',
    'LAA': 'angel_stadium',
    'SEA': 't-mobile_park',
    'SEA': 'tmobile_park',
    'TEX': 'globe_life_field',
    'ARI': 'chase_field',
    'COL': 'coors_field',
    'PIT': 'pnc_park',
    'STL': 'busch_stadium',
    'BAL': 'camden_yards',
    'WSH': 'nationals_park',
    'ATH': 'sutter_health_park' # For Las Vegas/Oakland alternate
}

import math
# 16-point compass for mapping wind/park directions to angles (in degrees)
compass_points = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]
compass_angles = {name: i * 22.5 for i, name in enumerate(compass_points)}
# Add 2-letter variants for weather API compatibility
compass_aliases = {
    'EN': 'NE', 'NE': 'NE', 'ES': 'SE', 'SE': 'SE',
    'WN': 'NW', 'NW': 'NW', 'WS': 'SW', 'SW': 'SW',
    'SN': 'N',  'NS': 'S',  'EW': 'E',  'WE': 'W',
    'N': 'N', 'S': 'S', 'E': 'E', 'W': 'W'
}

def direction_to_angle(dir_str):
    d = str(dir_str).upper().replace('-', '').strip()
    if d in compass_angles:
        return compass_angles[d]
    if d in compass_aliases and compass_aliases[d] in compass_angles:
        return compass_angles[compass_aliases[d]]
    # Try first letter fallback
    if d and d[0] in compass_angles:
        return compass_angles[d[0]]
    return None

def is_wind_out(wind_dir, park_orientation, tolerance=45):
    """
    Returns "out" if wind is blowing out to center, "in" if in, "side" otherwise.
    tolerance: Angle in degrees for 'out'/'in' (default 45°).
    """
    wind_angle = direction_to_angle(wind_dir)
    park_angle = direction_to_angle(park_orientation)
    if wind_angle is None or park_angle is None:
        return "unknown"
    diff = (wind_angle - park_angle) % 360
    if diff <= tolerance or diff >= (360 - tolerance):
        return "out"
    if abs(diff - 180) <= tolerance:
        return "in"
    return "side"

def get_weather(city, date, park_orientation, game_time, api_key=API_KEY):
    try:
        if not isinstance(city, str) or pd.isna(city):
            raise ValueError("City is not a valid string")
        city_clean = city.strip()
        data = cached_weather_api(city_clean, date, api_key)
        game_hour = int(str(game_time).split(":")[0]) if isinstance(game_time, str) and ":" in game_time else 14
        hours = data['forecast']['forecastday'][0]['hour']
        weather_hour = min(
            hours,
            key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - game_hour)
        )
        temp = weather_hour.get('temp_f')
        wind = weather_hour.get('wind_mph')
        wind_dir_full = weather_hour.get('wind_dir', '').strip().upper()
        humidity = weather_hour.get('humidity')
        condition = weather_hour.get('condition', {}).get('text')
        wind_effect = is_wind_out(wind_dir_full, park_orientation)
        if wind_effect == "unknown":
            log_error("Wind Effect Debug", f"wind_dir={wind_dir_full}, orientation={park_orientation}")
        return {
            "Temp": temp, "Wind": wind, "WindDir": wind_dir_full, "WindEffect": wind_effect,
            "Humidity": humidity, "Condition": condition
        }
    except Exception as e:
        log_error("Weather", e)
        return {"Temp": None, "Wind": None, "WindDir": None, "WindEffect": None, "Humidity": None, "Condition": None}
# ========= Normalization Functions =========
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
def norm_xwoba(x): return max(0, min((x - 0.250) / (0.400 - 0.250), 1)) if pd.notnull(x) else 0
def norm_sweetspot(x): return max(0, min((x - 0.25) / (0.45 - 0.25), 1)) if pd.notnull(x) else 0
def norm_hardhit(x): return max(0, min((x - 0.25) / (0.60 - 0.25), 1)) if pd.notnull(x) else 0
def norm_whiff(x): return max(0, min((x - 0.15) / (0.40 - 0.15), 1)) if pd.notnull(x) else 0

# --- Statcast Feature Functions: Batter ---
def get_batter_stats_multi(batter_id, windows=[3,5,7,14]):
    out = {}
    if not batter_id:
        for w in windows:
            for k in ['B_BarrelRate','B_EV','B_SLG','B_xSLG','B_xISO','B_xwoba','B_sweet_spot_pct','B_hardhit_pct']:
                out[f"{k}_{w}"] = None
        return out
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_batter(start, end, batter_id)
            df = df[df['type'] == 'X']
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            if df.empty: raise ValueError("No batted ball data")
            pa = df.shape[0]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            ev = df['launch_speed'].mean()
            xwoba = df['woba_value'].mean() if 'woba_value' in df.columns else None
            sweet_spot_pct = df[(df['launch_angle'].between(8, 32))].shape[0] / pa if pa > 0 else None
            hard_hit_pct = df[(df['launch_speed'] >= 95)].shape[0] / pa if pa > 0 else None
            events = df['events'].value_counts().to_dict()
            ab2 = sum(events.get(e, 0) for e in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
            total_bases = (events.get('single', 0) + 2 * events.get('double', 0) + 3 * events.get('triple', 0) + 4 * events.get('home_run', 0))
            slg = total_bases / ab2 if ab2 > 0 else None
            single = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'] < 15)].shape[0]
            double = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(15, 30))].shape[0]
            triple = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(30, 40))].shape[0]
            hr = df[(df['launch_angle'] > 35) & (df['launch_speed'] > 100)].shape[0]
            ab = single + double + triple + hr
            xslg = (1*single + 2*double + 3*triple + 4*hr) / ab if ab else None
            xba = df['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in df.columns else None
            xiso = (xslg - xba) if xslg and xba else None
            out[f'B_BarrelRate_{w}'] = round(barrels / pa, 3)
            out[f'B_EV_{w}'] = round(ev, 1)
            out[f'B_SLG_{w}'] = round(slg, 3) if slg is not None else None
            out[f'B_xSLG_{w}'] = round(xslg, 3) if xslg is not None else None
            out[f'B_xISO_{w}'] = round(xiso, 3) if xiso is not None else None
            out[f'B_xwoba_{w}'] = round(xwoba, 3) if xwoba is not None else None
            out[f'B_sweet_spot_pct_{w}'] = round(sweet_spot_pct, 3) if sweet_spot_pct is not None else None
            out[f'B_hardhit_pct_{w}'] = round(hard_hit_pct, 3) if hard_hit_pct is not None else None
        except Exception as e:
            log_error(f"Batter stats error ({batter_id}, {w}d)", e)
            for k in ['B_BarrelRate','B_EV','B_SLG','B_xSLG','B_xISO','B_xwoba','B_sweet_spot_pct','B_hardhit_pct']:
                out[f"{k}_{w}"] = None
    return out

def get_pitcher_stats_multi(pitcher_id, windows=[3,5,7,14]):
    out = {}
    if not pitcher_id:
        for w in windows:
            for k in ['P_BarrelRateAllowed','P_EVAllowed','P_SLG','P_xSLG','P_xISO','P_xwoba','P_sweet_spot_pct','P_hardhit_pct']:
                out[f"{k}_{w}"] = None
        return out
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_pitcher(start, end, pitcher_id)
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            if df.empty: raise ValueError("No batted ball data")
            total = df.shape[0]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            ev = df['launch_speed'].mean()
            xwoba = df['woba_value'].mean() if 'woba_value' in df.columns else None
            sweet_spot_pct = df[(df['launch_angle'].between(8, 32))].shape[0] / total if total > 0 else None
            hard_hit_pct = df[(df['launch_speed'] >= 95)].shape[0] / total if total > 0 else None
            events = df['events'].value_counts().to_dict()
            ab2 = sum(events.get(e, 0) for e in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
            total_bases = (events.get('single', 0) + 2 * events.get('double', 0) + 3 * events.get('triple', 0) + 4 * events.get('home_run', 0))
            slg = total_bases / ab2 if ab2 > 0 else None
            single = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'] < 15)].shape[0]
            double = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(15, 30))].shape[0]
            triple = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(30, 40))].shape[0]
            hr = df[(df['launch_angle'] > 35) & (df['launch_speed'] > 100)].shape[0]
            ab = single + double + triple + hr
            xslg = (1*single + 2*double + 3*triple + 4*hr) / ab if ab else None
            xba = df['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in df.columns else None
            xiso = (xslg - xba) if xslg and xba else None
            out[f'P_BarrelRateAllowed_{w}'] = round(barrels / total, 3)
            out[f'P_EVAllowed_{w}'] = round(ev, 1)
            out[f'P_SLG_{w}'] = round(slg, 3) if slg is not None else None
            out[f'P_xSLG_{w}'] = round(xslg, 3) if xslg is not None else None
            out[f'P_xISO_{w}'] = round(xiso, 3) if xiso is not None else None
            out[f'P_xwoba_{w}'] = round(xwoba, 3) if xwoba is not None else None
            out[f'P_sweet_spot_pct_{w}'] = round(sweet_spot_pct, 3) if sweet_spot_pct is not None else None
            out[f'P_hardhit_pct_{w}'] = round(hard_hit_pct, 3) if hard_hit_pct is not None else None
        except Exception as e:
            log_error(f"Pitcher stats error ({pitcher_id}, {w}d)", e)
            for k in ['P_BarrelRateAllowed','P_EVAllowed','P_SLG','P_xSLG','P_xISO','P_xwoba','P_sweet_spot_pct','P_hardhit_pct']:
                out[f"{k}_{w}"] = None
    return out

# --- Whiff, Spin, Mix, Platoon, etc ---
def get_batter_pitch_metrics(batter_id, windows=[3,5,7,14]):
    out = {}
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_batter(start, end, batter_id)
            if 'description' in df.columns and not df.empty:
                swing_mask = df['description'].isin([
                    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip"
                ])
                whiff_mask = df['description'].isin([
                    "swinging_strike", "swinging_strike_blocked"
                ])
                swings = df[swing_mask]
                whiffs = df[whiff_mask]
                whiff_rate = whiffs.shape[0] / swings.shape[0] if swings.shape[0] > 0 else None
            else:
                whiff_rate = None
            out[f'B_WhiffRate_{w}'] = round(whiff_rate, 3) if whiff_rate is not None else None
        except Exception as e:
            log_error(f"Batter pitch metric error ({batter_id}, {w}d)", e)
            out[f'B_WhiffRate_{w}'] = None
    return out

def get_pitcher_pitch_metrics(pitcher_id, windows=[3,5,7,14]):
    out = {}
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_pitcher(start, end, pitcher_id)
            if 'description' in df.columns and not df.empty:
                swing_mask = df['description'].isin([
                    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip"
                ])
                whiff_mask = df['description'].isin([
                    "swinging_strike", "swinging_strike_blocked"
                ])
                swings = df[swing_mask]
                whiffs = df[whiff_mask]
                whiff_rate = whiffs.shape[0] / swings.shape[0] if swings.shape[0] > 0 else None
            else:
                whiff_rate = None
            out[f'P_WhiffRate_{w}'] = round(whiff_rate, 3) if whiff_rate is not None else None
        except Exception as e:
            log_error(f"Pitcher pitch metric error ({pitcher_id}, {w}d)", e)
            out[f'P_WhiffRate_{w}'] = None
    return out

def get_pitcher_spin_metrics(pitcher_id, windows=[3,5,7,14]):
    out = {}
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_pitcher(start, end, pitcher_id)
            fb = df[df['pitch_type'] == 'FF']
            avg_spin = fb['release_spin_rate'].mean() if not fb.empty else None
            out[f'P_FF_Spin_{w}'] = round(avg_spin, 1) if avg_spin is not None else None
        except Exception as e:
            log_error(f"Pitcher spin metric error ({pitcher_id}, {w}d)", e)
            out[f'P_FF_Spin_{w}'] = None
    return out

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
        log_error("Pitch mix error", e)
        return {}

def get_batter_pitchtype_woba(batter_id, window=14):
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_batter(start, end, batter_id)
        if df.empty or 'pitch_type' not in df.columns: return {}
        result = {}
        for pt in df['pitch_type'].dropna().unique():
            result[pt] = round(df[df['pitch_type'] == pt]['woba_value'].mean(), 3)
        return result
    except Exception as e:
        log_error("Batter wOBA by pitch type error", e)
        return {}

def get_platoon_woba(batter_id, pitcher_hand, days=365):
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        df = cached_statcast_batter(start, end, batter_id)
        df = df[df['p_throws'] == pitcher_hand]
        return df['woba_value'].mean() if not df.empty else None
    except Exception as e:
        log_error("Platoon wOBA error", e)
        return None

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
        log_error("Pitch type matchup boost error", e)
        return 0

def load_and_standardize_handed_hr(handed_hr_file):
    """
    Load and normalize the Handed HR CSV to always produce:
    columns: ['BatterHandedness', 'PitcherHandedness', 'HandedHRRate']
    """
    df = pd.read_csv(handed_hr_file)
    # Lowercase and strip all columns
    df.columns = [c.strip().lower().replace(' ', '').replace('_', '') for c in df.columns]
    rename_map = {
        'stand': 'BatterHandedness',
        'batterhand': 'BatterHandedness',
        'batterhandedness': 'BatterHandedness',
        'pthrows': 'PitcherHandedness',
        'pitcherhand': 'PitcherHandedness',
        'pitcherhandedness': 'PitcherHandedness',
        'hroutcome': 'HandedHRRate',
        'hrrate': 'HandedHRRate',
        'handedhrrate': 'HandedHRRate'
    }
    # Rename all columns using mapping if they exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Check for all required columns
    required = ['BatterHandedness', 'PitcherHandedness', 'HandedHRRate']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Handed HR CSV missing required column: {col}. Columns found: {df.columns.tolist()}")
    # Only keep required columns
    return df[required]

def load_and_standardize_pitchtype_hr(pitchtype_hr_file):
    """
    Loads and robustly normalizes the Pitch Type HR CSV to always produce columns:
    ['pitch_type', 'PitchTypeHRRate']
    """
    df = pd.read_csv(pitchtype_hr_file)
    # Normalize all columns to lowercase, remove spaces/underscores
    df.columns = [c.strip().lower().replace(' ', '').replace('_', '') for c in df.columns]
    rename_map = {
        'pitchtype': 'pitch_type',
        'pitch_type': 'pitch_type',
        'pitch': 'pitch_type',
        'hroutcome': 'PitchTypeHRRate',
        'hrratepitch': 'PitchTypeHRRate',
        'hrrate': 'PitchTypeHRRate',
        'hr_rate': 'PitchTypeHRRate',
        'hr_rate_pitch': 'PitchTypeHRRate',
        'pitchtypehrrate': 'PitchTypeHRRate'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if 'pitch_type' not in df.columns or 'PitchTypeHRRate' not in df.columns:
        raise ValueError(f"PitchType HR CSV missing required column(s). Found: {df.columns.tolist()}")
    return df[['pitch_type', 'PitchTypeHRRate']]
# --- Custom 2025 Ballpark/Weather/Handedness Boosts ---
def custom_2025_boost(row):
    bonus = 0
    if row.get('park') == 'sutter_health_park' and row.get('WindEffect') == 'out': bonus += 0.02
    if row.get('park') == 'citi_field': bonus += 0.025
    if row.get('park') == 'comerica_park': bonus += 0.02
    if row.get('park') == 'wrigley_field' and row.get('WindEffect') == 'out': bonus += 0.03
    if row.get('park') in ['american_family_field','citizens_bank_park'] and row.get('WindEffect') == 'out': bonus += 0.015
    if row.get('park') == 'dodger_stadium' and row.get('BatterHandedness') == 'R': bonus += 0.01
    if row.get('Temp') and row.get('Temp') > 80: bonus += 0.01
    if row.get('BatterHandedness') == 'R' and row.get('park') in [
        "yankee_stadium","great_american_ball_park","guaranteed_rate_field"]: bonus += 0.012
    if row.get('Humidity') and row.get('Humidity') > 65 and row.get('park') in ["truist_park","loandepot_park"]: bonus += 0.01
    if row.get('park') in ["dodger_stadium","petco_park","oracle_park"]:
        game_time = row.get('time')
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

# ========== Batted Ball Profile Score Functions ==========
def calc_batted_ball_score(row):
    score = 0
    score += row.get('fb_rate', 0) * 0.09
    score += row.get('pull_air_rate', 0) * 0.09
    score += row.get('pull_rate', 0) * 0.04
    score += row.get('air_rate', 0) * 0.03
    score += row.get('ld_rate', 0) * 0.01
    score -= row.get('pu_rate', 0) * 0.07
    score -= row.get('gb_rate', 0) * 0.09
    score -= row.get('oppo_gb_rate', 0) * 0.04
    score -= row.get('pull_gb_rate', 0) * 0.01
    score -= row.get('straight_gb_rate', 0) * 0.01
    score -= row.get('oppo_air_rate', 0) * 0.02
    score -= row.get('straight_air_rate', 0) * 0.01
    if row.get('bbe', 0) < 10:
        score *= 0.85
    return score

def calc_pitcher_bb_score(row):
    score = 0
    score -= row.get('fb_rate_pbb', 0) * 0.09
    score -= row.get('pull_air_rate_pbb', 0) * 0.09
    score -= row.get('pull_rate_pbb', 0) * 0.04
    score -= row.get('air_rate_pbb', 0) * 0.03
    score -= row.get('ld_rate_pbb', 0) * 0.01
    score += row.get('pu_rate_pbb', 0) * 0.07
    score += row.get('gb_rate_pbb', 0) * 0.09
    score += row.get('oppo_gb_rate_pbb', 0) * 0.04
    score += row.get('pull_gb_rate_pbb', 0) * 0.01
    score += row.get('straight_gb_rate_pbb', 0) * 0.01
    score += row.get('oppo_air_rate_pbb', 0) * 0.02
    score += row.get('straight_air_rate_pbb', 0) * 0.01
    if row.get('bbe_pbb', 0) < 10:
        score *= 0.85
    return score

def hr_score_tier(score):
    if score >= 0.70: return "A (Elite)"
    elif score >= 0.50: return "B (Strong)"
    elif score >= 0.35: return "C (Solid)"
    elif score >= 0.22: return "D (Fringe)"
    else: return "E (Low)"

def get_sample_flag(val):
    return "Low" if val is not None and val < 10 else "OK"

def train_and_apply_model(df_leaderboard):
    features = [col for col in df_leaderboard.columns if col not in [
        'batter', 'pitcher', 'hr_score', 'rank', 'city', 'park', 'date', 'time',
        'hr_tier', 'hr_score_pctile'
    ]]
    df_ml = df_leaderboard.dropna(subset=features)
    if 'hr_outcome' not in df_ml.columns or df_ml.shape[0] < 50:
        st.warning("Not enough labeled data (with hr_outcome=1/0) for model. Add hr_outcome column to train ML model.")
        return None, None
    X = df_ml[features]
    y = df_ml['hr_outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)
    df_leaderboard['ML_HR_Prob'] = model.predict_proba(df_leaderboard[selected_features].fillna(0))[:, 1]
    importances = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return df_leaderboard, importances
    
def compute_hr_score_logit(row, logit_weights_dict):
    score = 0
    for feature, weight in logit_weights_dict.items():
        val = row.get(feature, 0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        score += float(val) * float(weight)
    return score
# ====================== STREAMLIT UI & LEADERBOARD ========================
st.title("⚾ MLB HR Matchup Leaderboard – Analyzer+ Statcast + Pitcher Trends + ML")
st.markdown("""
**Upload these 8 Analyzer CSVs for maximum prediction power:**  
1. **Lineups/Matchups** (confirmed, with MLB IDs)  
2. **xHR/HR Regression** (player, hr_total, xhr, xhr_diff)  
3. **Batter Batted-Ball Profile** (with id)  
4. **Pitcher Batted-Ball Profile** (with id)  
5. **Handedness HR Rates** (batter_id, pitcher_hand, hr_rate, etc.)  
6. **Pitch Type HR Rates** (batter_id, pitch_type, hr_rate, etc.)  
7. **Park HR Rates** (park, hr_rate, etc.)  
8. **Custom Logistic Weights** (feature, weight)  
""")

# --- File Uploaders for all 8 CSVs
lineup_file = st.file_uploader("1️⃣ Lineups/Matchups CSV (with MLB IDs)", type=["csv"])
xhr_file = st.file_uploader("2️⃣ xHR / HR Regression CSV", type=["csv"])
battedball_file = st.file_uploader("3️⃣ Batter batted-ball profile CSV", type=["csv"])
pitcher_battedball_file = st.file_uploader("4️⃣ Pitcher batted-ball profile CSV", type=["csv"])
handed_hr_file = st.file_uploader("5️⃣ Handedness HR Rates CSV", type=["csv"])
pitchtype_hr_file = st.file_uploader("6️⃣ Pitch Type HR Rates CSV", type=["csv"])
park_hr_file = st.file_uploader("7️⃣ Park HR Rates CSV", type=["csv"])

csvs_uploaded = [
    lineup_file, xhr_file, battedball_file, pitcher_battedball_file,
    handed_hr_file, pitchtype_hr_file, park_hr_file
]
all_files_uploaded = all(csvs_uploaded)

logit_weights_dict = STATIC_LOGIT_WEIGHTS

if all_files_uploaded:
    # --- Load and clean all uploaded data
    df_upload = pd.read_csv(lineup_file, sep=None, engine='python')
    df_upload.columns = (
        df_upload.columns
        .str.strip().str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    # Standardize lineup columns
    rename_dict = {
        'team_code': 'team_code', 'game_date': 'date', 'mlb_id': 'batter_id',
        'player_name': 'batter', 'batting_order': 'batting_order', 'confirmed': 'confirmed',
        'weather': 'weather', 'city': 'city', 'park': 'park', 'time': 'time'
    }
    df_upload.rename(columns={k: v for k, v in rename_dict.items() if k in df_upload.columns}, inplace=True)
    required_cols = ['batter', 'batter_id', 'team_code', 'date', 'batting_order', 'confirmed', 'city', 'park', 'time']
    missing = [col for col in required_cols if col not in df_upload.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    df_upload = df_upload[df_upload['confirmed'].astype(str).str.lower() == 'y']
    df_upload['norm_batter'] = df_upload['batter'].apply(normalize_name)
    df_upload['batter_id'] = df_upload['batter_id'].astype(str)
    # Assign pitcher per team (SP = starting pitcher)
    pitcher_rows = df_upload[df_upload['batting_order'].astype(str).str.lower() == 'sp']
    team_pitcher_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter_id']))
    pitcher_name_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter']))
    df_upload['pitcher_id'] = df_upload['team_code'].map(team_pitcher_map)
    df_upload['pitcher'] = df_upload['team_code'].map(pitcher_name_map)

    # Merge xHR regression data
    xhr_df = pd.read_csv(xhr_file)
    xhr_df.columns = xhr_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)
    df_upload['norm_batter'] = df_upload['batter'].apply(normalize_name)
    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='norm_batter', right_on='player_norm', how='left'
    )
    # Normalize park fields
    df_merged['park'] = (
        df_merged['park']
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    df_merged['parkfactor'] = df_merged['park'].map(park_factors)
    df_merged['parkorientation'] = df_merged['park'].map(ballpark_orientations)
    unmatched_parks = df_merged[df_merged['parkorientation'].isna()]['park'].unique()
    if len(unmatched_parks) > 0:
        log_error("Missing parkorientation mapping", unmatched_parks)
    df_merged['parkorientation'] = df_merged['parkorientation'].fillna('N')

    # Merge batter batted ball data
    batted = pd.read_csv(battedball_file)
    batted.columns = (
        batted.columns
        .str.strip().str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    batted = batted.rename(columns={"id": "batter_id"})
    df_merged['batter_id'] = df_merged['batter_id'].astype(str)
    batted['batter_id'] = batted['batter_id'].astype(str)
    df_merged = df_merged.merge(batted, on="batter_id", how="left")

    # Merge pitcher batted ball data
    pitcher_bb = pd.read_csv(pitcher_battedball_file)
    pitcher_bb.columns = (
        pitcher_bb.columns
        .str.strip().str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    pitcher_bb = pitcher_bb.rename(columns={"id": "pitcher_id", "bbe": "bbe_pbb"})
    pitcher_bb = pitcher_bb.rename(columns={c: f"{c}_pbb" for c in pitcher_bb.columns if c not in ["pitcher_id", "name_pbb"]})
    df_merged['pitcher_id'] = df_merged['pitcher_id'].astype(str)
    pitcher_bb['pitcher_id'] = pitcher_bb['pitcher_id'].astype(str)
    df_merged = df_merged.merge(pitcher_bb, on="pitcher_id", how="left")
    
    # Assign/normalize park using team_code if needed
    if 'park' not in df_merged.columns or df_merged['park'].isnull().any():
        if 'team_code' in df_merged.columns:
            df_merged['park'] = df_merged['team_code'].map(team_to_park)

    # --- Park HR Rate Merge (Team code to Park name and normalization) ---
    park_hr = pd.read_csv(park_hr_file)

    # Normalize column names
    park_hr.columns = (
        park_hr.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )

    possible_park_cols = ['park', 'parknames', 'ballpark', 'stadium']
    possible_rate_cols = ['parkrate', 'parkratehr', 'parkrate_homerun', 'hrrate', 'hrrate_park', 'parkhrrate']

    park_col = next((c for c in park_hr.columns if c in possible_park_cols), park_hr.columns[0])
    rate_col = next((c for c in park_hr.columns if c in possible_rate_cols), park_hr.columns[1] if len(park_hr.columns) > 1 else None)

    park_hr.rename(columns={park_col: 'park', rate_col: 'ParkHRRate'}, inplace=True)

    # Map codes (ATH, ATL, etc.) to full park names
    park_hr['park'] = park_hr['park'].map(team_code_to_park)

    # Normalize park names (so 'sutter_health_park', etc.)
    park_hr['park'] = park_hr['park'].apply(normalize_park_name)
    df_merged['park'] = df_merged['park'].apply(normalize_park_name)

    # Merge the ParkHRRate column onto your main dataframe
    df_merged = df_merged.merge(park_hr[['park', 'ParkHRRate']], on='park', how='left')

    # Debugging: check for parks with missing ParkHRRate
    missing_park_hr = df_merged[df_merged['ParkHRRate'].isnull()]['park'].unique()
    if len(missing_park_hr) > 0:
        st.warning(f"Park HR Rate missing for: {missing_park_hr}")

    df_merged['ParkHRRate'] = df_merged['ParkHRRate'].fillna(1.0)

    # --- Add BatterHandedness and PitcherHandedness columns to df_merged ---
    df_merged['BatterHandedness'] = df_merged['batter'].apply(
        lambda n: get_handedness(n)[0] if pd.notnull(n) else np.nan
    )
    df_merged['PitcherHandedness'] = df_merged['pitcher'].apply(
        lambda n: get_handedness(n)[1] if pd.notnull(n) else np.nan
    )

# --- Load and prepare handedness HR rate file ---
    handed_hr = load_and_standardize_handed_hr(handed_hr_file)
    df_merged = df_merged.merge(
        handed_hr,
        on=['BatterHandedness', 'PitcherHandedness'],
        how='left'
    )
    # Optional: Rename for clarity
    if 'hr_rate' in df_merged.columns:
        df_merged.rename(columns={'hr_rate': 'HandedHRRate'}, inplace=True)
    # Pitch Type HR Rate
    # --- Pitch Type HR Rate (robust assignment per matchup) ---
    pitchtype_hr = load_and_standardize_pitchtype_hr(pitchtype_hr_file)
    pitch_type_to_hr = dict(zip(
        pitchtype_hr['pitch_type'],
        pitchtype_hr['PitchTypeHRRate']
    ))
    def get_pitcher_primary_pitch(pitcher_id):
        try:
            row = pitcher_bb[pitcher_bb['pitcher_id'] == str(pitcher_id)]
            if not row.empty:
                pitch_cols = [col for col in row.columns if col.endswith('_pct')]
                if pitch_cols:
                    pitch_type = row[pitch_cols].iloc[0].idxmax().replace('_pct', '').upper()
                    return pitch_type
        except Exception as e:
            log_error("Primary pitch assignment", e)
        return 'FF'  # Fallback if unknown

    # <<--- Use it to fill PitchTypeHRRate
    df_merged['PitchTypeHRRate'] = df_merged['pitcher_id'].apply(
        lambda pid: pitch_type_to_hr.get(get_pitcher_primary_pitch(pid), 0)
    )
    # --- Load Logistic Weights (Logit) File ---
    logit_weights_dict = STATIC_LOGIT_WEIGHTS
    
    # --- Begin leaderboard row construction ---
    progress = st.progress(0)
    rows = []

    # --- Define key HR features and normalization functions (edit/expand as you want) ---
    feature_weights = {
        'B_BarrelRate_3': 0.12, 'B_BarrelRate_5': 0.09, 'B_BarrelRate_7': 0.07, 'B_BarrelRate_14': 0.05,
        'B_EV_3': 0.08, 'B_EV_5': 0.06, 'B_EV_7': 0.04, 'B_EV_14': 0.02,
        'B_xwoba_3': 0.08, 'B_xwoba_5': 0.06, 'B_xwoba_7': 0.04, 'B_xwoba_14': 0.02,
        'B_sweet_spot_pct_3': 0.06, 'B_sweet_spot_pct_5': 0.04, 'B_sweet_spot_pct_7': 0.03, 'B_sweet_spot_pct_14': 0.02,
        'B_hardhit_pct_3': 0.06, 'B_hardhit_pct_5': 0.04, 'B_hardhit_pct_7': 0.03, 'B_hardhit_pct_14': 0.02,
    }
    norm_funcs = {
        'B_BarrelRate_3': norm_barrel, 'B_BarrelRate_5': norm_barrel, 'B_BarrelRate_7': norm_barrel, 'B_BarrelRate_14': norm_barrel,
        'B_EV_3': norm_ev, 'B_EV_5': norm_ev, 'B_EV_7': norm_ev, 'B_EV_14': norm_ev,
        'B_xwoba_3': norm_xwoba, 'B_xwoba_5': norm_xwoba, 'B_xwoba_7': norm_xwoba, 'B_xwoba_14': norm_xwoba,
        'B_sweet_spot_pct_3': norm_sweetspot, 'B_sweet_spot_pct_5': norm_sweetspot, 'B_sweet_spot_pct_7': norm_sweetspot, 'B_sweet_spot_pct_14': norm_sweetspot,
        'B_hardhit_pct_3': norm_hardhit, 'B_hardhit_pct_5': norm_hardhit, 'B_hardhit_pct_7': norm_hardhit, 'B_hardhit_pct_14': norm_hardhit,
    }

    for idx, row in df_merged.iterrows():
        try:
            # Defensive weather logic
            city = str(row.get('city', '') or '').strip()
            date = str(row.get('date', '') or '').strip()
            parkorientation = str(row.get('parkorientation', '') or 'N').strip()
            time = str(row.get('time', '') or '14:00').strip()

            if not city or city.lower() in ["", "nan", "none"] or not date or date.lower() in ["", "nan", "none"]:
                weather = {"Temp": None, "Wind": None, "WindDir": None, "WindEffect": None, "Humidity": None, "Condition": None}
                log_error("Missing Weather Input", f"city={city}, date={date}, parkorientation={parkorientation}")
            else:
                weather = get_weather(city, date, parkorientation, time)
                if any(weather.get(k) is None for k in ["WindDir", "WindEffect", "Humidity"]):
                    log_error("Partial Weather Missing", f"{city} | {date} | {parkorientation} | {weather}")

            b_stats = get_batter_stats_multi(row['batter_id'])
            p_stats = get_pitcher_stats_multi(row['pitcher_id'])
            b_pitch_metrics = get_batter_pitch_metrics(row['batter_id'])
            p_pitch_metrics = get_pitcher_pitch_metrics(row['pitcher_id'])
            p_spin_metrics = get_pitcher_spin_metrics(row['pitcher_id'])
            b_bats, _ = get_handedness(row['batter'])
            _, p_throws = get_handedness(row['pitcher'])
            platoon_woba = get_platoon_woba(row['batter_id'], p_throws) if b_bats and p_throws else None
            pitch_mix = get_pitcher_pitch_mix(row['pitcher_id'])
            pitch_woba = get_batter_pitchtype_woba(row['batter_id'])
            pt_boost = calc_pitchtype_boost(pitch_woba, pitch_mix)
            record = row.to_dict()
            record.update(weather)
            record.update(b_stats)
            record.update(p_stats)
            record.update(b_pitch_metrics)
            record.update(p_pitch_metrics)
            record.update(p_spin_metrics)
            record['BatterHandedness'] = b_bats
            record['PitcherHandedness'] = p_throws
            record['PlatoonWoba'] = platoon_woba
            record['PitchMixBoost'] = pt_boost
            p_spin_metrics_30 = get_pitcher_spin_metrics(row['pitcher_id'], windows=[30])
            record.update(p_spin_metrics_30)
            record['HandedHRRate'] = row.get('HandedHRRate', np.nan)
            record['PitchTypeHRRate'] = row.get('PitchTypeHRRate', np.nan)

# --- Compute AnalyzerLogitScore using the correct features and weights ---
            analyzer_score = 0
            for feat, weight in logit_weights_dict.items():
                val = record.get(feat, 0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0
            record['HR_Score'] = compute_hr_score_logit(record, logit_weights_dict)
            # Optionally, create a simple data flag (for missing % of features)
            missing_count = sum(pd.isna(record.get(f, None)) or record.get(f, None) == '' for f in logit_weights_dict)
            total_feats = len(logit_weights_dict)
            record['DataFlag'] = "Low" if missing_count > total_feats * 0.4 else "OK"

            rows.append(record)

        except Exception as e:
            log_error(f"Row error ({row.get('batter','NA')} vs {row.get('pitcher','NA')})", e)

        progress.progress((idx + 1) / len(df_merged), text=f"Processing {int(100 * (idx + 1) / len(df_merged))}%")

    # --- Score & leaderboard construction ---
    df_final = pd.DataFrame(rows)
    df_final.reset_index(drop=True, inplace=True)
    df_final.insert(0, "rank", df_final.index + 1)
    df_final['BattedBallScore'] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final['PitcherBBScore'] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final['CustomBoost'] = df_final.apply(custom_2025_boost, axis=1)
    df_final['HR_Score_pctile'] = df_final['HR_Score'].rank(pct=True)
    df_final['HR_Tier'] = df_final['HR_Score'].apply(hr_score_tier)

    df_leaderboard = df_final.sort_values("HR_Score", ascending=False).reset_index(drop=True)
    df_leaderboard["rank"] = df_leaderboard.index + 1

    # --- Show the top leaderboard table, including DataFlag
    st.success("Leaderboard ready! Top HR Matchups below.")
    out_cols = [
        'rank', 'batter', 'pitcher', 'HR_Score', 'ParkHRRate', 'DataFlag'
    ] + [c for c in df_leaderboard.columns if c not in [
        'rank', 'batter', 'pitcher', 'HR_Score', 'ParkHRRate', 'DataFlag'
    ]]

    st.dataframe(df_leaderboard[out_cols].head(20), use_container_width=True)
else:
    st.info("📂 Upload all 7 files to generate the leaderboard.")
