# ===================== CHUNK 1: Imports, Config, Logging, Stat Functions ======================

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

# ========= Error Logging =========
def log_error(context, exception, level="ERROR"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] [{level}] {context}: {exception}"
    error_log.append(msg)
    print(msg)

# ========= Caching for APIs =========
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

# ========= Name & Handedness Helpers =========
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

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

# ========= Ballpark Dictionaries & Helpers =========
ballpark_orientations = {
    "sutter_health_park": "NE", "yankee_stadium": "N", "fenway_park": "N", "tropicana_field": "N",
    "camden_yards": "NE", "rogers_centre": "NE", "comerica_park": "N", "progressive_field": "NE",
    "target_field": "N", "kauffman_stadium": "NE", "guaranteed_rate_field": "NE", "angel_stadium": "NE",
    "minute_maid_park": "N", "oakland_coliseum": "N", "t-mobile_park": "N", "globe_life_field": "NE",
    "dodger_stadium": "NE", "chase_field": "N", "coors_field": "N", "oracle_park": "E", "wrigley_field": "NE",
    "great_american_ball_park": "N", "american_family_field": "NE", "pnc_park": "NE", "busch_stadium": "NE",
    "truist_park": "N", "loandepot_park": "N", "citi_field": "N", "nationals_park": "NE", "petco_park": "N",
    "citizens_bank_park": "NE"
}
park_factors = {
    "sutter_health_park": 1.12, "yankee_stadium": 1.19, "fenway_park": 0.97, "tropicana_field": 0.85,
    "camden_yards": 1.13, "rogers_centre": 1.10, "comerica_park": 0.96, "progressive_field": 1.01,
    "target_field": 1.04, "kauffman_stadium": 0.98, "guaranteed_rate_field": 1.18, "angel_stadium": 1.05,
    "minute_maid_park": 1.06, "oakland_coliseum": 0.82, "t-mobile_park": 0.86, "globe_life_field": 1.00,
    "dodger_stadium": 1.10, "chase_field": 1.06, "coors_field": 1.30, "oracle_park": 0.82, "wrigley_field": 1.12,
    "great_american_ball_park": 1.26, "american_family_field": 1.17, "pnc_park": 0.87, "busch_stadium": 0.87,
    "truist_park": 1.06, "loandepot_park": 0.86, "citi_field": 1.05, "nationals_park": 1.05, "petco_park": 0.85,
    "citizens_bank_park": 1.19
}
compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

def get_compass_idx(dir_str):
    dir_str = dir_str.upper()
    try: return compass.index(dir_str)
    except: return -1

def is_wind_out(wind_dir, park_orientation):
    if not isinstance(wind_dir, str) or not isinstance(park_orientation, str):
        return "unknown"
    wi = get_compass_idx(wind_dir)
    pi = get_compass_idx(park_orientation)
    if wi == -1 or pi == -1: return "unknown"
    if abs(wi - pi) <= 1 or abs(wi - pi) >= 7: return "out"
    elif abs(wi - pi) == 4: return "in"
    else: return "side"

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
        wind_dir = wind_dir_full if wind_dir_full in compass else wind_dir_full[:2]
        humidity = weather_hour.get('humidity')
        condition = weather_hour.get('condition', {}).get('text')
        wind_effect = is_wind_out(wind_dir, park_orientation)
        if wind_effect == "unknown":
            log_error("Wind Effect Debug", f"wind_dir={wind_dir}, orientation={park_orientation}")
        return {
            "Temp": temp, "Wind": wind, "WindDir": wind_dir, "WindEffect": wind_effect,
            "Humidity": humidity, "Condition": condition
        }
    except Exception as e:
        log_error("Weather", e)
        return {"Temp": None, "Wind": None, "WindDir": None, "WindEffect": None, "Humidity": None, "Condition": None}

# ========== Normalization Functions ==========
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
def norm_xwoba(x): 
    return max(0, min((x - 0.250) / (0.400 - 0.250), 1)) if pd.notnull(x) else 0
def norm_sweetspot(x):
    return max(0, min((x - 0.25) / (0.45 - 0.25), 1)) if pd.notnull(x) else 0
def norm_hardhit(x):
    return max(0, min((x - 0.25) / (0.60 - 0.25), 1)) if pd.notnull(x) else 0
def norm_whiff(x):
    return max(0, min((x - 0.15) / (0.40 - 0.15), 1)) if pd.notnull(x) else 0

# All core Statcast feature/stat functions follow in CHUNK 2!

# ===================== CHUNK 2: Statcast, Batted Ball, Boosts, HR Score, Model ======================

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

# --- Statcast Feature Functions: Pitcher ---
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

# --- Statcast Whiff, Spin, Pitch Mix, Platoon, Pitchtype Functions ---
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

# --- Custom 2025 Boosts, Batted Ball Profile Score, HR Score, Model Trainer ---
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

# ========== Batted Ball Profile Scores ==========
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

# ========== Robust, Advanced HR Scoring ==========
def calc_hr_score(row):
    batter_score = (
        norm_barrel(row.get('B_BarrelRate_3')) * 0.12 +
        norm_barrel(row.get('B_BarrelRate_5')) * 0.09 +
        norm_barrel(row.get('B_BarrelRate_7')) * 0.07 +
        norm_barrel(row.get('B_BarrelRate_14')) * 0.05 +

        norm_ev(row.get('B_EV_3')) * 0.08 +
        norm_ev(row.get('B_EV_5')) * 0.06 +
        norm_ev(row.get('B_EV_7')) * 0.04 +
        norm_ev(row.get('B_EV_14')) * 0.02 +

        norm_xwoba(row.get('B_xwoba_3')) * 0.08 +
        norm_xwoba(row.get('B_xwoba_5')) * 0.06 +
        norm_xwoba(row.get('B_xwoba_7')) * 0.04 +
        norm_xwoba(row.get('B_xwoba_14')) * 0.02 +

        norm_sweetspot(row.get('B_sweet_spot_pct_3')) * 0.06 +
        norm_sweetspot(row.get('B_sweet_spot_pct_5')) * 0.04 +
        norm_sweetspot(row.get('B_sweet_spot_pct_7')) * 0.03 +
        norm_sweetspot(row.get('B_sweet_spot_pct_14')) * 0.02 +

        norm_hardhit(row.get('B_hardhit_pct_3')) * 0.06 +
        norm_hardhit(row.get('B_hardhit_pct_5')) * 0.04 +
        norm_hardhit(row.get('B_hardhit_pct_7')) * 0.03 +
        norm_hardhit(row.get('B_hardhit_pct_14')) * 0.02 +

        (1 - norm_whiff(row.get('B_WhiffRate_3'))) * 0.06 +
        (1 - norm_whiff(row.get('B_WhiffRate_5'))) * 0.05 +
        (1 - norm_whiff(row.get('B_WhiffRate_7'))) * 0.04 +
        (1 - norm_whiff(row.get('B_WhiffRate_14'))) * 0.02 +

        (row.get('B_SLG_3') or 0) * 0.08 +
        (row.get('B_SLG_14') or 0) * 0.05 +
        (row.get('B_xSLG_3') or 0) * 0.08 +
        (row.get('B_xSLG_14') or 0) * 0.05 +
        (row.get('B_xISO_3') or 0) * 0.04 +
        (row.get('B_xISO_14') or 0) * 0.03
    )

    pitcher_score = (
        norm_barrel(row.get('P_BarrelRateAllowed_3')) * 0.07 +
        norm_barrel(row.get('P_BarrelRateAllowed_5')) * 0.05 +
        norm_barrel(row.get('P_BarrelRateAllowed_7')) * 0.03 +
        norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.02 +

        norm_ev(row.get('P_EVAllowed_3')) * 0.05 +
        norm_ev(row.get('P_EVAllowed_5')) * 0.03 +
        norm_ev(row.get('P_EVAllowed_7')) * 0.02 +
        norm_ev(row.get('P_EVAllowed_14')) * 0.01 +

        norm_xwoba(row.get('P_xwoba_3')) * -0.05 +
        norm_xwoba(row.get('P_xwoba_5')) * -0.03 +
        norm_xwoba(row.get('P_xwoba_7')) * -0.02 +
        norm_xwoba(row.get('P_xwoba_14')) * -0.01 +

        norm_sweetspot(row.get('P_sweet_spot_pct_3')) * -0.03 +
        norm_sweetspot(row.get('P_sweet_spot_pct_5')) * -0.02 +
        norm_sweetspot(row.get('P_sweet_spot_pct_7')) * -0.01 +
        norm_sweetspot(row.get('P_sweet_spot_pct_14')) * -0.01 +

        norm_hardhit(row.get('P_hardhit_pct_3')) * -0.03 +
        norm_hardhit(row.get('P_hardhit_pct_5')) * -0.02 +
        norm_hardhit(row.get('P_hardhit_pct_7')) * -0.01 +
        norm_hardhit(row.get('P_hardhit_pct_14')) * -0.01 +

        norm_whiff(row.get('P_WhiffRate_3')) * -0.01 +
        norm_whiff(row.get('P_WhiffRate_5')) * -0.008 +
        norm_whiff(row.get('P_WhiffRate_7')) * -0.006 +
        norm_whiff(row.get('P_WhiffRate_14')) * -0.004 +

        (row.get('P_SLG_3') or 0) * -0.07 +
        (row.get('P_SLG_14') or 0) * -0.04 +
        (row.get('P_xSLG_3') or 0) * -0.07 +
        (row.get('P_xSLG_14') or 0) * -0.04 +
        (row.get('P_xISO_3') or 0) * -0.04 +
        (row.get('P_xISO_14') or 0) * -0.03
    )

    spin_drop = 0
    try:
        spin_3 = row.get('P_FF_Spin_3')
        spin_30 = row.get('P_FF_Spin_30')
        if spin_3 and spin_30 and (spin_30 - spin_3) >= 100:
            spin_drop = 0.01
    except:
        pass

    park_score = norm_park(row.get('parkfactor', 1.0)) * 0.10
    weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.15
    regression_score = max(0, min((row.get('xhr_diff', 0) or 0) / 5, 0.12))
    platoon_score = ((row.get('PlatoonWoba') or 0.320) - 0.320) * 0.10
    pitchtype_boost = row.get('PitchMixBoost', 0)
    batted_ball_score = row.get('BattedBallScore', 0)
    pitcher_bb_score = row.get('PitcherBBScore', 0)
    custom_boost = custom_2025_boost(row)

    return round(
        batter_score + pitcher_score + spin_drop + park_score + weather_score +
        regression_score + batted_ball_score + pitcher_bb_score +
        platoon_score + pitchtype_boost + custom_boost,
        3
    )

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

    # ========== Streamlit UI Section ==========

st.title("⚾ MLB HR Matchup Leaderboard – Analyzer Integration + Statcast + ML")
st.markdown("""
Upload these 8 CSV files for full feature blending:
1. Lineups/Matchups
2. xHR/HR Regression
3. Batter Batted-Ball Profile
4. Pitcher Batted-Ball Profile
5. Analyzer Batted-Ball Profile (Batter)
6. Analyzer Batted-Ball Profile (Pitcher)
7. Logistic Regression Feature Weights (Analyzer)
8. Analyzer HR Rates (Handedness & Pitch Type)
""")

lineup_file = st.file_uploader("1️⃣ Lineups/Matchups CSV", type=["csv"])
xhr_file = st.file_uploader("2️⃣ xHR / HR Regression CSV", type=["csv"])
battedball_file = st.file_uploader("3️⃣ Batter batted-ball CSV", type=["csv"])
pitcher_battedball_file = st.file_uploader("4️⃣ Pitcher batted-ball CSV", type=["csv"])
an_battedball_file = st.file_uploader("5️⃣ Analyzer Batter Profile CSV", type=["csv"])
an_pitcherbb_file = st.file_uploader("6️⃣ Analyzer Pitcher Profile CSV", type=["csv"])
logit_weights_file = st.file_uploader("7️⃣ Logistic Regression Feature Weights CSV", type=["csv"])
hr_rates_file = st.file_uploader("8️⃣ Analyzer HR Rates CSV", type=["csv"])

# --- Logistic Weights Dict ---
logit_weights_dict = {}
if logit_weights_file is not None:
    df_logit_weights = pd.read_csv(logit_weights_file)
    feature_col = df_logit_weights.columns[0]
    weight_col = df_logit_weights.columns[1]
    for _, row in df_logit_weights.iterrows():
        f = row[feature_col]
        w = row[weight_col] if pd.notnull(row[weight_col]) else 1.0
        logit_weights_dict[f] = w

# --- Proceed when all 8 CSVs are uploaded ---
if (lineup_file and xhr_file and battedball_file and pitcher_battedball_file and
    an_battedball_file and an_pitcherbb_file and logit_weights_file and hr_rates_file):

    # Load dataframes
    df_lineup = pd.read_csv(lineup_file)
    df_xhr = pd.read_csv(xhr_file)
    df_battedball = pd.read_csv(battedball_file)
    df_pitcherbb = pd.read_csv(pitcher_battedball_file)
    df_an_battedball = pd.read_csv(an_battedball_file)
    df_an_pitcherbb = pd.read_csv(an_pitcherbb_file)
    df_hr_rates = pd.read_csv(hr_rates_file)

    # --- Column Normalization for main input ---
    df_lineup.columns = (
        df_lineup.columns
        .str.strip().str.lower().str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    # Example renames (edit as per your actual column names)
    df_lineup.rename(columns={
        'mlb_id': 'batter_id',
        'player_name': 'batter',
        'team_code': 'team_code',
        'game_date': 'date',
        'batting_order': 'batting_order',
        'confirmed': 'confirmed',
        'city': 'city',
        'park': 'park',
        'time': 'time'
    }, inplace=True)

    # Filter confirmed batters
    df_lineup = df_lineup[df_lineup['confirmed'].astype(str).str.lower() == 'y']

    # --- Add pitcher columns ---
    pitcher_rows = df_lineup[df_lineup['batting_order'].astype(str).str.lower() == 'sp']
    team_pitcher_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter_id']))
    pitcher_name_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter']))
    df_lineup['pitcher_id'] = df_lineup['team_code'].map(team_pitcher_map)
    df_lineup['pitcher'] = df_lineup['team_code'].map(pitcher_name_map)

    # --- Merge xHR regression ---
    df_xhr.columns = df_xhr.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    df_xhr['player_norm'] = df_xhr['player'].apply(normalize_name)
    df_lineup['norm_batter'] = df_lineup['batter'].apply(normalize_name)
    df_merged = df_lineup.merge(
        df_xhr[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='norm_batter', right_on='player_norm', how='left'
    )

    # --- Merge Statcast and Batted Ball CSVs ---
    df_merged['park'] = df_merged['park'].str.strip().str.lower().str.replace(' ', '_')
    df_merged['batter_id'] = df_merged['batter_id'].astype(str)
    df_merged['pitcher_id'] = df_merged['pitcher_id'].astype(str)
    df_battedball['batter_id'] = df_battedball['batter_id'].astype(str)
    df_pitcherbb['pitcher_id'] = df_pitcherbb['pitcher_id'].astype(str)

    # Merge in standard and analyzer profiles
    df_merged = df_merged.merge(df_battedball, on="batter_id", how="left", suffixes=('', '_bb'))
    df_merged = df_merged.merge(df_pitcherbb, on="pitcher_id", how="left", suffixes=('', '_pbb'))
    df_an_battedball['batter_id'] = df_an_battedball['batter_id'].astype(str)
    df_merged = df_merged.merge(df_an_battedball, on="batter_id", how="left", suffixes=('', '_anb'))
    df_an_pitcherbb['pitcher_id'] = df_an_pitcherbb['pitcher_id'].astype(str)
    df_merged = df_merged.merge(df_an_pitcherbb, on="pitcher_id", how="left", suffixes=('', '_anp'))

    # --- Merge HR Rates ---
    hr_rate_cols = df_hr_rates.columns.str.lower().tolist()
    if 'batter_id' in hr_rate_cols:
        df_hr_rates['batter_id'] = df_hr_rates['batter_id'].astype(str)
        df_merged = df_merged.merge(df_hr_rates, on='batter_id', how='left', suffixes=('', '_hrate'))

    # --- Logistic Regression Blended Score ---
    analyzer_features = [c for c in df_an_battedball.columns if c in logit_weights_dict]
    if analyzer_features:
        df_merged['AnalyzerLogitScore'] = (
            df_merged[analyzer_features].fillna(0) * pd.Series({k: logit_weights_dict.get(k, 1.0) for k in analyzer_features})
        ).sum(axis=1)
    else:
        df_merged['AnalyzerLogitScore'] = 0

    # --- HR Rates (Handed/PitchType) ---
    df_merged['HandedHRRate'] = df_merged['HandedHRRate'] if 'HandedHRRate' in df_merged.columns else 0
    df_merged['PitchTypeHRRate'] = df_merged['PitchTypeHRRate'] if 'PitchTypeHRRate' in df_merged.columns else 0

    # --- Calculate All Scores ---
    df_merged['HR_Score'] = df_merged.apply(calc_hr_score, axis=1)
    df_merged['Analyzer_Blend'] = (
        0.60 * df_merged['HR_Score'] +
        0.30 * df_merged['AnalyzerLogitScore'] +
        0.05 * df_merged['HandedHRRate'] +
        0.05 * df_merged['PitchTypeHRRate']
    )
    df_merged['HR_Score_pctile'] = df_merged['Analyzer_Blend'].rank(pct=True)
    df_merged['HR_Tier'] = df_merged['Analyzer_Blend'].apply(hr_score_tier)
    df_merged.reset_index(drop=True, inplace=True)
    df_merged['rank'] = df_merged['Analyzer_Blend'].rank(method='min', ascending=False).astype(int)

    # --- Output ---
    leaderboard_cols = [
        'rank', 'batter', 'pitcher', 'Analyzer_Blend', 'HR_Tier', 'HR_Score_pctile',
        'xhr_diff', 'xhr', 'hr_total', 'park', 'city', 'time',
        'HR_Score', 'AnalyzerLogitScore', 'HandedHRRate', 'PitchTypeHRRate'
    ] + [c for c in analyzer_features if c in df_merged.columns]
    leaderboard_cols = [c for c in leaderboard_cols if c in df_merged.columns]

    st.success("Leaderboard ready! Top Matchups:")
    st.dataframe(df_merged[leaderboard_cols].sort_values('Analyzer_Blend', ascending=False).head(15), use_container_width=True)
    st.subheader("Top 5 HR Scores")
    st.bar_chart(df_merged.set_index('batter')[['Analyzer_Blend']].head(5))
    csv_bytes = df_merged.sort_values('Analyzer_Blend', ascending=False).to_csv(index=False).encode()
    st.download_button("Download Full Leaderboard as CSV", csv_bytes, file_name="hr_leaderboard.csv")
    if error_log:
        with st.expander("⚠️ Errors and Warnings (Click to expand)"):
            for e in error_log[-30:]:
                st.text(e)
else:
    st.info("Upload all 8 files to generate the leaderboard with analyzer integration.")
