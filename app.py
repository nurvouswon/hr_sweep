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
logit_weights_file = st.file_uploader("8️⃣ Custom Logistic Weights CSV", type=["csv"])

# --- All files required for full Analyzer run
csvs_uploaded = [
    lineup_file, xhr_file, battedball_file, pitcher_battedball_file,
    handed_hr_file, pitchtype_hr_file, park_hr_file, logit_weights_file
]
all_files_uploaded = all(csvs_uploaded)

if all_files_uploaded:
    # --- Load all files with robust normalization
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

    # --- Merge xHR regression data
    xhr_df = pd.read_csv(xhr_file)
    xhr_df.columns = xhr_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)
    df_upload['norm_batter'] = df_upload['batter'].apply(normalize_name)
    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='norm_batter', right_on='player_norm', how='left'
    )
    df_merged['park'] = df_merged['park'].str.strip().str.lower().str.replace(' ', '_')
    df_merged['parkfactor'] = df_merged['park'].map(park_factors)
    df_merged['parkorientation'] = df_merged['park'].map(ballpark_orientations)

    # --- Merge batted ball profiles (batter & pitcher)
    batted = pd.read_csv(battedball_file)
    batted.columns = batted.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    batted = batted.rename(columns={"id": "batter_id"})
    df_merged['batter_id'] = df_merged['batter_id'].astype(str)
    batted['batter_id'] = batted['batter_id'].astype(str)
    df_merged = df_merged.merge(batted, on="batter_id", how="left")

    pitcher_bb = pd.read_csv(pitcher_battedball_file)
    pitcher_bb.columns = pitcher_bb.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    pitcher_bb = pitcher_bb.rename(columns={"id": "pitcher_id", 'bbe': 'bbe_pbb'})
    pitcher_bb = pitcher_bb.rename(columns={c: f"{c}_pbb" for c in pitcher_bb.columns if c not in ['pitcher_id', 'name_pbb']})
    df_merged['pitcher_id'] = df_merged['pitcher_id'].astype(str)
    pitcher_bb['pitcher_id'] = pitcher_bb['pitcher_id'].astype(str)
    df_merged = df_merged.merge(pitcher_bb, on="pitcher_id", how="left")

    # --- Merge extra Analyzer CSVs (Handedness, Pitch Type, Park HR rates, Logit weights)
    handed_hr = pd.read_csv(handed_hr_file)
    handed_hr.columns = handed_hr.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    pitchtype_hr = pd.read_csv(pitchtype_hr_file)
    pitchtype_hr.columns = pitchtype_hr.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    park_hr = pd.read_csv(park_hr_file)
    park_hr.columns = park_hr.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    logit_weights = pd.read_csv(logit_weights_file)
    logit_weights.columns = logit_weights.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

    # --- Apply and store HR rate and logistic weights columns
    # (If columns don't match, fillna, skip, or warn)
    if 'hr_rate' in handed_hr.columns:
        df_merged = df_merged.merge(
            handed_hr[['batter_id', 'pitcher_hand', 'hr_rate']],
            left_on=['batter_id', 'pitcher'],
            right_on=['batter_id', 'pitcher_hand'], how='left', suffixes=('', '_handed')
        )
    if 'hr_rate' in pitchtype_hr.columns and 'pitch_type' in pitchtype_hr.columns:
        df_merged = df_merged.merge(
            pitchtype_hr[['batter_id', 'pitch_type', 'hr_rate']],
            left_on=['batter_id'],
            right_on=['batter_id'], how='left', suffixes=('', '_pitch')
        )
    if 'hr_rate' in park_hr.columns and 'park' in park_hr.columns:
        df_merged = df_merged.merge(
            park_hr[['park', 'hr_rate']],
            left_on=['park'],
            right_on=['park'], how='left', suffixes=('', '_park')
        )
    # Save logistic weights as a dict for feature blending
    # Auto-detect feature and weight columns from CSV
logit_weights_dict = {}
if len(logit_weights.columns) >= 2:
    feature_col = logit_weights.columns[0]
    weight_col = logit_weights.columns[1]
    for _, row in logit_weights.iterrows():
        feature = row.get(feature_col)
        weight = row.get(weight_col, 1.0)
        if pd.notna(feature):
            logit_weights_dict[feature] = weight

    # --- Prepare leaderboard construction
    progress = st.progress(0)
    rows = []
    for idx, row in df_merged.iterrows():
        try:
            weather = get_weather(row.get('city',''), row.get('date',''), row.get('parkorientation',''), row.get('time',''))
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
            # Include analyzer CSV extra rates if available
            record['HandedHRRate'] = row.get('hr_rate', np.nan)
            record['PitchTypeHRRate'] = row.get('hr_rate_pitch', np.nan)
            record['ParkHRRate'] = row.get('hr_rate_park', np.nan)
            # Custom logistic scoring if weights given
            analyzer_score = 0
            for feat, weight in logit_weights_dict.items():
                analyzer_score += (record.get(feat, 0) or 0) * float(weight)
            record['AnalyzerLogitScore'] = analyzer_score
            rows.append(record)
        except Exception as e:
            log_error(f"Row error ({row.get('batter','NA')} vs {row.get('pitcher','NA')})", e)
        progress.progress((idx + 1) / len(df_merged), text=f"Processing {int(100 * (idx + 1) / len(df_merged))}%")
    df_final = pd.DataFrame(rows)

    # Add scores
if all_files_uploaded:
    df_final.reset_index(drop=True, inplace=True)
    df_final.insert(0, "rank", df_final.index + 1)
    df_final['BattedBallScore'] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final['PitcherBBScore'] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final['CustomBoost'] = df_final.apply(custom_2025_boost, axis=1)
    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_final['HR_Score_pctile'] = df_final['HR_Score'].rank(pct=True)
    df_final['HR_Tier'] = df_final['HR_Score'].apply(hr_score_tier)
    # Save both Analyzer and default model columns
    df_final['Analyzer_Blend'] = (
        0.60 * df_final['HR_Score'] +
        0.30 * df_final.get('AnalyzerLogitScore', 0) +
        0.05 * df_final.get('HandedHRRate', 0) +
        0.05 * df_final.get('PitchTypeHRRate', 0)
    )

    importances = None  # Set this if you run ML, else leave as None

    # Leaderboard sort and ranking by blended score (Analyzer_Blend)
    df_leaderboard = df_final.sort_values("Analyzer_Blend", ascending=False).reset_index(drop=True)
    df_leaderboard["rank"] = df_leaderboard.index + 1

    # Optionally display or use feature importances if you have them:
    if importances is not None:
        st.write("Feature importances:", importances)

else:
    st.info("Upload all 8 files to generate the leaderboard.")
