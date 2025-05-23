# MLB HR Matchup Leaderboard – Fully Enhanced (May 2025)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import unicodedata
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
import difflib

API_KEY = st.secrets["weather"]["api_key"]
error_log = []

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

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

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

def get_batter_stats_multi(batter_id, windows=[3,5,7,14]):
    out = {}
    if not batter_id:
        for w in windows:
            out[f"B_BarrelRate_{w}"] = None
            out[f"B_EV_{w}"] = None
            out[f"B_SLG_{w}"] = None
            out[f"B_xSLG_{w}"] = None
        return out
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_batter(start, end, batter_id)
            df = df[df['type'] == 'X']
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            ev = df['launch_speed'].mean() if len(df) > 0 else None
            singles = df[df['events'] == 'single'].shape[0]
            doubles = df[df['events'] == 'double'].shape[0]
            triples = df[df['events'] == 'triple'].shape[0]
            hrs = df[df['events'] == 'home_run'].shape[0]
            outs = df[df['events'].isin(['field_out','force_out','other_out'])].shape[0]
            ab = singles + doubles + triples + hrs + outs
            slg = (singles + 2*doubles + 3*triples + 4*hrs) / ab if ab > 0 else None
            xslg = df['estimated_slg'].mean() if 'estimated_slg' in df else None
            pa = df.shape[0]
            out[f"B_BarrelRate_{w}"] = round(barrels / pa, 3) if pa else None
            out[f"B_EV_{w}"] = round(ev, 1) if ev else None
            out[f"B_SLG_{w}"] = round(slg, 3) if slg else None
            out[f"B_xSLG_{w}"] = round(xslg, 3) if xslg is not None else None
        except Exception as e:
            error_log.append(f"Batter stat error ({batter_id}, {w}d): {e}")
    return out

def get_pitcher_stats_multi(pitcher_id, windows=[3,5,7,14]):
    out = {}
    if not pitcher_id:
        for w in windows:
            out[f"P_BarrelRateAllowed_{w}"] = None
            out[f"P_EVAllowed_{w}"] = None
            out[f"P_SLG_{w}"] = None
            out[f"P_xSLG_{w}"] = None
        return out
    for w in windows:
        try:
            start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
            end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = cached_statcast_pitcher(start, end, pitcher_id)
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            ev = df['launch_speed'].mean() if len(df) > 0 else None
            singles = df[df['events'] == 'single'].shape[0]
            doubles = df[df['events'] == 'double'].shape[0]
            triples = df[df['events'] == 'triple'].shape[0]
            hrs = df[df['events'] == 'home_run'].shape[0]
            outs = df[df['events'].isin(['field_out','force_out','other_out'])].shape[0]
            ab = singles + doubles + triples + hrs + outs
            slg = (singles + 2*doubles + 3*triples + 4*hrs) / ab if ab > 0 else None
            xslg = df['estimated_slg'].mean() if 'estimated_slg' in df else None
            total = len(df)
            out[f"P_BarrelRateAllowed_{w}"] = round(barrels / total, 3) if total else None
            out[f"P_EVAllowed_{w}"] = round(ev, 1) if ev else None
            out[f"P_SLG_{w}"] = round(slg, 3) if slg else None
            out[f"P_xSLG_{w}"] = round(xslg, 3) if xslg is not None else None
        except Exception as e:
            error_log.append(f"Pitcher stat error ({pitcher_id}, {w}d): {e}")
    return out

def get_batter_advanced_stats(batter_id, window=14):
    out = {}
    if not batter_id:
        for k in ['B_xwoba_14','B_sweet_spot_pct_14','B_pull_pct_14','B_oppo_pct_14','B_gbfb_14','B_hardhit_pct_14']:
            out[k] = None
        return out
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_batter(start, end, batter_id)
        df = df[df['type'] == 'X']
        xwoba = df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in df else None
        sweet = df['launch_angle'].between(8, 32).mean() if 'launch_angle' in df else None
        pull_pct, oppo_pct = None, None
        if 'hit_location' in df:
            pull_pct = (df['hit_location'] <= 3).mean()
            oppo_pct = (df['hit_location'] >= 8).mean()
        gb = (df['bb_type'] == 'ground_ball').sum() if 'bb_type' in df else 0
        fb = (df['bb_type'] == 'fly_ball').sum() if 'bb_type' in df else 0
        gbfb = gb / fb if fb > 0 else None
        hard = (df['launch_speed'] >= 95).mean() if 'launch_speed' in df else None
        out['B_xwoba_14'] = round(xwoba, 3) if xwoba is not None else None
        out['B_sweet_spot_pct_14'] = round(100 * sweet, 1) if sweet is not None else None
        out['B_pull_pct_14'] = round(100 * pull_pct, 1) if pull_pct is not None else None
        out['B_oppo_pct_14'] = round(100 * oppo_pct, 1) if oppo_pct is not None else None
        out['B_gbfb_14'] = round(gbfb, 2) if gbfb is not None else None
        out['B_hardhit_pct_14'] = round(100 * hard, 1) if hard is not None else None
        return out
    except Exception as e:
        error_log.append(f"Batter advanced stat error ({batter_id}): {e}")
        return out

def get_pitcher_advanced_stats(pitcher_id, window=14):
    out = {}
    if not pitcher_id:
        for k in ['P_xwoba_14','P_sweet_spot_pct_14','P_pull_pct_14','P_oppo_pct_14','P_gbfb_14','P_hardhit_pct_14']:
            out[k] = None
        return out
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_pitcher(start, end, pitcher_id)
        df = df[df['type'] == 'X']
        xwoba = df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in df else None
        sweet = df['launch_angle'].between(8, 32).mean() if 'launch_angle' in df else None
        pull_pct, oppo_pct = None, None
        if 'hit_location' in df:
            pull_pct = (df['hit_location'] <= 3).mean()
            oppo_pct = (df['hit_location'] >= 8).mean()
        gb = (df['bb_type'] == 'ground_ball').sum() if 'bb_type' in df else 0
        fb = (df['bb_type'] == 'fly_ball').sum() if 'bb_type' in df else 0
        gbfb = gb / fb if fb > 0 else None
        hard = (df['launch_speed'] >= 95).mean() if 'launch_speed' in df else None
        out['P_xwoba_14'] = round(xwoba, 3) if xwoba is not None else None
        out['P_sweet_spot_pct_14'] = round(100 * sweet, 1) if sweet is not None else None
        out['P_pull_pct_14'] = round(100 * pull_pct, 1) if pull_pct is not None else None
        out['P_oppo_pct_14'] = round(100 * oppo_pct, 1) if oppo_pct is not None else None
        out['P_gbfb_14'] = round(gbfb, 2) if gbfb is not None else None
        out['P_hardhit_pct_14'] = round(100 * hard, 1) if hard is not None else None
        return out
    except Exception as e:
        error_log.append(f"Pitcher advanced stat error ({pitcher_id}): {e}")
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
        error_log.append(f"Pitch mix error: {e}")
        return {}

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

def get_platoon_woba(batter_id, pitcher_hand, days=365):
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        df = cached_statcast_batter(start, end, batter_id)
        df = df[df['p_throws'] == pitcher_hand]
        return df['woba_value'].mean() if not df.empty else None
    except Exception as e:
        error_log.append(f"Platoon wOBA error: {e}")
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
        error_log.append(f"Pitch type matchup boost error: {e}")
        return 0

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
        (row.get('B_xSLG_14') or 0) * 0.09 +
        (row.get('B_xwoba_14') or 0) * 0.07 +
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
        -(row.get('P_xSLG_14') or 0) * 0.09 +
        -(row.get('P_xwoba_14') or 0) * 0.07 +
        -(row.get('P_sweet_spot_pct_14') or 0) * 0.02 +
        -(row.get('P_pull_pct_14') or 0) * 0.01 +
        -(row.get('P_oppo_pct_14') or 0) * 0.01 +
        -(row.get('P_gbfb_14') or 0) * 0.01 +
        -(row.get('P_hardhit_pct_14') or 0) * 0.02
    )
    park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.10
    weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.15
    regression_score = max(0, min((row.get('xhr_diff', 0) or 0) / 5, 0.12))
    platoon_score = ((row.get('PlatoonWoba') or 0.320) - 0.320) * 0.1
    pitchtype_boost = row.get("PitchMixBoost", 0)
    return round(
        batter_score + pitcher_score + park_score + weather_score +
        regression_score + row.get('BattedBallScore', 0) + row.get('PitcherBBScore', 0) +
        platoon_score + pitchtype_boost + custom_2025_boost(row),
        3
    )

# === Streamlit UI Setup ===
st.title("⚾ MLB HR Matchup Leaderboard – Advanced Statcast Scoring")
st.markdown("""
Upload the following 4 CSV files:
- **Matchups**: Batter, Pitcher, City, Park, Date, Time
- **xHR/HR Regression**: player, hr_total, xhr, xhr_diff
- **Batter Batted-Ball Profile** (with `id`)
- **Pitcher Batted-Ball Profile** (with `id`)
""")

uploaded_file = st.file_uploader("Matchups CSV", type=["csv"])
xhr_file = st.file_uploader("xHR / HR Regression CSV", type=["csv"])
battedball_file = st.file_uploader("Batter batted-ball CSV", type=["csv"])
pitcher_battedball_file = st.file_uploader("Pitcher batted-ball CSV", type=["csv"])

if uploaded_file and xhr_file and battedball_file and pitcher_battedball_file:
    df_upload = pd.read_csv(uploaded_file)
    for col in ['Batter', 'Pitcher', 'City', 'Park', 'Date', 'Time']:
        if col not in df_upload.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    xhr_df = pd.read_csv(xhr_file)
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)

    df_upload['norm_batter'] = df_upload['Batter'].apply(normalize_name)
    df_upload['batter_id'] = df_upload['Batter'].apply(get_player_id)
    df_upload['pitcher_id'] = df_upload['Pitcher'].apply(get_player_id)

    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='norm_batter', right_on='player_norm', how='left'
    )
    df_merged['ParkFactor'] = df_merged['Park'].map(park_factors)
    df_merged['ParkOrientation'] = df_merged['Park'].map(ballpark_orientations)

    progress = st.progress(0, text="Processing 0%")
    rows = []

    for idx, row in df_merged.iterrows():
        try:
            weather = get_weather(row['City'], row['Date'], row['ParkOrientation'], row['Time'])
            b_stats = get_batter_stats_multi(row['batter_id'])
            p_stats = get_pitcher_stats_multi(row['pitcher_id'])
            b_adv = get_batter_advanced_stats(row['batter_id'])
            p_adv = get_pitcher_advanced_stats(row['pitcher_id'])
            b_bats, _ = get_handedness(row['Batter'])
            _, p_throws = get_handedness(row['Pitcher'])
            platoon_woba = get_platoon_woba(row['batter_id'], p_throws) if b_bats and p_throws else None
            pitch_mix = get_pitcher_pitch_mix(row['pitcher_id'])
            pitch_woba = get_batter_pitchtype_woba(row['batter_id'])
            pt_boost = calc_pitchtype_boost(pitch_woba, pitch_mix)
            record = row.to_dict()
            record.update(weather)
            record.update(b_stats)
            record.update(p_stats)
            record.update(b_adv)
            record.update(p_adv)
            record['PlatoonWoba'] = platoon_woba
            record['PitchMixBoost'] = pt_boost
            rows.append(record)
        except Exception as e:
            error_log.append(f"Row error ({row['Batter']} vs {row['Pitcher']}): {e}")
        percent = int(100 * (idx + 1) / len(df_merged))
        progress.progress((idx + 1) / len(df_merged), text=f"Processing {percent}%")

    df_final = pd.DataFrame(rows)

    # Merge batted ball CSVs
    batted = pd.read_csv(battedball_file).rename(columns={"id": "batter_id"})
    df_final = df_final.merge(batted, on="batter_id", how="left")

    pitcher_bb = pd.read_csv(pitcher_battedball_file).rename(columns={"id": "pitcher_id", 'bbe': 'bbe_pbb'})
    pitcher_bb = pitcher_bb.rename(columns={c: f"{c}_pbb" for c in pitcher_bb.columns if c not in ['pitcher_id', 'name_pbb']})
    df_final = df_final.merge(pitcher_bb, on="pitcher_id", how="left")

    # Apply batted-ball and pitcher profile scores
    df_final['BattedBallScore'] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final['PitcherBBScore'] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final['CustomBoost'] = df_final.apply(custom_2025_boost, axis=1)
    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values("HR_Score", ascending=False)

    # Show Leaderboard
    st.success("Leaderboard ready! Top Matchups:")
    cols_to_show = [
        'Batter', 'Pitcher', 'HR_Score', 'xhr_diff', 'xhr', 'hr_total', 'Park', 'City', 'Time',
        'B_BarrelRate_14', 'B_EV_14', 'B_SLG_14', 'B_xSLG_14', 'B_xwoba_14', 'B_sweet_spot_pct_14',
        'B_pull_pct_14', 'B_oppo_pct_14', 'B_gbfb_14', 'B_hardhit_pct_14', 'PlatoonWoba', 'PitchMixBoost',
        'P_BarrelRateAllowed_14', 'P_EVAllowed_14', 'P_SLG_14', 'P_xSLG_14', 'P_xwoba_14',
        'P_sweet_spot_pct_14', 'P_pull_pct_14', 'P_oppo_pct_14', 'P_gbfb_14', 'P_hardhit_pct_14', 'Temp', 'Wind', 'WindEffect',
        'ParkFactor', 'BattedBallScore', 'PitcherBBScore', 'CustomBoost'
    ]
    cols_to_show = [col for col in cols_to_show if col in df_leaderboard.columns]
    st.dataframe(df_leaderboard[cols_to_show].head(15))

    # Chart Top 5
    st.subheader("Top 5 HR Scores")
    st.bar_chart(df_leaderboard.set_index('Batter')[['HR_Score']].head(5))

    # Download CSV
    csv_bytes = df_leaderboard.to_csv(index=False).encode()
    st.download_button("Download Full Leaderboard as CSV", csv_bytes, file_name="hr_leaderboard.csv")

    # Error Log Viewer
    if error_log:
        with st.expander("⚠️ Errors and Warnings"):
            for e in error_log:
                st.text(e)
else:
    st.info("Upload all 4 files to generate the leaderboard.")
