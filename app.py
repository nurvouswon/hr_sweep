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

# ========== Error Logging ==========
def log_error(context, exception, level="ERROR"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_log.append(f"[{timestamp}] [{level}] {context}: {exception}")

# ========== Caching ==========
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

# ========== Helpers ==========
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

# ========== Ballpark Dictionaries ==========
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
        wind_dir = weather_hour.get('wind_dir', '')[:2].upper()
        humidity = weather_hour.get('humidity')
        condition = weather_hour.get('condition', {}).get('text')
        wind_effect = is_wind_out(wind_dir, park_orientation)
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

# ========== Statcast Feature Functions ==========
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
            single = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'] < 15)].shape[0]
            double = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(15, 30))].shape[0]
            triple = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(30, 40))].shape[0]
            hr = df[(df['launch_angle'] > 35) & (df['launch_speed'] > 100)].shape[0]
            ab = single + double + triple + hr
            xslg = (1*single + 2*double + 3*triple + 4*hr) / ab if ab else None
            xba = df['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in df.columns else None
            xiso = (xslg - xba) if xslg and xba else None
            events = df['events'].value_counts().to_dict()
            ab2 = sum(events.get(e, 0) for e in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
            total_bases = (events.get('single', 0) + 2 * events.get('double', 0) + 3 * events.get('triple', 0) + 4 * events.get('home_run', 0))
            slg = total_bases / ab2 if ab2 > 0 else None
            xwoba = df['woba_value'].mean() if 'woba_value' in df.columns else None
            sweet_spot_pct = df[(df['launch_angle'].between(8, 32))].shape[0] / pa if pa > 0 else None
            hard_hit_pct = df[(df['launch_speed'] >= 95)].shape[0] / pa if pa > 0 else None
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
            single = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'] < 15)].shape[0]
            double = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(15, 30))].shape[0]
            triple = df[(df.get('estimated_ba_using_speedangle', 0) >= 0.5) & (df['launch_angle'].between(30, 40))].shape[0]
            hr = df[(df['launch_angle'] > 35) & (df['launch_speed'] > 100)].shape[0]
            ab = single + double + triple + hr
            xslg = (1*single + 2*double + 3*triple + 4*hr) / ab if ab else None
            xba = df['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in df.columns else None
            xiso = (xslg - xba) if xslg and xba else None
            events = df['events'].value_counts().to_dict()
            ab2 = sum(events.get(e, 0) for e in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
            total_bases = (events.get('single', 0) + 2 * events.get('double', 0) + 3 * events.get('triple', 0) + 4 * events.get('home_run', 0))
            slg = total_bases / ab2 if ab2 > 0 else None
            xwoba = df['woba_value'].mean() if 'woba_value' in df.columns else None
            sweet_spot_pct = df[(df['launch_angle'].between(8, 32))].shape[0] / total if total > 0 else None
            hard_hit_pct = df[(df['launch_speed'] >= 95)].shape[0] / total if total > 0 else None
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

# Pitch/whiff/spin/pitch-mix/platoon functions already included above

# ========== Streamlit UI Section ==========
st.title("⚾ MLB HR Matchup Leaderboard – Advanced Statcast Scoring + Pitcher Trends + ML")
st.markdown("""
Upload these 4 CSV files:
- **Lineups/Matchups**: Must have batter, pitcher, MLB IDs, city, park, date, time, etc.
- **xHR/HR Regression**: player, hr_total, xhr, xhr_diff
- **Batter Batted-Ball Profile** (with `id`)
- **Pitcher Batted-Ball Profile** (with `id`)
""")

lineup_file = st.file_uploader("Lineups/Matchups CSV (with MLB IDs)", type=["csv"])
xhr_file = st.file_uploader("xHR / HR Regression CSV", type=["csv"])
battedball_file = st.file_uploader("Batter batted-ball CSV", type=["csv"])
pitcher_battedball_file = st.file_uploader("Pitcher batted-ball CSV", type=["csv"])

if lineup_file and xhr_file and battedball_file and pitcher_battedball_file:
    df_upload = pd.read_csv(lineup_file)
    # Normalize all columns for robust joins
    df_upload.columns = (
        df_upload.columns
            .str.strip().str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w]', '', regex=True)
    )
    # Rename to standard app columns (edit if needed for your actual file!)
    df_upload.rename(columns={
        'player_name': 'batter',
        'mlb_id': 'batter_id',
        'team_code': 'team_code',
        'game_date': 'date',
        'batting_order': 'batting_order',
        'confirmed': 'confirmed',
        'weather': 'weather', # optional
        'park': 'park',
        'city': 'city',
        'time': 'time'
    }, inplace=True)
    # Validate columns and fix missing
    required_cols = ['batter', 'batter_id', 'team_code', 'date', 'batting_order', 'confirmed', 'city', 'park', 'time']
    missing = [col for col in required_cols if col not in df_upload.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    # Only confirmed starters
    df_upload = df_upload[df_upload['confirmed'].astype(str).str.lower() == 'y']
    # Use normalized names for joining
    df_upload['norm_batter'] = df_upload['batter'].apply(normalize_name)
    # Force correct id field
    df_upload['batter_id'] = df_upload['batter_id'].astype(str)
    # Get pitcher rows for mapping
    pitcher_rows = df_upload[df_upload['batting_order'].astype(str).str.lower() == 'sp']
    team_pitcher_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter_id']))
    pitcher_name_map = dict(zip(pitcher_rows['team_code'], pitcher_rows['batter']))
    df_upload['pitcher_id'] = df_upload['team_code'].map(team_pitcher_map)
    df_upload['pitcher'] = df_upload['team_code'].map(pitcher_name_map)

    # Merge xHR regression file
    xhr_df = pd.read_csv(xhr_file)
    xhr_df.columns = xhr_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)
    df_upload['norm_batter'] = df_upload['batter'].apply(normalize_name)
    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='norm_batter', right_on='player_norm', how='left'
    )
    df_merged['parkfactor'] = df_merged['park'].map(park_factors)
    df_merged['parkorientation'] = df_merged['park'].map(ballpark_orientations)
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
            # Spin drop logic for 30d spin
            p_spin_metrics_30 = get_pitcher_spin_metrics(row['pitcher_id'], windows=[30])
            record.update(p_spin_metrics_30)
            rows.append(record)
        except Exception as e:
            log_error(f"Row error ({row.get('batter','NA')} vs {row.get('pitcher','NA')})", e)
        progress.progress((idx + 1) / len(df_merged), text=f"Processing {int(100 * (idx + 1) / len(df_merged))}%")
    df_final = pd.DataFrame(rows)

    # Merge in batted ball CSVs (robust normalization)
    batted = pd.read_csv(battedball_file)
    batted.columns = batted.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    batted = batted.rename(columns={"id": "batter_id"})
    df_final = df_final.merge(batted, on="batter_id", how="left")
    pitcher_bb = pd.read_csv(pitcher_battedball_file)
    pitcher_bb.columns = pitcher_bb.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    pitcher_bb = pitcher_bb.rename(columns={"id": "pitcher_id", 'bbe': 'bbe_pbb'})
    pitcher_bb = pitcher_bb.rename(columns={c: f"{c}_pbb" for c in pitcher_bb.columns if c not in ['pitcher_id', 'name_pbb']})
    df_final = df_final.merge(pitcher_bb, on="pitcher_id", how="left")
    # Add scores
    df_final.reset_index(drop=True, inplace=True)
    df_final.insert(0, "rank", df_final.index + 1)
    df_final['BattedBallScore'] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final['PitcherBBScore'] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final['CustomBoost'] = df_final.apply(custom_2025_boost, axis=1)
    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_final['HR_Score_pctile'] = df_final['HR_Score'].rank(pct=True)
    df_final['HR_Tier'] = df_final['HR_Score'].apply(hr_score_tier)

    df_leaderboard, importances = train_and_apply_model(df_final)
    if df_leaderboard is None:
        df_leaderboard = df_final.sort_values("HR_Score", ascending=False).reset_index(drop=True)
        df_leaderboard["rank"] = df_leaderboard.index + 1
    else:
        st.write("Feature importances:", importances)
    st.success("Leaderboard ready! Top Matchups:")
    cols_to_show = [
        'rank', 'batter', 'pitcher', 'HR_Score', 'HR_Tier', 'HR_Score_pctile', 'xhr_diff', 'xhr', 'hr_total',
        'park', 'city', 'time', 'B_BarrelRate_14', 'B_EV_14', 'B_SLG_14', 'B_xSLG_14', 'B_xISO_14',
        'B_xwoba_14', 'B_sweet_spot_pct_14', 'B_hardhit_pct_14', 'B_WhiffRate_14',
        'P_BarrelRateAllowed_14', 'P_EVAllowed_14', 'P_SLG_14', 'P_xSLG_14', 'P_xISO_14', 'P_xwoba_14',
        'P_sweet_spot_pct_14', 'P_hardhit_pct_14', 'P_WhiffRate_14', 'P_FF_Spin_14', 'P_FF_Spin_30',
        'Temp', 'Wind', 'WindEffect', 'parkfactor', 'BattedBallScore', 'PitcherBBScore', 'CustomBoost', 'PlatoonWoba', 'PitchMixBoost'
    ]
    if 'ML_HR_Prob' in df_leaderboard.columns:
        cols_to_show.append('ML_HR_Prob')
    cols_to_show = [col for col in cols_to_show if col in df_leaderboard.columns]
    st.dataframe(df_leaderboard[cols_to_show].head(15), use_container_width=True)
    st.subheader("Top 5 HR Scores")
    st.bar_chart(df_leaderboard.set_index('batter')[['HR_Score']].head(5))
    csv_bytes = df_leaderboard.to_csv(index=False).encode()
    st.download_button("Download Full Leaderboard as CSV", csv_bytes, file_name="hr_leaderboard.csv")
    if error_log:
        with st.expander("⚠️ Errors and Warnings (Click to expand)"):
            for e in error_log[-30:]:
                st.text(e)
else:
    st.info("Upload all 4 files to generate the leaderboard.")
