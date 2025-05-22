import streamlit as st
import pandas as pd
import numpy as np
import requests
import unicodedata
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from pybaseball.lahman import people
import difflib

API_KEY = st.secrets["weather"]["api_key"]
error_log = []

@st.cache_data
def cached_statcast_batter(start, end, batter_id): return statcast_batter(start, end, batter_id)

@st.cache_data
def cached_statcast_pitcher(start, end, pitcher_id): return statcast_pitcher(start, end, pitcher_id)

@st.cache_data
def cached_playerid_lookup(last, first): return playerid_lookup(last, first)

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
    # Ballpark orientations and park factors
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

# Wind and weather logic
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

# Player ID + Handedness
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
    # Rolling batter statcast metrics (BarrelRate, EV, SLG, Barrel/PA)
def get_batter_stats_multi(batter_id, windows=[3, 5, 7, 14]):
    ...

# Rolling pitcher statcast metrics (BarrelRateAllowed, EVAllowed, SLG)
def get_pitcher_stats_multi(pitcher_id, windows=[3, 5, 7, 14]):
    ...

# Advanced batter stats
def get_batter_advanced_stats(batter_id, window=14):
    ...

# Advanced pitcher stats
def get_pitcher_advanced_stats(pitcher_id, window=14):
    ...

# Z-Contact% (Fangraphs)
@st.cache_data
def get_zcontact_data():
    try:
        from pybaseball import fangraphs_batting_stats
        z = fangraphs_batting_stats(2024, qual=0)
        z["norm_name"] = z["Name"].map(normalize_name)
        return z[["norm_name", "Z-Contact%"]]
    except Exception as e:
        error_log.append(f"Z-Contact fetch error: {e}")
        return pd.DataFrame(columns=["norm_name", "Z-Contact%"])
        def get_batter_rolling_advanced(batter_id, windows=[3, 5, 7, 14]):
    ...

def get_pitcher_rolling_advanced(pitcher_id, windows=[3, 5, 7, 14]):
    ...
    def get_pitcher_pitch_mix(pitcher_id, window=14):
    ...

def get_batter_pitchtype_woba(batter_id, window=14):
    ...

def get_platoon_woba(batter_id, pitcher_hand, days=365):
    ...

def calc_pitchtype_boost(batter_pitch_woba, pitcher_mix):
    ...
    # Plate Discipline Stats
def get_plate_discipline_stats(player_id, is_pitcher=False, window=14):
    try:
        start = (datetime.now() - timedelta(days=window)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        df = cached_statcast_pitcher(start, end, player_id) if is_pitcher else cached_statcast_batter(start, end, player_id)

        if df.empty: return {}

        swings = df['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'foul_bunt'])
        whiffs = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        chases = df[df['zone'].isin([11,12,13,14,15,16,17,18,19])]['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip'])

        total_pitches = len(df)
        swing_pct = swings.mean() if total_pitches > 0 else None
        whiff_pct = whiffs.mean() if total_pitches > 0 else None
        chase_pct = chases.mean() if len(df[df['zone'].isin([11,12,13,14,15,16,17,18,19])]) > 0 else None

        prefix = 'P_' if is_pitcher else 'B_'
        return {
            f'{prefix}ChasePct_14': round(100 * chase_pct, 1) if chase_pct is not None else None,
            f'{prefix}SwingPct_14': round(100 * swing_pct, 1) if swing_pct is not None else None,
            f'{prefix}WhiffPct_14': round(100 * whiff_pct, 1) if whiff_pct is not None else None
        }
    except Exception as e:
        who = 'Pitcher' if is_pitcher else 'Batter'
        error_log.append(f"Plate discipline error for {who} {player_id}: {e}")
        return {}
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
    ...
    return bonus
    def calc_hr_score(row):
    batter_score = (
        norm_barrel(row.get('B_BarrelRate_14')) * 0.12 +
        norm_ev(row.get('B_EV_14')) * 0.08 +
        (row.get('B_SLG_14') or 0) * 0.09 +
        (row.get('B_xwOBA_14') or 0) * 0.05 +
        (row.get('B_xSLG_14') or 0) * 0.05 +
        (row.get('B_sweet_14') or 0) * 0.02 +
        (row.get('B_pull_14') or 0) * 0.01 +
        (row.get('B_oppo_14') or 0) * 0.01 +
        (row.get('B_gbfb_14') or 0) * 0.01 +
        (row.get('B_WhiffPct_14') or 0) * -0.02 +
        (row.get('B_SwingPct_14') or 0) * 0.01 +
        (row.get('B_ChasePct_14') or 0) * -0.01
    )
    pitcher_score = (
        norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.07 +
        norm_ev(row.get('P_EVAllowed_14')) * 0.05 +
        -(row.get('P_SLG_14') or 0) * 0.09 +
        -(row.get('P_xwOBA_14') or 0) * 0.05 +
        -(row.get('P_xSLG_14') or 0) * 0.05 +
        -(row.get('P_sweet_14') or 0) * 0.02 +
        -(row.get('P_pull_14') or 0) * 0.01 +
        -(row.get('P_oppo_14') or 0) * 0.01 +
        -(row.get('P_gbfb_14') or 0) * 0.01 +
        (row.get('P_WhiffPct_14') or 0) * 0.02 +
        -(row.get('P_SwingPct_14') or 0) * 0.01 +
        (row.get('P_ChasePct_14') or 0) * 0.01
    )
    park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.10
    weather_score = norm_weather(row.get("Temp"), row.get("Wind"), row.get("WindEffect")) * 0.15
    regression_score = max(0, min((row.get('xhr_diff', 0) or 0) / 5, 0.12))
    platoon_score = ((row.get('PlatoonWoba') or 0.320) - 0.320) * 0.1
    pitchtype_boost = row.get("PitchMixBoost", 0)
    return round(
        batter_score + pitcher_score + park_score + weather_score + regression_score +
        row.get('BattedBallScore', 0) + row.get('PitcherBBScore', 0) +
        platoon_score + pitchtype_boost + custom_2025_boost(row),
        3
    )
    # === Streamlit UI ===
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
    xhr_df = pd.read_csv(xhr_file)
    batted_ball = pd.read_csv(battedball_file).rename(columns={"id": "batter_id"})
    pitcher_bb = pd.read_csv(pitcher_battedball_file).rename(columns={"id": "pitcher_id", "bbe": "bbe_pbb"})
    pitcher_bb = pitcher_bb.rename(columns={c: f"{c}_pbb" for c in pitcher_bb.columns if c not in ["pitcher_id", "name_pbb"]})

    xhr_df["player_norm"] = xhr_df["player"].apply(normalize_name)
    df_upload["norm_batter"] = df_upload["Batter"].apply(normalize_name)
    df_upload["batter_id"] = df_upload["Batter"].apply(get_player_id)
    df_upload["pitcher_id"] = df_upload["Pitcher"].apply(get_player_id)

    df_merged = df_upload.merge(
        xhr_df[["player_norm", "hr_total", "xhr", "xhr_diff"]],
        left_on="norm_batter", right_on="player_norm", how="left"
    )

    df_merged["ParkFactor"] = df_merged["Park"].map(park_factors)
    df_merged["ParkOrientation"] = df_merged["Park"].map(ballpark_orientations)

    progress = st.progress(0)
    rows = []

    for idx, row in df_merged.iterrows():
        try:
            weather = get_weather(row["City"], row["Date"], row["ParkOrientation"], row["Time"])
            b_stats = get_batter_stats_multi(row["batter_id"])
            p_stats = get_pitcher_stats_multi(row["pitcher_id"])
            b_adv = get_batter_advanced_stats(row["batter_id"])
            p_adv = get_pitcher_advanced_stats(row["pitcher_id"])
            b_rolling_adv = get_batter_rolling_advanced(row["batter_id"])
            p_rolling_adv = get_pitcher_rolling_advanced(row["pitcher_id"])
            b_plate = get_plate_discipline_stats(row["batter_id"], is_pitcher=False)
            p_plate = get_plate_discipline_stats(row["pitcher_id"], is_pitcher=True)

            z_df = get_zcontact_data()
            zcontact = z_df[z_df["norm_name"] == normalize_name(row["Batter"])]["Z-Contact%"].values[0] if not z_df.empty else None

            b_bats, _ = get_handedness(row["Batter"])
            _, p_throws = get_handedness(row["Pitcher"])
            platoon_woba = get_platoon_woba(row["batter_id"], p_throws) if b_bats and p_throws else None

            pitch_mix = get_pitcher_pitch_mix(row["pitcher_id"])
            pitch_woba = get_batter_pitchtype_woba(row["batter_id"])
            pt_boost = calc_pitchtype_boost(pitch_woba, pitch_mix)

            record = row.to_dict()
            record.update(weather)
            record.update(b_stats)
            record.update(p_stats)
            record.update(b_adv)
            record.update(p_adv)
            record.update(b_rolling_adv)
            record.update(p_rolling_adv)
            record.update(b_plate)
            record.update(p_plate)
            record["Z-Contact%"] = zcontact
            record["PlatoonWoba"] = platoon_woba
            record["PitchMixBoost"] = pt_boost
            rows.append(record)
        except Exception as e:
            error_log.append(f"Row error ({row['Batter']} vs {row['Pitcher']}): {e}")
        progress.progress((idx + 1) / len(df_merged))

    df_final = pd.DataFrame(rows)
    df_final = df_final.merge(batted_ball, on="batter_id", how="left")
    df_final = df_final.merge(pitcher_bb, on="pitcher_id", how="left")
    df_final["BattedBallScore"] = df_final.apply(calc_batted_ball_score, axis=1)
    df_final["PitcherBBScore"] = df_final.apply(calc_pitcher_bb_score, axis=1)
    df_final["CustomBoost"] = df_final.apply(custom_2025_boost, axis=1)
    df_final["HR_Score"] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values("HR_Score", ascending=False)
    st.success("Leaderboard ready! Top Matchups:")
    show_cols = [
        "Batter", "Pitcher", "HR_Score", "xhr", "xhr_diff", "hr_total", "Park", "City", "Time",
        "B_BarrelRate_14", "B_EV_14", "B_SLG_14", "B_xwOBA_14", "B_xSLG_14", "B_sweet_14", "B_pull_14", "B_oppo_14",
        "B_WhiffPct_14", "B_SwingPct_14", "B_ChasePct_14", "Z-Contact%",
        "P_BarrelRateAllowed_14", "P_EVAllowed_14", "P_SLG_14", "P_xwOBA_14", "P_xSLG_14", "P_sweet_14", "P_pull_14", "P_oppo_14",
        "P_WhiffPct_14", "P_SwingPct_14", "P_ChasePct_14", "PlatoonWoba", "PitchMixBoost",
        "Temp", "Wind", "WindEffect", "Humidity", "ParkFactor", "BattedBallScore", "PitcherBBScore", "CustomBoost"
    ]
    show_cols = [col for col in show_cols if col in df_leaderboard.columns]
    st.dataframe(df_leaderboard[show_cols].head(15))
    st.bar_chart(df_leaderboard.set_index("Batter")[["HR_Score"]].head(5))
    st.download_button("Download CSV", df_leaderboard.to_csv(index=False).encode(), file_name="hr_leaderboard.csv")

    if error_log:
        with st.expander("⚠️ Errors and Warnings"):
            for e in error_log:
                st.text(e)
else:
    st.info("Please upload all four required CSV files to begin.")
