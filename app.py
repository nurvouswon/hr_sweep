import streamlit as st
import pandas as pd
import requests
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from pybaseball.lahman import people
from datetime import datetime, timedelta
import unicodedata
import difflib

API_KEY = "11ac3c31fb664ba8971102152251805"

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
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
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        resp = requests.get(url)
        data = resp.json()
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
    except Exception:
        return {
            "Temp": None, "Wind": None, "WindDir": None, "WindEffect": None,
            "Humidity": None, "Condition": None
        }

def get_player_id(name):
    try:
        first, last = name.split(" ", 1)
        info = playerid_lookup(last, first)
        if not info.empty:
            return int(info.iloc[0]['key_mlbam'])
    except Exception:
        return None
    return None

# Manual overrides for new/missing players
MANUAL_HANDEDNESS = {
    'alexander canario': ('R', 'R'),
    'liam hicks': ('L', 'R'),
    'patrick bailey': ('B', 'R'),
    # Add more as needed
}
UNKNOWNS_LOG = set()

# Fangraphs info table (if available)
try:
    from pybaseball.fangraphs import fg_player_info
    FG_INFO = fg_player_info()
    FG_INFO['norm_name'] = FG_INFO['Name'].map(lambda x: x.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip())
except Exception:
    FG_INFO = pd.DataFrame()

def get_handedness(name):
    clean_name = normalize_name(name)
    parts = clean_name.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
    else:
        first, last = clean_name, ""
    # 1. Try MLB Stats API
    try:
        info = playerid_lookup(last.capitalize(), first.capitalize())
        if not info.empty and 'key_mlbam' in info.columns:
            mlbam_id = info.iloc[0]['key_mlbam']
            url = f'https://statsapi.mlb.com/api/v1/people/{mlbam_id}'
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                hand = data['people'][0]
                bats = hand['batSide']['code']
                throws = hand['pitchHand']['code']
                if bats and throws:
                    return bats, throws
    except Exception:
        pass
    if clean_name in MANUAL_HANDEDNESS:
        return MANUAL_HANDEDNESS[clean_name]
    try:
        if not FG_INFO.empty:
            fg_row = FG_INFO[FG_INFO['norm_name'] == clean_name]
            if not fg_row.empty:
                bats = fg_row.iloc[0].get('bats')
                throws = fg_row.iloc[0].get('throws')
                if pd.notnull(bats) and pd.notnull(throws):
                    return bats, throws
            fg_row = FG_INFO[FG_INFO['norm_name'].str.endswith(' ' + last)]
            if not fg_row.empty:
                bats = fg_row.iloc[0].get('bats')
                throws = fg_row.iloc[0].get('throws')
                if pd.notnull(bats) and pd.notnull(throws):
                    return bats, throws
    except Exception:
        pass
    try:
        df = people()
        df['nname'] = (df['name_first'].fillna('') + ' ' + df['name_last'].fillna('')).map(normalize_name)
        match = df[df['nname'] == clean_name]
        if not match.empty:
            return match.iloc[0].get('bats'), match.iloc[0].get('throws')
        close = difflib.get_close_matches(clean_name, df['nname'].tolist(), n=1, cutoff=0.85)
        if close:
            row = df[df['nname'] == close[0]].iloc[0]
            return row.get('bats'), row.get('throws')
    except Exception:
        pass
    UNKNOWNS_LOG.add(clean_name)
    return None, None

# NORMALIZERS for Advanced Stats
def norm_barrel(x):   return min(x / 0.15, 1) if pd.notnull(x) else 0
def norm_ev(x):       return max(0, min((x - 80) / (105 - 80), 1)) if pd.notnull(x) else 0
def norm_hardhit(x):  return max(0, min(x / 0.55, 1)) if pd.notnull(x) else 0
def norm_xba(x):      return max(0, min((x - 0.15) / (0.4 - 0.15), 1)) if pd.notnull(x) else 0
def norm_xslg(x):     return max(0, min((x - 0.250) / (0.800 - 0.250), 1)) if pd.notnull(x) else 0
def norm_woba(x):     return max(0, min((x - 0.220) / (0.500 - 0.220), 1)) if pd.notnull(x) else 0
def norm_park(x):     return max(0, min((x - 0.8) / (1.3 - 0.8), 1)) if pd.notnull(x) else 0
def norm_weather(temp, wind, wind_effect):
    score = 1
    if temp and temp > 80: score += 0.05
    if wind and wind > 10:
        if wind_effect == "out": score += 0.07
        elif wind_effect == "in": score -= 0.07
    return max(0.8, min(score, 1.2))

def custom_2025_boost(row):
    bonus = 0
    bonus += norm_barrel(row.get('B_BarrelRate_14')) * 0.01
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
                if hour < 17:
                    bonus -= 0.01
            except Exception:
                bonus -= 0.01
        else:
            bonus -= 0.01
    if row.get('PitcherHandedness') == 'L': bonus += 0.01
    return bonus

windows = [3, 5, 7, 14]

# NEW: ADVANCED STATS ADDED!
def get_batter_stats_multi(batter_name, windows):
    pid = get_player_id(batter_name)
    out = {}
    stat_fields = ['B_BarrelRate', 'B_EV', 'B_HardHit', 'B_xBA', 'B_xSLG', 'B_wOBA']
    if not pid:
        for w in windows:
            for f in stat_fields:
                out[f"{f}_{w}"] = None
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_batter(start, end, pid)
            if df.empty:
                for f in stat_fields:
                    out[f"{f}_{w}"] = None
                continue
            df = df[df['type'] == 'X']
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            total = len(df)
            hardhit = df[df['launch_speed'] >= 95].shape[0]
            out[f"B_BarrelRate_{w}"] = round(barrels / total, 3) if total > 0 else None
            out[f"B_EV_{w}"] = round(df['launch_speed'].mean(), 1) if total > 0 else None
            out[f"B_HardHit_{w}"] = round(hardhit / total, 3) if total > 0 else None
            out[f"B_xBA_{w}"] = round(df['estimated_ba_using_speedangle'].mean(), 3) if 'estimated_ba_using_speedangle' in df and total > 0 else None
            out[f"B_xSLG_{w}"] = round(df['estimated_slg_using_speedangle'].mean(), 3) if 'estimated_slg_using_speedangle' in df and total > 0 else None
            out[f"B_wOBA_{w}"] = round(df['estimated_woba_using_speedangle'].mean(), 3) if 'estimated_woba_using_speedangle' in df and total > 0 else None
        except Exception:
            for f in stat_fields:
                out[f"{f}_{w}"] = None
    return out

def get_pitcher_stats_multi(pitcher_name, windows):
    pid = get_player_id(pitcher_name)
    out = {}
    stat_fields = [
        "P_BarrelRateAllowed", "P_EVAllowed", "P_HardHitAllowed", "P_xBAAllowed", "P_xSLGAllowed", "P_wOBAAllowed",
        "P_HRAllowed", "P_BIP", "P_HardHitRate", "P_FlyBallRate", "P_KRate", "P_BBRate", "P_HR9"
    ]
    if not pid:
        for w in windows:
            for f in stat_fields:
                out[f"{f}_{w}"] = None
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_pitcher(start, end, pid)
            if df.empty:
                for f in stat_fields:
                    out[f"{f}_{w}"] = None
                continue
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            total = len(df)
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            hardhit = df[df['launch_speed'] >= 95].shape[0]
            out[f"P_BarrelRateAllowed_{w}"] = round(barrels / total, 3) if total > 0 else None
            out[f"P_EVAllowed_{w}"] = round(df['launch_speed'].mean(), 1) if total > 0 else None
            out[f"P_HardHitAllowed_{w}"] = round(hardhit / total, 3) if total > 0 else None
            out[f"P_xBAAllowed_{w}"] = round(df['estimated_ba_using_speedangle'].mean(), 3) if 'estimated_ba_using_speedangle' in df and total > 0 else None
            out[f"P_xSLGAllowed_{w}"] = round(df['estimated_slg_using_speedangle'].mean(), 3) if 'estimated_slg_using_speedangle' in df and total > 0 else None
            out[f"P_wOBAAllowed_{w}"] = round(df['estimated_woba_using_speedangle'].mean(), 3) if 'estimated_woba_using_speedangle' in df and total > 0 else None
            hrs = df[df['events'] == "home_run"].shape[0]
            fly_balls = df[df['bb_type'] == 'fly_ball'].shape[0]
            k = df[df['events'] == 'strikeout'].shape[0]
            bb = df[df['events'] == 'walk'].shape[0]
            outs = df['outs_when_up'].sum() if 'outs_when_up' in df.columns else 0
            innings = outs / 3 if outs else 0
            hr9 = (hrs / innings * 9) if innings > 0 else None
            out[f"P_HRAllowed_{w}"] = hrs
            out[f"P_BIP_{w}"] = total
            out[f"P_HardHitRate_{w}"] = round(hardhit / total, 3) if total > 0 else None
            out[f"P_FlyBallRate_{w}"] = round(fly_balls / total, 3) if total > 0 else None
            out[f"P_KRate_{w}"] = round(k / total, 3) if total > 0 else None
            out[f"P_BBRate_{w}"] = round(bb / total, 3) if total > 0 else None
            out[f"P_HR9_{w}"] = round(hr9, 2) if hr9 else None
        except Exception:
            for f in stat_fields:
                out[f"{f}_{w}"] = None
    return out

# HR SCORE EXTENDED WITH ADVANCED STATS
def calc_hr_score(row):
    # Rolling window weights
    batter_score = (
        norm_barrel(row.get('B_BarrelRate_14')) * 0.12 +
        norm_ev(row.get('B_EV_14')) * 0.08 +
        norm_hardhit(row.get('B_HardHit_14')) * 0.08 +
        norm_xba(row.get('B_xBA_14')) * 0.07 +
        norm_xslg(row.get('B_xSLG_14')) * 0.07 +
        norm_woba(row.get('B_wOBA_14')) * 0.07 +
        norm_barrel(row.get('B_BarrelRate_7')) * 0.08 +
        norm_ev(row.get('B_EV_7')) * 0.06 +
        norm_hardhit(row.get('B_HardHit_7')) * 0.06 +
        norm_xba(row.get('B_xBA_7')) * 0.05 +
        norm_xslg(row.get('B_xSLG_7')) * 0.05 +
        norm_woba(row.get('B_wOBA_7')) * 0.05
    )
    pitcher_score = (
        norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.07 +
        norm_ev(row.get('P_EVAllowed_14')) * 0.05 +
        norm_hardhit(row.get('P_HardHitAllowed_14')) * 0.05 +
        norm_xba(row.get('P_xBAAllowed_14')) * 0.04 +
        norm_xslg(row.get('P_xSLGAllowed_14')) * 0.04 +
        norm_woba(row.get('P_wOBAAllowed_14')) * 0.04 +
        norm_barrel(row.get('P_BarrelRateAllowed_7')) * 0.05 +
        norm_ev(row.get('P_EVAllowed_7')) * 0.04 +
        norm_hardhit(row.get('P_HardHitAllowed_7')) * 0.04 +
        norm_xba(row.get('P_xBAAllowed_7')) * 0.03 +
        norm_xslg(row.get('P_xSLGAllowed_7')) * 0.03 +
        norm_woba(row.get('P_wOBAAllowed_7')) * 0.03
    )
    park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.1
    weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.1
    regression_score = max(0, min((row.get('Reg_xHR', 0) or 0) / 5, 0.15))
    total = batter_score + pitcher_score + park_score + weather_score + regression_score
    total += custom_2025_boost(row)
    return round(total, 3)

# Streamlit UI
st.title("⚾️ MLB HR Predictor & Matchup Leaderboard (Full Advanced Stat Version)")

st.markdown("""
**Upload your daily matchup CSV:**  
`Batter,Pitcher,City,Park,Date,Time`  
And a **Baseball Savant xHR/HR CSV** (with columns `player`, `hr_total`, `xhr`, `xhr_diff`)
""")

uploaded_file = st.file_uploader("Upload your daily CSV:", type=["csv"])
xhr_file = st.file_uploader("Upload Baseball Savant xHR/HR CSV:", type=["csv"])

if uploaded_file and xhr_file:
    df_upload = pd.read_csv(uploaded_file)
    for col in ['Batter','Pitcher','City','Park','Date','Time']:
        if col not in df_upload.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
    xhr_df = pd.read_csv(xhr_file)
    df_upload['batter_norm'] = df_upload['Batter'].apply(normalize_name)
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)

    unmatched = df_upload[~df_upload['batter_norm'].isin(xhr_df['player_norm'])]
    if not unmatched.empty:
        st.write("DEBUG xHR Merge — Unmatched Batter Names (not found in xHR):")
        st.dataframe(unmatched[['Batter', 'batter_norm']])

    # Merge xHR just once!
    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='batter_norm', right_on='player_norm', how='left'
    )

    weather_rows, stat_rows, park_factor_rows = [], [], []
    st.write("Fetching Statcast, weather (game time), park factor, and merging xHR (may take a few minutes)...")
    progress = st.progress(0, text="Starting...")
    for idx, row in df_merged.iterrows():
        city = row['City']
        date = row['Date']
        park = row['Park']
        batter = row['Batter']
        pitcher = row['Pitcher']
        game_time = row['Time']
        park_orientation = ballpark_orientations.get(park, "N")
        park_factor = park_factors.get(park, 1.0)
        weather = get_weather(city, date, park_orientation, game_time)
        batter_stats = get_batter_stats_multi(batter, windows)
        pitcher_stats = get_pitcher_stats_multi(pitcher, windows)
        weather_rows.append(weather)
        stat_row = {}
        stat_row.update(batter_stats)
        stat_row.update(pitcher_stats)
        stat_rows.append(stat_row)
        park_factor_rows.append({"ParkFactor": park_factor})
        percent = int(100 * (idx+1) / len(df_merged))
        progress.progress((idx+1)/len(df_merged), text=f"Progress: {percent}%")

    weather_df = pd.DataFrame(weather_rows)
    stat_df = pd.DataFrame(stat_rows)
    park_df = pd.DataFrame(park_factor_rows)
    df_final = pd.concat([df_merged.reset_index(drop=True), weather_df, park_df, stat_df], axis=1)

    # Add handedness
    batter_handedness = []
    pitcher_handedness = []
    for idx, row in df_final.iterrows():
        b_bats, _ = get_handedness(row['Batter'])
        _, p_throws = get_handedness(row['Pitcher'])
        batter_handedness.append(b_bats)
        pitcher_handedness.append(p_throws)
    df_final['BatterHandedness'] = [b if b is not None else "UNK" for b in batter_handedness]
    df_final['PitcherHandedness'] = [p if p is not None else "UNK" for p in pitcher_handedness]

    # Calculate Reg_xHR using the merged columns
    df_final['Reg_xHR'] = df_final['xhr'] - df_final['hr_total']

    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values('HR_Score', ascending=False)

    st.success("All done! Top matchups below:")

    # Show columns
    show_cols = [
        'Batter','Pitcher','BatterHandedness','PitcherHandedness','Park','City','Time','HR_Score','Reg_xHR','ParkFactor','Temp','Wind','WindEffect'
    ]
    # Add advanced batter and pitcher metrics for all windows (just as example, you can show more/less)
    for w in windows:
        show_cols += [
            f'B_BarrelRate_{w}',f'B_EV_{w}',f'B_HardHit_{w}',f'B_xBA_{w}',f'B_xSLG_{w}',f'B_wOBA_{w}',
            f'P_BarrelRateAllowed_{w}',f'P_EVAllowed_{w}',f'P_HardHitAllowed_{w}',f'P_xBAAllowed_{w}',f'P_xSLGAllowed_{w}',f'P_wOBAAllowed_{w}'
        ]
    show_cols += [
        'P_HRAllowed_14','P_BIP_14','P_HardHitRate_14','P_FlyBallRate_14','P_KRate_14','P_BBRate_14','P_HR9_14',
        'xhr','hr_total','xhr_diff'
    ]
    show_cols = [c for c in show_cols if c in df_leaderboard.columns]

    top5 = df_leaderboard.head(5)
    st.dataframe(top5[show_cols])

    # Bar chart for top 5 (HR_Score and Reg_xHR)
    if 'Reg_xHR' in top5.columns:
        st.bar_chart(top5.set_index('Batter')[['HR_Score','Reg_xHR']])
    else:
        st.bar_chart(top5.set_index('Batter')[['HR_Score']])

    # Show all data and allow download
    st.dataframe(df_leaderboard[show_cols])
    csv_out = df_leaderboard.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", csv_out, "hr_leaderboard_all_advanced_stats.csv")

else:
    st.info("Please upload your daily CSV and Savant xHR/HR CSV to begin.")

st.caption("""
- All rolling batter and pitcher advanced stats (Barrel%, EV, HardHit%, xBA, xSLG, wOBA: 3,5,7,14d windows) and classic pitcher stats (K%, BB%, HR9, etc) are included.
- Park factors, park orientations, city, weather, wind, game time, handedness, and xHR regression included.
- Top-5 leaderboard and download available.
""")
