import streamlit as st
import pandas as pd
import requests
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from pybaseball.lahman import people
from datetime import datetime, timedelta
import numpy as np

API_KEY = "11ac3c31fb664ba8971102152251805"

# --- PARK DICTS ---
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

# --- HANDEDNESS ---
MANUAL_HANDEDNESS = {
    'alexander canario': ('R', 'R'),
    'liam hicks': ('L', 'R'),
    'patrick bailey': ('B', 'R'),
}
try:
    from pybaseball.fangraphs import fg_player_info
    FG_INFO = fg_player_info()
    FG_INFO['norm_name'] = FG_INFO['Name'].map(lambda x: x.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip())
except Exception:
    FG_INFO = pd.DataFrame()

def get_handedness(name):
    # (same as before, or you can streamline)
    return 'UNK', 'UNK'  # replace with previous logic if you want

# --- STATCAST & ROLLING SLG ---
def get_batter_stats_multi(batter_name, windows):
    pid = get_player_id(batter_name)
    out = {}
    if not pid:
        for w in windows:
            out[f"B_BarrelRate_{w}"] = None
            out[f"B_EV_{w}"] = None
            out[f"B_SLG_{w}"] = None
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_batter(start, end, pid)
            if df.empty:
                out[f"B_BarrelRate_{w}"] = None
                out[f"B_EV_{w}"] = None
                out[f"B_SLG_{w}"] = None
                continue
            df = df[df['type'] == 'X']
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            total = len(df)
            barrel_rate = barrels / total if total > 0 else 0
            ev = df['launch_speed'].mean() if total > 0 else None
            # ROLLING SLG
            if 'events' in df.columns:
                single = df[df['events'] == 'single'].shape[0]
                double = df[df['events'] == 'double'].shape[0]
                triple = df[df['events'] == 'triple'].shape[0]
                hr = df[df['events'] == 'home_run'].shape[0]
                ab = single + double + triple + hr + df[df['events'] == 'field_out'].shape[0] + df[df['events'] == 'force_out'].shape[0] + df[df['events'] == 'other_out'].shape[0]
                slg = (single + 2*double + 3*triple + 4*hr)/ab if ab > 0 else None
            else:
                slg = None
            out[f"B_BarrelRate_{w}"] = round(barrel_rate,3)
            out[f"B_EV_{w}"] = round(ev,1) if ev else None
            out[f"B_SLG_{w}"] = round(slg,3) if slg else None
        except Exception:
            out[f"B_BarrelRate_{w}"] = None
            out[f"B_EV_{w}"] = None
            out[f"B_SLG_{w}"] = None
    return out

def get_pitcher_stats_multi(pitcher_name, windows):
    pid = get_player_id(pitcher_name)
    out = {}
    stats_list = [
        "BarrelRateAllowed", "EVAllowed", "HRAllowed", "BIP", "HardHitRate",
        "FlyBallRate", "KRate", "BBRate", "HR9"
    ]
    if not pid:
        for w in windows:
            for stat in stats_list:
                out[f"P_{stat}_{w}"] = None
            out[f"P_SLG_{w}"] = None
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_pitcher(start, end, pid)
            if df.empty:
                for stat in stats_list:
                    out[f"P_{stat}_{w}"] = None
                out[f"P_SLG_{w}"] = None
                continue
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            total = len(df)
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            ev_allowed = df['launch_speed'].mean() if total > 0 else None
            hrs = df[df['events'] == "home_run"].shape[0]
            hard_hit = df[df['launch_speed'] > 95].shape[0]
            fly_balls = df[df['bb_type'] == 'fly_ball'].shape[0]
            k = df[df['events'] == 'strikeout'].shape[0]
            bb = df[df['events'] == 'walk'].shape[0]
            outs = df['outs_when_up'].sum() if 'outs_when_up' in df.columns else 0
            innings = outs / 3 if outs else 0
            hr9 = (hrs / innings * 9) if innings > 0 else None
            # ROLLING SLG AGAINST
            if 'events' in df.columns:
                single = df[df['events'] == 'single'].shape[0]
                double = df[df['events'] == 'double'].shape[0]
                triple = df[df['events'] == 'triple'].shape[0]
                hr = df[df['events'] == 'home_run'].shape[0]
                ab = single + double + triple + hr + df[df['events'] == 'field_out'].shape[0] + df[df['events'] == 'force_out'].shape[0] + df[df['events'] == 'other_out'].shape[0]
                slg = (single + 2*double + 3*triple + 4*hr)/ab if ab > 0 else None
            else:
                slg = None
            out[f"P_BarrelRateAllowed_{w}"] = round(barrels / total, 3) if total > 0 else None
            out[f"P_EVAllowed_{w}"] = round(ev_allowed, 1) if ev_allowed else None
            out[f"P_HRAllowed_{w}"] = hrs
            out[f"P_BIP_{w}"] = total
            out[f"P_HardHitRate_{w}"] = round(hard_hit / total, 3) if total > 0 else None
            out[f"P_FlyBallRate_{w}"] = round(fly_balls / total, 3) if total > 0 else None
            out[f"P_KRate_{w}"] = round(k / total, 3) if total > 0 else None
            out[f"P_BBRate_{w}"] = round(bb / total, 3) if total > 0 else None
            out[f"P_HR9_{w}"] = round(hr9, 2) if hr9 else None
            out[f"P_SLG_{w}"] = round(slg,3) if slg else None
        except Exception:
            for stat in stats_list:
                out[f"P_{stat}_{w}"] = None
            out[f"P_SLG_{w}"] = None
    return out

# --- BATTED BALL PROFILE BY PLAYER ID ---
def get_bb_id_map(bb_df, colname):
    # Returns a dict of player_id (int) -> row
    id_map = {}
    for _, row in bb_df.iterrows():
        try:
            player_id = int(row[colname])
            id_map[player_id] = row
        except Exception:
            continue
    return id_map

def merge_bb_stats_by_id(pid, id_map, prefix):
    # Use player_id as key for batted-ball stats
    d = id_map.get(pid)
    if d is not None:
        # Flexible: pull_rate/pull% etc
        def first_of(*candidates):
            for c in candidates:
                if c in d and pd.notnull(d[c]):
                    return d[c]
            return None
        return {
            f"{prefix}_pull_pct": first_of('pull_rate', 'pull%', 'Pull%', 'pull'),
            f"{prefix}_oppo_pct": first_of('oppo_rate', 'oppo%', 'Oppo%', 'oppo'),
            f"{prefix}_gb_pct": first_of('gb_rate', 'gb%', 'GB%', 'gb'),
            f"{prefix}_fb_pct": first_of('fb_rate', 'fb%', 'FB%', 'fb'),
            f"{prefix}_ld_pct": first_of('ld_rate', 'ld%', 'LD%', 'ld'),
            f"{prefix}_pop_pct": first_of('pu_rate', 'pop%', 'POP%', 'pop'),
            f"{prefix}_hr_fb_pct": first_of('hr_fb_rate', 'hr/fb', 'HR/FB', 'hr_fb'),
            f"{prefix}_hardhit_pct": first_of('hardhit_rate', 'hardhit%', 'HardHit%', 'hardhit'),
            f"{prefix}_barrel_pct": first_of('barrel_rate', 'barrel%', 'Barrel%', 'barrel'),
        }
    # else
    return {
        f"{prefix}_pull_pct": None, f"{prefix}_oppo_pct": None, f"{prefix}_gb_pct": None, f"{prefix}_fb_pct": None,
        f"{prefix}_ld_pct": None, f"{prefix}_pop_pct": None, f"{prefix}_hr_fb_pct": None,
        f"{prefix}_hardhit_pct": None, f"{prefix}_barrel_pct": None
    }

# --- NORMALIZATION & SCORING ---
def norm_barrel(x):   return min(x / 0.15, 1) if pd.notnull(x) else 0
def norm_ev(x):       return max(0, min((x - 80) / (105 - 80), 1)) if pd.notnull(x) else 0
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

st.title("⚾️ MLB HR Matchup Leaderboard (Rolling Statcast, SLG, Advanced Batted Ball, Park, Weather)")

st.markdown("""
**Upload your daily matchup CSV:**  
`Batter,Pitcher,City,Park,Date,Time`  
**Upload Batter & Pitcher Batted Ball CSVs:**  
- Must contain: `id` (MLBAM), plus columns like `pull_rate`, `oppo_rate`, `gb_rate`, `fb_rate`, `ld_rate`, `pu_rate`, `hr_fb_rate`, `hardhit_rate`, `barrel_rate`
**Upload Savant xHR/HR CSV**  
""")

uploaded_file = st.file_uploader("Upload your daily matchup CSV:", type=["csv"])
xhr_file = st.file_uploader("Upload Baseball Savant xHR/HR CSV:", type=["csv"])
batter_bb_file = st.file_uploader("Upload Batter batted-ball CSV:", type=["csv"])
pitcher_bb_file = st.file_uploader("Upload Pitcher batted-ball CSV:", type=["csv"])

if uploaded_file and xhr_file and batter_bb_file and pitcher_bb_file:
    df_upload = pd.read_csv(uploaded_file)
    xhr_df = pd.read_csv(xhr_file)
    batter_bb = pd.read_csv(batter_bb_file)
    pitcher_bb = pd.read_csv(pitcher_bb_file)
    # Map: id -> row
    batter_bb_id_map = get_bb_id_map(batter_bb, 'id')
    pitcher_bb_id_map = get_bb_id_map(pitcher_bb, 'id')
    # Get batter/pitcher MLBAM id for merge
    df_upload['batter_id'] = df_upload['Batter'].apply(get_player_id)
    df_upload['pitcher_id'] = df_upload['Pitcher'].apply(get_player_id)
    # Merge xHR (by normalized name, as xHR CSV may not have ids)
    def normalize_name(name):
        if not isinstance(name, str):
            return ""
        import unicodedata
        name = ''.join(c for c in unicodedata.normalize('NFD', name)
                    if unicodedata.category(c) != 'Mn')
        name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
        if ',' in name:
            last, first = name.split(',', 1)
            name = f"{first.strip()} {last.strip()}"
        return ' '.join(name.split())
    xhr_df['player_norm'] = xhr_df['player'].apply(normalize_name)
    df_upload['batter_norm'] = df_upload['Batter'].apply(normalize_name)
    df_upload = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='batter_norm', right_on='player_norm', how='left'
    )
    # Main processing loop
    weather_rows, stat_rows, park_factor_rows, batter_bb_rows, pitcher_bb_rows = [], [], [], [], []
    missing_batter_ids, missing_pitcher_ids = set(), set()
    st.write("Fetching Statcast, batted ball stats, weather, park factor, and merging xHR (may take a few minutes)...")
    progress = st.progress(0)
    for idx, row in df_upload.iterrows():
        city = row['City']
        date = row['Date']
        park = row['Park']
        batter = row['Batter']
        pitcher = row['Pitcher']
        batter_id = row['batter_id']
        pitcher_id = row['pitcher_id']
        game_time = row['Time']
        park_orientation = ballpark_orientations.get(park, "N")
        park_factor = park_factors.get(park, 1.0)
        # Weather
        weather = get_weather(city, date, park_orientation, game_time)
        # Statcast/SLG
        batter_stats = get_batter_stats_multi(batter, windows)
        pitcher_stats = get_pitcher_stats_multi(pitcher, windows)
        # Batted ball profiles by id
        batter_bb_stats = merge_bb_stats_by_id(batter_id, batter_bb_id_map, "B")
        pitcher_bb_stats = merge_bb_stats_by_id(pitcher_id, pitcher_bb_id_map, "P")
        if batter_id not in batter_bb_id_map:
            missing_batter_ids.add(batter_id)
        if pitcher_id not in pitcher_bb_id_map:
            missing_pitcher_ids.add(pitcher_id)
        # Collect
        weather_rows.append(weather)
        stat_row = {}
        stat_row.update(batter_stats)
        stat_row.update(pitcher_stats)
        stat_rows.append(stat_row)
        park_factor_rows.append({"ParkFactor": park_factor, "BallparkCity": city})
        batter_bb_rows.append(batter_bb_stats)
        pitcher_bb_rows.append(pitcher_bb_stats)
        pct = int(100 * (idx+1)/len(df_upload))
        progress.progress((idx+1)/len(df_upload), text=f"Processing {pct}%")
    # Build dataframes
    weather_df = pd.DataFrame(weather_rows)
    stat_df = pd.DataFrame(stat_rows)
    park_df = pd.DataFrame(park_factor_rows)
    batterbb_df = pd.DataFrame(batter_bb_rows)
    pitcherbb_df = pd.DataFrame(pitcher_bb_rows)
    df_final = pd.concat([df_upload.reset_index(drop=True), weather_df, park_df, stat_df, batterbb_df, pitcherbb_df], axis=1)
    # Handedness (simple placeholder)
    df_final['BatterHandedness'] = "UNK"
    df_final['PitcherHandedness'] = "UNK"
    # xHR Regression
    df_final['Reg_xHR'] = df_final['xhr'] - df_final['hr_total']
    # --- SCORING FUNCTION ---
    def calc_hr_score(row):
        batter_score = (
            norm_barrel(row.get('B_BarrelRate_14')) * 0.13 +
            norm_barrel(row.get('B_BarrelRate_7')) * 0.11 +
            norm_ev(row.get('B_EV_14')) * 0.08 +
            (float(row.get('B_SLG_14')) if row.get('B_SLG_14') else 0) * 0.08 +
            # Advanced batted ball
            (float(row.get('B_hardhit_pct')) if row.get('B_hardhit_pct') else 0) * 0.02 +
            (float(row.get('B_barrel_pct')) if row.get('B_barrel_pct') else 0) * 0.03 +
            (float(row.get('B_pull_pct')) if row.get('B_pull_pct') else 0) * 0.01 +
            (float(row.get('B_oppo_pct')) if row.get('B_oppo_pct') else 0) * 0.01 +
            (float(row.get('B_gb_pct')) if row.get('B_gb_pct') else 0) * 0.01 +
            (float(row.get('B_fb_pct')) if row.get('B_fb_pct') else 0) * 0.01 +
            (float(row.get('B_ld_pct')) if row.get('B_ld_pct') else 0) * 0.01 +
            (float(row.get('B_hr_fb_pct')) if row.get('B_hr_fb_pct') else 0) * 0.02
        )
        pitcher_score = (
            norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.07 +
            norm_barrel(row.get('P_BarrelRateAllowed_7')) * 0.05 +
            norm_ev(row.get('P_EVAllowed_14')) * 0.05 +
            (float(row.get('P_SLG_14')) if row.get('P_SLG_14') else 0) * -0.08 +
            # Advanced batted ball (penalty)
            (float(row.get('P_hardhit_pct')) if row.get('P_hardhit_pct') else 0) * -0.01 +
            (float(row.get('P_barrel_pct')) if row.get('P_barrel_pct') else 0) * -0.02 +
            (float(row.get('P_pull_pct')) if row.get('P_pull_pct') else 0) * -0.01 +
            (float(row.get('P_oppo_pct')) if row.get('P_oppo_pct') else 0) * -0.01 +
            (float(row.get('P_gb_pct')) if row.get('P_gb_pct') else 0) * -0.01 +
            (float(row.get('P_fb_pct')) if row.get('P_fb_pct') else 0) * -0.01 +
            (float(row.get('P_ld_pct')) if row.get('P_ld_pct') else 0) * -0.01 +
            (float(row.get('P_hr_fb_pct')) if row.get('P_hr_fb_pct') else 0) * -0.02
        )
        park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.10
        weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.13
        regression_score = max(0, min((row.get('Reg_xHR', 0) or 0) / 5, 0.12))
        total = batter_score + pitcher_score + park_score + weather_score + regression_score
        total += custom_2025_boost(row)
        return round(total, 3)
    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values('HR_Score', ascending=False)
    st.success("All done! Top matchups below:")
    show_cols = [
        'Batter','Pitcher','BatterHandedness','PitcherHandedness','Park','BallparkCity','Time','HR_Score','Reg_xHR',
        'B_BarrelRate_14','B_EV_14','B_SLG_14','ParkFactor','Temp','Wind','WindEffect',
        'P_BarrelRateAllowed_14','P_EVAllowed_14','P_SLG_14',
        'B_hardhit_pct','B_barrel_pct','B_pull_pct','B_oppo_pct','B_gb_pct','B_fb_pct','B_ld_pct','B_hr_fb_pct',
        'P_hardhit_pct','P_barrel_pct','P_pull_pct','P_oppo_pct','P_gb_pct','P_fb_pct','P_ld_pct','P_hr_fb_pct',
        'xhr','hr_total','xhr_diff'
    ]
    show_cols = [c for c in show_cols if c in df_leaderboard.columns]
    top5 = df_leaderboard.head(5)
    st.dataframe(top5[show_cols])
    # Bar chart for top 5
    if 'Reg_xHR' in top5.columns:
        st.bar_chart(top5.set_index('Batter')[['HR_Score','Reg_xHR']])
    else:
        st.bar_chart(top5.set_index('Batter')[['HR_Score']])
    st.dataframe(df_leaderboard[show_cols])
    csv_out = df_leaderboard.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", csv_out, "hr_leaderboard_all_pitcher_stats.csv")
    # --- Debug missing player IDs ---
    st.write(f"Batted-ball data missing for these batters: {sorted([pid for pid in missing_batter_ids if pid is not None])}")
    st.write(f"Batted-ball data missing for these pitchers: {sorted([pid for pid in missing_pitcher_ids if pid is not None])}")
else:
    st.info("Please upload all required CSVs: daily matchup, Savant xHR/HR, batter & pitcher batted-ball.")

st.caption("""
- All rolling Statcast and SLG, advanced batted ball stats (by MLBAM id!), park, weather, and xHR regression.
- Full leaderboard, city, chart, CSV export. Missing batted-ball data shown by player id.
""")
