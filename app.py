import streamlit as st
import pandas as pd
import requests
import unicodedata
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from pybaseball.lahman import people
from datetime import datetime, timedelta
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

MANUAL_HANDEDNESS = {
    'alexander canario': ('R', 'R'),
    'liam hicks': ('L', 'R'),
    'patrick bailey': ('B', 'R'),
    # Add more as needed
}
UNKNOWNS_LOG = set()
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

# --- BATTER/PITCHER BATTED BALL PROFILE HANDLING ---

def get_batter_battedball_stats(batter, bb_df):
    n = normalize_name(batter)
    row = bb_df[bb_df['batter_norm'] == n]
    out = {}
    for col in ['xwOBA', 'SLG', 'Sweet Spot %', 'Pull %', 'Oppo %', 'GB/FB', 'HardHit %']:
        k = f"B_{col.replace(' ', '_').replace('%','pct').lower()}_14"
        out[k] = row[col].values[0] if not row.empty and col in row else None
    return out

def get_pitcher_battedball_stats(pitcher, bb_df):
    n = normalize_name(pitcher)
    row = bb_df[bb_df['pitcher_norm'] == n]
    out = {}
    for col in ['xwOBA', 'SLG', 'Sweet Spot %', 'Pull %', 'Oppo %', 'GB/FB', 'HardHit %']:
        k = f"P_{col.replace(' ', '_').replace('%','pct').lower()}_14"
        out[k] = row[col].values[0] if not row.empty and col in row else None
    return out

# --- HR SCORE NORMALIZERS ---
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
                if hour < 17:   # Before 5 PM local
                    bonus -= 0.01
            except Exception:
                bonus -= 0.01
        else:
            bonus -= 0.01
    if row.get('PitcherHandedness') == 'L': bonus += 0.01
    return bonus

windows = [3, 5, 7, 14]

st.title("⚾️ MLB HR Matchup Leaderboard (All Stats, Batted Ball, Park, Weather, 2025 Micro-Trends)")

st.markdown("""
Upload:
- Daily matchup CSV: `Batter,Pitcher,City,Park,Date,Time`
- xHR/HR CSV: `player,hr_total,xhr,xhr_diff`
- Batter batted-ball profile: **rolling 14d** (CSV, stathead/fangraphs export ok)
- Pitcher batted-ball profile: **rolling 14d** (CSV)
""")

uploaded_file = st.file_uploader("Upload daily matchup CSV", type=["csv"])
xhr_file = st.file_uploader("Upload Savant xHR/HR CSV", type=["csv"])
batter_bb_file = st.file_uploader("Upload Batter Batted Ball CSV", type=["csv"])
pitcher_bb_file = st.file_uploader("Upload Pitcher Batted Ball CSV", type=["csv"])

if uploaded_file and xhr_file and batter_bb_file and pitcher_bb_file:
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
    df_merged = df_upload.merge(
        xhr_df[['player_norm', 'hr_total', 'xhr', 'xhr_diff']],
        left_on='batter_norm', right_on='player_norm', how='left'
    )

    batter_bb = pd.read_csv(batter_bb_file)
    pitcher_bb = pd.read_csv(pitcher_bb_file)
    batter_bb['batter_norm'] = batter_bb['Name'].apply(normalize_name)
    pitcher_bb['pitcher_norm'] = pitcher_bb['Name'].apply(normalize_name)

    weather_rows, stat_rows, bb_rows = [], [], []
    st.write("Fetching batted-ball stats, weather, park, xHR... (this may take a minute)")
    progress = st.progress(0)
    for idx, row in df_merged.iterrows():
        city, date, park = row['City'], row['Date'], row['Park']
        batter, pitcher, game_time = row['Batter'], row['Pitcher'], row['Time']
        park_orientation = ballpark_orientations.get(park, "N")
        park_factor = park_factors.get(park, 1.0)
        weather = get_weather(city, date, park_orientation, game_time)
        batter_bb_stats = get_batter_battedball_stats(batter, batter_bb)
        pitcher_bb_stats = get_pitcher_battedball_stats(pitcher, pitcher_bb)
        weather_rows.append(weather)
        stat_row = {
            "ParkFactor": park_factor,
            "BallparkCity": city
        }
        stat_row.update(batter_bb_stats)
        stat_row.update(pitcher_bb_stats)
        stat_rows.append(stat_row)
        progress.progress((idx+1)/len(df_merged))
    stat_df = pd.DataFrame(stat_rows)
    weather_df = pd.DataFrame(weather_rows)
    df_final = pd.concat([df_merged.reset_index(drop=True), weather_df, stat_df], axis=1)

    # Handedness
    batter_handedness, pitcher_handedness = [], []
    for idx, row in df_final.iterrows():
        b_bats, _ = get_handedness(row['Batter'])
        _, p_throws = get_handedness(row['Pitcher'])
        batter_handedness.append(b_bats)
        pitcher_handedness.append(p_throws)
    df_final['BatterHandedness'] = [b if b is not None else "UNK" for b in batter_handedness]
    df_final['PitcherHandedness'] = [p if p is not None else "UNK" for p in pitcher_handedness]
    df_final['Reg_xHR'] = df_final['xhr'] - df_final['hr_total']

    # Calculate HR Score
    def calc_hr_score(row):
        # Use xwOBA, SLG, Sweet Spot, Pull, Oppo, GB/FB, HardHit pct for batter/pitcher!
        batter_score = (
            (row.get('B_xwoba_14') or 0) * 0.10 +
            (row.get('B_slg_14') or 0) * 0.07 +
            (row.get('B_sweet_spot_pct_14') or 0) * 0.03 +
            (row.get('B_pull_pct_14') or 0) * 0.01 +
            (row.get('B_oppo_pct_14') or 0) * 0.01 +
            (row.get('B_gb_fb_14') or 0) * 0.01 +
            (row.get('B_hardhit_pct_14') or 0) * 0.02
        )
        pitcher_score = (
            -(row.get('P_xwoba_14') or 0) * 0.08 +
            -(row.get('P_slg_14') or 0) * 0.06 +
            -(row.get('P_sweet_spot_pct_14') or 0) * 0.02 +
            -(row.get('P_pull_pct_14') or 0) * 0.01 +
            -(row.get('P_oppo_pct_14') or 0) * 0.01 +
            -(row.get('P_gb_fb_14') or 0) * 0.01 +
            -(row.get('P_hardhit_pct_14') or 0) * 0.02
        )
        # Park, weather, xHR regression
        park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.10
        weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.13
        regression_score = max(0, min((row.get('Reg_xHR', 0) or 0) / 5, 0.10))
        total = batter_score + pitcher_score + park_score + weather_score + regression_score
        total += custom_2025_boost(row)
        return round(total, 3)

    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values('HR_Score', ascending=False)

    st.success("All done! Top matchups below:")

    show_cols = [
        'Batter','Pitcher','BatterHandedness','PitcherHandedness','Park','BallparkCity','Time','HR_Score','Reg_xHR',
        # Batter
        'B_xwoba_14','B_slg_14','B_sweet_spot_pct_14','B_pull_pct_14','B_oppo_pct_14','B_gb_fb_14','B_hardhit_pct_14',
        # Pitcher
        'P_xwoba_14','P_slg_14','P_sweet_spot_pct_14','P_pull_pct_14','P_oppo_pct_14','P_gb_fb_14','P_hardhit_pct_14',
        # Park/Weather/xHR
        'ParkFactor','Temp','Wind','WindEffect','xhr','hr_total','xhr_diff'
    ]
    show_cols = [c for c in show_cols if c in df_leaderboard.columns]

    top5 = df_leaderboard.head(5)
    st.dataframe(top5[show_cols])

    # Bar chart for top 5 (HR_Score and Reg_xHR)
    if 'Reg_xHR' in top5.columns:
        st.bar_chart(top5.set_index('Batter')[['HR_Score','Reg_xHR']])
    else:
        st.bar_chart(top5.set_index('Batter')[['HR_Score']])

    st.dataframe(df_leaderboard[show_cols])
    csv_out = df_leaderboard.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", csv_out, "hr_leaderboard_all_stats.csv")

else:
    st.info("Please upload your daily matchup, xHR/HR CSV, and batted-ball profile CSVs to begin.")

st.caption("""
- **All rolling batter and pitcher advanced stats per window (xwOBA, SLG, Sweet Spot%, Pull%, Oppo%, GB/FB, HardHit%, etc) included.**
- Weather, wind, park factor, handedness, and xHR regression automated.
- Ballpark city and factors included. Top-5 leaderboard chart and CSV download included.
""")
