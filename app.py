import streamlit as st
import pandas as pd
import requests
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup
from datetime import datetime, timedelta

API_KEY = "11ac3c31fb664ba8971102152251805"

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

# Updated 2025 park factors (adjusted where applicable)
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
        if game_time:
            game_hour = int(game_time.split(":")[0])
        else:
            game_hour = 14  # Default to 2pm
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

def get_handedness(name):
    try:
        first, last = name.split(" ", 1)
        lookup = playerid_lookup(last, first)
        if not lookup.empty:
            throws = lookup.iloc[0]['throws']
            bats = lookup.iloc[0]['bats']
            return bats, throws
    except Exception:
        return None, None
    return None, None

def get_batter_stats_multi(batter_name, windows):
    pid = get_player_id(batter_name)
    out = {}
    if not pid:
        for w in windows:
            out[f"B_BarrelRate_{w}"] = None
            out[f"B_EV_{w}"] = None
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_batter(start, end, pid)
            if df.empty:
                out[f"B_BarrelRate_{w}"] = None
                out[f"B_EV_{w}"] = None
                continue
            df = df[df['type'] == 'X']
            df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
            barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
            total = len(df)
            barrel_rate = barrels / total if total > 0 else 0
            ev = df['launch_speed'].mean() if total > 0 else None
            out[f"B_BarrelRate_{w}"] = round(barrel_rate,3)
            out[f"B_EV_{w}"] = round(ev,1) if ev else None
        except Exception:
            out[f"B_BarrelRate_{w}"] = None
            out[f"B_EV_{w}"] = None
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
        return out
    for w in windows:
        start = (datetime.now() - timedelta(days=w)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = statcast_pitcher(start, end, pid)
            if df.empty:
                for stat in stats_list:
                    out[f"P_{stat}_{w}"] = None
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
            out[f"P_BarrelRateAllowed_{w}"] = round(barrels / total, 3) if total > 0 else None
            out[f"P_EVAllowed_{w}"] = round(ev_allowed, 1) if ev_allowed else None
            out[f"P_HRAllowed_{w}"] = hrs
            out[f"P_BIP_{w}"] = total
            out[f"P_HardHitRate_{w}"] = round(hard_hit / total, 3) if total > 0 else None
            out[f"P_FlyBallRate_{w}"] = round(fly_balls / total, 3) if total > 0 else None
            out[f"P_KRate_{w}"] = round(k / total, 3) if total > 0 else None
            out[f"P_BBRate_{w}"] = round(bb / total, 3) if total > 0 else None
            out[f"P_HR9_{w}"] = round(hr9, 2) if hr9 else None
        except Exception:
            for stat in stats_list:
                out[f"P_{stat}_{w}"] = None
    return out

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

# -------- 2025 MICRO-TRENDS INTEGRATION --------
def custom_2025_boost(row):
    bonus = 0
    # A. League-wide: More barrels = more HRs
    bonus += norm_barrel(row.get('B_BarrelRate_14')) * 0.01
    # B. Citi Field up
    if row.get('Park') == 'Citi Field': bonus += 0.025
    # B. Comerica up
    if row.get('Park') == 'Comerica Park': bonus += 0.02
    # C. Wrigley wind-out super-boost
    if row.get('Park') == 'Wrigley Field' and row.get('WindEffect') == 'out': bonus += 0.03
    # D. Milwaukee/Philly wind-out bonus
    if row.get('Park') in ['American Family Field','Citizens Bank Park'] and row.get('WindEffect') == 'out': bonus += 0.015
    # C. Dodger RHB pull HR bonus
    if row.get('Park') == 'Dodger Stadium' and row.get('BatterHandedness') == 'R': bonus += 0.01
    # D. Extra warm weather HR boost
    if row.get('Temp') and row.get('Temp') > 80: bonus += 0.01
    # F. Pull-side RHB HR-friendly parks
    if row.get('BatterHandedness') == 'R' and row.get('Park') in [
        "Yankee Stadium","Great American Ball Park","Guaranteed Rate Field"]: bonus += 0.012
    # G. High humidity, east/south park
    if row.get('Humidity') and row.get('Humidity') > 65 and row.get('Park') in ["Truist Park","LoanDepot Park"]: bonus += 0.01
    # H. West coast (marine layer dampening), use time of day
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
    # J. Pitcher is LH
    if row.get('PitcherHandedness') == 'L': bonus += 0.01
    return bonus

windows = [3, 5, 7, 14]

st.title("⚾️ MLB HR Matchup Leaderboard (All Pitcher Stats, Handedness, 2025 Micro-Trends, Game Time Weather)")

st.markdown("""
**Upload your daily matchup CSV:**  
`Batter,Pitcher,City,Park,Date,Time`  
And a **Baseball Savant xHR/HR CSV** (with columns `player_name`, `hr`, `xhr`)
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
    xhr_df = xhr_df.rename(columns={c: c.lower() for c in xhr_df.columns})

    # Auto-fill handedness columns
    batter_handedness = []
    pitcher_handedness = []
    for idx, row in df_upload.iterrows():
        b_bats, _ = get_handedness(row['Batter'])
        _, p_throws = get_handedness(row['Pitcher'])
        batter_handedness.append(b_bats)
        pitcher_handedness.append(p_throws)
    df_upload['BatterHandedness'] = batter_handedness
    df_upload['PitcherHandedness'] = pitcher_handedness

    weather_rows, stat_rows, park_factor_rows = [], [], []
    st.write("Fetching Statcast, weather (game time), park factor, and merging xHR (may take a few minutes)...")
    progress = st.progress(0)
    for idx, row in df_upload.iterrows():
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
        progress.progress((idx+1)/len(df_upload))
    weather_df = pd.DataFrame(weather_rows)
    stat_df = pd.DataFrame(stat_rows)
    park_df = pd.DataFrame(park_factor_rows)
    df_final = pd.concat([df_upload.reset_index(drop=True), weather_df, park_df, stat_df], axis=1)

    # Merge in xHR/HR regression from Savant leaderboard
    df_final = df_final.merge(
        xhr_df[['player_name','hr','xhr']],
        left_on='Batter', right_on='player_name', how='left'
    )
    df_final['Reg_xHR'] = df_final['xhr'] - df_final['hr']

    def calc_hr_score(row):
        batter_score = (
            norm_barrel(row.get('B_BarrelRate_14')) * 0.15 +
            norm_barrel(row.get('B_BarrelRate_7')) * 0.12 +
            norm_barrel(row.get('B_BarrelRate_5')) * 0.08 +
            norm_barrel(row.get('B_BarrelRate_3')) * 0.05 +
            norm_ev(row.get('B_EV_14')) * 0.10 +
            norm_ev(row.get('B_EV_7')) * 0.07 +
            norm_ev(row.get('B_EV_5')) * 0.05 +
            norm_ev(row.get('B_EV_3')) * 0.03
        )
        pitcher_score = (
            norm_barrel(row.get('P_BarrelRateAllowed_14')) * 0.07 +
            norm_barrel(row.get('P_B
        norm_barrel(row.get('P_BarrelRateAllowed_7')) * 0.05 +
            norm_barrel(row.get('P_BarrelRateAllowed_5')) * 0.03 +
            norm_barrel(row.get('P_BarrelRateAllowed_3')) * 0.02 +
            norm_ev(row.get('P_EVAllowed_14')) * 0.05 +
            norm_ev(row.get('P_EVAllowed_7')) * 0.03 +
            norm_ev(row.get('P_EVAllowed_5')) * 0.02 +
            norm_ev(row.get('P_EVAllowed_3')) * 0.01
        )
        park_score = norm_park(row.get('ParkFactor', 1.0)) * 0.1
        weather_score = norm_weather(row.get('Temp'), row.get('Wind'), row.get('WindEffect')) * 0.15
        regression_score = max(0, min((row.get('Reg_xHR', 0) or 0) / 5, 0.15))  # cap boost

        # ----- MICRO-TRENDS 2025 BOOST -----
        total = batter_score + pitcher_score + park_score + weather_score + regression_score
        total += custom_2025_boost(row)
        return round(total, 3)

    df_final['HR_Score'] = df_final.apply(calc_hr_score, axis=1)
    df_leaderboard = df_final.sort_values('HR_Score', ascending=False)

    st.success("All done! Top matchups below:")

    show_cols = [
        'Batter','Pitcher','BatterHandedness','PitcherHandedness','Park','Time','HR_Score','Reg_xHR',
        'B_BarrelRate_14','B_EV_14','ParkFactor','Temp','Wind','WindEffect',
        'P_BarrelRateAllowed_14','P_EVAllowed_14','P_HRAllowed_14','P_BIP_14','P_HardHitRate_14',
        'P_FlyBallRate_14','P_KRate_14','P_BBRate_14','P_HR9_14'
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
    st.download_button("Download Results as CSV", csv_out, "hr_leaderboard_all_pitcher_stats.csv")
else:
    st.info("Please upload your daily CSV and Savant xHR/HR CSV to begin.")

st.caption("""
- **All rolling batter and pitcher stats (3, 5, 7, 14 days) and all advanced pitcher stats per window (Barrel%, EV, HR, BIP, HardHit%, FlyBall%, K%, BB%, HR/9) are included.**
- Weather, wind (at game time!), park factor, handedness, and xHR regression are all automated.
- Latest 2025 micro-trends: park upgrades, humidity, wind, warm weather, pitcher/batter splits, and more.
- CSV download and top-5 leaderboard chart included.
""")
