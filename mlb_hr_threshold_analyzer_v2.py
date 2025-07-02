import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import gc

st.set_page_config("MLB HR Analyzer – Parquet Tools", layout="wide")

# ===================== CONTEXT MAPS & RATES =====================
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82,
    'wrigley_field': 1.12, 'great_american_ball_park': 1.26, 'american_family_field': 1.17,
    'pnc_park': 0.87, 'busch_stadium': 0.87, 'truist_park': 1.06, 'loan_depot_park': 0.86,
    'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05, 'petco_park': 0.85,
    'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12, 'target_field': 1.05
}
park_altitude_map = {
    'coors_field': 5280, 'chase_field': 1100, 'dodger_stadium': 338, 'minute_maid_park': 50,
    'fenway_park': 19, 'wrigley_field': 594, 'great_american_ball_park': 489, 'oracle_park': 10,
    'petco_park': 62, 'yankee_stadium': 55, 'citizens_bank_park': 30, 'kauffman_stadium': 750,
    'guaranteed_rate_field': 600, 'progressive_field': 650, 'busch_stadium': 466, 'camden_yards': 40,
    'rogers_centre': 250, 'angel_stadium': 160, 'tropicana_field': 3, 'citi_field': 3,
    'oakland_coliseum': 50, 'globe_life_field': 560, 'pnc_park': 725, 'loan_depot_park': 7,
    'loandepot_park': 7, 'nationals_park': 25, 'american_family_field': 633, 'sutter_health_park': 20,
    'target_field': 830
}
roof_status_map = {
    'rogers_centre': 'closed', 'chase_field': 'open', 'minute_maid_park': 'open',
    'loan_depot_park': 'closed', 'loandepot_park': 'closed', 'globe_life_field': 'open',
    'tropicana_field': 'closed', 'american_family_field': 'open'
}
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field',
    'BOS': 'fenway_park', 'NYY': 'yankee_stadium', 'CHC': 'wrigley_field',
    'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park', 'ATH': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park',
    'MIA': 'loandepot_park', 'TB': 'tropicana_field', 'MIL': 'american_family_field',
    'SD': 'petco_park', 'SF': 'oracle_park', 'TOR': 'rogers_centre', 'CLE': 'progressive_field',
    'MIN': 'target_field', 'KC': 'kauffman_stadium', 'CWS': 'guaranteed_rate_field',
    'CHW': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park',
    'TEX': 'globe_life_field', 'ARI': 'chase_field', 'AZ': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park',
    'STL': 'busch_stadium', 'BAL': 'camden_yards', 'WSH': 'nationals_park', 'WAS': 'nationals_park'
}
mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'AZ': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'CHW': 'Chicago', 'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim',
    'LAD': 'Los Angeles', 'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York',
    'NYY': 'New York', 'OAK': 'Oakland', 'ATH': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
    'SD': 'San Diego', 'SEA': 'Seattle', 'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg',
    'TEX': 'Arlington', 'TOR': 'Toronto', 'WSH': 'Washington', 'WAS': 'Washington'
}
park_hand_hr_rate_map = {
    'angels_stadium': {'L': 1.09, 'R': 1.02}, 'angel_stadium': {'L': 1.09, 'R': 1.02},
    'minute_maid_park': {'L': 1.13, 'R': 1.06}, 'coors_field': {'L': 1.38, 'R': 1.24},
    'yankee_stadium': {'L': 1.47, 'R': 0.98}, 'fenway_park': {'L': 1.04, 'R': 0.97},
    'rogers_centre': {'L': 1.08, 'R': 1.12}, 'tropicana_field': {'L': 0.84, 'R': 0.89},
    'camden_yards': {'L': 0.98, 'R': 1.27}, 'guaranteed_rate_field': {'L': 1.25, 'R': 1.11},
    'progressive_field': {'L': 0.99, 'R': 1.02}, 'comerica_park': {'L': 1.10, 'R': 0.91},
    'kauffman_stadium': {'L': 0.90, 'R': 1.03}, 'globe_life_field': {'L': 1.01, 'R': 0.98},
    'dodger_stadium': {'L': 1.02, 'R': 1.18}, 'oakland_coliseum': {'L': 0.81, 'R': 0.85},
    't-mobile_park': {'L': 0.81, 'R': 0.92}, 'tmobile_park': {'L': 0.81, 'R': 0.92},
    'oracle_park': {'L': 0.67, 'R': 0.99}, 'wrigley_field': {'L': 1.10, 'R': 1.16},
    'great_american_ball_park': {'L': 1.30, 'R': 1.23}, 'american_family_field': {'L': 1.25, 'R': 1.13},
    'pnc_park': {'L': 0.76, 'R': 0.92}, 'busch_stadium': {'L': 0.78, 'R': 0.91},
    'truist_park': {'L': 1.00, 'R': 1.09}, 'loan_depot_park': {'L': 0.83, 'R': 0.91},
    'loandepot_park': {'L': 0.83, 'R': 0.91}, 'citi_field': {'L': 1.11, 'R': 0.98},
    'nationals_park': {'L': 1.04, 'R': 1.06}, 'petco_park': {'L': 0.90, 'R': 0.88},
    'chase_field': {'L': 1.16, 'R': 1.05}, 'citizens_bank_park': {'L': 1.22, 'R': 1.20},
    'sutter_health_park': {'L': 1.12, 'R': 1.12}, 'target_field': {'L': 1.09, 'R': 1.01}
}
# ========== DEEP RESEARCH HR MULTIPLIERS: BATTER SIDE ===============
park_hr_percent_map_all = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CWS': 1.25,
    'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09,
    'WAS': 1.00, 'WSH': 1.00
}
park_hr_percent_map_rhb = {
    'ARI': 1.00, 'AZ': 1.00, 'ATL': 0.93, 'BAL': 1.09, 'BOS': 0.90, 'CHC': 1.09, 'CHW': 1.26, 'CWS': 1.26,
    'CIN': 1.27, 'CLE': 0.91, 'COL': 1.05, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.84, 'MIL': 1.12, 'MIN': 0.95, 'NYM': 1.11, 'NYY': 1.15, 'OAK': 0.91, 'ATH': 0.91,
    'PHI': 1.18, 'PIT': 0.80, 'SD': 1.02, 'SEA': 1.03, 'SF': 0.76, 'STL': 0.84, 'TB': 0.94, 'TEX': 1.06, 'TOR': 1.11,
    'WAS': 1.02, 'WSH': 1.02
}
park_hr_percent_map_lhb = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.99, 'BAL': 1.13, 'BOS': 0.75, 'CHC': 0.93, 'CHW': 1.23, 'CWS': 1.23,
    'CIN': 1.29, 'CLE': 1.01, 'COL': 1.07, 'DET': 0.96, 'HOU': 1.09, 'KC': 0.81, 'LAA': 1.00, 'LAD': 1.12,
    'MIA': 0.87, 'MIL': 1.19, 'MIN': 0.91, 'NYM': 1.06, 'NYY': 1.28, 'OAK': 0.87, 'ATH': 0.87,
    'PHI': 1.19, 'PIT': 0.90, 'SD': 0.98, 'SEA': 0.96, 'SF': 0.73, 'STL': 0.90, 'TB': 0.99, 'TEX': 1.11, 'TOR': 1.05,
    'WAS': 0.96, 'WSH': 0.96
}
# ========== DEEP RESEARCH HR MULTIPLIERS: PITCHER SIDE ===============
park_hr_percent_map_pitcher_all = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CWS': 1.25,
    'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09,
    'WAS': 1.00, 'WSH': 1.00
}
park_hr_percent_map_rhp = {
    'ARI': 0.97, 'AZ': 0.97, 'ATL': 1.01, 'BAL': 1.16, 'BOS': 0.84, 'CHC': 1.02, 'CHW': 1.28, 'CWS': 1.28,
    'CIN': 1.27, 'CLE': 0.98, 'COL': 1.06, 'DET': 0.95, 'HOU': 1.11, 'KC': 0.84, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.84, 'MIL': 1.14, 'MIN': 0.96, 'NYM': 1.07, 'NYY': 1.24, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.19, 'PIT': 0.85, 'SD': 1.02, 'SEA': 1.01, 'SF': 0.73, 'STL': 0.84, 'TB': 0.97, 'TEX': 1.10, 'TOR': 1.11,
    'WAS': 1.03, 'WSH': 1.03
}
park_hr_percent_map_lhp = {
    'ARI': 0.99, 'AZ': 0.99, 'ATL': 0.79, 'BAL': 0.97, 'BOS': 0.83, 'CHC': 1.03, 'CHW': 1.18, 'CWS': 1.18,
    'CIN': 1.27, 'CLE': 0.89, 'COL': 1.05, 'DET': 0.97, 'HOU': 1.07, 'KC': 0.79, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.90, 'MIL': 1.14, 'MIN': 0.89, 'NYM': 1.05, 'NYY': 1.12, 'OAK': 0.89, 'ATH': 0.89,
    'PHI': 1.16, 'PIT': 0.78, 'SD': 1.02, 'SEA': 0.97, 'SF': 0.82, 'STL': 0.96, 'TB': 0.94, 'TEX': 1.01, 'TOR': 1.06,
    'WAS': 0.90, 'WSH': 0.90
}

# ========================= UTILITY FUNCTIONS =========================

def dedup_columns(df):
    """Remove duplicate columns after merging."""
    return df.loc[:, ~df.columns.duplicated()]

def downcast_numeric(df):
    """Downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def parse_custom_weather_string_v2(s):
    if pd.isna(s):
        return pd.Series([np.nan]*7, index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])
    s = str(s)
    temp_match = re.search(r'(\d{2,3})\s*[OI°]?\s', s)
    temp = int(temp_match.group(1)) if temp_match else np.nan
    wind_vector_match = re.search(r'\d{2,3}\s*([OI])\s', s)
    wind_vector = wind_vector_match.group(1) if wind_vector_match else np.nan
    wind_field_dir_match = re.search(r'\s([A-Z]{2})\s*\d', s)
    wind_field_dir = wind_field_dir_match.group(1) if wind_field_dir_match else np.nan
    mph = re.search(r'(\d{1,3})\s*-\s*(\d{1,3})', s)
    if mph:
        wind_mph = (int(mph.group(1)) + int(mph.group(2))) / 2
    else:
        mph = re.search(r'([1-9][0-9]?)\s*(?:mph)?', s)
        wind_mph = int(mph.group(1)) if mph else np.nan
    humidity_match = re.search(r'(\d{1,3})%', s)
    humidity = int(humidity_match.group(1)) if humidity_match else np.nan
    condition = "outdoor" if "outdoor" in s.lower() else ("indoor" if "indoor" in s.lower() else np.nan)
    wind_dir_string = f"{wind_vector} {wind_field_dir}".strip()
    return pd.Series([temp, wind_vector, wind_field_dir, wind_mph, humidity, condition, wind_dir_string],
                     index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])

# ========================= STREAMLIT APP =========================

tab1, tab2 = st.tabs(["1️⃣ Combine Parquet Files", "2️⃣ Generate TODAY CSV"])

# ---------------------- TAB 1: Combine Parquet Files ----------------------
with tab1:
    st.header("Combine Two Event-Level Parquet Files")
    p1 = st.file_uploader("Upload First Event-Level Parquet", type="parquet", key="p1")
    p2 = st.file_uploader("Upload Second Event-Level Parquet", type="parquet", key="p2")
    if p1 and p2:
        df1 = pd.read_parquet(p1)
        df2 = pd.read_parquet(p2)
        st.write("[Diagnostics] First file shape:", df1.shape)
        st.write("[Diagnostics] Second file shape:", df2.shape)
        combined = pd.concat([df1, df2], ignore_index=True)
        st.write("[Diagnostics] Combined shape:", combined.shape)
        st.dataframe(combined.head(20), use_container_width=True)
        # Download combined
        out_parquet = io.BytesIO()
        combined.to_parquet(out_parquet, index=False)
        st.download_button("⬇️ Download Combined Parquet", data=out_parquet.getvalue(),
                           file_name="event_level_combined.parquet", mime="application/octet-stream")
    else:
        st.info("Upload two event-level Parquet files to combine.")

# ---------------------- TAB 2: Generate TODAY CSV ----------------------
if run_btn and p_event and lineup_csv:
        df = pd.read_parquet(p_event)
        st.write("[Diagnostics] Loaded event-level shape:", df.shape)
        st.write("[Diagnostics] Columns:", list(df.columns))

        lineup_df = pd.read_csv(lineup_csv)
        lineup_df.columns = [str(c).strip().lower().replace(" ", "_") for c in lineup_df.columns]
        if "park" in lineup_df.columns:
            lineup_df["park"] = lineup_df["park"].astype(str).str.lower().str.replace(" ", "_")
        for col in ['mlb_id', 'batter_id']:
            if col in lineup_df.columns and 'batter_id' not in lineup_df.columns:
                lineup_df['batter_id'] = lineup_df[col]
        for col in ['player_name', 'player name', 'name']:
            if col in lineup_df.columns and 'player_name' not in lineup_df.columns:
                lineup_df['player_name'] = lineup_df[col]
        if 'game_date' not in lineup_df.columns:
            for date_col in ['game_date', 'game date']:
                if date_col in lineup_df.columns:
                    lineup_df['game_date'] = lineup_df[date_col]
        if 'game_date' in lineup_df.columns:
            lineup_df['game_date'] = pd.to_datetime(lineup_df['game_date'], errors='coerce').dt.strftime("%Y-%m-%d")
        if 'batting_order' in lineup_df.columns:
            lineup_df['batting_order'] = lineup_df['batting_order'].astype(str).str.upper().str.strip()
        if 'team_code' in lineup_df.columns:
            lineup_df['team_code'] = lineup_df['team_code'].astype(str).str.strip().str.upper()
        if 'game_number' in lineup_df.columns:
            lineup_df['game_number'] = lineup_df['game_number'].astype(str).str.strip()
        for col in ['batter_id', 'mlb_id']:
            if col in lineup_df.columns:
                lineup_df[col] = lineup_df[col].astype(str).str.replace('.0','',regex=False).str.strip()
        if 'weather' in lineup_df.columns:
            wx_parsed = lineup_df['weather'].apply(parse_custom_weather_string_v2)
            lineup_df = pd.concat([lineup_df, wx_parsed], axis=1)

        # ==== Assign Opposing SP for Each Batter ====
        lineup_df['pitcher_id'] = np.nan
        grouped = lineup_df.groupby(['game_date', 'park', 'time']) if 'time' in lineup_df.columns else lineup_df.groupby(['game_date', 'park'])
        for group_key, group in grouped:
            if 'team_code' not in group.columns: continue
            teams = group['team_code'].unique()
            if len(teams) < 2: continue
            team_sps = {}
            for team in teams:
                sp_row = group[(group['team_code'] == team) & (group['batting_order'] == "SP")]
                if not sp_row.empty:
                    team_sps[team] = str(sp_row.iloc[0]['batter_id'])
            for team in teams:
                opp_teams = [t for t in teams if t != team]
                if not opp_teams: continue
                opp_sp = team_sps.get(opp_teams[0], np.nan)
                idx = group[group['team_code'] == team].index
                lineup_df.loc[idx, 'pitcher_id'] = opp_sp

        # ========== BUILD TODAY CSV ==============
        roll_windows = [3, 5, 7, 14, 20, 30, 60]
        rolling_feature_cols = [col for col in df.columns if (
            col.startswith('b_') or col.startswith('p_')
        ) and any(str(w) in col for w in roll_windows)]

        extra_context_cols = [
            'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status', 'city',
            'batter_hand', 'pitcher_hand',
            'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
            'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'stand'
        ] + extra_context_cols + rolling_feature_cols

        pitcher_hand_map = {}
        if 'pitcher_id' in df.columns and 'pitcher_hand' in df.columns:
            pitcher_hand_statcast = df[['pitcher_id', 'pitcher_hand']].drop_duplicates().dropna()
            for _, row_p in pitcher_hand_statcast.iterrows():
                pid = str(row_p['pitcher_id'])
                hand = row_p['pitcher_hand']
                if pid not in pitcher_hand_map and pd.notna(hand):
                    pitcher_hand_map[pid] = hand
        if 'pitcher_id' in lineup_df.columns:
            for _, row_p in lineup_df.dropna(subset=['pitcher_id']).drop_duplicates(['pitcher_id']).iterrows():
                pid = str(row_p['pitcher_id'])
                hand = row_p.get('p_throws') or row_p.get('stand') or row_p.get('pitcher_hand')
                if pid not in pitcher_hand_map and pd.notna(hand):
                    pitcher_hand_map[pid] = hand

        today_rows = []
        for idx, row in lineup_df.iterrows():
            this_batter_id = str(row['batter_id']).split(".")[0]
            park = row.get("park", np.nan)
            city = row.get("city", np.nan)
            team_code = row.get("team_code", np.nan)
            game_date = row.get("game_date", np.nan)
            pitcher_id = str(row.get("pitcher_id", np.nan))
            player_name = row.get("player_name", np.nan)
            stand = row.get("stand", np.nan)
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                row_out = {c: last_row.get(c, np.nan) for c in rolling_feature_cols + [
                    'batter_hand', 'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status',
                    'city', 'pitcher_hand',
                    'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
                    'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ]}
            else:
                row_out = {c: np.nan for c in rolling_feature_cols + [
                    'batter_hand', 'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status',
                    'city', 'pitcher_hand',
                    'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
                    'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ]}

            batter_hand = row.get('stand', row_out.get('batter_hand', np.nan))
            pitcher_hand = pitcher_hand_map.get(pitcher_id, np.nan)
            park_hand_rate = 1.0
            if not pd.isna(park) and not pd.isna(batter_hand):
                park_hand_rate = park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(batter_hand).upper(), 1.0)
            if not pd.isna(team_code):
                park_hr_pct_all = park_hr_percent_map_all.get(team_code, 1.0)
                park_hr_pct_rhb = park_hr_percent_map_rhb.get(team_code, 1.0)
                park_hr_pct_lhb = park_hr_percent_map_lhb.get(team_code, 1.0)
                if str(batter_hand).upper() == "R":
                    park_hr_pct_hand = park_hr_pct_rhb
                elif str(batter_hand).upper() == "L":
                    park_hr_pct_hand = park_hr_pct_lhb
                else:
                    park_hr_pct_hand = park_hr_pct_all
            else:
                park_hr_pct_all = park_hr_pct_rhb = park_hr_pct_lhb = park_hr_pct_hand = 1.0

            pitcher_team_code = row.get("pitcher_team_code", np.nan)
            if pd.isna(pitcher_team_code):
                if 'pitcher_team_code' in row_out and pd.notna(row_out['pitcher_team_code']):
                    pitcher_team_code = row_out['pitcher_team_code']
                elif 'team_code' in row and pd.notna(row['team_code']):
                    pitcher_team_code = row['team_code']
                else:
                    pitcher_team_code = np.nan
            pitcher_hand_val = str(pitcher_hand).upper() if pd.notna(pitcher_hand) else ""
            if not pd.isna(pitcher_team_code):
                pitcher_park_hr_pct_all = park_hr_percent_map_pitcher_all.get(pitcher_team_code, 1.0)
                pitcher_park_hr_pct_rhp = park_hr_percent_map_rhp.get(pitcher_team_code, 1.0)
                pitcher_park_hr_pct_lhp = park_hr_percent_map_lhp.get(pitcher_team_code, 1.0)
                if pitcher_hand_val == "R":
                    pitcher_park_hr_pct_hand = pitcher_park_hr_pct_rhp
                elif pitcher_hand_val == "L":
                    pitcher_park_hr_pct_hand = pitcher_park_hr_pct_lhp
                else:
                    pitcher_park_hr_pct_hand = pitcher_park_hr_pct_all
            else:
                pitcher_park_hr_pct_all = pitcher_park_hr_pct_rhp = pitcher_park_hr_pct_lhp = pitcher_park_hr_pct_hand = 1.0

            row_out.update({
                "game_date": game_date,
                "batter_id": this_batter_id,
                "player_name": player_name,
                "pitcher_id": pitcher_id,
                "park": park,
                "park_hr_rate": park_hr_rate_map.get(str(park).lower(), 1.0) if not pd.isna(park) else 1.0,
                "park_hand_hr_rate": park_hand_rate,
                "park_altitude": park_altitude_map.get(str(park).lower(), 0) if not pd.isna(park) else 0,
                "roof_status": roof_status_map.get(str(park).lower(), "open") if not pd.isna(park) else "open",
                "city": city if not pd.isna(city) else mlb_team_city_map.get(team_code, ""),
                "stand": batter_hand,
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand,
                "park_hr_pct_all": park_hr_pct_all,
                "park_hr_pct_rhb": park_hr_pct_rhb,
                "park_hr_pct_lhb": park_hr_pct_lhb,
                "park_hr_pct_hand": park_hr_pct_hand,
                "pitcher_team_code": pitcher_team_code,
                "pitcher_park_hr_pct_all": pitcher_park_hr_pct_all,
                "pitcher_park_hr_pct_rhp": pitcher_park_hr_pct_rhp,
                "pitcher_park_hr_pct_lhp": pitcher_park_hr_pct_lhp,
                "pitcher_park_hr_pct_hand": pitcher_park_hr_pct_hand,
            })
            for c in ['temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition']:
                row_out[c] = row.get(c, np.nan)
            today_rows.append(row_out)

        today_df = pd.DataFrame(today_rows, columns=today_cols)
        today_df = dedup_columns(today_df)
        today_df = downcast_numeric(today_df)

        st.write("TODAY CSV (sample):", today_df.head(20))
        st.markdown("#### Download TODAY CSV / Parquet (1 row per batter, matchup, rolling features & weather):")
        st.dataframe(today_df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download TODAY CSV",
            data=today_df.to_csv(index=False),
            file_name="today_hr_features.csv",
            key="download_today_csv"
        )
        today_parquet = io.BytesIO()
        today_df.to_parquet(today_parquet, index=False)
        st.download_button(
            "⬇️ Download TODAY Parquet",
            data=today_parquet.getvalue(),
            file_name="today_hr_features.parquet",
            mime="application/octet-stream",
            key="download_today_parquet"
        )
        st.success("All files and debug outputs ready.")
        gc.collect()
