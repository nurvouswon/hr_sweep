import streamlit as st import pandas as pd import numpy as np import snowflake.connector import io import gc

st.set_page_config("MLB HR Analyzer – Parquet Tools", layout="wide")

-----------------------------------

Snowflake Connection

-----------------------------------

conn = snowflake.connector.connect( user=st.secrets["snowflake"]["user"], password=st.secrets["snowflake"]["password"], account=st.secrets["snowflake"]["account"], warehouse=st.secrets["snowflake"]["warehouse"], database=st.secrets["snowflake"]["database"], schema=st.secrets["snowflake"]["schema"] ) cursor = conn.cursor()

-----------------------------------

File Upload Interface

-----------------------------------

df_hr, df_matchups, df_7, df_14 = None, None, None, None

st.header("Upload Daily Files")

parquet_file = st.file_uploader("Upload Parquet File (daily HR data)", type=["parquet"]) matchup_file = st.file_uploader("Upload Matchup CSV", type=["csv"]) batted7_file = st.file_uploader("Upload 7-Day Batted Ball CSV", type=["csv"]) batted14_file = st.file_uploader("Upload 14-Day Batted Ball CSV", type=["csv"])

if parquet_file is not None: df_hr = pd.read_parquet(parquet_file) if matchup_file is not None: df_matchups = pd.read_csv(matchup_file) if batted7_file is not None: df_7 = pd.read_csv(batted7_file) if batted14_file is not None: df_14 = pd.read_csv(batted14_file)

-----------------------------------

Snowflake Table Upload Function

-----------------------------------

def upload_df_to_snowflake(df, table_name): if df is not None and not df.empty: placeholders = ", ".join(["%s"] * len(df.columns)) col_names = ", ".join([f'"{col}"' for col in df.columns]) for _, row in df.iterrows(): cursor.execute( f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})", tuple(row) ) conn.commit()

Upload to Snowflake tables

if st.button("Upload All to Snowflake"): if df_hr is not None: upload_df_to_snowflake(df_hr, "daily_hr_data") if df_matchups is not None: upload_df_to_snowflake(df_matchups, "matchups") if df_7 is not None: upload_df_to_snowflake(df_7, "batted_7") if df_14 is not None: upload_df_to_snowflake(df_14, "batted_14") st.success("All files uploaded to Snowflake successfully.")

-----------------------------------

Load From Snowflake and Merge

-----------------------------------

@st.cache_data def load_snowflake_table(table_name): return pd.read_sql(f"SELECT * FROM {table_name}", conn)

if st.button("Load and Merge Data"): df_hr = load_snowflake_table("daily_hr_data") df_matchups = load_snowflake_table("matchups") df_7 = load_snowflake_table("batted_7") df_14 = load_snowflake_table("batted_14")

# Merge step
df = df_hr.merge(df_matchups, on="batter_id", how="left")
df = df.merge(df_7, on="batter_id", how="left")
df = df.merge(df_14, on="batter_id", how="left")

# Final memory cleanup
gc.collect()

# Show preview
st.subheader("Merged Data Preview")
st.dataframe(df.head(50))

# You can now run any scoring/model logic on df below
# Placeholder for model logic:
# df["hr_score"] = model_score(df)

# Display final results (sorted by score or any criteria)
# st.dataframe(df.sort_values("hr_score", ascending=False).head(10))



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
    'NYY': 'New York', 'OAK': 'West Sacramento', 'ATH': 'West Sacramento', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
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
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def downcast_numeric(df):
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

def add_rolling_hr_features(df, id_col, date_col, outcome_col='hr_outcome', windows=[3, 5, 7, 14, 20, 30, 60], prefix=""):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([id_col, date_col])
    results = []
    for name, group in df.groupby(id_col):
        group = group.sort_values(date_col)
        for w in windows:
            col_count = f"{prefix}hr_count_{w}"
            col_rate = f"{prefix}hr_rate_{w}"
            group[col_count] = group[outcome_col].rolling(w, min_periods=1).sum()
            group[col_rate] = group[outcome_col].rolling(w, min_periods=1).mean()
        results.append(group)
    df = pd.concat(results)
    return df

def get_wind_edge(row, batter_profile, pitcher_profile):
    wind_dir = str(row.get('wind_dir_string', '')).lower()
    batter_id = str(row.get('batter_id', ''))
    pitcher_id = str(row.get('pitcher_id', ''))

    b_pull = batter_profile.get(batter_id, {}).get('pull_rate', np.nan)
    b_oppo = batter_profile.get(batter_id, {}).get('oppo_rate', np.nan)
    b_fb = batter_profile.get(batter_id, {}).get('fb_rate', np.nan)
    p_gb = pitcher_profile.get(pitcher_id, {}).get('gb_rate', np.nan)
    p_fb = pitcher_profile.get(pitcher_id, {}).get('fb_rate', np.nan)

    hand = str(row.get('stand', row.get('batter_hand', ''))).upper()

    edge = 1.0
    if not isinstance(wind_dir, str) or wind_dir.strip() == "" or wind_dir == "nan":
        return edge

    if "out" in wind_dir or "o" in wind_dir:
        if "rf" in wind_dir and b_oppo is not np.nan and hand == "R" and b_oppo > 0.28:
            edge *= 1.07
        if "lf" in wind_dir and b_pull is not np.nan and hand == "R" and b_pull > 0.37:
            edge *= 1.10
        if "cf" in wind_dir and b_fb is not np.nan and b_fb > 0.23:
            edge *= 1.05
        if "rf" in wind_dir and b_pull is not np.nan and hand == "L" and b_pull > 0.37:
            edge *= 1.10
        if "lf" in wind_dir and b_oppo is not np.nan and hand == "L" and b_oppo > 0.28:
            edge *= 1.07
    elif "in" in wind_dir or "i" in wind_dir:
        if "rf" in wind_dir and b_oppo is not np.nan and hand == "R" and b_oppo > 0.28:
            edge *= 0.93
        if "lf" in wind_dir and b_pull is not np.nan and hand == "R" and b_pull > 0.37:
            edge *= 0.90
        if "cf" in wind_dir and b_fb is not np.nan and b_fb > 0.23:
            edge *= 0.94
        if "rf" in wind_dir and b_pull is not np.nan and hand == "L" and b_pull > 0.37:
            edge *= 0.90
        if "lf" in wind_dir and b_oppo is not np.nan and hand == "L" and b_oppo > 0.28:
            edge *= 0.93

    if p_fb is not np.nan and p_fb > 0.24:
        if "out" in wind_dir or "o" in wind_dir:
            edge *= 1.05
        elif "in" in wind_dir or "i" in wind_dir:
            edge *= 0.97
    if p_gb is not np.nan and p_gb > 0.49:
        edge *= 0.97

    return edge

tab1, tab2 = st.tabs(["1️⃣ Combine Parquet Files", "2️⃣ Generate TODAY CSV + Batted Ball Profile Overlay"])

# ---------------- TAB 1: Combine Parquet Files ----------------
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
        out_parquet = io.BytesIO()
        combined.to_parquet(out_parquet, index=False)
        st.download_button(
            "⬇️ Download Combined Parquet",
            data=out_parquet.getvalue(),
            file_name="event_level_combined.parquet",
            mime="application/octet-stream"
        )
    else:
        st.info("Upload two event-level Parquet files to combine.")

# ---------------- TAB 2: Generate TODAY CSV + Overlay ----------------
with tab2:
    st.header("Generate TODAY CSV with Weather, Context, & Batted Ball Overlay")
    p_event = st.file_uploader("Upload Event-Level Parquet", type=["parquet"], key="event_parquet")
    lineup_csv = st.file_uploader("Upload Today's Matchups CSV", type=["csv"], key="lineup_csv")
    bb_batter_csv = st.file_uploader("Upload Batter Batted Ball Profiles CSV (optional)", type=["csv"], key="bb_batter_csv")
    bb_pitcher_csv = st.file_uploader("Upload Pitcher Batted Ball Profiles CSV (optional)", type=["csv"], key="bb_pitcher_csv")
    run_btn = st.button("Generate TODAY CSV", key="run_btn")
    if run_btn and p_event and lineup_csv:
        df = pd.read_parquet(p_event)
        st.write("[Diagnostics] Loaded event-level shape:", df.shape)
        st.write("[Diagnostics] Columns:", list(df.columns))

        roll_windows = [3, 5, 7, 14, 20, 30, 60]
        if 'hr_outcome' in df.columns:
            df = add_rolling_hr_features(df, id_col='batter_id', date_col='game_date', outcome_col='hr_outcome', windows=roll_windows, prefix='b_')
            df = add_rolling_hr_features(df, id_col='pitcher_id', date_col='game_date', outcome_col='hr_outcome', windows=roll_windows, prefix='p_')

        lineup_df = pd.read_csv(lineup_csv)
        lineup_df.columns = [str(c).strip().lower().replace(" ", "_") for c in lineup_df.columns]

        if 'team_code' not in lineup_df.columns:
            st.error("Missing 'team code' column in your matchup CSV.")
            st.stop()
        if 'time' not in lineup_df.columns:
            st.error("Missing 'time' column in your matchup CSV.")
            st.stop()

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

        lineup_df['pitcher_id'] = np.nan
        grouped = lineup_df.groupby(['game_date', 'park'])
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

        batter_profile = {}
        if bb_batter_csv:
            bb_bat = pd.read_csv(bb_batter_csv)
            for _, row in bb_bat.iterrows():
                pid = str(row['batter_id']) if 'batter_id' in row else str(row.get('player_id', ''))
                batter_profile[pid] = row.to_dict()
        pitcher_profile = {}
        if bb_pitcher_csv:
            bb_pitch = pd.read_csv(bb_pitcher_csv)
            for _, row in bb_pitch.iterrows():
                pid = str(row['pitcher_id']) if 'pitcher_id' in row else str(row.get('player_id', ''))
                pitcher_profile[pid] = row.to_dict()

        all_event_cols = list(df.columns)
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status', 'city',
            'batter_hand', 'pitcher_hand',
            'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
            'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'stand',
            'team_code', 'time'
        ] + extra_context_cols

        rolling_feature_cols = [col for col in all_event_cols if (
            (col.startswith('b_') or col.startswith('p_')) or ('rolling_' in col) or ('barrel_rate' in col) or ('hard_hit_rate' in col)
        )]
        today_cols += [c for c in all_event_cols if c not in today_cols and c in rolling_feature_cols]

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
                    'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp',
                    'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ] if c in all_event_cols or c in extra_context_cols}
            else:
                row_out = {c: np.nan for c in rolling_feature_cols + [
                    'batter_hand', 'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status',
                    'city', 'pitcher_hand',
                    'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
                    'pitcher_team_code', 'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp',
                    'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ] if c in all_event_cols or c in extra_context_cols}

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

            # --------- Overlay Multiplier Calculation (Wind/Batted Ball Profile Edge) ----------
            try:
                overlay_multiplier = get_wind_edge(
                    row, batter_profile, pitcher_profile
                )
            except Exception:
                overlay_multiplier = 1.0

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
                "team_code": team_code,
                "time": row.get("time", np.nan),
                "temp": row.get("temp", np.nan),
                "humidity": row.get("humidity", np.nan),
                "wind_mph": row.get("wind_mph", np.nan),
                "wind_dir_string": row.get("wind_dir_string", np.nan),
                "condition": row.get("condition", np.nan),
                "overlay_multiplier": overlay_multiplier
            })
            today_rows.append(row_out)

        today_df = pd.DataFrame(today_rows)
        today_df = dedup_columns(today_df)
        today_df = downcast_numeric(today_df)

        # Ensure columns are sorted as in event-level if possible
        event_col_set = set(df.columns)
        today_ordered_cols = [col for col in df.columns if col in today_df.columns] + [col for col in today_df.columns if col not in df.columns]
        if "team_code" not in today_ordered_cols:
            today_ordered_cols.append("team_code")
        if "time" not in today_ordered_cols:
            today_ordered_cols.append("time")
        if "overlay_multiplier" not in today_ordered_cols:
            today_ordered_cols.append("overlay_multiplier")
        today_df = today_df[today_ordered_cols]

        # Show and offer downloads
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
    else:
        st.info("Upload event-level Parquet, lineup CSV, and (optionally) batted ball profiles, then click 'Generate TODAY CSV'.")
