import streamlit as st
import pandas as pd
import numpy as np

# ----- 1. HARD CODE YOUR LOGISTIC WEIGHTS -----
LOGIT_WEIGHTS = {
    "iso_value": 5.757820079, "hit_distance_sc": 0.6411852127, "pull_side": 0.5569402386,
    "launch_speed_angle": 0.5280235471, "B_pitch_pct_CH_5": 0.3858783912,
    "park_handed_hr_rate": 0.3438658641, "B_median_ev_7": 0.33462617, "B_pitch_pct_CU_3": 0.3280395666,
    "P_max_ev_5": 0.3113203434, "P_pitch_pct_SV_3": 0.2241205438, "B_pitch_pct_EP_5": 0.2163322514,
    "P_pitch_pct_ST_14": 0.2052831283, "P_rolling_hr_rate_7": 0.1877664166, "P_pitch_pct_FF_5": 0.1783978536,
    "P_median_ev_3": 0.1752142738, "groundball": 0.1719989086, "B_pitch_pct_KC_5": 0.1615036223,
    "B_pitch_pct_FS_3": 0.1595644445, "P_pitch_pct_FC_14": 0.1591148241, "B_pitch_pct_SI_14": 0.1570044892,
    "B_max_ev_5": 0.1540596514, "P_pitch_pct_CU_7": 0.1524371468, "P_pitch_pct_SL_3": 0.1429928993,
    "P_pitch_pct_FO_14": 0.1332430394, "B_pitch_pct_SV_5": 0.1257929016, "P_hit_distance_sc_7": 0.1236586016,
    "B_iso_value_14": 0.1199768939, "P_woba_value_5": 0.1175567692, "B_pitch_pct_CS_14": 0.1137568069,
    "pitch_pct_FO": 0.1124543401, "B_pitch_pct_FF_7": 0.105404093, "is_barrel": 0.1044204311,
    "B_pitch_pct_FA_7": 0.1041956255, "pitch_pct_FF": 0.1041947265, "B_pitch_pct_ST_3": 0.1016502344,
    # ... (add all features from your posted weights here)
    "line_drive": -0.7114540736
}
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----- 2. FILE UPLOADS -----
st.title("MLB HR Predictor Leaderboard (Player + Event CSV, Full Detail)")
player_file = st.file_uploader("Player-level CSV (from Analyzer)", type=["csv"])
event_file = st.file_uploader("Event-level CSV (from Analyzer)", type=["csv"])
if player_file is None or event_file is None:
    st.stop()

df_player = pd.read_csv(player_file)
df_event = pd.read_csv(event_file)

# ----- 3. FIND ALL COLUMNS NEEDED -----
player_cols = set(df_player.columns)
event_cols = set(df_event.columns)
needed_features = set(LOGIT_WEIGHTS.keys())

missing_in_player = needed_features - player_cols
merge_features = list(missing_in_player & event_cols)

# ----- 4. AGGREGATE LATEST EVENT FEATURES PER BATTER -----
if merge_features:
    df_event['game_date'] = pd.to_datetime(df_event['game_date'], errors='coerce')
    latest_events = df_event.sort_values(['batter_id','game_date']).groupby('batter_id').last().reset_index()
    event_merge = latest_events[['batter_id'] + merge_features]
    df_player = df_player.merge(event_merge, on='batter_id', how='left')

# ----- 5. RENAME/ENSURE PLAYER NAME COLUMNS -----
# Use 'batter' as the display name (if it's just an ID, try event-level for names)
if 'batter' not in df_player.columns or df_player['batter'].astype(str).str.isnumeric().all():
    if 'batter' in df_event.columns:
        batter_names = df_event.groupby('batter_id')['batter'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else x.iloc[0])
        df_player = df_player.set_index('batter_id')
        df_player['batter'] = batter_names
        df_player = df_player.reset_index()

# Clean up player names
if 'batter' in df_player.columns:
    df_player['batter'] = df_player['batter'].astype(str).str.title().str.replace('_',' ')

# ----- 6. USE ONLY AVAILABLE WEIGHTS -----
used_weights = {k: v for k,v in LOGIT_WEIGHTS.items() if k in df_player.columns}
missing = sorted(set(LOGIT_WEIGHTS)-set(used_weights))
if missing:
    st.info(f"Missing features (not in either CSV): {missing}")

def calc_hr_logit_score(row):
    score = 0
    for f, w in used_weights.items():
        v = row.get(f, 0)
        if pd.isnull(v): v = 0
        score += float(v) * w
    return score

df_player['hr_logit_score'] = df_player.apply(calc_hr_logit_score, axis=1)
df_player['prob_hr'] = df_player['hr_logit_score'].apply(sigmoid)

# ----- 7. ENHANCED LEADERBOARD COLUMNS -----
summary_cols = [
    'Rank', 'batter_id', 'batter', 'hr_logit_score', 'prob_hr',
    # Showcase most important raw features from player-level (edit this for your context!)
    'B_launch_speed_7','B_launch_angle_7','B_hit_distance_sc_7','B_woba_value_7','B_iso_value_7','B_xwoba_7','B_max_ev_7','B_median_ev_7',
    'B_pitch_pct_CH_5','B_pitch_pct_CU_3','B_pitch_pct_KC_5','B_pitch_pct_SI_14','B_pitch_pct_FS_3','B_pitch_pct_SL_3','B_pitch_pct_EP_5',
    'pull_side','line_drive','groundball','is_barrel','is_sweet_spot','is_hard_hit','flyball'
]
# Add context features if present
for col in ['park_handed_hr_rate','park_hr_rate','park_altitude','platoon']:
    if col in df_player.columns and col not in summary_cols:
        summary_cols.append(col)
# Ensure only available columns are used
summary_cols = [col for col in summary_cols if col in df_player.columns]

# ----- 8. CREATE & SHOW LEADERBOARD -----
leaderboard = df_player.copy()
leaderboard = leaderboard.sort_values('hr_logit_score', ascending=False).reset_index(drop=True)
leaderboard.insert(0,'Rank', leaderboard.index+1)
leaderboard_display = leaderboard[summary_cols]
st.dataframe(
    leaderboard_display.style.format({"prob_hr":"{:.3f}","hr_logit_score":"{:.2f}"}),
    use_container_width=True
)

# ----- 9. FULL DOWNLOADABLE VERSION (all scored features included) -----
leaderboard_all = leaderboard[['Rank','batter_id','batter','hr_logit_score','prob_hr'] + list(used_weights.keys())]
st.download_button("Download Full Leaderboard", leaderboard_all.to_csv(index=False), file_name="hr_leaderboard_full.csv")
