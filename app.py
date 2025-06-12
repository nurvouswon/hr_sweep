import streamlit as st
import pandas as pd
import numpy as np

# --- 1. HARDCODE LOGISTIC WEIGHTS ---
LOGIT_WEIGHTS = {
    # Your full logistic weights dictionary here, truncated for brevity:
    "iso_value": 5.757820079,
    "hit_distance_sc": 0.6411852127,
    "pull_side": 0.5569402386,
    # ... (add all your features and weights)
    "line_drive": -0.7114540736,
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

st.title("MLB HR Predictor – Player + Event CSV Integration")

# --- 2. UPLOAD FILES ---
player_file = st.file_uploader("Upload Player-level CSV (from Analyzer)", type=["csv"])
event_file = st.file_uploader("Upload Event-level CSV (from Analyzer)", type=["csv"])

if player_file is None or event_file is None:
    st.warning("Please upload BOTH the player-level and event-level CSVs from Analyzer.")
    st.stop()

# --- 3. LOAD DATA ---
df_player = pd.read_csv(player_file, low_memory=False)
df_event = pd.read_csv(event_file, low_memory=False)

# --- 4. Find all features needed by weights ---
needed_features = set(LOGIT_WEIGHTS.keys())
player_cols = set(df_player.columns)
event_cols = set(df_event.columns)

# Features NOT in player-level
missing_in_player = needed_features - player_cols

# --- 5. For each missing feature, try to get from event-level (latest event per batter) ---
event_features_for_merge = list(missing_in_player & event_cols)
if event_features_for_merge:
    # Pick latest event for each batter
    df_event['game_date'] = pd.to_datetime(df_event['game_date'], errors='coerce')
    df_event_sorted = df_event.sort_values(['batter_id', 'game_date'])
    # We'll use the last event for each batter (latest date)
    event_features = (df_event_sorted.groupby('batter_id')
                      .last()[event_features_for_merge]
                      .reset_index())
    # Merge them up to player-level
    df_player = df_player.merge(event_features, on='batter_id', how='left')

# --- 6. After merge, get the FINAL set of columns available for scoring ---
final_cols = set(df_player.columns)
used_weights = {f: w for f, w in LOGIT_WEIGHTS.items() if f in final_cols}

if len(used_weights) < len(LOGIT_WEIGHTS):
    missing_feats = set(LOGIT_WEIGHTS) - set(used_weights)
    st.info(f"Missing features not in either CSV (ignored in score): {sorted(missing_feats)}")

# --- 7. Score function ---
def calc_hr_logit_score(row):
    score = 0
    for feature, weight in used_weights.items():
        v = row.get(feature, 0)
        if pd.isnull(v):
            v = 0
        score += float(v) * weight
    return score

df_player['hr_logit_score'] = df_player.apply(calc_hr_logit_score, axis=1)
df_player['prob_hr'] = df_player['hr_logit_score'].apply(sigmoid)

# --- 8. LEADERBOARD OUTPUT ---
out_cols = ['batter_id', 'batter', 'hr_logit_score', 'prob_hr'] + [f for f in used_weights if f in df_player.columns]
leaderboard = df_player[out_cols].copy()
leaderboard = leaderboard.sort_values('hr_logit_score', ascending=False).reset_index(drop=True)
leaderboard.insert(0, 'Rank', leaderboard.index + 1)

# --- 9. Show leaderboard
st.dataframe(leaderboard.style.format({"prob_hr": "{:.3f}", "hr_logit_score": "{:.2f}"}), use_container_width=True)

# --- 10. Download leaderboard
st.download_button(
    "Download HR Leaderboard CSV",
    leaderboard.to_csv(index=False),
    file_name="hr_leaderboard.csv"
)
