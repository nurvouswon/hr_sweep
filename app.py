import streamlit as st
import pandas as pd
from pybaseball import statcast, playerid_lookup
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.title("⚾ HR Predictor")

player_name = st.text_input("Batter Name:", "Shohei Ohtani")
days_back = st.slider("Days Back", 5, 30, 14)

def get_player_id(name):
    try:
        first, last = name.split(" ", 1)
        player_info = playerid_lookup(last, first)
        if not player_info.empty:
            return int(player_info.iloc[0]['key_mlbam'])
        else:
            return None
    except Exception:
        return None

def get_recent_data(player_id, days_back):
    try:
        days = int(days_back)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        from pybaseball import statcast_batter
        df = statcast_batter(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), player_id)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()
if 'type' in data.columns:
    data = data[data['type'] == "X"]
data = data[data['launch_speed'].notnull() & data['launch_angle'].notnull()]
if st.button("Predict HR"):
    player_id = get_player_id(player_name)
if player_id:
    data = get_recent_data(player_id, days_back)
    st.write(f"Player ID: {player_id}")
    st.write("First 5 data rows (before filtering):")
    st.dataframe(data.head())

    if not data.empty and 'type' in data.columns:
        data = data[data['type'] == "X"]
        data = data[data['launch_speed'].notnull() & data['launch_angle'].notnull()]
        st.write("First 5 data rows (after filtering for batted balls):")
        st.dataframe(data.head())

        if data.empty:
            st.warning("No batted ball events found after filtering.")
        else:
            # --- HR calculation logic goes here ---
            ev = data['launch_speed'].dropna().mean()
            barrels = data[(data['launch_speed'] > 95) & (data['launch_angle'].between(20, 35))].shape[0]
            total = len(data)
            barrel_rate = barrels / total if total > 0 else 0

            if ev is None or pd.isna(ev):
                ev = 0
            # MLB typical exit velo: 80-105, barrel rate: 0-15%
            ev_norm = (ev - 80) / (105 - 80)
            ev_norm = max(0, min(ev_norm, 1))
            br_norm = min(barrel_rate / 0.15, 1)
            probability = (ev_norm * 0.6 + br_norm * 0.4) * 100

            if len(data) < 3:
                st.warning("Warning: Very little recent Statcast data for this player. Probability may be unreliable.")

            st.success(f"HR Probability: {probability:.2f}%")
            st.write(f"Avg Exit Velo: {ev:.1f} mph | Barrel Rate: {barrel_rate:.2%}")
    else:
        st.write("No batted ball events found for this player/date range.")
else:
    st.error("Player not found. Please check the spelling.")
