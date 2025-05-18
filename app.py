import streamlit as st
import pandas as pd
from pybaseball import statcast, playerid_lookup
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.title("⚾ HR Predictor")

player_name = st.text_input("Batter Name:", "Shohei Ohtani")
days_back = st.slider("Days Back", 5, 30, 14)

def get_player_id(name):
    first, last = name.split(" ", 1)
    player_info = playerid_lookup(last, first)
    return player_info.iloc[0]['key_mlbam'] if not player_info.empty else None

def get_recent_data(player_id, days_back):
    try:
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=days_back)
        df = statcast(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), player_id=player_id)
        return df
    except Exception as e:
        st.error(f"Data retrieval issue: {e}")
        return pd.DataFrame()

if st.button("Predict HR"):
    player_id = get_player_id(player_name)

    if player_id:
        data = get_recent_data(player_id, days_back)
        if data.empty:
            st.warning("No recent data or retrieval error.")
        else:
            ev = data['launch_speed'].mean()
            barrels = data[(data['launch_angle'] > 20) & (data['launch_speed'] > 95)].shape[0]
            barrel_rate = barrels / len(data) if len(data) > 0 else 0

            score = MinMaxScaler().fit_transform([[ev, barrel_rate]])[0]
            probability = (score[0]*0.6 + score[1]*0.4)*100

            st.success(f"HR Chance: {probability:.2f}%")
            st.write(f"Exit Velo: {ev:.1f} mph, Barrel: {barrel_rate:.2%}")
    else:
        st.error("Player not found.")
