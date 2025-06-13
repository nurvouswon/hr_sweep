import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Predictor ⚾️")

# ------------ FULL LOGISTIC WEIGHTS (YOUR TABLE) ------------
LOGISTIC_WEIGHTS = {
    'iso_value': 5.757820079, 'hit_distance_sc': 0.6411852127, 'pull_side': 0.5569402386, 'launch_speed_angle': 0.5280235471,
    'B_pitch_pct_CH_5': 0.3858783912, 'park_handed_hr_rate': 0.3438658641, 'B_median_ev_7': 0.33462617, 'B_pitch_pct_CU_3': 0.3280395666,
    'P_max_ev_5': 0.3113203434, 'P_pitch_pct_SV_3': 0.2241205438, 'B_pitch_pct_EP_5': 0.2163322514, 'P_pitch_pct_ST_14': 0.2052831283,
    'P_rolling_hr_rate_7': 0.1877664166, 'P_pitch_pct_FF_5': 0.1783978536, 'P_median_ev_3': 0.1752142738, 'groundball': 0.1719989086,
    'B_pitch_pct_KC_5': 0.1615036223, 'B_pitch_pct_FS_3': 0.1595644445, 'P_pitch_pct_FC_14': 0.1591148241, 'B_pitch_pct_SI_14': 0.1570044892,
    'B_max_ev_5': 0.1540596514, 'P_pitch_pct_CU_7': 0.1524371468, 'P_pitch_pct_SL_3': 0.1429928993, 'P_pitch_pct_FO_14': 0.1332430394,
    'B_pitch_pct_SV_5': 0.1257929016, 'P_hit_distance_sc_7': 0.1236586016, 'B_iso_value_14': 0.1199768939, 'P_woba_value_5': 0.1175567692,
    'B_pitch_pct_CS_14': 0.1137568069, 'pitch_pct_FO': 0.1124543401, 'B_pitch_pct_FF_7': 0.105404093, 'is_barrel': 0.1044204311,
    'B_pitch_pct_FA_7': 0.1041956255, 'pitch_pct_FF': 0.1041947265, 'B_pitch_pct_ST_3': 0.1016502344, 'pitch_pct_ST': 0.09809980426,
    'pitch_pct_CH': 0.09588455603, 'B_pitch_pct_SL_3': 0.09395294235, 'P_rolling_hr_rate_5': 0.09176055559, 'B_pitch_pct_SC_14': 0.08671517652,
    'platoon': 0.08601459992, 'P_pitch_pct_FS_3': 0.08464192523, 'B_iso_value_7': 0.08090866123, 'B_pitch_pct_KC_7': 0.08079362526,
    'B_median_ev_14': 0.07898600411, 'B_pitch_pct_KN_7': 0.07368063279, 'B_pitch_pct_SL_14': 0.07334392117, 'P_pitch_pct_SV_5': 0.06890378686,
    'P_pitch_pct_CH_3': 0.06804529698, 'P_woba_value_7': 0.0674790282, 'B_launch_angle_7': 0.06733255236, 'P_pitch_pct_ST_7': 0.06545350898,
    'B_pitch_pct_FF_14': 0.06491620372, 'P_max_ev_7': 0.06116445719, 'P_max_ev_3': 0.05980174448, 'B_pitch_pct_FC_7': 0.05788952516,
    'B_pitch_pct_FA_3': 0.05587337787, 'pitch_pct_FC': 0.05483038609, 'P_pitch_pct_KC_7': 0.05350923671, 'B_max_ev_3': 0.05203847819,
    'P_launch_angle_5': 0.05141139562, 'P_pitch_pct_CS_14': 0.05139024478, 'B_pitch_pct_FA_14': 0.05021331706, 'P_pitch_pct_CU_14': 0.05020601371,
    'P_rolling_hr_rate_3': 0.04837416267, 'P_pitch_pct_EP_3': 0.04716192902, 'B_pitch_pct_EP_7': 0.04703265604, 'P_iso_value_7': 0.04279584322,
    'P_pitch_pct_CS_7': 0.04223520154, 'B_hit_distance_sc_7': 0.04213173751, 'P_hit_distance_sc_14': 0.04051098632, 'pitch_pct_EP': 0.04016871102,
    'B_pitch_pct_FS_5': 0.03855898953, 'B_max_ev_14': 0.03737112154, 'P_hit_distance_sc_5': 0.03624982534, 'B_pitch_pct_ST_7': 0.03422107548,
    'P_pitch_pct_FA_14': 0.03370091448, 'P_pitch_pct_SI_3': 0.0330997414, 'P_pitch_pct_SC_3': 0.0323674025, 'P_pitch_pct_FA_5': 0.03217237942,
    'P_pitch_pct_FA_7': 0.03040455729, 'B_pitch_pct_CS_7': 0.03032370172, 'P_pitch_pct_FS_14': 0.02975351665, 'P_pitch_pct_CH_5': 0.02916198552,
    'P_launch_angle_3': 0.02898747384, 'api_break_x_arm': 0.02580362561, 'P_pitch_pct_KN_14': 0.02437049636, 'B_launch_angle_5': 0.02279752661,
    'P_pitch_pct_KN_7': 0.02198258989, 'pitch_pct_FA': 0.02116715063, 'P_pitch_pct_CS_3': 0.02028996709, 'P_launch_speed_5': 0.02025459348,
    'B_pitch_pct_FS_7': 0.01914504011, 'B_iso_value_3': 0.01880782539, 'B_pitch_pct_KN_3': 0.01866745556, 'P_pitch_pct_KN_5': 0.01832716716,
    'P_pitch_pct_EP_5': 0.01796925344, 'P_pitch_pct_FA_3': 0.01765343611, 'B_pitch_pct_SC_3': 0.01704581005, 'B_woba_value_14': 0.01625324119,
    'P_pitch_pct_CH_7': 0.0159123482, 'P_iso_value_5': 0.01527328467, 'P_pitch_pct_SC_5': 0.01459290553, 'P_pitch_pct_EP_7': 0.01245969926,
    'P_pitch_pct_EP_14': 0.01239925854, 'B_pitch_pct_KN_5': 0.01238071672, 'P_pitch_pct_KN_3': 0.01108048233, 'B_pitch_pct_FO_5': 0.009812235107,
    'B_pitch_pct_CH_14': 0.006891595653, 'P_launch_speed_7': 0.00587642924, 'P_pitch_pct_SI_5': 0.004140622116, 'P_iso_value_3': 0.003317464841,
    'B_pitch_pct_SC_7': 0.002973730736, 'P_pitch_pct_SI_7': 0.002545559744, 'B_pitch_pct_CS_5': 0.001400601092, 'P_pitch_pct_CU_5': 4.06E-05,
    'pitch_pct_SC': 0, 'pitch_pct_KN': 0, 'B_pitch_pct_FO_7': -0.0007714881065, 'B_pitch_pct_SV_14': -0.001823116203, 'P_pitch_pct_SC_14': -0.00383505602,
    'B_pitch_pct_FO_3': -0.005145249883, 'P_pitch_pct_SC_7': -0.005206073559, 'P_median_ev_14': -0.005612233359, 'P_pitch_pct_CS_5': -0.006176652001,
    'B_pitch_pct_FA_5': -0.006331705902, 'B_hit_distance_sc_5': -0.006943213796, 'B_pitch_pct_SC_5': -0.01205200704, 'B_pitch_pct_FC_14': -0.0143497921,
    'pitch_pct_SL': -0.01510900719, 'P_rolling_hr_rate_14': -0.01595557821, 'P_pitch_pct_KC_5': -0.02349274133, 'B_pitch_pct_CH_7': -0.02379703807,
    'P_pitch_pct_FS_7': -0.02455482911, 'P_median_ev_5': -0.025323513, 'P_launch_angle_7': -0.02534123972, 'B_pitch_pct_KN_14': -0.02610112462,
    'B_hit_distance_sc_3': -0.02717841116, 'P_pitch_pct_SI_14': -0.02729122758, 'B_pitch_pct_FC_3': -0.02743044863, 'P_pitch_pct_FC_7': -0.02813827819,
    'api_break_x_batter_in': -0.03119408952, 'P_pitch_pct_SV_7': -0.03257910372, 'P_pitch_pct_FF_3': -0.03271543951, 'B_iso_value_5': -0.03577165976,
    'P_pitch_pct_KC_14': -0.04097640307, 'B_pitch_pct_FO_14': -0.04146002355, 'B_pitch_pct_CU_5': -0.04183288301, 'B_pitch_pct_SI_3': -0.04391242165,
    'P_pitch_pct_SL_7': -0.04539241009, 'P_pitch_pct_FS_5': -0.04720379886, 'B_pitch_pct_SV_3': -0.04966895579, 'pitch_pct_CU': -0.05060363486,
    'B_pitch_pct_CU_7': -0.05188060492, 'B_woba_value_7': -0.05352041865, 'B_median_ev_5': -0.05445014211, 'P_pitch_pct_FO_3': -0.05565757264,
    'B_pitch_pct_SI_5': -0.05597580306, 'B_launch_speed_3': -0.05725070846, 'P_pitch_pct_FC_5': -0.05818885708, 'P_pitch_pct_KC_3': -0.0609302114,
    'B_pitch_pct_FF_5': -0.07026136308, 'P_pitch_pct_FF_7': -0.07054096955, 'P_median_ev_7': -0.07249860479, 'P_pitch_pct_SV_14': -0.07401508716,
    'pitch_pct_CS': -0.0750167943, 'P_pitch_pct_SL_5': -0.07516276101, 'P_pitch_pct_FO_5': -0.07608663534, 'B_launch_angle_14': -0.07802255023,
    'B_launch_speed_7': -0.0805917092, 'P_pitch_pct_SL_14': -0.08198276207, 'P_pitch_pct_FO_7': -0.08472122915, 'B_pitch_pct_SI_7': -0.08603262392,
    'B_pitch_pct_FF_3': -0.08711461233, 'B_pitch_pct_KC_14': -0.08749208913, 'B_hit_distance_sc_14': -0.08941394423, 'B_pitch_pct_SL_5': -0.08944923535,
    'B_woba_value_5': -0.09131252715, 'P_launch_angle_14': -0.09339176458, 'B_pitch_pct_KC_3': -0.09380795248, 'woba_value': -0.09486853811,
    'B_max_ev_7': -0.09570913208, 'B_pitch_pct_SL_7': -0.09651967847, 'P_pitch_pct_CH_14': -0.09732621466, 'P_pitch_pct_ST_3': -0.0983690184,
    'pitch_pct_KC': -0.09887480393, 'pitch_pct_FS': -0.09913272207, 'P_pitch_pct_FC_3': -0.1007290791, 'P_pitch_pct_FF_14': -0.1041925119,
    'B_launch_speed_14': -0.1044138585, 'B_launch_speed_5': -0.1047430693, 'P_hit_distance_sc_3': -0.1095165747, 'pitch_pct_SV': -0.1135124529,
    'P_launch_speed_14': -0.1154621075, 'B_launch_angle_3': -0.1182057938, 'P_iso_value_14': -0.1239357984, 'B_pitch_pct_SV_7': -0.1243399089,
    'P_woba_value_14': -0.1300451803, 'B_pitch_pct_FS_14': -0.1308265914, 'intercept_ball_minus_batter_pos_x_inches': -0.1311741983,
    'B_pitch_pct_ST_5': -0.1317139074, 'B_pitch_pct_ST_14': -0.1365470891, 'P_max_ev_14': -0.1365872313, 'park_hr_rate': -0.1393701286,
    'B_pitch_pct_CS_3': -0.1396998173, 'B_pitch_pct_FC_5': -0.1420174063, 'park_altitude': -0.1571715354, 'B_median_ev_3': -0.1597647764,
    'P_pitch_pct_ST_5': -0.1644178515, 'pull_air': -0.1646662335, 'launch_speed': -0.1671652738, 'pitch_pct_SI': -0.1750704955,
    'B_pitch_pct_EP_3': -0.1803807091, 'flyball': -0.183110744, 'B_woba_value_3': -0.1878784838, 'B_pitch_pct_EP_14': -0.187883453,
    'P_pitch_pct_CU_3': -0.1965025187, 'P_woba_value_3': -0.2020609691, 'P_launch_speed_3': -0.2064394068, 'B_pitch_pct_CU_14': -0.2071642082,
    'launch_angle': -0.2249806408, 'B_pitch_pct_CH_3': -0.2283291737, 'is_hard_hit': -0.4349162272, 'is_sweet_spot': -0.5773584022,
    'line_drive': -0.7114540736
}
INTERCEPT = 0

# ------------ PROGRESS BAR ------------
progress = st.progress(0, text="Starting...")

# ------------ FILE UPLOADS ------------
st.sidebar.header("Step 1: Upload Data")
player_csv = st.sidebar.file_uploader("Player-Level CSV (required)", type=["csv"], key="player")
event_csv = st.sidebar.file_uploader("Event-Level CSV (required)", type=["csv"], key="event")
progress.progress(10, text="Waiting for both player-level and event-level CSVs...")

if not player_csv or not event_csv:
    st.warning("⬆️ Upload **both** player-level and event-level CSVs to begin!")
    st.stop()

player_df = pd.read_csv(player_csv)
event_df = pd.read_csv(event_csv)
progress.progress(25, text="Files uploaded, reading player/event CSVs...")

# ------------ WEATHER LOOKUP (event_df -> park/city) ------------
def fetch_weather(city, date_str):
    try:
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"
        resp = requests.get(url, timeout=7)
        data = resp.json()
        day = data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        temp = day.get("avgtemp_f", 72)
        humidity = day.get("avghumidity", 55)
        wind_mph = day.get("maxwind_mph", 7)
        return temp, humidity, wind_mph
    except Exception:
        return 72, 55, 7  # fallback/defaults

# Use city if present in event_df, otherwise park, otherwise manual input.
if "city" in event_df.columns and not event_df.empty:
    city = event_df["city"].mode()[0]
elif "park" in event_df.columns and not event_df.empty:
    city = event_df["park"].mode()[0]
else:
    city = st.sidebar.text_input("Ballpark/City for Weather Lookup", "Baltimore,MD")

date_guess = event_df["game_date"].mode()[0] if "game_date" in event_df.columns and not event_df.empty else datetime.now().strftime("%Y-%m-%d")

weather_temp, weather_humidity, weather_wind_mph = fetch_weather(city, date_guess)
player_df["weather_temp"] = weather_temp
player_df["weather_humidity"] = weather_humidity
player_df["weather_wind_mph"] = weather_wind_mph
progress.progress(40, text="Weather loaded and merged...")

# ------------ FILL MISSING FEATURES WITH ZEROS ------------
for col in LOGISTIC_WEIGHTS:
    if col not in player_df.columns:
        player_df[col] = 0
progress.progress(55, text="Model features prepared...")

# ------------ LOGIT CALCULATION ------------
def calc_hr_logit(row):
    score = INTERCEPT
    for feat, wt in LOGISTIC_WEIGHTS.items():
        score += wt * row.get(feat, 0)
    return score

player_df["HR Logit Score"] = player_df.apply(calc_hr_logit, axis=1)
player_df["HR Probability"] = 1 / (1 + np.exp(-player_df["HR Logit Score"]))
progress.progress(70, text="Logistic scoring complete...")

# ------------ MLB PLAYER NAME LOOKUP BY MLB ID ------------
@st.cache_data(ttl=86400)
def get_mlb_player_names(mlb_id_list):
    out_map = {}
    ids = [int(i) for i in mlb_id_list if not pd.isna(i)]
    CHUNK = 100
    for i in range(0, len(ids), CHUNK):
        id_str = ",".join(str(j) for j in ids[i:i+CHUNK])
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={id_str}"
        try:
            resp = requests.get(url, timeout=7)
            data = resp.json().get("people", [])
            for p in data:
                out_map[str(p["id"])] = p.get("fullName", "")
        except Exception:
            continue
    return out_map

unique_ids = player_df["batter_id"].dropna().astype(int).unique().tolist()
name_map = get_mlb_player_names(unique_ids)
player_df["Player"] = player_df["batter_id"].astype(int).astype(str).map(name_map).fillna(player_df["batter_id"])
progress.progress(85, text="Player names resolved using MLB API...")

# ------------ LEADERBOARD: ONE ROW PER BATTER ------------
player_df = player_df.copy()
player_df = player_df.drop_duplicates(subset="batter_id")  # one row per batter

player_df = player_df.sort_values("HR Logit Score", ascending=False).reset_index(drop=True)
player_df["Rank"] = np.arange(1, len(player_df) + 1)
progress.progress(95, text="Leaderboard ranked...")

# ------------ HIGHEST WEIGHTED FEATURES FOR EACH BATTER ------------
TOP_N_FEATS = 10

def get_top_feats(row):
    feat_scores = []
    for feat, wt in LOGISTIC_WEIGHTS.items():
        val = row.get(feat, 0)
        contrib = wt * val
        feat_scores.append((feat, val, wt, contrib))
    # Sort by absolute value of contribution (impact on score)
    feat_scores.sort(key=lambda x: abs(x[3]), reverse=True)
    return feat_scores[:TOP_N_FEATS]

progress.progress(100, text="Leaderboard complete! 🎉")

st.success(f"Done! Leaderboard generated for {date_guess} ({city}) with {len(player_df)} batters.")

st.header("Predicted Home Run Leaderboard")
leaderboard_cols = [
    "Rank", "Player", "batter_id", "HR Logit Score", "HR Probability",
    "weather_temp", "weather_humidity", "weather_wind_mph"
]
for col in ["iso_value", "hit_distance_sc", "pull_side", "launch_speed_angle", "B_pitch_pct_CH_5", "park_handed_hr_rate"]:
    if col in player_df.columns and col not in leaderboard_cols:
        leaderboard_cols.append(col)

st.dataframe(player_df[leaderboard_cols].head(15), use_container_width=True)
st.caption("Sorted by HR Logit Score (most predictive for home runs). Player names are from MLB.com API.")

# ------------ SHOW TOP FEATURES PER PLAYER ------------
st.header("🔍 Top Features by Contribution (per player)")
with st.expander("Show Top Weighted Features for Each Player in Leaderboard"):
    for idx, row in player_df.head(15).iterrows():
        st.subheader(f"{int(row['Rank'])}. {row['Player']} ({row['batter_id']})")
        feat_scores = get_top_feats(row)
        feat_table = pd.DataFrame(feat_scores, columns=["Feature", "Value", "Weight", "Score Contribution"])
        feat_table["Score Contribution"] = feat_table["Score Contribution"].round(4)
        st.dataframe(feat_table, use_container_width=True)

st.caption(
    "• Top features are shown by largest score impact (weight × value). Logistic score determines probability of HR. "
    "Upload new CSVs to refresh leaderboard."
)
