import streamlit as st
import pandas as pd

st.header("MLB HR Feature Debugger")

event_file = st.file_uploader("Upload Event-Level Parquet", type=["parquet"])
today_file = st.file_uploader("Upload TODAY CSV", type=["csv"])

if event_file and today_file:
    df_event = pd.read_parquet(event_file)
    df_today = pd.read_csv(today_file)

    st.write("Event-level shape:", df_event.shape)
    st.write("TODAY shape:", df_today.shape)

    TARGET_COL = st.selectbox("Pick column to debug (from TODAY)", df_today.columns)
    
    # Print non-null counts
    st.write(f"Non-null in EVENT for `{TARGET_COL}`:", df_event.get(TARGET_COL, pd.Series(dtype=float)).notnull().sum() if TARGET_COL in df_event else "N/A")
    st.write(f"Non-null in TODAY for `{TARGET_COL}`:", df_today[TARGET_COL].notnull().sum())

    # Print sample batter_ids
    key = st.selectbox("Key column", ["batter_id", "batter", "player_name"])
    if TARGET_COL in df_event:
        st.write("Sample EVENT rows with data for", TARGET_COL)
        st.dataframe(df_event.loc[df_event[TARGET_COL].notnull(), [key, TARGET_COL]].head())
    st.write("Sample TODAY rows for", TARGET_COL)
    st.dataframe(df_today[[key, TARGET_COL]].head())

    # Optional: Test merge logic
    merge_cols = ['game_date', key]  # adjust as needed
    if (TARGET_COL in df_event) and (key in df_event.columns and key in df_today.columns):
        tmp_event = df_event[merge_cols + [TARGET_COL]].drop_duplicates(subset=merge_cols)
        test_merge = df_today.merge(tmp_event, on=merge_cols, how='left', suffixes=('', '_event'))
        st.write("Merge check (TODAY vs EVENT):")
        st.dataframe(test_merge[[TARGET_COL, f"{TARGET_COL}_event"]].head(10))
else:
    st.info("Upload both files to begin.")
