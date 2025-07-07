import pandas as pd
import streamlit as st

# --- Uploaders ---
event_file = st.file_uploader("Upload event-level Parquet", type=["parquet"])
today_file = st.file_uploader("Upload TODAY CSV", type=["csv"])

if event_file and today_file:
    df_event = pd.read_parquet(event_file)
    df_today = pd.read_csv(today_file)
    
    event_cols = set(df_event.columns)
    today_cols = set(df_today.columns)

    # --- 1. Compare Headers ---
    missing_from_event = sorted(today_cols - event_cols)
    missing_from_today = sorted(event_cols - today_cols)
    common = sorted(today_cols & event_cols)

    st.write("## Columns in TODAY but NOT in Event-level:", missing_from_event)
    st.write("## Columns in Event-level but NOT in TODAY:", missing_from_today)
    st.write("## Columns present in BOTH:", common[:50], "…", len(common), "total")
    
    # --- 2. Diagnose ALL NULL columns in TODAY ---
    all_null_cols = [col for col in df_today.columns if df_today[col].isnull().all()]
    st.write("## Columns in TODAY that are ALL NULL after merge:", all_null_cols)

    # --- 3. (Optional) List shape and sample rows for quick eyeball ---
    st.write("Event-level Parquet shape:", df_event.shape)
    st.write("TODAY CSV shape:", df_today.shape)
    st.write("First 3 rows of TODAY:", df_today.head(3))
else:
    st.info("Upload both event-level Parquet and TODAY CSV to compare columns.")
