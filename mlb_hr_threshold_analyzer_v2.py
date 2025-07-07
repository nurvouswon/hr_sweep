import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("MLB HR Feature Diagnostic App", layout="wide")
st.header("MLB HR Feature Debugger & Merge Inspector")

# Upload files
col1, col2 = st.columns(2)
with col1:
    event_parquet = st.file_uploader("Upload Event-Level Parquet", type=["parquet"])
with col2:
    today_csv = st.file_uploader("Upload TODAY CSV", type=["csv"])

if event_parquet and today_csv:
    df_event = pd.read_parquet(event_parquet)
    df_today = pd.read_csv(today_csv)

    st.write(f"Event-level shape: {df_event.shape}")
    st.write(f"TODAY shape: {df_today.shape}")

    # Show columns present only in one, and in both
    event_cols = set(df_event.columns)
    today_cols = set(df_today.columns)
    in_today_not_event = sorted(list(today_cols - event_cols))
    in_event_not_today = sorted(list(event_cols - today_cols))
    in_both = sorted(list(event_cols & today_cols))

    st.markdown("**Columns in TODAY but NOT in Event-level:**")
    st.code(in_today_not_event)
    st.markdown("**Columns in Event-level but NOT in TODAY:**")
    st.code(in_event_not_today[:100])  # Only show first 100 if too many

    st.markdown("**Columns in BOTH:**")
    st.code(in_both[:20])  # Only show first 20

    # Merge on game_date + batter_id (as string)
    key_cols = ['game_date', 'batter_id']
    for key in key_cols:
        if key in df_event.columns:
            df_event[key] = df_event[key].astype(str)
        if key in df_today.columns:
            df_today[key] = df_today[key].astype(str)

    merged = pd.merge(df_today, df_event, on=key_cols, how='left', suffixes=('_today', '_event'))
    st.write(f"Merged shape: {merged.shape}")

    # Columns all NULL after merge
    null_cols = merged.columns[merged.isnull().all()]
    st.write(f"Total columns ALL NULL after merge: {len(null_cols)}")
    st.code(list(null_cols)[:50])  # Show first 50

    # Select column for diagnostic
    sel_col = st.selectbox("Pick column to debug (from TODAY)", sorted(list(df_today.columns)))
    st.write(f"Non-null in EVENT for {sel_col} :", df_event[sel_col].notnull().sum() if sel_col in df_event.columns else "N/A")
    st.write(f"Non-null in TODAY for {sel_col} :", df_today[sel_col].notnull().sum() if sel_col in df_today.columns else "N/A")

    st.markdown("**Sample EVENT rows for this column:**")
    st.dataframe(df_event[[sel_col]].dropna().head(10) if sel_col in df_event.columns else pd.DataFrame(), use_container_width=True)
    st.markdown("**Sample TODAY rows for this column:**")
    st.dataframe(df_today[[sel_col]].dropna().head(10), use_container_width=True)

    # Rows in TODAY not matched in EVENT-level
    merged_nulls = merged[merged[sel_col].isnull()]
    st.write(f"Rows in TODAY not matched in EVENT-level (showing first 20):")
    st.dataframe(merged_nulls[key_cols + [sel_col]].head(20), use_container_width=True)
else:
    st.info("Please upload both Event-Level Parquet and TODAY CSV files.")
