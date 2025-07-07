import streamlit as st
import pandas as pd

st.title("Parquet File Uploader & Pitch Type Inspector")

# File uploader
parquet_file = st.file_uploader("Upload your Parquet file", type=["parquet"])

if parquet_file:
    df = pd.read_parquet(parquet_file)
    st.write(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show pitch types
    if "pitch_type" in df.columns:
        pitch_types = df["pitch_type"].dropna().unique()
        st.write("Unique pitch types:", pitch_types)

        selected_pitch_type = st.selectbox("Select a pitch type to preview rows:", pitch_types)
        st.write(df[df["pitch_type"] == selected_pitch_type].head(20))
    else:
        st.write("No 'pitch_type' column found in this file.")

    # DataFrame preview
    st.write("Full DataFrame preview:")
    st.dataframe(df.head(100))
