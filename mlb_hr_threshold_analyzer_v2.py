import streamlit as st
import pandas as pd

st.title("🔍 Parquet File Explorer (Column Names, Dtypes, Sample Rows)")

parquet_file = st.file_uploader("Upload Parquet File (.parquet)", type=["parquet"])
if parquet_file is not None:
    try:
        df = pd.read_parquet(parquet_file)
        st.success(f"Loaded file! Shape: {df.shape}")
        st.write("### Columns:")
        st.write(list(df.columns))

        st.write("### Dtypes:")
        st.write(df.dtypes.astype(str))

        st.write("### Sample (first 10 rows):")
        st.dataframe(df.head(10))

        if st.checkbox("Show Unique Value Counts for All Columns (slow!)"):
            st.write(df.nunique())

        col = st.selectbox("Pick a column to explore:", list(df.columns))
        if col:
            st.write(f"#### Unique values in `{col}`:")
            st.write(df[col].unique())
            st.write(f"#### Value counts for `{col}`:")
            st.write(df[col].value_counts(dropna=False).reset_index().rename(columns={'index': col, col:'count'}))

        if st.checkbox("Show entire DataFrame (⚠️ SLOW for big files)"):
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load parquet: {e}")
