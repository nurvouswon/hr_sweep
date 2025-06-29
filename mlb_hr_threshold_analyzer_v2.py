import streamlit as st
import pandas as pd

file = st.file_uploader("Upload Parquet", type="parquet")
if file is not None:
    df = pd.read_parquet(file)
    st.write(df.info())
    st.dataframe(df.head(20))
