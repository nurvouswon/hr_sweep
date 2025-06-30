import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Upload Parquet", type=["parquet"])
if uploaded_file:
    df = pd.read_parquet(uploaded_file)
    st.write("Columns:", list(df.columns))
    if "hr_outcome" in df.columns:
        st.success("✅ 'hr_outcome' column found!")
        st.write(df["hr_outcome"].value_counts())
    else:
        st.error("❌ 'hr_outcome' column NOT found!")
