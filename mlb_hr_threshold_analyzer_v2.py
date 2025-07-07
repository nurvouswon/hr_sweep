import streamlit as st
import pandas as pd
import io

st.set_page_config("Parquet Preview Tool", layout="wide")
st.title("📝 Parquet File Previewer")

uploaded_file = st.file_uploader("Upload a Parquet file", type="parquet")

if uploaded_file:
    df = pd.read_parquet(uploaded_file)
    st.write("**File shape:**", df.shape)
    st.dataframe(df.head(50), use_container_width=True)

    # Download CSV sample
    csv_sample = df.head(20).to_csv(index=False)
    st.download_button(
        label="⬇️ Download Top 20 Rows as CSV",
        data=csv_sample,
        file_name="sample_top20.csv",
        mime="text/csv"
    )
else:
    st.info("Upload a Parquet file to preview its data.")
