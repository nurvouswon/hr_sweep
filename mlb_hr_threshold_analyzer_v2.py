import streamlit as st
import pandas as pd

st.set_page_config("Parquet Column Viewer", layout="wide")

st.title("Parquet File Column Viewer")

# File uploader for Parquet
parquet_file = st.file_uploader("Upload a Parquet file", type=["parquet"])

if parquet_file:
    df = pd.read_parquet(parquet_file)
    st.success(f"Loaded Parquet file with shape: {df.shape}")
    st.write("### Column Headers:")
    st.dataframe(pd.DataFrame({"Column": df.columns}))
    
    # Also show as a plain list for copy/paste
    st.write("### Columns as Python list")
    st.code(list(df.columns))
    
    # Optional: Download the headers as a .txt file
    import io
    output = io.StringIO()
    output.write('\n'.join(df.columns))
    st.download_button(
        label="Download column headers as .txt",
        data=output.getvalue(),
        file_name="parquet_headers.txt",
        mime="text/plain"
    )
else:
    st.info("Upload a Parquet file to see its columns.")
