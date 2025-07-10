import streamlit as st
import pandas as pd
import io

st.set_page_config("🟣 Parquet Reader", layout="wide")

st.title("🟣 Parquet File Reader & Inspector")

# --- Parquet File Upload ---
parquet_file = st.file_uploader("Upload a Parquet file", type=["parquet"])

if parquet_file is not None:
    st.info("⚡ [DEBUG] About to read uploaded Parquet...")
    try:
        df = pd.read_parquet(parquet_file)
        st.success(f"[DEBUG] Successfully loaded Parquet! Shape: {df.shape}")
        
        # Show diagnostic info
        st.write("**[Diagnostics] DataFrame Info:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("**[Diagnostics] First 20 rows:**")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download sample as CSV
        sample_csv = df.head(100).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Top 100 as CSV", sample_csv, file_name="sample_top100.csv")
        
        # Show all columns, chunked by 50
        st.write("**[Diagnostics] Column List:**")
        col_chunks = [df.columns[i:i+50] for i in range(0, len(df.columns), 50)]
        for i, chunk in enumerate(col_chunks):
            st.text(f"[Columns {i*50}-{i*50 + len(chunk) - 1}]: " + ", ".join(chunk))
            
    except Exception as e:
        st.error(f"[ERROR] Could not load Parquet file: {e}")

else:
    st.info("Upload a Parquet file to inspect its structure.")
