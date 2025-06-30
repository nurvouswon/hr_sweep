import streamlit as st
import pandas as pd

st.title("🧹 Parquet Duplicate Column Fixer")

uploaded_file = st.file_uploader("Upload Parquet File to Fix", type=["parquet"])
if uploaded_file is not None:
    try:
        # Trick: read columns using pyarrow to inspect duplicates, then reload via pandas.
        import pyarrow.parquet as pq
        table = pq.read_table(uploaded_file)
        cols = list(table.schema.names)
        st.write("All column names:", cols)
        from collections import Counter
        counts = Counter(cols)
        dupes = [c for c, n in counts.items() if n > 1]
        if not dupes:
            st.success("No duplicate columns found!")
        else:
            st.warning(f"Duplicate columns: {dupes}")
            # Now reload with pandas, dropping later duplicates
            df = pd.read_parquet(uploaded_file)
            df = df.loc[:, ~df.columns.duplicated()]
            st.write("Fixed columns:", list(df.columns))
            st.write("Sample head:", df.head(5))
            # Option to download
            outbuf = st.download_button(
                "⬇️ Download Cleaned Parquet",
                data=df.to_parquet(index=False),
                file_name="fixed_"+uploaded_file.name,
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"Error: {e}")
