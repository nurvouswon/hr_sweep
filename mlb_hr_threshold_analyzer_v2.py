import streamlit as st
import pandas as pd

st.title("MLB HR Feature Debugger & Merge Inspector")

# --- File Uploaders
st.header("Upload Files")
event_file = st.file_uploader("Upload Event-Level Parquet", type=["parquet"])
today_file = st.file_uploader("Upload TODAY CSV", type=["csv"])

if event_file and today_file:
    df_event = pd.read_parquet(event_file)
    df_today = pd.read_csv(today_file)

    st.write("Event-level shape:", df_event.shape)
    st.write("TODAY shape:", df_today.shape)
    
    # Guess merge keys (update as needed)
    possible_keys = ['game_date', 'batter_id']
    st.write("Default merge keys:", possible_keys)
    
    # --- Check for duplicates
    event_dups = df_event.duplicated(subset=possible_keys).sum()
    today_dups = df_today.duplicated(subset=possible_keys).sum()
    st.write(f"Event-level duplicates on {possible_keys}: {event_dups}")
    st.write(f"TODAY duplicates on {possible_keys}: {today_dups}")

    # --- Group event-level by merge keys if there are dups
    if event_dups > 0:
        st.warning("Duplicates found in Event-level. Aggregating (using first row per key)...")
        df_event_grouped = df_event.groupby(possible_keys).first().reset_index()
    else:
        df_event_grouped = df_event.copy()

    # --- Data type & unique check for keys
    st.subheader("Key Diagnostics")
    for key in possible_keys:
        st.write(f"TODAY {key} dtype:", df_today[key].dtype)
        st.write(f"EVENT {key} dtype:", df_event_grouped[key].dtype)
        st.write(f"Sample TODAY {key}:", df_today[key].unique()[:5])
        st.write(f"Sample EVENT {key}:", df_event_grouped[key].unique()[:5])

    # --- Try to align dtypes if needed
    for key in possible_keys:
        # Try to convert to string if either side is object/string
        if df_today[key].dtype != df_event_grouped[key].dtype:
            try:
                df_today[key] = df_today[key].astype(str)
                df_event_grouped[key] = df_event_grouped[key].astype(str)
                st.info(f"Auto-converted {key} to string on both files for merge.")
            except:
                pass
        # Also try to align dates (if parseable)
        if 'date' in key:
            df_today[key] = pd.to_datetime(df_today[key], errors='coerce').dt.date
            df_event_grouped[key] = pd.to_datetime(df_event_grouped[key], errors='coerce').dt.date

    # --- Column overlap check
    today_cols = set(df_today.columns)
    event_cols = set(df_event_grouped.columns)
    only_in_today = list(today_cols - event_cols)
    only_in_event = list(event_cols - today_cols)
    in_both = list(today_cols & event_cols)
    st.write("Columns in TODAY but NOT in Event-level:", only_in_today)
    st.write("Columns in Event-level but NOT in TODAY (showing first 100):", only_in_event[:100])
    st.write("Columns in BOTH (showing first 20):", in_both[:20])

    # --- Merge and diagnose nulls
    merged = df_today.merge(df_event_grouped, on=possible_keys, how='left', suffixes=('', '_event'))
    st.write("Merged shape:", merged.shape)

    # Columns ALL NULL after merge
    all_null_cols = [col for col in merged.columns if merged[col].isnull().all()]
    st.write(f"Columns ALL NULL after merge (showing first 50): {all_null_cols[:50]}")
    st.write(f"Total columns ALL NULL after merge: {len(all_null_cols)}")

    # --- Find which TODAY rows failed to match EVENT (key-based)
    merged_ind = df_today.merge(df_event_grouped, on=possible_keys, how='left', indicator=True)
    missing_today_rows = merged_ind[merged_ind['_merge'] == 'left_only']
    st.write(f"Rows in TODAY not matched in EVENT-level (showing first 20):")
    st.dataframe(missing_today_rows.head(20))

    # --- Allow user to pick a column to inspect values in both files
    st.subheader("Pick column to debug (from TODAY)")
    debug_col = st.selectbox("Pick column", sorted(df_today.columns))
    st.write("Non-null in EVENT for", debug_col, ":", df_event_grouped[debug_col].notnull().sum() if debug_col in df_event_grouped else "N/A")
    st.write("Non-null in TODAY for", debug_col, ":", df_today[debug_col].notnull().sum())

    st.write("Sample EVENT rows for", debug_col, ":", df_event_grouped[[debug_col]].dropna().head() if debug_col in df_event_grouped else "N/A")
    st.write("Sample TODAY rows for", debug_col, ":", df_today[[debug_col]].dropna().head())

else:
    st.info("Upload both Event-level Parquet and TODAY CSV to begin debug.")
