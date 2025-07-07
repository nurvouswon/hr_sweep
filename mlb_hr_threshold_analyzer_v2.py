import streamlit as st
import pandas as pd

st.title("MLB HR Feature File Diagnostic")

# Uploaders
event_file = st.file_uploader("Upload event-level Parquet file", type=["parquet"])
today_file = st.file_uploader("Upload TODAY CSV", type=["csv"])

if event_file and today_file:
    # Load files
    df_event = pd.read_parquet(event_file)
    df_today = pd.read_csv(today_file)

    # Merge keys
    merge_keys = ['game_date', 'batter_id']

    st.write("### File Shapes")
    st.write(f"Event-level shape: {df_event.shape}")
    st.write(f"TODAY shape: {df_today.shape}")

    # Key detection
    key_status = []
    for k in merge_keys:
        if k not in df_event.columns or k not in df_today.columns:
            key_status.append(f"WARNING: Key column '{k}' not found in both dataframes.")
    if key_status:
        st.write("#### Merge Key Warnings")
        for msg in key_status:
            st.warning(msg)

    # Columns in TODAY not in EVENT
    cols_today_not_event = [col for col in df_today.columns if col not in df_event.columns]
    st.write("### Columns in TODAY but NOT in EVENT-level:")
    st.code(cols_today_not_event)

    # Columns in EVENT not in TODAY
    cols_event_not_today = [col for col in df_event.columns if col not in df_today.columns]
    st.write(f"### Columns in EVENT but NOT in TODAY: (showing first 100 of {len(cols_event_not_today)})")
    st.code(cols_event_not_today[:100])

    # Columns present in BOTH
    cols_both = [col for col in df_today.columns if col in df_event.columns]
    st.write(f"### Columns present in BOTH: (showing first 100 of {len(cols_both)})")
    st.code(cols_both[:100])

    # Merge diagnostic
    st.write("### Checking merge key uniqueness...")
    dups_event = df_event.duplicated(subset=merge_keys).sum()
    dups_today = df_today.duplicated(subset=merge_keys).sum()
    st.write(f"Event-level dups by key: {dups_event}")
    st.write(f"TODAY dups by key: {dups_today}")

    if dups_event > 0:
        st.warning("WARNING: Duplicate keys found in EVENT. Aggregation recommended.")

    # Merge and check nulls
    merged = df_today.merge(
        df_event[merge_keys + cols_both],
        on=merge_keys,
        how='left',
        suffixes=('', '_event')
    )

    all_null_cols = [col for col in cols_both if merged[col].isnull().all()]
    st.write("### Columns in TODAY that are ALL NULL after merge:")
    st.code(all_null_cols)
